"""Ingestão: parse paralelo → chunking semântico vetorizado → COPY binário.

Onde a ingestão costuma morrer, e o que se faz aqui:

| Sintoma                              | Causa                          | Aqui                       |
|--------------------------------------|--------------------------------|----------------------------|
| 40 min para 5k PDFs                  | parse serial preso na GIL      | ProcessPoolExecutor        |
| chunking O(n²) em sentenças          | loop Python de similaridade    | einsum + percentile        |
| 200k INSERTs, WAL saturado           | executemany                    | COPY binário               |
| GPU a 8% de uso                      | embed de 1 chunk por vez       | batch grande + dynamic batching |

O único laço Python que sobra percorre *fronteiras de chunk* (dezenas),
nunca tokens (milhões).
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Final

import numpy as np
import orjson
import polars as pl
from numpy.typing import NDArray
from pgvector import HalfVector
from tokenizers import Tokenizer

from rag_graph.encoders import TritonEncoders
from rag_graph.store import VectorStore

# Fim de frase: pontuação + espaço + maiúscula/dígito. Evita abreviações
# comuns em documento corporativo brasileiro (art., inc., Ltda., R$).
_SENT_RE: Final = re.compile(
    r"(?<![A-Z][a-z]\.)(?<!\bart\.)(?<!\binc\.)(?<!\bLtda\.)(?<=[.!?;])\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ0-9])"
)
_EMBED_BATCH: Final = 256


# --------------------------------------------------------------------- #
# 1. parse (CPU-bound, sai da GIL via processos)
# --------------------------------------------------------------------- #
def _parse_file(path_str: str) -> tuple[str, str, bytes]:
    """Roda em subprocesso. Substitua o corpo pelo seu extrator (pymupdf etc.)."""
    path = Path(path_str)
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    return path_str, text, hashlib.sha256(raw).digest()


async def parse_many(paths: Sequence[Path], *, workers: int | None = None) -> pl.DataFrame:
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = await asyncio.gather(
            *(loop.run_in_executor(pool, _parse_file, str(p)) for p in paths)
        )
    return pl.DataFrame(
        {
            "uri": [r[0] for r in results],
            "text": [r[1] for r in results],
            "sha256": [r[2] for r in results],
        }
    )


# --------------------------------------------------------------------- #
# 2. chunking semântico — todo o custo em NumPy
# --------------------------------------------------------------------- #
def _boundaries(
    sims: NDArray[np.float32],
    token_counts: NDArray[np.int64],
    *,
    percentile: float,
    max_tokens: int,
) -> NDArray[np.int64]:
    """Fronteiras = quedas de similaridade + estouros de orçamento de tokens."""
    semantic = (
        np.flatnonzero(sims < np.percentile(sims, percentile)) + 1
        if sims.size
        else np.empty(0, dtype=np.int64)
    )

    # corte duro por tamanho: caminha no cumsum com searchsorted (busca
    # binária em C), não somando sentença a sentença em Python
    cum = np.cumsum(token_counts)
    hard: list[int] = []
    start = 0
    base = 0  # tokens já fechados nos chunks anteriores
    while start < cum.size:
        nxt = int(np.searchsorted(cum, base + max_tokens, side="right"))
        if nxt <= start:
            nxt = start + 1  # uma única sentença estoura o orçamento
        if nxt >= cum.size:
            break
        hard.append(nxt)
        base = int(cum[nxt - 1])
        start = nxt

    return np.unique(np.concatenate([semantic, np.asarray(hard, dtype=np.int64)])).astype(np.int64)


async def chunk_text(
    text: str,
    encoders: TritonEncoders,
    tokenizer: Tokenizer,
    *,
    percentile: float = 22.0,
    max_tokens: int = 480,
    overlap: int = 1,
) -> list[str]:
    sents = [s for s in _SENT_RE.split(text) if s.strip()]
    if len(sents) <= 1:
        return sents

    # embeddings de sentença em lotes grandes — o dynamic batching do Triton
    # agrega ainda mais entre requisições concorrentes
    mats = [
        await encoders.embed(sents[i : i + _EMBED_BATCH])
        for i in range(0, len(sents), _EMBED_BATCH)
    ]
    emb = np.vstack(mats)

    # vetores já são L2-normalizados: cosseno adjacente é um produto interno.
    # einsum faz N produtos em uma passada, sem matriz N×N.
    sims = np.einsum("ij,ij->i", emb[:-1], emb[1:], dtype=np.float32)

    tokenizer.enable_truncation(max_tokens)
    counts = np.fromiter(
        (len(e.ids) for e in tokenizer.encode_batch(sents)), dtype=np.int64, count=len(sents)
    )

    cuts = _boundaries(sims, counts, percentile=percentile, max_tokens=max_tokens)
    groups = np.split(np.arange(len(sents)), cuts)

    out: list[str] = []
    for gi, g in enumerate(groups):
        if g.size == 0:
            continue
        lo = max(0, int(g[0]) - overlap) if gi else int(g[0])  # janela de overlap
        out.append(" ".join(sents[lo : int(g[-1]) + 1]))
    return out


# --------------------------------------------------------------------- #
# 3. persistência
# --------------------------------------------------------------------- #
async def ingest(
    paths: Sequence[Path],
    *,
    tenant_id: str,
    encoders: TritonEncoders,
    store: VectorStore,
    tokenizer_path: str,
    meta: dict[str, Any] | None = None,
) -> int:
    tokenizer = Tokenizer.from_file(tokenizer_path)
    frame = await parse_many(paths)

    records: list[tuple[Any, ...]] = []
    for row in frame.iter_rows(named=True):  # laço por documento, não por token
        chunks = await chunk_text(row["text"], encoders, tokenizer)
        if not chunks:
            continue

        vecs = np.vstack(
            [
                await encoders.embed(chunks[i : i + _EMBED_BATCH])
                for i in range(0, len(chunks), _EMBED_BATCH)
            ]
        )
        doc_meta = orjson.dumps({"uri": row["uri"], **(meta or {})}).decode()
        doc_id = await store.upsert_document(tenant_id, row["uri"], row["sha256"], doc_meta)

        records.extend(
            (tenant_id, doc_id, ord_, text, doc_meta, HalfVector(vecs[ord_]))
            for ord_, text in enumerate(chunks)
        )

    if records:
        await store.copy_chunks(records)
    return len(records)


__all__ = ["chunk_text", "ingest", "parse_many"]
