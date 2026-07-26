"""Embeddings e reranking no Triton (gRPC) — zero forward pass em Python.

Divisão de trabalho:
  * tokenização  -> `tokenizers` (Rust, libera a GIL em `encode_batch`,
                    paraleliza internamente por rayon);
  * padding      -> NumPy, uma alocação + fancy indexing, sem loop por item;
  * forward      -> Triton, TensorRT/ONNX engine, dynamic batching no servidor;
  * normalização -> NumPy vetorizado, in-place.

O Python neste arquivo só empacota buffers contíguos e espera o socket.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import tritonclient.grpc.aio as tgrpc
from numpy.typing import NDArray
from tokenizers import Tokenizer

_I64: Final = "INT64"
_FP32: Final = np.float32


def _pad_batch(
    tok: Tokenizer, texts: list[str], max_len: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Tokeniza e faz padding sem um único loop Python sobre tokens."""
    tok.enable_truncation(max_len)
    encs = tok.encode_batch(texts)  # Rust, paralelo, sem GIL

    lengths = np.fromiter((len(e.ids) for e in encs), dtype=np.int64, count=len(encs))
    width = int(lengths.max())

    ids = np.zeros((len(encs), width), dtype=np.int64)
    for row, enc in enumerate(encs):  # memcpy por linha; sem trabalho por token
        ids[row, : lengths[row]] = enc.ids

    # máscara por broadcast: (B,1) < (1,W) -> (B,W) em uma passada C
    mask = (np.arange(width, dtype=np.int64)[None, :] < lengths[:, None]).astype(np.int64)
    return ids, mask


class TritonEncoders:
    """Um cliente gRPC compartilhado para embedder e reranker."""

    __slots__ = ("_c", "_embed_model", "_rerank_model", "_tok", "_rtok", "_max_len", "_dim")

    def __init__(
        self,
        url: str,
        *,
        embed_model: str,
        rerank_model: str,
        tokenizer_path: str,
        rerank_tokenizer_path: str,
        dim: int,
        max_tokens: int = 512,
    ) -> None:
        self._c = tgrpc.InferenceServerClient(url=url, verbose=False)
        self._embed_model = embed_model
        self._rerank_model = rerank_model
        self._tok = Tokenizer.from_file(tokenizer_path)
        self._rtok = Tokenizer.from_file(rerank_tokenizer_path)
        self._max_len = max_tokens
        self._dim = dim

    async def aclose(self) -> None:
        await self._c.close()

    # ------------------------------------------------------------------ #
    async def _infer(
        self, model: str, ids: NDArray[np.int64], mask: NDArray[np.int64], out_name: str
    ) -> NDArray[np.float32]:
        inputs = []
        for name, arr in (("input_ids", ids), ("attention_mask", mask)):
            t = tgrpc.InferInput(name, list(arr.shape), _I64)
            t.set_data_from_numpy(np.ascontiguousarray(arr))
            inputs.append(t)
        result = await self._c.infer(
            model_name=model,
            inputs=inputs,
            outputs=[tgrpc.InferRequestedOutput(out_name)],
        )
        return result.as_numpy(out_name)

    # ------------------------------------------------------------------ #
    async def embed(self, texts: list[str], *, query: bool = False) -> NDArray[np.float32]:
        """Retorna (N, dim) float32 L2-normalizado, pronto para `<=>`."""
        if not texts:
            return np.empty((0, self._dim), dtype=_FP32)
        if query:
            # BGE-M3 não exige prefixo de query; modelos e5/gte exigem.
            texts = [f"query: {t}" for t in texts]

        ids, mask = _pad_batch(self._tok, texts, self._max_len)
        out = await self._infer(self._embed_model, ids, mask, "sentence_embedding")

        vecs = np.ascontiguousarray(out, dtype=_FP32)
        if vecs.ndim == 3:  # o modelo devolveu last_hidden_state -> mean pooling
            m = mask.astype(_FP32)[:, :, None]
            vecs = (vecs * m).sum(1) / np.maximum(m.sum(1), 1e-9)

        # normalização in-place: nenhuma cópia intermediária
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        np.maximum(norms, 1e-12, out=norms)
        vecs /= norms
        return vecs

    async def rerank(self, query: str, passages: list[str]) -> NDArray[np.float32]:
        """Cross-encoder: (query, passage) -> score. Um batch, um round-trip."""
        if not passages:
            return np.empty(0, dtype=_FP32)

        self._rtok.enable_truncation(self._max_len)
        encs = self._rtok.encode_batch([(query, p) for p in passages])
        lengths = np.fromiter((len(e.ids) for e in encs), dtype=np.int64, count=len(encs))
        width = int(lengths.max())

        ids = np.zeros((len(encs), width), dtype=np.int64)
        types = np.zeros_like(ids)
        for row, enc in enumerate(encs):
            n = lengths[row]
            ids[row, :n] = enc.ids
            types[row, :n] = enc.type_ids
        mask = (np.arange(width, dtype=np.int64)[None, :] < lengths[:, None]).astype(np.int64)

        inputs = []
        for name, arr in (
            ("input_ids", ids),
            ("attention_mask", mask),
            ("token_type_ids", types),
        ):
            t = tgrpc.InferInput(name, list(arr.shape), _I64)
            t.set_data_from_numpy(np.ascontiguousarray(arr))
            inputs.append(t)
        result = await self._c.infer(
            model_name=self._rerank_model,
            inputs=inputs,
            outputs=[tgrpc.InferRequestedOutput("logits")],
        )

        logits = result.as_numpy("logits").astype(_FP32).reshape(len(passages), -1)[:, 0]
        # sigmoid vetorizado, estável para logit negativo grande
        return np.where(
            logits >= 0,
            1.0 / (1.0 + np.exp(-logits)),
            np.exp(logits) / (1.0 + np.exp(logits)),
        ).astype(_FP32)
