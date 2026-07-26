"""Retrieval híbrido em pgvector — a fusão acontece dentro do Postgres.

O erro comum é trazer 60 candidatos densos e 60 lexicais para o Python e
fundir com um dict. Isso são duas viagens de rede, duas desserializações e
um loop interpretado por query. Aqui o RRF é um `FULL JOIN` + aritmética em
C, num único round-trip: o Python recebe só os 8 finais.

pgvector 0.8: `halfvec` corta índice e heap pela metade (fp16 basta para
ranquear; o reranker corrige a cauda) e `hnsw.iterative_scan` mantém o
recall quando há filtro por tenant.
"""

from __future__ import annotations

from typing import Any, Final, Self, TypedDict

import asyncpg
import numpy as np
import orjson
from numpy.typing import NDArray
from pgvector.asyncpg import register_vector


class Doc(TypedDict):
    id: str
    doc_id: str
    content: str
    score: float
    meta: dict[str, Any]


# Uma query. Duas pernas. Fusão em C.
_HYBRID: Final = """
WITH dense AS (
    SELECT id, row_number() OVER (ORDER BY embedding <=> $1) AS rnk
    FROM rag.chunk
    WHERE tenant_id = $2
    ORDER BY embedding <=> $1
    LIMIT $3
),
sparse AS (
    SELECT c.id, row_number() OVER (ORDER BY ts_rank_cd(c.tsv, q, 32) DESC) AS rnk
    FROM rag.chunk c, websearch_to_tsquery($6, $4) AS q
    WHERE c.tenant_id = $2 AND c.tsv @@ q
    ORDER BY ts_rank_cd(c.tsv, q, 32) DESC
    LIMIT $3
),
fused AS (
    SELECT
        coalesce(d.id, s.id) AS id,
        coalesce($7 / ($5 + d.rnk), 0.0) + coalesce($8 / ($5 + s.rnk), 0.0) AS score
    FROM dense d FULL JOIN sparse s ON d.id = s.id
)
SELECT c.id::text, c.doc_id::text, c.content, c.meta, f.score
FROM fused f
JOIN rag.chunk c ON c.id = f.id
ORDER BY f.score DESC
LIMIT $9
"""


async def _init_conn(conn: asyncpg.Connection) -> None:
    await register_vector(conn)  # codecs binários p/ vector/halfvec
    await conn.set_type_codec(
        "jsonb", encoder=orjson.dumps, decoder=orjson.loads, schema="pg_catalog", format="binary"
    )


class VectorStore:
    __slots__ = ("_pool", "_cfg", "_fts", "_ef")

    def __init__(self, pool: asyncpg.Pool, *, fts_config: str, ef_search: int) -> None:
        self._pool = pool
        self._fts = fts_config
        self._ef = ef_search

    @classmethod
    async def connect(
        cls,
        dsn: str,
        *,
        min_size: int,
        max_size: int,
        stmt_cache: int,
        fts_config: str,
        ef_search: int,
    ) -> Self:
        pool = await asyncpg.create_pool(
            dsn,
            min_size=min_size,
            max_size=max_size,
            max_inactive_connection_lifetime=300.0,
            statement_cache_size=stmt_cache,  # prepared statements reaproveitados
            init=_init_conn,
        )
        assert pool is not None
        return cls(pool, fts_config=fts_config, ef_search=ef_search)

    async def aclose(self) -> None:
        await self._pool.close()

    # ------------------------------------------------------------------ #
    async def hybrid_search(
        self,
        embedding: NDArray[np.float32],
        query_text: str,
        tenant_id: str,
        *,
        per_leg: int,
        rrf_k: int,
        w_dense: float,
        w_sparse: float,
        limit: int,
    ) -> list[Doc]:
        async with self._pool.acquire() as conn, conn.transaction():
            # SET LOCAL: escopo da transação, não vaza para a próxima conexão
            await conn.execute(f"SET LOCAL hnsw.ef_search = {self._ef}")
            await conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
            rows = await conn.fetch(
                _HYBRID,
                embedding,
                tenant_id,
                per_leg,
                query_text,
                float(rrf_k),
                self._fts,
                float(w_dense),
                float(w_sparse),
                limit,
            )
        return [
            Doc(
                id=r["id"],
                doc_id=r["doc_id"],
                content=r["content"],
                score=float(r["score"]),
                meta=r["meta"] or {},
            )
            for r in rows
        ]

    async def ping(self) -> bool:
        try:
            async with self._pool.acquire() as conn:
                return await conn.fetchval("SELECT 1") == 1
        except Exception:
            return False

    async def upsert_document(self, tenant_id: str, uri: str, sha256: bytes, meta: str) -> Any:
        """Idempotente por (tenant, sha256): reingerir o mesmo arquivo não duplica."""
        async with self._pool.acquire() as conn:
            return await conn.fetchval(
                """
                INSERT INTO rag.document (tenant_id, uri, sha256, meta)
                VALUES ($1, $2, $3, $4::jsonb)
                ON CONFLICT (tenant_id, sha256) DO UPDATE SET uri = EXCLUDED.uri
                RETURNING id
                """,
                tenant_id,
                uri,
                sha256,
                meta,
            )

    async def copy_chunks(self, records: list[tuple[Any, ...]]) -> None:
        """Ingestão via protocolo COPY binário — ~20x um `executemany` de INSERT."""
        async with self._pool.acquire() as conn:
            await conn.copy_records_to_table(
                "chunk",
                schema_name="rag",
                columns=["tenant_id", "doc_id", "ord", "content", "meta", "embedding"],
                records=records,
            )
