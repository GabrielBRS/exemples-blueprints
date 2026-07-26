-- pgvector >= 0.8
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE SCHEMA IF NOT EXISTS rag;

CREATE TABLE IF NOT EXISTS rag.document (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id  text        NOT NULL,
    uri        text        NOT NULL,
    sha256     bytea       NOT NULL,
    meta       jsonb       NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, sha256)          -- reingestão idempotente
);

CREATE TABLE IF NOT EXISTS rag.chunk (
    id        uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id text  NOT NULL,
    doc_id    uuid  NOT NULL REFERENCES rag.document(id) ON DELETE CASCADE,
    ord       int   NOT NULL,
    content   text  NOT NULL,
    meta      jsonb NOT NULL DEFAULT '{}'::jsonb,

    -- halfvec: fp16. Metade do heap e metade do índice; a perda de precisão
    -- some no reranking. Use vector(1024) se precisar de recall exato.
    embedding halfvec(1024) NOT NULL,

    -- coluna gerada: o tsvector nunca sai de sincronia com o content
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('portuguese', content)) STORED
);

-- Perna densa. m/ef_construction altos = build lento, recall alto: um índice
-- de RAG é escrito uma vez e lido um milhão de vezes.
CREATE INDEX IF NOT EXISTS chunk_embedding_hnsw
    ON rag.chunk USING hnsw (embedding halfvec_cosine_ops)
    WITH (m = 24, ef_construction = 200);

-- Perna lexical.
CREATE INDEX IF NOT EXISTS chunk_tsv_gin ON rag.chunk USING gin (tsv);

-- Filtro multi-tenant: com iterative_scan o HNSW aceita o predicado sem
-- colapsar o recall, mas o índice B-tree ainda ajuda a perna lexical.
CREATE INDEX IF NOT EXISTS chunk_tenant ON rag.chunk (tenant_id);
CREATE INDEX IF NOT EXISTS chunk_doc_ord ON rag.chunk (doc_id, ord);

-- Estatísticas: o planner precisa saber que tenant_id e doc_id correlacionam.
CREATE STATISTICS IF NOT EXISTS chunk_stats (dependencies)
    ON tenant_id, doc_id FROM rag.chunk;

-- Tabelas do checkpointer do LangGraph são criadas por
-- `AsyncPostgresSaver.setup()` no boot.
