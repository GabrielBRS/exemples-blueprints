-- Migration inicial: tabela de exemplo.
-- gen_random_uuid() vem do modulo pgcrypto (Postgres 13+ ja o traz nativo,
-- mas garantimos a extensao para ambientes mais antigos).
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS examples (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        VARCHAR(120) NOT NULL,
    description TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_examples_created_at ON examples (created_at DESC);
