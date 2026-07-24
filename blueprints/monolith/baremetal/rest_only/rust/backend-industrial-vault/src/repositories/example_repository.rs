//! "example repository": contrato de persistencia + implementacao Postgres/SQLx.
//!
//! A camada de servico depende do trait (abstracao), nunca da struct concreta.
//! Isso permite mockar o banco nos testes (ver services/example.rs).

use async_trait::async_trait;
use sqlx::PgPool;
use uuid::Uuid;
use crate::dto::example_dto::UpsertExampleRequest;
use crate::error::AppResult;
use crate::models::example::Example;

#[async_trait]
pub trait ExampleRepository: Send + Sync + 'static {
    async fn list(&self) -> AppResult<Vec<Example>>;
    async fn find(&self, id: Uuid) -> AppResult<Option<Example>>;
    async fn create(&self, input: &UpsertExampleRequest) -> AppResult<Example>;
    async fn update(&self, id: Uuid, input: &UpsertExampleRequest) -> AppResult<Option<Example>>;
    async fn delete(&self, id: Uuid) -> AppResult<bool>;
}

/// Implementacao concreta sobre Postgres.
///
/// Usa `query_as` em runtime (nao a macro `query_as!`) de proposito: o scaffold
/// compila SEM um banco no ar. Ao migrar para macros compile-time, rode
/// `cargo sqlx prepare` e versione o diretorio `.sqlx/`.
pub struct PgExampleRepository {
    pool: PgPool,
}

impl PgExampleRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

const COLUMNS: &str = "id, name, description, created_at, updated_at";

#[async_trait]
impl ExampleRepository for PgExampleRepository {
    async fn list(&self) -> AppResult<Vec<Example>> {
        let rows = sqlx::query_as::<_, Example>(&format!(
            "SELECT {COLUMNS} FROM examples ORDER BY created_at DESC"
        ))
            .fetch_all(&self.pool)
            .await?;
        Ok(rows)
    }

    async fn find(&self, id: Uuid) -> AppResult<Option<Example>> {
        let row = sqlx::query_as::<_, Example>(&format!(
            "SELECT {COLUMNS} FROM examples WHERE id = $1"
        ))
            .bind(id)
            .fetch_optional(&self.pool)
            .await?;
        Ok(row)
    }

    async fn create(&self, input: &UpsertExampleRequest) -> AppResult<Example> {
        let row = sqlx::query_as::<_, Example>(&format!(
            "INSERT INTO examples (name, description) VALUES ($1, $2) RETURNING {COLUMNS}"
        ))
            .bind(&input.name)
            .bind(&input.description)
            .fetch_one(&self.pool)
            .await?;
        Ok(row)
    }

    async fn update(&self, id: Uuid, input: &UpsertExampleRequest) -> AppResult<Option<Example>> {
        let row = sqlx::query_as::<_, Example>(&format!(
            "UPDATE examples SET name = $2, description = $3, updated_at = now() \
             WHERE id = $1 RETURNING {COLUMNS}"
        ))
            .bind(id)
            .bind(&input.name)
            .bind(&input.description)
            .fetch_optional(&self.pool)
            .await?;
        Ok(row)
    }

    async fn delete(&self, id: Uuid) -> AppResult<bool> {
        let result = sqlx::query("DELETE FROM examples WHERE id = $1")
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(result.rows_affected() > 0)
    }
}
