//! Entidade de dominio, mapeada 1:1 com a tabela `examples`.

use chrono::{DateTime, Utc};
use serde::Serialize;
use sqlx::FromRow;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, FromRow)]
pub struct Example {
    #[sqlx(rename = "id")]
    pub id: Uuid,
    #[sqlx(rename = "name")]
    pub name: String,
    #[sqlx(rename = "description")]
    pub description: Option<String>,
    #[sqlx(rename = "created_at")]
    pub created_at: DateTime<Utc>,
    #[sqlx(rename = "updated_at")]
    pub updated_at: DateTime<Utc>,
}
