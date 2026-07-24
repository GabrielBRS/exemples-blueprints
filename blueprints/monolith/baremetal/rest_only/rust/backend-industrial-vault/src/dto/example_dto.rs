//! Contratos de entrada/saida (desacoplam o dominio do formato HTTP).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::models::example::Example;

/// Payload para criar/atualizar um Example.
#[derive(Debug, Deserialize)]
pub struct UpsertExampleRequest {
    pub name: String,
    pub description: Option<String>,
}

impl UpsertExampleRequest {
    /// Validacao minima de exemplo. Troque por `validator`/`garde` se precisar de mais.
    pub fn validate(&self) -> Result<(), String> {
        if self.name.trim().is_empty() {
            return Err("o campo 'name' nao pode ser vazio".into());
        }
        if self.name.chars().count() > 120 {
            return Err("o campo 'name' excede 120 caracteres".into());
        }
        Ok(())
    }
}

/// Representacao de saida (o que a API expoe).
#[derive(Debug, Serialize)]
pub struct ExampleResponse {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl From<Example> for ExampleResponse {
    fn from(e: Example) -> Self {
        Self {
            id: e.id,
            name: e.name,
            description: e.description,
            created_at: e.created_at,
            updated_at: e.updated_at,
        }
    }
}
