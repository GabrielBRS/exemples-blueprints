//! Erro de aplicacao unificado + conversao automatica para resposta HTTP JSON.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("recurso nao encontrado")]
    NotFound,

    #[error("requisicao invalida: {0}")]
    BadRequest(String),

    #[error(transparent)]
    Database(#[from] sqlx::Error),

    #[error(transparent)]
    Unexpected(#[from] anyhow::Error),
}

impl AppError {
    fn status(&self) -> StatusCode {
        match self {
            AppError::NotFound => StatusCode::NOT_FOUND,
            AppError::BadRequest(_) => StatusCode::BAD_REQUEST,
            AppError::Database(sqlx::Error::RowNotFound) => StatusCode::NOT_FOUND,
            AppError::Database(_) | AppError::Unexpected(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = self.status();

        // 5xx: registra o detalhe interno, mas nao vaza pro cliente.
        if status.is_server_error() {
            tracing::error!(error = ?self, "erro interno");
        }

        let message = if status == StatusCode::INTERNAL_SERVER_ERROR {
            "erro interno do servidor".to_string()
        } else {
            self.to_string()
        };

        let body = Json(json!({
            "error": { "status": status.as_u16(), "message": message }
        }));

        (status, body).into_response()
    }
}

pub type AppResult<T> = Result<T, AppError>;
