//! "example handler": adapta HTTP <-> service. Sem regra de negocio aqui.
//!
//! Regra de ouro dos extractors do Axum: o que consome o corpo (`Json`) vem por
//! ULTIMO na assinatura; `State` e `Path` vem antes.

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use uuid::Uuid;
use crate::dto::example_dto::{ExampleResponse, UpsertExampleRequest};
use crate::error::AppResult;

/// GET /api/v1/examples
pub async fn list(State(state): State<AppState>) -> AppResult<Json<Vec<ExampleResponse>>> {
    Ok(Json(state.example_service.list().await?))
}

/// GET /api/v1/examples/{id}
pub async fn get(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> AppResult<Json<ExampleResponse>> {
    Ok(Json(state.example_service.get(id).await?))
}

/// POST /api/v1/examples
pub async fn create(
    State(state): State<AppState>,
    Json(payload): Json<UpsertExampleRequest>,
) -> AppResult<(StatusCode, Json<ExampleResponse>)> {
    let created = state.example_service.create(payload).await?;
    Ok((StatusCode::CREATED, Json(created)))
}

/// PUT /api/v1/examples/{id}
pub async fn update(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Json(payload): Json<UpsertExampleRequest>,
) -> AppResult<Json<ExampleResponse>> {
    Ok(Json(state.example_service.update(id, payload).await?))
}

/// DELETE /api/v1/examples/{id}
pub async fn delete(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> AppResult<StatusCode> {
    state.example_service.delete(id).await?;
    Ok(StatusCode::NO_CONTENT)
}
