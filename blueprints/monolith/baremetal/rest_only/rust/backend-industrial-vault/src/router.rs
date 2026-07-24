//! "example router": composicao das rotas + middlewares globais.

use std::time::Duration;

use axum::routing::get;
use axum::Router;
use tower_http::cors::CorsLayer;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;
use crate::handlers::example_handler;

/// Router raiz com middlewares aplicados a tudo.
pub fn build(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health::health))
        .nest("/api/v1", api_v1())
        .layer(TraceLayer::new_for_http())
        .layer(TimeoutLayer::new(Duration::from_secs(30)))
        .layer(CorsLayer::permissive()) // PRODUCAO: restrinja origem/metodos/headers.
        .with_state(state)
}

/// Agrupador versionado. Registre novos recursos da v1 aqui.
fn api_v1() -> Router<AppState> {
    Router::new().nest("/examples", example_routes())
}

/// Rotas do recurso Example (CRUD completo).
fn example_routes() -> Router<AppState> {
    Router::new()
        .route("/", get(example_handler::list).post(example_handler::create))
        .route(
            "/{id}",
            get(example_handler::get).put(example_handler::update).delete(example_handler::delete),
        )
}
