//! Estado compartilhado (container de dependencias injetado nos handlers).

use std::sync::Arc;

use crate::services::example::ExampleService;

/// Clonado por requisicao pelo Axum; campos em `Arc` tornam o clone barato.
#[derive(Clone)]
pub struct AppState {
    pub example_service: Arc<ExampleService>,
}
