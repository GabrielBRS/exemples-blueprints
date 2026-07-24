//! Ponto de entrada do monolito REST.
//!
//! Fluxo: carrega `.env` -> inicializa tracing -> le `Config` -> abre o pool
//! Postgres -> roda migrations -> monta o composition root (DI) -> sobe o Axum
//! com graceful shutdown.

mod config;
mod dto;
mod error;
mod handlers;
mod models;
mod repositories;
mod router;
mod services;
mod state;

use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use sqlx::postgres::PgPoolOptions;
use tokio::net::TcpListener;
use tokio::signal;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use crate::config::Config;
use crate::repositories::example::PgExampleRepository;
use crate::services::example::ExampleService;
use crate::state::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Carrega variaveis do .env (silencioso se o arquivo nao existir).
    dotenvy::dotenv().ok();

    init_tracing();

    let config = Config::load().await.context("falha ao carregar configuracao")?;
    tracing::info!(?config, "configuracao carregada");

    // Pool de conexoes Postgres.
    let pool = PgPoolOptions::new()
        .max_connections(config.db_max_connections)
        .acquire_timeout(Duration::from_secs(config.db_acquire_timeout_secs))
        .connect(&config.database_url)
        .await
        .context("falha ao conectar no Postgres")?;

    // Migrations embarcadas (pasta ./migrations, compiladas no binario).
    sqlx::migrate!("./migrations")
        .run(&pool)
        .await
        .context("falha ao rodar migrations")?;

    // --- Composition root: monta a arvore de dependencias (DI) ---
    // handler -> service -> repository. Troque a implementacao do repository
    // aqui (ex.: um mock) sem tocar em service nem handler.
    let example_repository = Arc::new(PgExampleRepository::new(pool.clone()));
    let example_service = Arc::new(ExampleService::new(example_repository));
    let state = AppState { example_service };

    let app = router::build(state);

    let addr = format!("{}:{}", config.host, config.port);
    let listener = TcpListener::bind(&addr)
        .await
        .with_context(|| format!("falha ao bindar em {addr}"))?;

    tracing::info!("servidor ouvindo em http://{addr}");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("erro no servidor http")?;

    Ok(())
}

/// Configura o `tracing`. Nivel controlado por `RUST_LOG` (fallback sensato).
fn init_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,tower_http=debug,sqlx=warn"));

    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer())
        .init();
}

/// Aguarda Ctrl+C (e SIGTERM no Unix) para desligar graciosamente.
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("falha ao instalar handler de Ctrl+C");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("falha ao instalar handler de SIGTERM")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    tracing::info!("sinal de shutdown recebido, encerrando...");
}
