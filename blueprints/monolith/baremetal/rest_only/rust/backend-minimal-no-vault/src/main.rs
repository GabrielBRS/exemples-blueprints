//! Monolito REST **minimal** — tudo em um arquivo, sobe em segundos.
//!
//! Stack: Axum + Tokio + SQLx (Postgres) + dotenv. Sem camadas, sem DI, sem
//! traits, sem testes: e o ponto de partida cru. Quando o projeto crescer, va
//! extraindo modulos (handler/service/repository) — o irmao `backend-industrial-no-vault`
//! ja mostra esse destino, em camadas.

use std::env;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;
use uuid::Uuid;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    // Nivel de log via RUST_LOG (fallback sensato).
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,sqlx=warn"));
    tracing_subscriber::fmt().with_env_filter(filter).init();

    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL ausente");
    let port = env::var("PORT").unwrap_or_else(|_| "8080".into());

    let pool = PgPoolOptions::new()
        .max_connections(10)
        .connect(&database_url)
        .await?;

    // Migrations embarcadas (pasta ./migrations), rodam no boot.
    sqlx::migrate!("./migrations").run(&pool).await?;

    let app = Router::new()
        .route("/health", get(health))
        .route("/examples", get(list_examples).post(create_example))
        .with_state(pool);

    let addr = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("ouvindo em http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health() -> Json<serde_json::Value> {
    Json(json!({ "status": "ok" }))
}

#[derive(Serialize, sqlx::FromRow)]
struct Example {
    id: Uuid,
    name: String,
}

#[derive(Deserialize)]
struct NewExample {
    name: String,
}

/// GET /examples
async fn list_examples(State(pool): State<PgPool>) -> Result<Json<Vec<Example>>, AppError> {
    let rows = sqlx::query_as::<_, Example>("SELECT id, name FROM examples ORDER BY name")
        .fetch_all(&pool)
        .await?;
    Ok(Json(rows))
}

/// POST /examples  { "name": "..." }
async fn create_example(
    State(pool): State<PgPool>,
    Json(body): Json<NewExample>,
) -> Result<(StatusCode, Json<Example>), AppError> {
    if body.name.trim().is_empty() {
        return Err(AppError::BadRequest("o campo 'name' nao pode ser vazio".into()));
    }
    let row = sqlx::query_as::<_, Example>(
        "INSERT INTO examples (name) VALUES ($1) RETURNING id, name",
    )
        .bind(&body.name)
        .fetch_one(&pool)
        .await?;
    Ok((StatusCode::CREATED, Json(row)))
}

/// Erro minimo: BadRequest -> 400; qualquer erro de SQLx -> 500 (logado, nao vazado).
enum AppError {
    BadRequest(String),
    Db(sqlx::Error),
}

impl From<sqlx::Error> for AppError {
    fn from(e: sqlx::Error) -> Self {
        AppError::Db(e)
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, msg) = match self {
            AppError::BadRequest(m) => (StatusCode::BAD_REQUEST, m),
            AppError::Db(e) => {
                tracing::error!(error = ?e, "erro de banco");
                (StatusCode::INTERNAL_SERVER_ERROR, "erro interno".to_string())
            }
        };
        (status, Json(json!({ "error": msg }))).into_response()
    }
}
