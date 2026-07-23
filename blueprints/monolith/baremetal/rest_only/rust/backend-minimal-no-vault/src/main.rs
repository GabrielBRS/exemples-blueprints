//! Monolito REST **minimal** — tudo em um arquivo, sobe em segundos.
//!
//! Stack: Axum + Tokio + SQLx (Postgres) + dotenv. Sem camadas, sem DI, sem
//! traits, sem testes: e o ponto de partida cru. Quando o projeto crescer, va
//! extraindo modulos (handler/service/repository) — o irmao `backend-industrial-no-vault`
//! ja mostra esse destino, em camadas.

use std::env;

use axum::extract::State;

#[tokio::main]
async fn main() -> anyhow::Result<()> {

    dotenvy::dotenv().ok();

    


    Ok(())
}
