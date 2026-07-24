//! Configuracao **env-aware**.
//!
//! Regra: o *ambiente* decide de onde vem cada segredo.
//! - `development` (default) -> segredos do ambiente/`.env`. Zero atrito pro dev.
//! - `staging`/`production`  -> segredos do **Vault** (KV v2, auth AppRole). Fail-closed:
//!   se o Vault falhar ou faltar um segredo, o boot aborta — nunca cai pro `.env`.
//!
//! Nao-segredos (HOST, PORT, tamanho do pool...) vem SEMPRE do ambiente, em qualquer
//! modo — quem seta isso e o orquestrador (k3s/systemd), nao o secret manager.

use std::collections::HashMap;
use std::env;
use std::fmt;

use async_trait::async_trait;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppEnv {
    Development,
    Staging,
    Production,
}

impl AppEnv {
    /// Controlado por APP_ENV. Ausente/desconhecido -> Development (protege a DX).
    fn from_env() -> Self {
        match env::var("APP_ENV").ok().as_deref().map(str::trim) {
            Some("production") | Some("prod") => Self::Production,
            Some("staging") | Some("homolog") | Some("homologacao") | Some("hml") => Self::Staging,
            _ => Self::Development,
        }
    }

    pub fn uses_vault(self) -> bool {
        matches!(self, Self::Staging | Self::Production)
    }
}

/// Seam de segredos: uma impl le do ambiente, outra do Vault. O resto do app
/// depende so deste trait — nao sabe (nem se importa) de onde o segredo veio.
#[async_trait]
pub trait SecretProvider: Send + Sync {
    async fn get(&self, key: &str) -> anyhow::Result<String>;
}

/// development: le do ambiente (o `.env` ja foi carregado pelo dotenvy no main).
pub struct EnvSecrets;

#[async_trait]
impl SecretProvider for EnvSecrets {
    async fn get(&self, key: &str) -> anyhow::Result<String> {
        env::var(key).map_err(|_| anyhow::anyhow!("segredo ausente no ambiente: {key}"))
    }
}

/// staging/production: autentica no Vault via AppRole e le TODOS os segredos de um
/// path KV v2 uma vez no boot, cacheando em memoria.
///
/// AppRole e universal (bare-metal e k8s). No k3s, injete VAULT_ROLE_ID/VAULT_SECRET_ID
/// via Secret do cluster. (Login Kubernetes nativo nao existe no vaultrs-login 0.2 —
/// para usa-lo, faca o POST manual em auth/kubernetes/login com o JWT da service
/// account, ou deixe o Vault Agent/VSO injetar os segredos como env, caindo no
/// caminho de EnvSecrets.)
pub struct VaultSecrets {
    data: HashMap<String, String>,
}

impl VaultSecrets {
    pub async fn connect() -> anyhow::Result<Self> {
        use vaultrs::client::{VaultClient, VaultClientSettingsBuilder};
        use vaultrs_login::engines::approle::AppRoleLogin;
        use vaultrs_login::LoginClient;

        let addr = require_env("VAULT_ADDR")?;
        let mount = env::var("VAULT_KV_MOUNT").unwrap_or_else(|_| "secret".to_string());
        let path = require_env("VAULT_SECRET_PATH")?;
        let role_id = require_env("VAULT_ROLE_ID")?;
        let secret_id = require_env("VAULT_SECRET_ID")?;

        let mut client = VaultClient::new(
            VaultClientSettingsBuilder::default()
                .address(addr)
                .build()?,
        )?;

        // AppRole login: o token e setado automaticamente no client.
        let login = AppRoleLogin { role_id, secret_id };
        client.login("approle", &login).await?;

        // KV v2: le o secret inteiro do path como mapa chave->valor.
        let data: HashMap<String, String> = vaultrs::kv2::read(&client, &mount, &path).await?;

        tracing::info!(%mount, %path, keys = data.len(), "segredos carregados do Vault");
        Ok(Self { data })
    }
}

#[async_trait]
impl SecretProvider for VaultSecrets {
    async fn get(&self, key: &str) -> anyhow::Result<String> {
        self.data
            .get(key)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("segredo ausente no Vault: {key}"))
    }
}

#[derive(Clone)]
pub struct Config {
    pub env: AppEnv,
    pub host: String,
    pub port: u16,
    pub database_url: String,
    pub db_max_connections: u32,
    pub db_acquire_timeout_secs: u64,
}

impl Config {
    pub async fn load() -> anyhow::Result<Self> {
        let app_env = AppEnv::from_env();

        // Nao-segredos: SEMPRE do ambiente.
        let host = optional("HOST", "0.0.0.0");
        let port = parse_or("PORT", 8080)?;
        let db_max_connections = parse_or("DB_MAX_CONNECTIONS", 10)?;
        let db_acquire_timeout_secs = parse_or("DB_ACQUIRE_TIMEOUT_SECS", 5)?;

        // Segredos: provider conforme o ambiente. Em staging/prod, erro aqui aborta o boot.
        let secrets: Box<dyn SecretProvider> = if app_env.uses_vault() {
            Box::new(VaultSecrets::connect().await?) // fail-closed
        } else {
            Box::new(EnvSecrets)
        };

        let database_url = secrets.get("DATABASE_URL").await?;

        tracing::info!(env = ?app_env, uses_vault = app_env.uses_vault(), "configuracao carregada");

        Ok(Self {
            env: app_env,
            host,
            port,
            database_url,
            db_max_connections,
            db_acquire_timeout_secs,
        })
    }
}

// Debug manual: nunca imprime a URL do banco (contem senha).
impl fmt::Debug for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Config")
            .field("env", &self.env)
            .field("database_url", &"<redacted>")
            .field("host", &self.host)
            .field("port", &self.port)
            .field("db_max_connections", &self.db_max_connections)
            .field("db_acquire_timeout_secs", &self.db_acquire_timeout_secs)
            .finish()
    }
}

fn require_env(key: &str) -> anyhow::Result<String> {
    env::var(key).map_err(|_| anyhow::anyhow!("variavel de ambiente obrigatoria ausente: {key}"))
}

fn optional(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}

fn parse_or<T>(key: &str, default: T) -> anyhow::Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match env::var(key) {
        Ok(v) => v
            .parse::<T>()
            .map_err(|e| anyhow::anyhow!("valor invalido para {key}: {e}")),
        Err(_) => Ok(default),
    }
}
