use std::ffi::OsString;
use std::sync::{Mutex, OnceLock};

use api::{
    read_groq_base_url, read_xai_base_url, ApiError, AuthSource, ProviderClient, ProviderKind,
};

#[test]
fn provider_client_routes_grok_aliases_through_xai() {
    let _lock = env_lock();
    let _xai_api_key = EnvVarGuard::set("XAI_API_KEY", Some("xai-test-key"));

    let client = ProviderClient::from_model("grok-mini").expect("grok alias should resolve");

    assert_eq!(client.provider_kind(), ProviderKind::Xai);
}

#[test]
fn provider_client_routes_groq_alias_through_groq() {
    let _lock = env_lock();
    let _groq_api_key = EnvVarGuard::set("GROQ_API_KEY", Some("groq-test-key"));

    let client = ProviderClient::from_model("groq").expect("groq alias should resolve");

    assert_eq!(client.provider_kind(), ProviderKind::Groq);
}

#[test]
fn provider_client_routes_groq_slash_models_through_groq_even_with_other_keys() {
    let _lock = env_lock();
    let _anthropic_api_key = EnvVarGuard::set("ANTHROPIC_API_KEY", Some("anthropic-test-key"));
    let _openai_api_key = EnvVarGuard::set("OPENAI_API_KEY", Some("openai-test-key"));
    let _groq_api_key = EnvVarGuard::set("GROQ_API_KEY", Some("groq-test-key"));

    let client = ProviderClient::from_model("groq/llama-3.1-8b-instant")
        .expect("explicit Groq prefix should force Groq routing");

    assert_eq!(client.provider_kind(), ProviderKind::Groq);
}

#[test]
fn provider_client_reports_missing_xai_credentials_for_grok_models() {
    let _lock = env_lock();
    let _xai_api_key = EnvVarGuard::set("XAI_API_KEY", None);

    let error = ProviderClient::from_model("grok-3")
        .expect_err("grok requests without XAI_API_KEY should fail fast");

    match error {
        ApiError::MissingCredentials { provider, env_vars } => {
            assert_eq!(provider, "xAI");
            assert_eq!(env_vars, &["XAI_API_KEY"]);
        }
        other => panic!("expected missing xAI credentials, got {other:?}"),
    }
}

#[test]
fn provider_client_reports_missing_groq_credentials_for_groq_models() {
    let _lock = env_lock();
    let _groq_api_key = EnvVarGuard::set("GROQ_API_KEY", None);

    let error = ProviderClient::from_model("groq")
        .expect_err("groq requests without GROQ_API_KEY should fail fast");

    match error {
        ApiError::MissingCredentials { provider, env_vars } => {
            assert_eq!(provider, "Groq");
            assert_eq!(env_vars, &["GROQ_API_KEY"]);
        }
        other => panic!("expected missing Groq credentials, got {other:?}"),
    }
}

#[test]
fn provider_client_rejects_empty_groq_slash_model() {
    let _lock = env_lock();

    let error = ProviderClient::from_model("groq/")
        .expect_err("empty explicit Groq model should fail fast");

    match error {
        ApiError::InvalidModel(message) => {
            assert!(message.contains("non-empty model name"));
        }
        other => panic!("expected invalid model error, got {other:?}"),
    }
}

#[test]
fn provider_client_uses_explicit_auth_without_env_lookup() {
    let _lock = env_lock();
    let _api_key = EnvVarGuard::set("ANTHROPIC_API_KEY", None);
    let _auth_token = EnvVarGuard::set("ANTHROPIC_AUTH_TOKEN", None);

    let client = ProviderClient::from_model_with_default_auth(
        "claude-sonnet-4-6",
        Some(AuthSource::ApiKey("claw-test-key".to_string())),
    )
    .expect("explicit auth should avoid env lookup");

    assert_eq!(client.provider_kind(), ProviderKind::ClawApi);
}

#[test]
fn read_groq_base_url_uses_default_when_env_missing() {
    let _lock = env_lock();
    let _groq_base_url = EnvVarGuard::set("GROQ_BASE_URL", None);

    assert_eq!(read_groq_base_url(), "https://api.groq.com/openai/v1");
}

#[test]
fn read_xai_base_url_prefers_env_override() {
    let _lock = env_lock();
    let _xai_base_url = EnvVarGuard::set("XAI_BASE_URL", Some("https://example.xai.test/v1"));

    assert_eq!(read_xai_base_url(), "https://example.xai.test/v1");
}

#[test]
fn read_groq_base_url_prefers_env_override() {
    let _lock = env_lock();
    let _groq_base_url = EnvVarGuard::set("GROQ_BASE_URL", Some("https://example.groq.test/v1"));

    assert_eq!(read_groq_base_url(), "https://example.groq.test/v1");
}

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

struct EnvVarGuard {
    key: &'static str,
    original: Option<OsString>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: Option<&str>) -> Self {
        let original = std::env::var_os(key);
        match value {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
        Self { key, original }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match &self.original {
            Some(value) => std::env::set_var(self.key, value),
            None => std::env::remove_var(self.key),
        }
    }
}
