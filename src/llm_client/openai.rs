//! OpenAI LLM client implementation.
//!
//! Uses `async-openai` for API calls, `moka` for response caching, and
//! `backoff` for exponential-backoff retry on rate limits / transient errors.

use std::sync::Arc;
use std::time::Duration;

use backoff::ExponentialBackoffBuilder;
use moka::future::Cache;
use serde_json::json;
use tracing::{debug, warn};

use crate::errors::{GraphitiError, LlmError, Result};

use super::token_tracker::{TokenTracker, TokenUsage};
use super::{LlmClient, Message, Role};

// ── Cache configuration ───────────────────────────────────────────────────────

/// Configuration for the in-process response cache.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of entries held in memory.
    pub max_capacity: u64,
    /// How long each entry lives before eviction.
    pub ttl: Duration,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_capacity: 1_000,
            ttl: Duration::from_secs(3_600), // 1 hour
        }
    }
}

// ── Backoff configuration ─────────────────────────────────────────────────────

/// Configuration for exponential-backoff retry.
#[derive(Debug, Clone)]
pub struct BackoffConfig {
    /// Initial delay before the first retry.
    pub initial_interval: Duration,
    /// Maximum delay between retries.
    pub max_interval: Duration,
    /// Maximum total time to keep retrying before giving up.
    pub max_elapsed_time: Duration,
}

impl Default for BackoffConfig {
    fn default() -> Self {
        Self {
            initial_interval: Duration::from_millis(500),
            max_interval: Duration::from_secs(60),
            max_elapsed_time: Duration::from_secs(300),
        }
    }
}

// ── Client struct ─────────────────────────────────────────────────────────────

/// OpenAI LLM client implementing [`LlmClient`].
pub struct OpenAiClient {
    client: async_openai::Client<async_openai::config::OpenAIConfig>,
    model: String,
    temperature: f32,
    max_tokens: u32,
    backoff_config: BackoffConfig,
    /// Keyed by `{type_name}:{md5(model + temperature + max_tokens + messages)}` → serialised response text.
    cache: Cache<String, String>,
    /// Cumulative token usage counters shared via `Arc` for cheap cloning.
    token_tracker: Arc<TokenTracker>,
}

impl OpenAiClient {
    /// Create a new client.
    ///
    /// # Arguments
    /// * `api_key` – OpenAI secret key.
    /// * `model`   – Model name (e.g. `"gpt-4o"`).
    /// * `cache_config` – Cache capacity and TTL.
    pub fn new(
        api_key: impl Into<String>,
        model: impl Into<String>,
        cache_config: CacheConfig,
    ) -> Self {
        let config = async_openai::config::OpenAIConfig::new().with_api_key(api_key);
        let client = async_openai::Client::with_config(config);

        let cache = Cache::builder()
            .max_capacity(cache_config.max_capacity)
            .time_to_live(cache_config.ttl)
            .build();

        Self {
            client,
            model: model.into(),
            temperature: 0.0,
            max_tokens: 8_192,
            backoff_config: BackoffConfig::default(),
            cache,
            token_tracker: Arc::new(TokenTracker::new()),
        }
    }

    /// Override the sampling temperature (default `0.0`).
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Override the max output token limit (default `8192`).
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Override the retry backoff configuration (default: 500 ms initial, 60 s max, 5 min total).
    pub fn with_backoff_config(mut self, backoff_config: BackoffConfig) -> Self {
        self.backoff_config = backoff_config;
        self
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Compute an MD5 cache key from model, temperature, max_tokens, and message sequence.
    ///
    /// Including `temperature` and `max_tokens` prevents cache collisions between
    /// requests that share the same messages but differ in sampling parameters.
    fn cache_key(&self, prefix: &str, messages: &[Message]) -> String {
        use md5::{Digest, Md5};
        let mut h = Md5::new();
        h.update(prefix.as_bytes());
        h.update(self.model.as_bytes());
        h.update(self.temperature.to_bits().to_le_bytes());
        h.update(self.max_tokens.to_le_bytes());
        for m in messages {
            let role = role_str(&m.role);
            h.update(role.as_bytes());
            h.update(m.content.as_bytes());
        }
        format!("{:x}", h.finalize())
    }

    /// Serialise our [`Message`] slice into the JSON array expected by the API.
    fn messages_to_json(messages: &[Message]) -> Vec<serde_json::Value> {
        messages
            .iter()
            .map(|m| {
                json!({
                    "role": role_str(&m.role),
                    "content": m.content,
                })
            })
            .collect()
    }

    /// Call the chat completions endpoint with exponential-backoff retry.
    ///
    /// Retries on [`LlmError::RateLimit`] (HTTP 429) and transient 5xx errors.
    async fn call_with_retry(&self, request: serde_json::Value) -> Result<serde_json::Value> {
        let backoff = ExponentialBackoffBuilder::new()
            .with_initial_interval(self.backoff_config.initial_interval)
            .with_max_interval(self.backoff_config.max_interval)
            .with_max_elapsed_time(Some(self.backoff_config.max_elapsed_time))
            .build();

        backoff::future::retry(backoff, || async {
            let outcome: std::result::Result<serde_json::Value, async_openai::error::OpenAIError> =
                self.client.chat().create_byot(request.clone()).await;

            match outcome {
                Ok(response) => Ok(response),
                Err(e) => {
                    let llm_err = map_openai_error(e);
                    match &llm_err {
                        LlmError::RateLimit => {
                            warn!("OpenAI rate limit hit — retrying with backoff");
                            Err(backoff::Error::transient(llm_err))
                        }
                        LlmError::Api { status, .. } if *status >= 500 => {
                            warn!("OpenAI transient server error ({}) — retrying", status);
                            Err(backoff::Error::transient(llm_err))
                        }
                        _ => Err(backoff::Error::permanent(llm_err)),
                    }
                }
            }
        })
        .await
        .map_err(GraphitiError::Llm)
    }

    /// Extract the assistant message text from a chat-completions response.
    fn extract_content(response: &serde_json::Value) -> Result<String> {
        response["choices"][0]["message"]["content"]
            .as_str()
            .map(ToOwned::to_owned)
            .ok_or(GraphitiError::Llm(LlmError::EmptyResponse))
    }

    /// Record token usage from an API response into the tracker.
    ///
    /// Silently ignores missing or malformed `usage` fields — the counters
    /// simply won't be incremented for that call.
    fn record_usage(&self, response: &serde_json::Value) {
        let prompt = response["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
        let completion = response["usage"]["completion_tokens"].as_u64().unwrap_or(0);
        self.token_tracker.record(prompt, completion);
    }

    /// Return a reference to the underlying [`TokenTracker`].
    pub fn tracker(&self) -> &Arc<TokenTracker> {
        &self.token_tracker
    }
}

// ── LlmClient implementation ──────────────────────────────────────────────────

#[async_trait::async_trait]
impl LlmClient for OpenAiClient {
    async fn generate(&self, messages: &[Message]) -> Result<String> {
        let key = self.cache_key("text", messages);

        if let Some(cached) = self.cache.get(&key).await {
            debug!("LLM cache hit (text)");
            return Ok(cached);
        }

        let max_tokens_key = max_tokens_param(&self.model);
        let mut request = json!({
            "model": self.model,
            "messages": Self::messages_to_json(messages),
            "temperature": self.temperature,
        });
        request[max_tokens_key] = json!(self.max_tokens);

        let response = self.call_with_retry(request).await?;
        self.record_usage(&response);
        let content = Self::extract_content(&response)?;

        self.cache.insert(key, content.clone()).await;

        Ok(content)
    }

    async fn generate_structured_json(
        &self,
        messages: &[Message],
        schema: serde_json::Value,
    ) -> Result<String> {
        // Use a hash of the schema as part of the cache key so different schemas
        // for the same messages don't collide.
        let schema_str = schema.to_string();
        let key = self.cache_key(&schema_str, messages);

        if let Some(cached) = self.cache.get(&key).await {
            debug!("LLM cache hit (structured)");
            return Ok(cached);
        }

        let mut schema_value = schema;

        // OpenAI's strict mode requires:
        //   1. `additionalProperties: false` on every object schema in the tree.
        //   2. All property keys listed in `required` (Option<_> fields are omitted
        //      by schemars but must be present for the API to accept the schema).
        add_no_additional_properties(&mut schema_value);
        ensure_all_properties_required(&mut schema_value);

        let max_tokens_key = max_tokens_param(&self.model);
        let mut request = json!({
            "model": self.model,
            "messages": Self::messages_to_json(messages),
            "temperature": self.temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema_value,
                    "strict": true,
                }
            }
        });
        request[max_tokens_key] = json!(self.max_tokens);

        let response = self.call_with_retry(request).await?;
        self.record_usage(&response);
        let content = Self::extract_content(&response)?;

        self.cache.insert(key, content.clone()).await;

        Ok(content)
    }

    fn token_usage(&self) -> TokenUsage {
        self.token_tracker.snapshot()
    }

    fn reset_token_usage(&self) {
        self.token_tracker.reset();
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn role_str(role: &Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
    }
}

/// Recursively add `"additionalProperties": false` to every JSON object schema
/// that has a `"properties"` field.
///
/// Always overwrites any pre-existing `additionalProperties` value because
/// OpenAI strict mode requires it to be exactly `false` — a pre-existing
/// `true` (e.g. from a map-like `schemars` type) would cause an API rejection.
fn add_no_additional_properties(schema: &mut serde_json::Value) {
    match schema {
        serde_json::Value::Object(obj) => {
            if obj.contains_key("properties") {
                obj.insert(
                    "additionalProperties".to_string(),
                    serde_json::Value::Bool(false),
                );
            }
            for v in obj.values_mut() {
                add_no_additional_properties(v);
            }
        }
        serde_json::Value::Array(arr) => {
            for item in arr.iter_mut() {
                add_no_additional_properties(item);
            }
        }
        _ => {}
    }
}

/// Recursively ensure every object schema has `"required"` equal to the full
/// list of its `"properties"` keys.
///
/// OpenAI strict mode requires BOTH `additionalProperties: false` AND all
/// properties listed in `required`.  `schemars` omits `Option<_>` fields from
/// `required`, which causes the OpenAI API to reject the schema at runtime.
fn ensure_all_properties_required(schema: &mut serde_json::Value) {
    match schema {
        serde_json::Value::Object(obj) => {
            if let Some(props) = obj.get("properties") {
                if let Some(props_map) = props.as_object() {
                    let keys: Vec<serde_json::Value> = props_map
                        .keys()
                        .map(|k| serde_json::Value::String(k.clone()))
                        .collect();
                    obj.insert("required".to_string(), serde_json::Value::Array(keys));
                }
            }
            // Recurse into nested schemas (including $defs, allOf, etc.)
            for v in obj.values_mut() {
                ensure_all_properties_required(v);
            }
        }
        serde_json::Value::Array(arr) => {
            for item in arr.iter_mut() {
                ensure_all_properties_required(item);
            }
        }
        _ => {}
    }
}

/// Return the correct token-limit parameter name for the given model.
///
/// OpenAI deprecated `"max_tokens"` for newer models (gpt-4o, gpt-4-turbo,
/// o1, o3 and later families); those models expect `"max_completion_tokens"`.
/// The old field may be silently ignored, causing unbounded output.
fn max_tokens_param(model: &str) -> &'static str {
    if model.contains("gpt-4o")
        || model.contains("gpt-4-turbo")
        || model.starts_with("o1")
        || model.starts_with("o3")
        || model.starts_with("o4")
    {
        "max_completion_tokens"
    } else {
        "max_tokens"
    }
}

/// Map an [`async_openai::error::OpenAIError`] to our [`LlmError`] domain type.
///
/// `async_openai::error::ApiError` does not carry an HTTP status field.
/// Instead, we infer the error class from the `code` field for known values.
/// Server errors (5xx) are wrapped by async-openai as `ApiError` with both
/// `code` and `r#type` set to `None`, so we assign them `status = 500` to
/// allow the caller's retry logic to detect them.
fn map_openai_error(err: async_openai::error::OpenAIError) -> LlmError {
    use async_openai::error::OpenAIError;

    match err {
        OpenAIError::ApiError(api_err) => match api_err.code.as_deref() {
            Some("invalid_api_key") => LlmError::Authentication,
            Some("rate_limit_exceeded") => LlmError::RateLimit,
            _ => {
                // async-openai formats 5xx errors as ApiError with code=None and
                // type=None. Assign status=500 so callers can treat them as transient.
                // TODO: revisit once async-openai exposes HTTP status on ApiError
                // directly — an API error with a genuinely unknown code/type would
                // also land here and be misclassified as a transient 5xx.
                let status = if api_err.code.is_none() && api_err.r#type.is_none() {
                    500
                } else {
                    0
                };
                LlmError::Api { status, message: api_err.message }
            }
        },
        OpenAIError::Reqwest(e) => LlmError::Api {
            status: e.status().map(|s| s.as_u16()).unwrap_or(0),
            message: e.to_string(),
        },
        other => LlmError::Api {
            status: 0,
            message: other.to_string(),
        },
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    // ── helpers ───────────────────────────────────────────────────────────────

    /// Build a client pointing at an arbitrary base URL (mock server).
    fn client_for(base_url: &str) -> OpenAiClient {
        let config =
            async_openai::config::OpenAIConfig::new().with_api_key("test-key").with_api_base(base_url);
        let inner = async_openai::Client::with_config(config);
        OpenAiClient {
            client: inner,
            model: "gpt-4o".to_string(),
            temperature: 0.0,
            max_tokens: 512,
            backoff_config: BackoffConfig {
                initial_interval: Duration::from_millis(50),
                max_interval: Duration::from_secs(1),
                max_elapsed_time: Duration::from_secs(10),
            },
            cache: Cache::builder()
                .max_capacity(100)
                .time_to_live(Duration::from_secs(60))
                .build(),
            token_tracker: Arc::new(TokenTracker::new()),
        }
    }

    fn chat_completions_response(content: &str) -> serde_json::Value {
        json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1700000000_u64,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            }
        })
    }

    fn user_messages(text: &str) -> Vec<Message> {
        vec![Message {
            role: Role::User,
            content: text.to_string(),
        }]
    }

    // ── generate() ───────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_generate_returns_content() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(chat_completions_response("Hello, world!")),
            )
            .mount(&server)
            .await;

        let client = client_for(&server.uri());
        let msgs = user_messages("Say hello");
        let result = client.generate(&msgs).await.expect("generate should succeed");

        assert_eq!(result, "Hello, world!");
    }

    #[tokio::test]
    async fn test_generate_uses_cache_on_second_call() {
        let server = MockServer::start().await;

        // Register the mock for exactly one request.
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(chat_completions_response("cached response")),
            )
            .expect(1) // must be called exactly once
            .mount(&server)
            .await;

        let client = client_for(&server.uri());
        let msgs = user_messages("Same question");

        let r1 = client.generate(&msgs).await.expect("first call");
        let r2 = client.generate(&msgs).await.expect("second call");

        assert_eq!(r1, "cached response");
        assert_eq!(r2, "cached response");
        // wiremock verifies the `expect(1)` on drop
    }

    #[tokio::test]
    async fn test_generate_maps_auth_error() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(401).set_body_json(json!({
                "error": {
                    "message": "Incorrect API key",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key"
                }
            })))
            .mount(&server)
            .await;

        let client = client_for(&server.uri());
        let msgs = user_messages("Hello");
        let err = client.generate(&msgs).await.expect_err("should fail");

        assert!(
            matches!(err, GraphitiError::Llm(LlmError::Authentication)),
            "expected Authentication, got {:?}",
            err
        );
    }

    #[tokio::test]
    async fn test_generate_retries_on_rate_limit() {
        let server = MockServer::start().await;

        // First call returns 429; second call succeeds.
        // async-openai handles 429 retry internally, so the test client's
        // own backoff may not be exercised, but the net result should succeed.
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(429).set_body_json(json!({
                "error": {
                    "message": "Rate limit exceeded",
                    "type": "requests",
                    "code": "rate_limit_exceeded"
                }
            })))
            .up_to_n_times(1)
            .mount(&server)
            .await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(chat_completions_response("after retry")),
            )
            .mount(&server)
            .await;

        let client = client_for(&server.uri());
        let msgs = user_messages("Hello after rate limit");
        let result = client.generate(&msgs).await.expect("should succeed after retry");
        assert_eq!(result, "after retry");
    }

    // ── generate_structured() ────────────────────────────────────────────────

    #[derive(Debug, Deserialize, Serialize, JsonSchema, PartialEq)]
    struct Person {
        name: String,
        age: u32,
    }

    #[tokio::test]
    async fn test_generate_structured_deserializes() {
        let server = MockServer::start().await;

        let body = serde_json::to_string(&Person {
            name: "Alice".to_string(),
            age: 30,
        })
        .expect("Person serialization is infallible");

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(chat_completions_response(&body)),
            )
            .mount(&server)
            .await;

        let client = client_for(&server.uri());
        let msgs = user_messages("Give me a person");
        let person: Person = client
            .generate_structured(&msgs)
            .await
            .expect("structured generation should succeed");

        assert_eq!(person, Person { name: "Alice".to_string(), age: 30 });
    }

    #[tokio::test]
    async fn test_generate_structured_uses_cache() {
        let server = MockServer::start().await;

        let body = serde_json::to_string(&Person {
            name: "Bob".to_string(),
            age: 25,
        })
        .expect("Person serialization is infallible");

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(chat_completions_response(&body)),
            )
            .expect(1)
            .mount(&server)
            .await;

        let client = client_for(&server.uri());
        let msgs = user_messages("Give me Bob");

        let p1: Person = client.generate_structured(&msgs).await.expect("first");
        let p2: Person = client.generate_structured(&msgs).await.expect("cached");

        assert_eq!(p1, p2);
    }

    // ── token tracking ────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_generate_records_token_usage() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(chat_completions_response("tracked")),
            )
            .mount(&server)
            .await;

        let client = client_for(&server.uri());
        let msgs = user_messages("Track tokens");
        client.generate(&msgs).await.expect("generate should succeed");

        let usage = client.token_usage();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 20);
        assert_eq!(usage.total_tokens, 30);
    }

    #[tokio::test]
    async fn test_generate_does_not_record_tokens_on_cache_hit() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(chat_completions_response("cached")),
            )
            .expect(1)
            .mount(&server)
            .await;

        let client = client_for(&server.uri());
        let msgs = user_messages("Cache hit tokens");

        client.generate(&msgs).await.expect("first call");
        client.generate(&msgs).await.expect("second call (cached)");

        // Tokens should only have been recorded once (the first API call).
        let usage = client.token_usage();
        assert_eq!(usage.prompt_tokens, 10, "tokens should only be counted for real API calls");
        assert_eq!(usage.completion_tokens, 20);
    }

    #[tokio::test]
    async fn test_generate_structured_records_token_usage() {
        let server = MockServer::start().await;

        let body = serde_json::to_string(&Person { name: "Carol".to_string(), age: 40 })
            .expect("Person serialization is infallible");

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(chat_completions_response(&body)),
            )
            .mount(&server)
            .await;

        let client = client_for(&server.uri());
        let msgs = user_messages("Give me Carol");
        let _: Person = client.generate_structured(&msgs).await.expect("structured should succeed");

        let usage = client.token_usage();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 20);
        assert_eq!(usage.total_tokens, 30);
    }

    #[tokio::test]
    async fn test_reset_token_usage_clears_counters() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(chat_completions_response("reset test")),
            )
            .mount(&server)
            .await;

        let client = client_for(&server.uri());
        let msgs = user_messages("Before reset");
        client.generate(&msgs).await.expect("generate should succeed");

        let before = client.token_usage();
        assert_eq!(before.total_tokens, 30);

        client.reset_token_usage();

        let after = client.token_usage();
        assert_eq!(after.prompt_tokens, 0);
        assert_eq!(after.completion_tokens, 0);
        assert_eq!(after.total_tokens, 0);
    }

    // ── cache key ─────────────────────────────────────────────────────────────

    #[test]
    fn test_cache_key_differs_by_content() {
        let client = OpenAiClient::new("key", "gpt-4o", CacheConfig::default());
        let msgs_a = user_messages("hello");
        let msgs_b = user_messages("world");
        assert_ne!(
            client.cache_key("text", &msgs_a),
            client.cache_key("text", &msgs_b)
        );
    }

    #[test]
    fn test_cache_key_differs_by_prefix() {
        let client = OpenAiClient::new("key", "gpt-4o", CacheConfig::default());
        let msgs = user_messages("hello");
        assert_ne!(
            client.cache_key("text", &msgs),
            client.cache_key("structured", &msgs)
        );
    }

    #[test]
    fn test_cache_key_differs_by_temperature() {
        let client_a = OpenAiClient::new("key", "gpt-4o", CacheConfig::default());
        let client_b =
            OpenAiClient::new("key", "gpt-4o", CacheConfig::default()).with_temperature(0.7);
        let msgs = user_messages("hello");
        assert_ne!(
            client_a.cache_key("text", &msgs),
            client_b.cache_key("text", &msgs),
        );
    }

    #[test]
    fn test_cache_key_differs_by_max_tokens() {
        let client_a = OpenAiClient::new("key", "gpt-4o", CacheConfig::default());
        let client_b =
            OpenAiClient::new("key", "gpt-4o", CacheConfig::default()).with_max_tokens(100);
        let msgs = user_messages("hello");
        assert_ne!(
            client_a.cache_key("text", &msgs),
            client_b.cache_key("text", &msgs),
        );
    }

    // ── schema helpers ────────────────────────────────────────────────────────

    #[test]
    fn test_add_no_additional_properties_adds_field() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            }
        });
        add_no_additional_properties(&mut schema);
        assert_eq!(schema["additionalProperties"], serde_json::Value::Bool(false));
    }

    #[test]
    fn test_add_no_additional_properties_overwrites_existing_true() {
        // For OpenAI strict mode, additionalProperties MUST be false.
        // Even a pre-existing `true` must be overwritten to avoid API rejection.
        let mut schema = json!({
            "type": "object",
            "properties": {},
            "additionalProperties": true
        });
        add_no_additional_properties(&mut schema);
        assert_eq!(schema["additionalProperties"], serde_json::Value::Bool(false));
    }

    #[test]
    fn test_add_no_additional_properties_recurses_into_nested() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "street": { "type": "string" }
                    }
                }
            }
        });
        add_no_additional_properties(&mut schema);
        assert_eq!(schema["additionalProperties"], serde_json::Value::Bool(false));
        assert_eq!(
            schema["properties"]["address"]["additionalProperties"],
            serde_json::Value::Bool(false)
        );
    }

    // ── ensure_all_properties_required() ─────────────────────────────────────

    #[test]
    fn test_ensure_all_properties_required_sets_required() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "age": { "type": "integer" }
            }
        });
        ensure_all_properties_required(&mut schema);
        let required = schema["required"].as_array().expect("required should be an array");
        let mut keys: Vec<&str> = required.iter().map(|v| v.as_str().unwrap()).collect();
        keys.sort_unstable();
        assert_eq!(keys, vec!["age", "name"]);
    }

    #[test]
    fn test_ensure_all_properties_required_overwrites_existing_partial_required() {
        // schemars omits Option<_> fields from required; we must overwrite.
        let mut schema = json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "nickname": { "type": ["string", "null"] }
            },
            "required": ["name"]
        });
        ensure_all_properties_required(&mut schema);
        let required = schema["required"].as_array().expect("required should be an array");
        let mut keys: Vec<&str> = required.iter().map(|v| v.as_str().unwrap()).collect();
        keys.sort_unstable();
        assert_eq!(keys, vec!["name", "nickname"]);
    }

    #[test]
    fn test_ensure_all_properties_required_recurses_into_nested() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "street": { "type": "string" },
                        "zip": { "type": ["string", "null"] }
                    }
                }
            }
        });
        ensure_all_properties_required(&mut schema);
        let inner_required = schema["properties"]["address"]["required"]
            .as_array()
            .expect("nested required should be an array");
        let mut keys: Vec<&str> = inner_required.iter().map(|v| v.as_str().unwrap()).collect();
        keys.sort_unstable();
        assert_eq!(keys, vec!["street", "zip"]);
    }

    // ── max_tokens_param() ────────────────────────────────────────────────────

    #[test]
    fn test_max_tokens_param_gpt4o_uses_completion_tokens() {
        assert_eq!(max_tokens_param("gpt-4o"), "max_completion_tokens");
        assert_eq!(max_tokens_param("gpt-4o-mini"), "max_completion_tokens");
    }

    #[test]
    fn test_max_tokens_param_gpt4_turbo_uses_completion_tokens() {
        assert_eq!(max_tokens_param("gpt-4-turbo"), "max_completion_tokens");
        assert_eq!(max_tokens_param("gpt-4-turbo-preview"), "max_completion_tokens");
    }

    #[test]
    fn test_max_tokens_param_o_series_uses_completion_tokens() {
        assert_eq!(max_tokens_param("o1"), "max_completion_tokens");
        assert_eq!(max_tokens_param("o1-mini"), "max_completion_tokens");
        assert_eq!(max_tokens_param("o3"), "max_completion_tokens");
        assert_eq!(max_tokens_param("o4-mini"), "max_completion_tokens");
    }

    #[test]
    fn test_max_tokens_param_legacy_uses_max_tokens() {
        assert_eq!(max_tokens_param("gpt-3.5-turbo"), "max_tokens");
        assert_eq!(max_tokens_param("gpt-4"), "max_tokens");
    }

    // ── error mapping ─────────────────────────────────────────────────────────

    #[test]
    fn test_map_openai_error_rate_limit() {
        let api_err = async_openai::error::ApiError {
            message: "rate limited".to_string(),
            r#type: Some("requests".to_string()),
            param: None,
            code: Some("rate_limit_exceeded".to_string()),
        };
        let err = map_openai_error(async_openai::error::OpenAIError::ApiError(api_err));
        assert!(matches!(err, LlmError::RateLimit));
    }

    #[test]
    fn test_map_openai_error_auth() {
        let api_err = async_openai::error::ApiError {
            message: "invalid key".to_string(),
            r#type: Some("invalid_request_error".to_string()),
            param: None,
            code: Some("invalid_api_key".to_string()),
        };
        let err = map_openai_error(async_openai::error::OpenAIError::ApiError(api_err));
        assert!(matches!(err, LlmError::Authentication));
    }

    #[test]
    fn test_map_openai_error_server_error_gets_status_500() {
        // async-openai wraps 5xx with code=None, type=None
        let api_err = async_openai::error::ApiError {
            message: "Internal Server Error".to_string(),
            r#type: None,
            param: None,
            code: None,
        };
        let err = map_openai_error(async_openai::error::OpenAIError::ApiError(api_err));
        assert!(matches!(err, LlmError::Api { status: 500, .. }));
    }
}
