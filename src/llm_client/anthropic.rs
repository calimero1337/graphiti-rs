//! Anthropic LLM client — direct HTTP calls to the Messages API.
//!
//! Uses `reqwest` to call `https://api.anthropic.com/v1/messages` directly,
//! avoiding the need for the `claude` CLI or Node.js in the container.
//!
//! This is the preferred backend for production/K8s deployments.

use std::sync::Arc;
use std::time::Duration;

use backoff::ExponentialBackoffBuilder;
use moka::future::Cache;
use serde::Deserialize;
use serde_json::json;
use tracing::{debug, warn};

use crate::errors::{GraphitiError, LlmError, Result};

use super::openai::CacheConfig;
use super::token_tracker::{TokenTracker, TokenUsage};
use super::{LlmClient, Message, Role};

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Anthropic Messages API client implementing [`LlmClient`].
pub struct AnthropicClient {
    http: reqwest::Client,
    api_key: String,
    model: String,
    max_tokens: u32,
    cache: Cache<String, String>,
    token_tracker: Arc<TokenTracker>,
}

/// Response from the Anthropic Messages API.
#[derive(Debug, Deserialize)]
struct MessagesResponse {
    content: Vec<ContentBlock>,
    usage: Option<Usage>,
    #[serde(default)]
    #[allow(dead_code)]
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Usage {
    input_tokens: u64,
    output_tokens: u64,
}

impl AnthropicClient {
    /// Create a new Anthropic client.
    ///
    /// # Arguments
    /// * `api_key` — Anthropic API key (starts with `sk-ant-`).
    /// * `model`   — Model name (e.g. `"claude-sonnet-4-6"`).
    /// * `cache_config` — Cache capacity and TTL.
    pub fn new(
        api_key: impl Into<String>,
        model: impl Into<String>,
        cache_config: CacheConfig,
    ) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("reqwest client should build");

        let cache = Cache::builder()
            .max_capacity(cache_config.max_capacity)
            .time_to_live(cache_config.ttl)
            .build();

        Self {
            http,
            api_key: api_key.into(),
            model: model.into(),
            max_tokens: 8_192,
            cache,
            token_tracker: Arc::new(TokenTracker::new()),
        }
    }

    /// Override the max output token limit (default `8192`).
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Build the request body for the Messages API.
    fn build_request(
        &self,
        messages: &[Message],
        system: Option<&str>,
    ) -> serde_json::Value {
        let api_messages: Vec<serde_json::Value> = messages
            .iter()
            .filter(|m| !matches!(m.role, Role::System))
            .map(|m| {
                json!({
                    "role": role_str(&m.role),
                    "content": m.content,
                })
            })
            .collect();

        let mut body = json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": api_messages,
        });

        // Collect system messages or use override.
        let system_text = if let Some(s) = system {
            Some(s.to_string())
        } else {
            let parts: Vec<&str> = messages
                .iter()
                .filter(|m| matches!(m.role, Role::System))
                .map(|m| m.content.as_str())
                .collect();
            if parts.is_empty() {
                None
            } else {
                Some(parts.join("\n\n"))
            }
        };

        if let Some(sys) = system_text {
            body["system"] = json!(sys);
        }

        body
    }

    /// Send a request to the Messages API with retry on rate limits / 5xx.
    async fn call_with_retry(&self, body: serde_json::Value) -> Result<MessagesResponse> {
        let backoff = ExponentialBackoffBuilder::new()
            .with_initial_interval(Duration::from_millis(500))
            .with_max_interval(Duration::from_secs(60))
            .with_max_elapsed_time(Some(Duration::from_secs(300)))
            .build();

        backoff::future::retry(backoff, || async {
            let response = self
                .http
                .post(ANTHROPIC_API_URL)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", ANTHROPIC_VERSION)
                .header("content-type", "application/json")
                .json(&body)
                .send()
                .await
                .map_err(|e| {
                    backoff::Error::transient(LlmError::Api {
                        status: e.status().map(|s| s.as_u16()).unwrap_or(0),
                        message: e.to_string(),
                    })
                })?;

            let status = response.status().as_u16();

            if status == 200 {
                let msg_response: MessagesResponse =
                    response.json().await.map_err(|e| {
                        backoff::Error::permanent(LlmError::Api {
                            status: 0,
                            message: format!("Failed to parse response: {e}"),
                        })
                    })?;
                Ok(msg_response)
            } else if status == 429 {
                warn!("Anthropic rate limit hit — retrying with backoff");
                let _body = response.text().await.unwrap_or_default();
                Err(backoff::Error::transient(LlmError::RateLimit))
            } else if status == 401 {
                Err(backoff::Error::permanent(LlmError::Authentication))
            } else if status >= 500 {
                let text = response.text().await.unwrap_or_default();
                warn!("Anthropic server error ({status}) — retrying: {text}");
                Err(backoff::Error::transient(LlmError::Api {
                    status,
                    message: text,
                }))
            } else {
                let text = response.text().await.unwrap_or_default();
                Err(backoff::Error::permanent(LlmError::Api {
                    status,
                    message: text,
                }))
            }
        })
        .await
        .map_err(GraphitiError::Llm)
    }

    /// Extract the text content from a Messages response.
    fn extract_text(response: &MessagesResponse) -> Result<String> {
        let text_parts: Vec<&str> = response
            .content
            .iter()
            .filter(|b| b.block_type == "text")
            .filter_map(|b| b.text.as_deref())
            .collect();

        if text_parts.is_empty() {
            return Err(GraphitiError::Llm(LlmError::EmptyResponse));
        }

        Ok(text_parts.join(""))
    }

    /// Record token usage from a Messages response.
    fn record_usage(&self, response: &MessagesResponse) {
        if let Some(usage) = &response.usage {
            self.token_tracker
                .record(usage.input_tokens, usage.output_tokens);
        }
    }

    /// Compute a cache key from model + messages.
    fn cache_key(&self, prefix: &str, messages: &[Message]) -> String {
        use md5::{Digest, Md5};
        let mut h = Md5::new();
        h.update(prefix.as_bytes());
        h.update(self.model.as_bytes());
        h.update(self.max_tokens.to_le_bytes());
        for m in messages {
            let role = role_str(&m.role);
            h.update(role.as_bytes());
            h.update(m.content.as_bytes());
        }
        format!("{:x}", h.finalize())
    }
}

#[async_trait::async_trait]
impl LlmClient for AnthropicClient {
    async fn generate(&self, messages: &[Message]) -> Result<String> {
        let key = self.cache_key("text", messages);

        if let Some(cached) = self.cache.get(&key).await {
            debug!("Anthropic cache hit (text)");
            return Ok(cached);
        }

        let body = self.build_request(messages, None);
        debug!(model = %self.model, "anthropic generate");

        let response = self.call_with_retry(body).await?;
        self.record_usage(&response);
        let content = Self::extract_text(&response)?;

        self.cache.insert(key, content.clone()).await;
        Ok(content)
    }

    async fn generate_structured_json(
        &self,
        messages: &[Message],
        schema: serde_json::Value,
    ) -> Result<String> {
        let schema_str = serde_json::to_string_pretty(&schema)
            .unwrap_or_else(|_| schema.to_string());
        let key = self.cache_key(&schema_str, messages);

        if let Some(cached) = self.cache.get(&key).await {
            debug!("Anthropic cache hit (structured)");
            return Ok(cached);
        }

        // Build system prompt with JSON schema constraint.
        let system_parts: Vec<&str> = messages
            .iter()
            .filter(|m| matches!(m.role, Role::System))
            .map(|m| m.content.as_str())
            .collect();

        let base_system = if system_parts.is_empty() {
            "You are a helpful assistant.".to_string()
        } else {
            system_parts.join("\n\n")
        };

        let structured_system = format!(
            "{base_system}\n\n\
            CRITICAL: You MUST respond with ONLY valid JSON matching this schema. \
            No markdown, no code fences, no explanation — ONLY the JSON object.\n\n\
            JSON Schema:\n```json\n{schema_str}\n```"
        );

        let body = self.build_request(messages, Some(&structured_system));
        debug!(model = %self.model, "anthropic generate_structured_json");

        let response = self.call_with_retry(body).await?;
        self.record_usage(&response);
        let mut content = Self::extract_text(&response)?;

        // Strip markdown code fences if present.
        content = strip_code_fences(&content);

        if content.is_empty() {
            return Err(GraphitiError::Pipeline(
                "Anthropic returned empty structured response".into(),
            ));
        }

        // Validate JSON.
        if let Err(e) = serde_json::from_str::<serde_json::Value>(&content) {
            return Err(GraphitiError::Pipeline(format!(
                "Anthropic response is not valid JSON: {e}\nResponse: {content}"
            )));
        }

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

fn role_str(role: &Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
    }
}

/// Strip markdown code fences from a JSON response.
fn strip_code_fences(s: &str) -> String {
    let trimmed = s.trim();
    if trimmed.starts_with("```") {
        let without_opening = if let Some(rest) = trimmed.strip_prefix("```json") {
            rest
        } else if let Some(rest) = trimmed.strip_prefix("```JSON") {
            rest
        } else {
            trimmed.strip_prefix("```").unwrap_or(trimmed)
        };
        let without_closing = without_opening
            .trim()
            .strip_suffix("```")
            .unwrap_or(without_opening);
        return without_closing.trim().to_string();
    }
    trimmed.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, method};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn anthropic_response(text: &str) -> serde_json::Value {
        json!({
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": text,
            }],
            "model": "claude-sonnet-4-6",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 15,
                "output_tokens": 25,
            }
        })
    }

    fn client_for(base_url: &str) -> AnthropicClient {
        let http = reqwest::Client::new();
        AnthropicClient {
            http,
            api_key: "test-key".to_string(),
            model: "claude-sonnet-4-6".to_string(),
            max_tokens: 512,
            cache: Cache::builder()
                .max_capacity(100)
                .time_to_live(Duration::from_secs(60))
                .build(),
            token_tracker: Arc::new(TokenTracker::new()),
        }
    }

    /// Helper: replace the API URL for testing (uses a mock server).
    /// Since ANTHROPIC_API_URL is a const, we test via the reqwest client directly.
    /// For integration tests we'd override the URL; for unit tests we verify
    /// the request building and response parsing logic.

    fn user_messages(text: &str) -> Vec<Message> {
        vec![Message {
            role: Role::User,
            content: text.to_string(),
        }]
    }

    #[test]
    fn build_request_basic() {
        let client = AnthropicClient::new("key", "claude-sonnet-4-6", CacheConfig::default());
        let msgs = vec![
            Message { role: Role::System, content: "You extract entities.".into() },
            Message { role: Role::User, content: "Alice is an engineer.".into() },
        ];
        let body = client.build_request(&msgs, None);

        assert_eq!(body["model"], "claude-sonnet-4-6");
        assert_eq!(body["system"], "You extract entities.");
        let api_msgs = body["messages"].as_array().expect("messages array");
        assert_eq!(api_msgs.len(), 1); // system filtered out
        assert_eq!(api_msgs[0]["role"], "user");
    }

    #[test]
    fn build_request_with_system_override() {
        let client = AnthropicClient::new("key", "claude-sonnet-4-6", CacheConfig::default());
        let msgs = vec![
            Message { role: Role::System, content: "Original system.".into() },
            Message { role: Role::User, content: "Hello".into() },
        ];
        let body = client.build_request(&msgs, Some("Override system"));
        assert_eq!(body["system"], "Override system");
    }

    #[test]
    fn extract_text_from_response() {
        let resp = MessagesResponse {
            content: vec![ContentBlock {
                block_type: "text".into(),
                text: Some("Hello world".into()),
            }],
            usage: Some(Usage { input_tokens: 10, output_tokens: 5 }),
            stop_reason: Some("end_turn".into()),
        };
        let text = AnthropicClient::extract_text(&resp).expect("should extract text");
        assert_eq!(text, "Hello world");
    }

    #[test]
    fn extract_text_empty_response() {
        let resp = MessagesResponse {
            content: vec![],
            usage: None,
            stop_reason: None,
        };
        let result = AnthropicClient::extract_text(&resp);
        assert!(result.is_err());
    }

    #[test]
    fn strip_code_fences_json() {
        assert_eq!(strip_code_fences("```json\n{\"a\": 1}\n```"), "{\"a\": 1}");
    }

    #[test]
    fn strip_code_fences_plain() {
        assert_eq!(strip_code_fences("```\n{\"a\": 1}\n```"), "{\"a\": 1}");
    }

    #[test]
    fn strip_code_fences_no_fences() {
        assert_eq!(strip_code_fences("{\"a\": 1}"), "{\"a\": 1}");
    }

    #[test]
    fn cache_key_differs_by_content() {
        let client = AnthropicClient::new("key", "claude-sonnet-4-6", CacheConfig::default());
        let msgs_a = user_messages("hello");
        let msgs_b = user_messages("world");
        assert_ne!(
            client.cache_key("text", &msgs_a),
            client.cache_key("text", &msgs_b),
        );
    }

    #[test]
    fn cache_key_differs_by_prefix() {
        let client = AnthropicClient::new("key", "claude-sonnet-4-6", CacheConfig::default());
        let msgs = user_messages("hello");
        assert_ne!(
            client.cache_key("text", &msgs),
            client.cache_key("structured", &msgs),
        );
    }

    #[test]
    fn token_tracking() {
        let client = AnthropicClient::new("key", "claude-sonnet-4-6", CacheConfig::default());
        let resp = MessagesResponse {
            content: vec![],
            usage: Some(Usage { input_tokens: 100, output_tokens: 50 }),
            stop_reason: None,
        };
        client.record_usage(&resp);
        let usage = client.token_usage();
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 50);
        assert_eq!(usage.total_tokens, 150);

        client.reset_token_usage();
        let usage = client.token_usage();
        assert_eq!(usage.total_tokens, 0);
    }
}
