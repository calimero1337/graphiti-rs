//! OpenAI embedding client implementation.
//!
//! Wraps [`async_openai`] to provide [`EmbedderClient`] for the OpenAI
//! Embeddings API, with chunked batch support and exponential-backoff retry.

use async_openai::{
    config::OpenAIConfig,
    error::OpenAIError,
    types::embeddings::CreateEmbeddingRequestArgs,
    Client,
};
use backoff::{future::retry, ExponentialBackoffBuilder};
use std::time::Duration;

use crate::embedder::{EmbedderClient, Embedding};
use crate::errors::{GraphitiError, Result};

/// Default embedding model name.
pub const DEFAULT_MODEL: &str = "text-embedding-3-small";

/// Maximum number of inputs per OpenAI embeddings API call.
const BATCH_CHUNK_SIZE: usize = 2048;

/// Return the embedding dimension for a given model name.
///
/// Falls back to 1536 (the `text-embedding-3-small` dimension) for
/// unrecognised models.
fn model_dim(model: &str) -> usize {
    match model {
        "text-embedding-3-small" | "text-embedding-ada-002" => 1536,
        "text-embedding-3-large" => 3072,
        _ => 1536,
    }
}

/// Classify an [`OpenAIError`] as transient (should retry) or permanent.
///
/// `async-openai` 0.33 removed the `status` field from `ApiError`.
/// We infer transience from the `code` field instead.
fn classify_error(err: OpenAIError) -> backoff::Error<GraphitiError> {
    let msg = err.to_string();
    match err {
        // Network-level failures (timeouts, connection refused) are transient.
        OpenAIError::Reqwest(e) if e.is_timeout() || e.is_connect() => {
            backoff::Error::transient(GraphitiError::Embedder(msg))
        }
        // Rate-limit and server errors are transient.
        OpenAIError::ApiError(ref api_err) => {
            let is_rate_limit = api_err.code.as_deref() == Some("rate_limit_exceeded");
            // async-openai wraps 5xx errors as ApiError with code=None and type=None.
            let is_server_error = api_err.code.is_none() && api_err.r#type.is_none();
            if is_rate_limit || is_server_error {
                backoff::Error::transient(GraphitiError::Embedder(msg))
            } else {
                backoff::Error::permanent(GraphitiError::Embedder(msg))
            }
        }
        // Everything else (auth errors, bad requests, ...) is permanent.
        _ => backoff::Error::permanent(GraphitiError::Embedder(msg)),
    }
}

/// OpenAI embedding client that implements [`EmbedderClient`].
pub struct OpenAiEmbedder {
    client: Client<OpenAIConfig>,
    model: String,
    dim: usize,
}

impl OpenAiEmbedder {
    /// Create a new embedder.
    ///
    /// # Arguments
    /// * `api_key` – OpenAI API key (`sk-…`).
    /// * `model`   – Embedding model name (e.g. [`DEFAULT_MODEL`]).
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let model = model.into();
        let dim = model_dim(&model);
        let config = OpenAIConfig::new().with_api_key(api_key.into());
        Self {
            client: Client::with_config(config),
            model,
            dim,
        }
    }

    /// Create a new embedder pointing at a custom API base URL.
    ///
    /// Used in unit tests where a [`wiremock`] server acts as the API endpoint.
    #[cfg(test)]
    fn with_base_url(
        api_key: impl Into<String>,
        model: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        let model = model.into();
        let dim = model_dim(&model);
        let config = OpenAIConfig::new()
            .with_api_key(api_key.into())
            .with_api_base(base_url.into());
        Self {
            client: Client::with_config(config),
            model,
            dim,
        }
    }

    /// Issue a single embeddings API call for up to [`BATCH_CHUNK_SIZE`] texts.
    ///
    /// Retries on transient network failures with exponential back-off
    /// (initial 500 ms, cap 10 s, total budget 60 s).
    async fn embed_chunk(&self, texts: &[&str]) -> Result<Vec<Embedding>> {
        let backoff_policy = ExponentialBackoffBuilder::new()
            .with_initial_interval(Duration::from_millis(500))
            .with_max_interval(Duration::from_secs(10))
            .with_max_elapsed_time(Some(Duration::from_secs(60)))
            .build();

        // Materialise owned data before entering the retry closure.
        let input: Vec<String> = texts.iter().map(|s| (*s).to_owned()).collect();
        let model = self.model.clone();
        let client = self.client.clone();

        retry(backoff_policy, move || {
            let input = input.clone();
            let model = model.clone();
            let client = client.clone();
            async move {
                let request = CreateEmbeddingRequestArgs::default()
                    .model(model.as_str())
                    .input(input)
                    .build()
                    .map_err(|e| {
                        backoff::Error::permanent(GraphitiError::Embedder(e.to_string()))
                    })?;

                let response = client
                    .embeddings()
                    .create(request)
                    .await
                    .map_err(classify_error)?;

                let embeddings: Vec<Embedding> = response
                    .data
                    .into_iter()
                    .map(|item| item.embedding)
                    .collect();

                Ok(embeddings)
            }
        })
        .await
    }
}

#[async_trait::async_trait]
impl EmbedderClient for OpenAiEmbedder {
    /// Embed a single text string.
    async fn embed(&self, text: &str) -> Result<Embedding> {
        let mut embeddings = self.embed_chunk(&[text]).await?;
        embeddings
            .pop()
            .ok_or_else(|| GraphitiError::Embedder("empty response from embedding API".to_string()))
    }

    /// Embed multiple texts, automatically splitting into chunks of at most
    /// [`BATCH_CHUNK_SIZE`] items to respect OpenAI's per-call limits.
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>> {
        let mut result = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(BATCH_CHUNK_SIZE) {
            let chunk_embeddings = self.embed_chunk(chunk).await?;
            result.extend(chunk_embeddings);
        }
        Ok(result)
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::{
        matchers::{method, path},
        Mock, MockServer, ResponseTemplate,
    };

    /// Build a JSON body mimicking a real OpenAI embeddings response.
    fn make_response(count: usize, dim: usize) -> serde_json::Value {
        let data: Vec<serde_json::Value> = (0..count)
            .map(|i| {
                serde_json::json!({
                    "object": "embedding",
                    "index": i,
                    "embedding": vec![0.1_f32; dim],
                })
            })
            .collect();
        serde_json::json!({
            "object": "list",
            "data": data,
            "model": "text-embedding-3-small",
            "usage": { "prompt_tokens": 8, "total_tokens": 8 },
        })
    }

    /// Mount a successful `POST /embeddings` mock returning `count` embeddings
    /// of `dim` dimensions each.
    async fn mount_ok(server: &MockServer, count: usize, dim: usize) {
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(make_response(count, dim)))
            .mount(server)
            .await;
    }

    fn embedder(server: &MockServer) -> OpenAiEmbedder {
        OpenAiEmbedder::with_base_url("sk-test", DEFAULT_MODEL, server.uri())
    }

    // ── dim() ──────────────────────────────────────────────────────────────

    #[test]
    fn dim_small_model() {
        assert_eq!(
            OpenAiEmbedder::new("key", "text-embedding-3-small").dim(),
            1536
        );
    }

    #[test]
    fn dim_large_model() {
        assert_eq!(
            OpenAiEmbedder::new("key", "text-embedding-3-large").dim(),
            3072
        );
    }

    #[test]
    fn dim_ada_model() {
        assert_eq!(
            OpenAiEmbedder::new("key", "text-embedding-ada-002").dim(),
            1536
        );
    }

    #[test]
    fn dim_unknown_model_defaults_to_1536() {
        assert_eq!(OpenAiEmbedder::new("key", "some-future-model").dim(), 1536);
    }

    // ── embed() ────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn embed_returns_vector_of_correct_length() {
        let server = MockServer::start().await;
        mount_ok(&server, 1, 4).await;

        let embedding = embedder(&server).embed("hello world").await.unwrap();
        assert_eq!(embedding.len(), 4);
    }

    #[tokio::test]
    async fn embed_values_match_mocked_response() {
        let server = MockServer::start().await;
        mount_ok(&server, 1, 3).await;

        let embedding = embedder(&server).embed("test").await.unwrap();
        for &v in &embedding {
            assert!((v - 0.1_f32).abs() < 1e-5, "expected ≈0.1, got {v}");
        }
    }

    #[tokio::test]
    async fn embed_empty_data_is_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "object": "list",
                "data": [],
                "model": "text-embedding-3-small",
                "usage": { "prompt_tokens": 0, "total_tokens": 0 },
            })))
            .mount(&server)
            .await;

        let result = embedder(&server).embed("test").await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GraphitiError::Embedder(_)));
    }

    // ── embed_batch() ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn embed_batch_returns_one_embedding_per_input() {
        let server = MockServer::start().await;
        mount_ok(&server, 3, 4).await;

        let texts = ["alpha", "beta", "gamma"];
        let embeddings = embedder(&server).embed_batch(&texts).await.unwrap();
        assert_eq!(embeddings.len(), 3);
        for emb in &embeddings {
            assert_eq!(emb.len(), 4);
        }
    }

    #[tokio::test]
    async fn embed_batch_empty_slice_returns_empty_vec() {
        // No HTTP call should be made for an empty input slice.
        let server = MockServer::start().await;
        let embeddings = embedder(&server).embed_batch(&[]).await.unwrap();
        assert!(embeddings.is_empty());
    }

    // ── retry behaviour ────────────────────────────────────────────────────

    #[tokio::test]
    async fn embed_retries_on_rate_limit() {
        let server = MockServer::start().await;

        // First call returns 429 — should be retried.
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(429).set_body_json(serde_json::json!({
                "error": {
                    "message": "Rate limit exceeded",
                    "type": "requests",
                    "code": "rate_limit_exceeded",
                }
            })))
            .up_to_n_times(1)
            .mount(&server)
            .await;

        // Second call succeeds.
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(make_response(1, 4)))
            .mount(&server)
            .await;

        // The initial backoff interval is 500 ms, so this test takes ~500 ms.
        let embedding = embedder(&server).embed("retry test").await.unwrap();
        assert_eq!(embedding.len(), 4);
    }

    // ── error mapping ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn api_error_maps_to_embedder_variant() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(401).set_body_json(serde_json::json!({
                "error": {
                    "message": "Incorrect API key provided.",
                    "type": "authentication_error",
                    "param": null,
                    "code": "invalid_api_key",
                }
            })))
            .mount(&server)
            .await;

        let result = embedder(&server).embed("test").await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GraphitiError::Embedder(_)));
    }
}
