//! HTTP embedding client for the local `embedding-server`.
//!
//! [`HttpEmbedder`] talks to a locally-running embedding server that exposes
//! an OpenAI-compatible embeddings API:
//!
//! ```text
//! POST /v1/embeddings
//! { "model": "...", "input": "..." | ["...", ...] }
//! →
//! { "object": "list", "data": [{ "index": 0, "embedding": [...] }], ... }
//! ```
//!
//! # Example
//! ```rust,no_run
//! use graphiti_rs::embedder::http::HttpEmbedder;
//! use graphiti_rs::embedder::EmbedderClient;
//!
//! # #[tokio::main] async fn main() -> graphiti_rs::Result<()> {
//! let embedder = HttpEmbedder::new("http://localhost:8080", "bge-base-en-v1.5", 768)?;
//! let v = embedder.embed("hello world").await?;
//! assert_eq!(v.len(), 768);
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};

use crate::embedder::{EmbedderClient, Embedding};
use crate::errors::{GraphitiError, Result};

/// Default base URL for the K8s-hosted embedding server.
pub const DEFAULT_BASE_URL: &str = "http://embedding-server.claude.svc.cluster.local:8080";

/// Default embedding model served by the embedding server.
pub const DEFAULT_MODEL: &str = "bge-base-en-v1.5";

/// Default embedding dimension for [`DEFAULT_MODEL`] (`bge-base-en-v1.5`).
pub const DEFAULT_DIM: usize = 768;

// ── Request / response shapes ──────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedData>,
}

#[derive(Debug, Deserialize)]
struct EmbedData {
    /// Positional index of the corresponding input text, as returned by the
    /// server.  The OpenAI embeddings API does **not** guarantee that `data`
    /// entries appear in the same order as the inputs, so we sort by this
    /// field before extracting embeddings.
    index: usize,
    embedding: Vec<f32>,
}

// ── HttpEmbedder ──────────────────────────────────────────────────────────────

/// HTTP embedding client that speaks the OpenAI embeddings API.
pub struct HttpEmbedder {
    client: reqwest::Client,
    base_url: String,
    model: String,
    dim: usize,
}

impl HttpEmbedder {
    /// Create a new [`HttpEmbedder`] with default timeouts (connect: 5s, request: 30s).
    ///
    /// * `base_url` – e.g. `"http://localhost:8080"`
    /// * `model`    – model name sent in every request body
    /// * `dim`      – expected embedding dimensionality (used by [`EmbedderClient::dim`])
    pub fn new(base_url: impl Into<String>, model: impl Into<String>, dim: usize) -> Result<Self> {
        Self::with_timeout(
            base_url,
            model,
            dim,
            std::time::Duration::from_secs(30),
        )
    }

    /// Create an [`HttpEmbedder`] with a custom request timeout.
    ///
    /// A connect timeout of 5 seconds is always applied; `timeout` controls the
    /// total time allowed for the entire request (including response body).
    pub fn with_timeout(
        base_url: impl Into<String>,
        model: impl Into<String>,
        dim: usize,
        timeout: std::time::Duration,
    ) -> Result<Self> {
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(5))
            .timeout(timeout)
            .build()
            .map_err(|e| GraphitiError::Embedder(format!("failed to build HTTP client: {e}")))?;
        Ok(Self {
            client,
            base_url: base_url.into(),
            model: model.into(),
            dim,
        })
    }

    /// Create an [`HttpEmbedder`] using the default K8s embedding-server settings.
    ///
    /// Equivalent to `HttpEmbedder::new(DEFAULT_BASE_URL, DEFAULT_MODEL, DEFAULT_DIM)`.
    pub fn local() -> Result<Self> {
        Self::new(DEFAULT_BASE_URL, DEFAULT_MODEL, DEFAULT_DIM)
    }

    /// POST to `{base_url}/v1/embeddings` and return the raw response.
    async fn post_embeddings(&self, input: serde_json::Value) -> Result<EmbedResponse> {
        let url = format!("{}/v1/embeddings", self.base_url);
        let body = EmbedRequest {
            model: &self.model,
            input,
        };
        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| GraphitiError::Embedder(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response.text().await.unwrap_or_default();
            return Err(GraphitiError::Embedder(format!("HTTP {status}: {text}")));
        }

        response
            .json::<EmbedResponse>()
            .await
            .map_err(|e| GraphitiError::Embedder(e.to_string()))
    }
}

#[async_trait::async_trait]
impl EmbedderClient for HttpEmbedder {
    async fn embed(&self, text: &str) -> Result<Embedding> {
        let resp = self
            .post_embeddings(serde_json::Value::String(text.to_owned()))
            .await?;

        resp.data
            .into_iter()
            .next()
            .map(|d| d.embedding)
            .ok_or_else(|| GraphitiError::Embedder("empty data array in response".into()))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let input = serde_json::Value::Array(
            texts
                .iter()
                .map(|t| serde_json::Value::String(t.to_string()))
                .collect(),
        );
        let resp = self.post_embeddings(input).await?;

        // The OpenAI embeddings API does **not** guarantee that `data` entries
        // are returned in the same order as the input texts.  Each entry
        // carries an `index` that maps it back to its input position.  We sort
        // by that index before collecting so that the returned `Vec<Embedding>`
        // is always aligned with the original `texts` slice.
        let mut data = resp.data;
        data.sort_unstable_by_key(|d| d.index);
        Ok(data.into_iter().map(|d| d.embedding).collect())
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::{
        matchers::{body_partial_json, header, method, path},
        Mock, MockServer, ResponseTemplate,
    };

    use crate::embedder::EmbedderClient;
    use crate::errors::GraphitiError;

    // ── helpers ────────────────────────────────────────────────────────────

    /// Build an OpenAI-compatible embedding response JSON.
    fn make_response(count: usize, dim: usize) -> serde_json::Value {
        let data: Vec<serde_json::Value> = (0..count)
            .map(|i| {
                serde_json::json!({
                    "object": "embedding",
                    "index": i,
                    "embedding": vec![0.42_f32; dim],
                })
            })
            .collect();
        serde_json::json!({
            "object": "list",
            "model": "bge-base-en-v1.5",
            "data": data,
            "usage": { "prompt_tokens": count * 4, "total_tokens": count * 4 },
        })
    }

    /// Mount a successful `POST /v1/embeddings` mock.
    async fn mount_ok(server: &MockServer, count: usize, dim: usize) {
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(make_response(count, dim)))
            .mount(server)
            .await;
    }

    /// Construct an [`HttpEmbedder`] pointing at the mock server.
    fn embedder(server: &MockServer, dim: usize) -> HttpEmbedder {
        HttpEmbedder::new(server.uri(), "bge-small-en-v1.5", dim).unwrap()
    }

    // ── dim() ──────────────────────────────────────────────────────────────

    #[test]
    fn dim_returns_configured_value() {
        let e = HttpEmbedder::new("http://localhost:8080", "bge-base-en-v1.5", 768).unwrap();
        assert_eq!(e.dim(), 768);
    }

    #[test]
    fn dim_768_is_preserved() {
        let e = HttpEmbedder::new("http://localhost:8080", "bge-large-en-v1.5", 768).unwrap();
        assert_eq!(e.dim(), 768);
    }

    // ── embed() ────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn embed_returns_vector_of_correct_length() {
        let server = MockServer::start().await;
        mount_ok(&server, 1, 4).await;

        let embedding = embedder(&server, 4).embed("hello world").await.unwrap();
        assert_eq!(embedding.len(), 4);
    }

    #[tokio::test]
    async fn embed_values_match_mocked_response() {
        let server = MockServer::start().await;
        mount_ok(&server, 1, 3).await;

        let embedding = embedder(&server, 3).embed("test text").await.unwrap();
        for &v in &embedding {
            assert!((v - 0.42_f32).abs() < 1e-5, "expected ≈0.42, got {v}");
        }
    }

    #[tokio::test]
    async fn embed_posts_to_v1_embeddings_path() {
        let server = MockServer::start().await;

        // Only match the exact path — if HttpEmbedder hits a different path,
        // the mock won't match and the request will 404.
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(make_response(1, 2)))
            .expect(1)
            .mount(&server)
            .await;

        let _ = embedder(&server, 2).embed("path check").await.unwrap();
        server.verify().await;
    }

    #[tokio::test]
    async fn embed_sends_correct_content_type() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .and(header("content-type", "application/json"))
            .respond_with(ResponseTemplate::new(200).set_body_json(make_response(1, 2)))
            .expect(1)
            .mount(&server)
            .await;

        let _ = embedder(&server, 2)
            .embed("content-type check")
            .await
            .unwrap();
        server.verify().await;
    }

    #[tokio::test]
    async fn embed_empty_data_response_is_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "object": "list",
                "model": "bge-small-en-v1.5",
                "data": [],
                "usage": { "prompt_tokens": 0, "total_tokens": 0 },
            })))
            .mount(&server)
            .await;

        let result = embedder(&server, 4).embed("test").await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GraphitiError::Embedder(_)));
    }

    #[tokio::test]
    async fn embed_http_500_is_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
            .mount(&server)
            .await;

        let result = embedder(&server, 4).embed("fail").await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GraphitiError::Embedder(_)));
    }

    #[tokio::test]
    async fn embed_http_404_is_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(404).set_body_string("Not Found"))
            .mount(&server)
            .await;

        let result = embedder(&server, 4).embed("not found").await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GraphitiError::Embedder(_)));
    }

    #[tokio::test]
    async fn embed_connection_refused_is_error() {
        // Point at a port nothing is listening on.
        let e = HttpEmbedder::new("http://127.0.0.1:19999", "model", 4).unwrap();
        let result = e.embed("unreachable").await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GraphitiError::Embedder(_)));
    }

    // ── embed_batch() ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn embed_batch_returns_one_embedding_per_input() {
        let server = MockServer::start().await;
        mount_ok(&server, 3, 4).await;

        let texts = ["alpha", "beta", "gamma"];
        let embeddings = embedder(&server, 4).embed_batch(&texts).await.unwrap();
        assert_eq!(embeddings.len(), 3);
        for emb in &embeddings {
            assert_eq!(emb.len(), 4);
        }
    }

    #[tokio::test]
    async fn embed_batch_empty_slice_returns_empty_vec_without_http_call() {
        // No mock is registered — if any HTTP call is made the server will
        // return a 404 and the test will fail.
        let server = MockServer::start().await;
        let embeddings = embedder(&server, 4).embed_batch(&[]).await.unwrap();
        assert!(embeddings.is_empty());
    }

    #[tokio::test]
    async fn embed_batch_single_element_works() {
        let server = MockServer::start().await;
        mount_ok(&server, 1, 8).await;

        let embeddings = embedder(&server, 8).embed_batch(&["only"]).await.unwrap();
        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), 8);
    }

    #[tokio::test]
    async fn embed_batch_http_error_propagates() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(503).set_body_string("unavailable"))
            .mount(&server)
            .await;

        let result = embedder(&server, 4).embed_batch(&["a", "b"]).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GraphitiError::Embedder(_)));
    }

    /// The OpenAI spec does not guarantee that `data` entries are returned in
    /// input order.  This test simulates a server that returns the three
    /// embeddings in reverse order (index 2, 1, 0) and asserts that
    /// `embed_batch` re-sorts them so the caller receives them in input order.
    #[tokio::test]
    async fn embed_batch_sorts_by_index_when_response_is_out_of_order() {
        let server = MockServer::start().await;

        // Build a response where `data` is deliberately in reverse order.
        // Each embedding is filled with a distinct sentinel value (0.0, 1.0, 2.0)
        // keyed to its logical input position so we can assert correct ordering.
        let reversed_response = serde_json::json!({
            "object": "list",
            "model": "bge-small-en-v1.5",
            "data": [
                { "object": "embedding", "index": 2, "embedding": [2.0_f32, 2.0_f32] },
                { "object": "embedding", "index": 0, "embedding": [0.0_f32, 0.0_f32] },
                { "object": "embedding", "index": 1, "embedding": [1.0_f32, 1.0_f32] },
            ],
            "usage": { "prompt_tokens": 6, "total_tokens": 6 },
        });

        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(reversed_response))
            .mount(&server)
            .await;

        let embeddings = embedder(&server, 2)
            .embed_batch(&["first", "second", "third"])
            .await
            .unwrap();

        assert_eq!(embeddings.len(), 3);
        // After sorting by index the embeddings must be in input order.
        assert!((embeddings[0][0] - 0.0_f32).abs() < 1e-6, "index-0 embedding should come first");
        assert!((embeddings[1][0] - 1.0_f32).abs() < 1e-6, "index-1 embedding should come second");
        assert!((embeddings[2][0] - 2.0_f32).abs() < 1e-6, "index-2 embedding should come third");
    }

    // ── request body shape ─────────────────────────────────────────────────

    #[tokio::test]
    async fn embed_request_body_contains_model_field() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .and(body_partial_json(serde_json::json!({
                "model": "bge-small-en-v1.5",
                "input": "body check"
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(make_response(1, 2)))
            .expect(1)
            .mount(&server)
            .await;

        let _ = embedder(&server, 2).embed("body check").await.unwrap();
        server.verify().await;
    }

    // ── timeout ────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn embed_timeout_is_error() {
        let server = MockServer::start().await;
        // Respond with a 2-minute delay — well beyond the short timeout used below.
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(make_response(1, 4))
                    .set_delay(std::time::Duration::from_secs(120)),
            )
            .mount(&server)
            .await;

        let e = HttpEmbedder::with_timeout(
            server.uri(),
            "model",
            4,
            std::time::Duration::from_millis(100),
        )
        .unwrap();
        let result = e.embed("slow").await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GraphitiError::Embedder(_)));
    }
}
