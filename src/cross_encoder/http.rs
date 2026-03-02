//! HTTP cross-encoder client for remote reranking services.
//!
//! [`HttpCrossEncoder`] talks to any service that implements the Cohere
//! reranking API schema:
//!
//! ```text
//! POST /v1/rerank
//! { "model": "...", "query": "...", "documents": ["..."], "top_n": N }
//! →
//! { "results": [{ "index": 0, "relevance_score": 0.95 }, ...] }
//! ```
//!
//! Compatible services include Cohere, Jina AI, and local servers such as
//! [text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference).
//!
//! # Example
//! ```rust,no_run
//! use graphiti_rs::cross_encoder::{CrossEncoderClient, HttpCrossEncoder};
//!
//! # #[tokio::main] async fn main() -> graphiti_rs::Result<()> {
//! let reranker = HttpCrossEncoder::new(
//!     "http://localhost:8080",
//!     "cross-encoder/ms-marco-MiniLM-L-6-v2",
//! )?;
//! let scores = reranker
//!     .score("What is Rust?", &["Rust is a systems language", "Python is dynamic"])
//!     .await?;
//! // scores[0] corresponds to the first document
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};

use crate::cross_encoder::CrossEncoderClient;
use crate::errors::{GraphitiError, Result};

/// Default API path for the reranking endpoint.
pub const DEFAULT_RERANK_PATH: &str = "/v1/rerank";

/// Default base URL for a locally-hosted reranking server.
pub const DEFAULT_BASE_URL: &str = "http://reranker.claude.svc.cluster.local:8080";

// ── Request / response shapes ──────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct RerankRequest<'a> {
    model: &'a str,
    query: &'a str,
    documents: &'a [&'a str],
    /// Request all documents so we can map scores back by index.
    top_n: usize,
}

#[derive(Debug, Deserialize)]
struct RerankResponse {
    results: Vec<RerankResult>,
}

#[derive(Debug, Deserialize)]
struct RerankResult {
    /// Position in the original `documents` input slice.
    index: usize,
    /// Relevance score for this `(query, document)` pair.
    relevance_score: f32,
}

// ── HttpCrossEncoder ──────────────────────────────────────────────────────────

/// HTTP cross-encoder client that speaks the Cohere reranking API.
pub struct HttpCrossEncoder {
    client: reqwest::Client,
    /// Full URL including path, e.g. `http://localhost:8080/v1/rerank`.
    url: String,
    model: String,
}

impl HttpCrossEncoder {
    /// Create an [`HttpCrossEncoder`] with default timeouts (connect: 5 s, request: 30 s).
    ///
    /// * `base_url` – e.g. `"http://localhost:8080"`
    /// * `model`    – model name sent in every request body
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Result<Self> {
        Self::with_timeout(base_url, model, std::time::Duration::from_secs(30))
    }

    /// Create an [`HttpCrossEncoder`] with a custom request timeout.
    ///
    /// A connect timeout of 5 seconds is always applied; `timeout` controls the
    /// total time allowed for the entire request (including response body).
    pub fn with_timeout(
        base_url: impl Into<String>,
        model: impl Into<String>,
        timeout: std::time::Duration,
    ) -> Result<Self> {
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(5))
            .timeout(timeout)
            .build()
            .map_err(|e| GraphitiError::Reranker(format!("failed to build HTTP client: {e}")))?;
        let base_url = base_url.into();
        let url = format!("{base_url}{DEFAULT_RERANK_PATH}");
        Ok(Self {
            client,
            url,
            model: model.into(),
        })
    }

    /// Create an [`HttpCrossEncoder`] pointing at the default K8s reranking service.
    pub fn local(model: impl Into<String>) -> Result<Self> {
        Self::new(DEFAULT_BASE_URL, model)
    }
}

#[async_trait::async_trait]
impl CrossEncoderClient for HttpCrossEncoder {
    async fn score(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let body = RerankRequest {
            model: &self.model,
            query,
            documents,
            top_n: documents.len(),
        };

        let response = self
            .client
            .post(&self.url)
            .json(&body)
            .send()
            .await
            .map_err(|e| GraphitiError::Reranker(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response.text().await.unwrap_or_default();
            return Err(GraphitiError::Reranker(format!("HTTP {status}: {text}")));
        }

        let resp: RerankResponse = response
            .json()
            .await
            .map_err(|e| GraphitiError::Reranker(e.to_string()))?;

        // Map each result back to its original document position.
        // The API may return results in any order; `index` links each score
        // back to its input document.  Out-of-bounds indices are silently
        // ignored to guard against malformed server responses.
        let mut scores = vec![0.0_f32; documents.len()];
        for result in resp.results {
            if result.index < scores.len() {
                scores[result.index] = result.relevance_score;
            }
        }
        Ok(scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::{
        matchers::{body_partial_json, header, method, path},
        Mock, MockServer, ResponseTemplate,
    };

    use crate::cross_encoder::CrossEncoderClient;
    use crate::errors::GraphitiError;

    // ── helpers ────────────────────────────────────────────────────────────

    /// Build a Cohere-compatible rerank response JSON.
    fn make_response(scores: &[(usize, f32)]) -> serde_json::Value {
        let results: Vec<serde_json::Value> = scores
            .iter()
            .map(|(idx, score)| {
                serde_json::json!({
                    "index": idx,
                    "relevance_score": score,
                })
            })
            .collect();
        serde_json::json!({ "results": results })
    }

    /// Construct an [`HttpCrossEncoder`] pointing at the mock server.
    fn reranker(server: &MockServer) -> HttpCrossEncoder {
        HttpCrossEncoder::new(server.uri(), "cross-encoder/ms-marco-MiniLM-L-6-v2").unwrap()
    }

    // ── score() — success paths ────────────────────────────────────────────

    #[tokio::test]
    async fn score_returns_one_score_per_document() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/rerank"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(make_response(&[(0, 0.9), (1, 0.3), (2, 0.6)])),
            )
            .mount(&server)
            .await;

        let scores = reranker(&server)
            .score("query", &["doc0", "doc1", "doc2"])
            .await
            .unwrap();

        assert_eq!(scores.len(), 3);
    }

    #[tokio::test]
    async fn score_values_match_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/rerank"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(make_response(&[(0, 0.8), (1, 0.2)])),
            )
            .mount(&server)
            .await;

        let scores = reranker(&server)
            .score("query", &["relevant", "irrelevant"])
            .await
            .unwrap();

        assert!((scores[0] - 0.8_f32).abs() < 1e-5, "index-0 score mismatch");
        assert!((scores[1] - 0.2_f32).abs() < 1e-5, "index-1 score mismatch");
    }

    /// The server may return results in a different order than the input.
    /// Scores must be reordered to match the original document positions.
    #[tokio::test]
    async fn score_reorders_by_index_when_response_is_out_of_order() {
        let server = MockServer::start().await;
        // Server returns results in reverse document order.
        Mock::given(method("POST"))
            .and(path("/v1/rerank"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "results": [
                        { "index": 2, "relevance_score": 0.3 },
                        { "index": 0, "relevance_score": 0.9 },
                        { "index": 1, "relevance_score": 0.6 },
                    ]
                })),
            )
            .mount(&server)
            .await;

        let scores = reranker(&server)
            .score("query", &["doc0", "doc1", "doc2"])
            .await
            .unwrap();

        assert_eq!(scores.len(), 3);
        assert!((scores[0] - 0.9_f32).abs() < 1e-5, "index-0 should be 0.9");
        assert!((scores[1] - 0.6_f32).abs() < 1e-5, "index-1 should be 0.6");
        assert!((scores[2] - 0.3_f32).abs() < 1e-5, "index-2 should be 0.3");
    }

    #[tokio::test]
    async fn score_empty_documents_returns_empty_without_http_call() {
        // No mock registered — an HTTP call would result in a 404 error.
        let server = MockServer::start().await;
        let scores = reranker(&server).score("query", &[]).await.unwrap();
        assert!(scores.is_empty());
    }

    // ── score() — request shape ────────────────────────────────────────────

    #[tokio::test]
    async fn score_posts_to_v1_rerank_path() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/rerank"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(make_response(&[(0, 0.5)])),
            )
            .expect(1)
            .mount(&server)
            .await;

        let _ = reranker(&server).score("q", &["doc"]).await.unwrap();
        server.verify().await;
    }

    #[tokio::test]
    async fn score_sends_json_content_type() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/rerank"))
            .and(header("content-type", "application/json"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(make_response(&[(0, 0.5)])),
            )
            .expect(1)
            .mount(&server)
            .await;

        let _ = reranker(&server).score("q", &["doc"]).await.unwrap();
        server.verify().await;
    }

    #[tokio::test]
    async fn score_request_body_has_required_fields() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/rerank"))
            .and(body_partial_json(serde_json::json!({
                "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "query": "q",
                "documents": ["doc"],
                "top_n": 1
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(make_response(&[(0, 0.5)])),
            )
            .expect(1)
            .mount(&server)
            .await;

        let _ = reranker(&server).score("q", &["doc"]).await.unwrap();
        server.verify().await;
    }

    // ── score() — error paths ──────────────────────────────────────────────

    #[tokio::test]
    async fn score_http_500_is_reranker_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/rerank"))
            .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
            .mount(&server)
            .await;

        let result = reranker(&server).score("q", &["doc"]).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GraphitiError::Reranker(_)));
    }

    #[tokio::test]
    async fn score_http_404_is_reranker_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/rerank"))
            .respond_with(ResponseTemplate::new(404).set_body_string("Not Found"))
            .mount(&server)
            .await;

        let result = reranker(&server).score("q", &["doc"]).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GraphitiError::Reranker(_)));
    }

    #[tokio::test]
    async fn score_connection_refused_is_reranker_error() {
        let r =
            HttpCrossEncoder::new("http://127.0.0.1:19998", "model").unwrap();
        let result = r.score("q", &["doc"]).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GraphitiError::Reranker(_)));
    }

    #[tokio::test]
    async fn score_timeout_is_reranker_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/rerank"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(make_response(&[(0, 0.5)]))
                    .set_delay(std::time::Duration::from_secs(120)),
            )
            .mount(&server)
            .await;

        let r = HttpCrossEncoder::with_timeout(
            server.uri(),
            "model",
            std::time::Duration::from_millis(100),
        )
        .unwrap();
        let result = r.score("q", &["doc"]).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GraphitiError::Reranker(_)));
    }
}
