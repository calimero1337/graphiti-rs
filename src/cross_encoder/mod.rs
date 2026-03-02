//! Cross-encoder module for semantic reranking of search results.
//!
//! Cross-encoders score `(query, document)` pairs jointly, providing
//! better relevance ranking than bi-encoder cosine similarity alone.
//! They are slower than bi-encoders but more accurate, making them ideal
//! as an optional post-retrieval reranking step.
//!
//! # Architecture
//!
//! - [`CrossEncoderClient`] — async trait for scoring `(query, document)` pairs.
//! - [`HttpCrossEncoder`] — HTTP client for remote reranking services (Cohere/Jina/TEI-compatible).
//! - [`NoopCrossEncoder`] — passthrough that assigns uniform scores (no reranking).
//!
//! # Usage
//!
//! ```rust,no_run
//! use graphiti_rs::cross_encoder::{CrossEncoderClient, HttpCrossEncoder};
//!
//! # #[tokio::main] async fn main() -> graphiti_rs::Result<()> {
//! let reranker = HttpCrossEncoder::new("http://localhost:8080", "cross-encoder/ms-marco-MiniLM-L-6-v2")?;
//! let scores = reranker.score("What is Rust?", &["Rust is a systems language", "Python is dynamic"]).await?;
//! // scores[0] > scores[1] — first document is more relevant
//! # Ok(())
//! # }
//! ```

use crate::errors::Result;

pub mod http;
pub mod noop;

pub use http::HttpCrossEncoder;
pub use noop::NoopCrossEncoder;

/// Trait for cross-encoder reranking clients.
///
/// Implementations score `(query, document)` pairs jointly, returning one
/// relevance score per document in the same order as the input `documents` slice.
///
/// Higher scores indicate greater relevance to the query.
#[async_trait::async_trait]
pub trait CrossEncoderClient: Send + Sync {
    /// Score each `(query, document)` pair and return one relevance score per document.
    ///
    /// The returned `Vec<f32>` has the same length as `documents`, with
    /// `scores[i]` corresponding to `documents[i]`.  Higher scores indicate
    /// greater relevance.
    ///
    /// Returns an empty `Vec` when `documents` is empty.
    async fn score(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>>;
}
