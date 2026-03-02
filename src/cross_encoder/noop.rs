//! No-op cross-encoder — returns uniform scores for all documents.
//!
//! Useful as a default when no reranking service is configured, preserving
//! the upstream RRF ranking order without any I/O overhead.

use crate::cross_encoder::CrossEncoderClient;
use crate::errors::Result;

/// A [`CrossEncoderClient`] that returns a constant score of `1.0` for every
/// document without performing any I/O.
///
/// Use this when reranking is not required but the interface must be satisfied,
/// or in tests where a real reranker is unavailable.
pub struct NoopCrossEncoder;

#[async_trait::async_trait]
impl CrossEncoderClient for NoopCrossEncoder {
    async fn score(&self, _query: &str, documents: &[&str]) -> Result<Vec<f32>> {
        Ok(vec![1.0_f32; documents.len()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn noop_returns_one_score_per_document() {
        let reranker = NoopCrossEncoder;
        let docs = ["alpha", "beta", "gamma"];
        let scores = reranker.score("query", &docs).await.unwrap();
        assert_eq!(scores.len(), 3);
    }

    #[tokio::test]
    async fn noop_all_scores_are_1_0() {
        let reranker = NoopCrossEncoder;
        let docs = ["a", "b"];
        let scores = reranker.score("q", &docs).await.unwrap();
        for &s in &scores {
            assert!((s - 1.0_f32).abs() < f32::EPSILON, "expected 1.0, got {s}");
        }
    }

    #[tokio::test]
    async fn noop_empty_documents_returns_empty_vec() {
        let reranker = NoopCrossEncoder;
        let scores = reranker.score("query", &[]).await.unwrap();
        assert!(scores.is_empty());
    }
}
