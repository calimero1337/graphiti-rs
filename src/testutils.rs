//! Shared test mocks for the graphiti-rs crate.
//!
//! This module is compiled only when running tests (`#[cfg(test)]`).
//! Import specific types via `use crate::testutils::{MockDriver, MockEmbedder, MockLlmClient};`.
//!
//! # Available mocks
//!
//! | Type | Behaviour |
//! |------|-----------|
//! | [`MockDriver`] | No-op [`GraphDriver`] — all writes succeed; all reads return empty/`None` |
//! | [`MockEmbedder`] | Returns a zero vector; dimension is configurable (default 1536) |
//! | [`MockLlmClient`] | `generate` returns `"mock"`; `generate_structured` panics |

use crate::driver::GraphDriver;
use crate::edges::{CommunityEdge, EntityEdge, EpisodicEdge};
use crate::embedder::{Embedding, EmbedderClient};
use crate::errors::Result;
use crate::llm_client::token_tracker::TokenTracker;
use crate::llm_client::{LlmClient, Message, TokenUsage};
use crate::nodes::{CommunityNode, EntityNode, EpisodicNode};
use chrono::{DateTime, Utc};
use uuid::Uuid;

// ── MockDriver ─────────────────────────────────────────────────────────────

/// A no-op [`GraphDriver`] that returns trivial successes for every operation.
///
/// Use this in tests that exercise code above the driver layer and don't need
/// real database interactions.
pub struct MockDriver;

#[async_trait::async_trait]
impl GraphDriver for MockDriver {
    async fn ping(&self) -> Result<()> {
        Ok(())
    }
    async fn close(&self) -> Result<()> {
        Ok(())
    }
    async fn save_entity_node(&self, _: &EntityNode) -> Result<()> {
        Ok(())
    }
    async fn get_entity_node(&self, _: &Uuid) -> Result<Option<EntityNode>> {
        Ok(None)
    }
    async fn delete_entity_node(&self, _: &Uuid) -> Result<()> {
        Ok(())
    }
    async fn save_episodic_node(&self, _: &EpisodicNode) -> Result<()> {
        Ok(())
    }
    async fn get_episodic_node(&self, _: &Uuid) -> Result<Option<EpisodicNode>> {
        Ok(None)
    }
    async fn delete_episodic_node(&self, _: &Uuid) -> Result<()> {
        Ok(())
    }
    async fn list_episodic_nodes(&self, _: &str) -> Result<Vec<EpisodicNode>> {
        Ok(vec![])
    }
    async fn list_entity_nodes(&self, _: &str) -> Result<Vec<EntityNode>> {
        Ok(vec![])
    }
    async fn list_entity_edges(&self, _: &str) -> Result<Vec<EntityEdge>> {
        Ok(vec![])
    }
    async fn save_community_node(&self, _: &CommunityNode) -> Result<()> {
        Ok(())
    }
    async fn save_entity_edge(&self, _: &EntityEdge) -> Result<()> {
        Ok(())
    }
    async fn get_entity_edge(&self, _: &Uuid) -> Result<Option<EntityEdge>> {
        Ok(None)
    }
    async fn save_episodic_edge(&self, _: &EpisodicEdge) -> Result<()> {
        Ok(())
    }
    async fn save_community_edge(&self, _: &CommunityEdge) -> Result<()> {
        Ok(())
    }
    async fn search_entity_nodes_by_name(
        &self,
        _: &str,
        _: &str,
        _: usize,
    ) -> Result<Vec<EntityNode>> {
        Ok(vec![])
    }
    async fn search_entity_nodes_by_embedding(
        &self,
        _: &[f32],
        _: &str,
        _: usize,
    ) -> Result<Vec<EntityNode>> {
        Ok(vec![])
    }
    async fn search_entity_edges_by_fact(
        &self,
        _: &[f32],
        _: &str,
        _: usize,
    ) -> Result<Vec<EntityEdge>> {
        Ok(vec![])
    }
    async fn bm25_search_edges(&self, _: &str, _: &str, _: usize) -> Result<Vec<EntityEdge>> {
        Ok(vec![])
    }
    async fn build_indices(&self) -> Result<()> {
        Ok(())
    }
    async fn get_entity_edges_between(&self, _: &Uuid, _: &Uuid) -> Result<Vec<EntityEdge>> {
        Ok(vec![])
    }
    async fn invalidate_edge(&self, _: &Uuid, _: DateTime<Utc>) -> Result<()> {
        Ok(())
    }
}

// ── MockEmbedder ───────────────────────────────────────────────────────────

/// An [`EmbedderClient`] that always returns a zero vector.
///
/// The embedding dimension is configurable; the default (1536) matches
/// `text-embedding-3-small`.
pub struct MockEmbedder {
    dim: usize,
}

impl MockEmbedder {
    /// Create a `MockEmbedder` with the default dimension (1536).
    pub fn new() -> Self {
        Self { dim: 1536 }
    }

    /// Create a `MockEmbedder` with a custom dimension.
    pub fn with_dim(dim: usize) -> Self {
        Self { dim }
    }
}

impl Default for MockEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl EmbedderClient for MockEmbedder {
    async fn embed(&self, _: &str) -> Result<Embedding> {
        Ok(vec![0.0_f32; self.dim])
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>> {
        Ok(texts.iter().map(|_| vec![0.0_f32; self.dim]).collect())
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

// ── MockLlmClient ──────────────────────────────────────────────────────────

/// A trivial [`LlmClient`] for tests that don't exercise LLM calls.
///
/// - `generate` returns the literal string `"mock"`.
/// - `generate_structured` panics — use [`crate::pipeline`]'s own
///   queue-backed mock when structured output is required.
pub struct MockLlmClient;

#[async_trait::async_trait]
impl LlmClient for MockLlmClient {
    async fn generate(&self, _: &[Message]) -> Result<String> {
        Ok("mock".to_string())
    }

    async fn generate_structured_json(
        &self,
        _: &[Message],
        _schema: serde_json::Value,
    ) -> Result<String> {
        // Return empty entities so the pipeline early-returns successfully.
        Ok(r#"{"entities":[]}"#.to_string())
    }
}

// ── TokenTrackingMockLlmClient ──────────────────────────────────────────────

/// An [`LlmClient`] mock that exposes a pre-seeded [`TokenTracker`].
///
/// Use this in tests that need to verify token-usage delegation without making
/// real LLM calls.  Pre-seed counts with [`Self::with_usage`]:
///
/// ```ignore
/// let llm = Arc::new(TokenTrackingMockLlmClient::with_usage(100, 50));
/// ```
pub struct TokenTrackingMockLlmClient {
    tracker: TokenTracker,
}

impl TokenTrackingMockLlmClient {
    /// Create a mock whose tracker already holds `prompt` + `completion` tokens.
    pub fn with_usage(prompt: u64, completion: u64) -> Self {
        let client = Self {
            tracker: TokenTracker::new(),
        };
        client.tracker.record(prompt, completion);
        client
    }
}

#[async_trait::async_trait]
impl LlmClient for TokenTrackingMockLlmClient {
    async fn generate(&self, _: &[Message]) -> Result<String> {
        Ok("mock".to_string())
    }

    async fn generate_structured_json(
        &self,
        _: &[Message],
        _schema: serde_json::Value,
    ) -> Result<String> {
        unimplemented!("TokenTrackingMockLlmClient::generate_structured_json not implemented")
    }

    fn token_usage(&self) -> TokenUsage {
        self.tracker.snapshot()
    }

    fn reset_token_usage(&self) {
        self.tracker.reset();
    }
}
