//! LLM-based community summarization with hierarchical reduction.
//!
//! When a community contains more than [`CHUNK_SIZE`] entities, summaries are
//! split into chunks, each chunk is summarized independently, and the resulting
//! summaries are fed back into the same process until a single summary remains.
//! This mirrors the tree-based reduction used in the Python Graphiti reference.

use crate::errors::Result;
use crate::llm_client::{generate_structured_via_dyn, LlmClient};
use crate::prompts::summarize::{build_community_messages, CommunitySummaryContext, Summary};

/// Maximum number of entity summaries processed in a single LLM call.
const CHUNK_SIZE: usize = 10;

/// Generate a community summary from the entity summaries of its members.
///
/// * If `entity_summaries` is empty, returns an empty string immediately.
/// * If `entity_summaries.len() <= CHUNK_SIZE`, a single LLM call is made.
/// * Otherwise, summaries are chunked, each chunk summarized, and the process
///   repeats on the resulting (shorter) list until convergence to a single call.
pub(super) async fn summarize_community(
    llm: &dyn LlmClient,
    entity_summaries: &[String],
) -> Result<String> {
    if entity_summaries.is_empty() {
        return Ok(String::new());
    }

    // Start with a copy we can replace each reduction round.
    let mut current: Vec<String> = entity_summaries.to_vec();

    // Hierarchically reduce until we're within one chunk.
    while current.len() > CHUNK_SIZE {
        let mut reduced: Vec<String> = Vec::new();
        for chunk in current.chunks(CHUNK_SIZE) {
            let ctx = CommunitySummaryContext { entity_summaries: chunk };
            let msgs = build_community_messages(&ctx);
            let summary: Summary = generate_structured_via_dyn(llm, &msgs).await?;
            reduced.push(summary.summary);
        }
        current = reduced;
    }

    // Final (or only) summarization call.
    let ctx = CommunitySummaryContext { entity_summaries: &current };
    let msgs = build_community_messages(&ctx);
    let summary: Summary = generate_structured_via_dyn(llm, &msgs).await?;
    Ok(summary.summary)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::GraphitiError;
    use crate::llm_client::{LlmClient, Message};


    /// Mock LLM that records how many times it was called and returns a fixed summary.
    struct CountingMockLlm {
        response: String,
        call_count: std::sync::atomic::AtomicUsize,
    }

    impl CountingMockLlm {
        fn new(response: &str) -> Self {
            Self {
                response: response.to_string(),
                call_count: std::sync::atomic::AtomicUsize::new(0),
            }
        }

        fn calls(&self) -> usize {
            self.call_count.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    #[async_trait::async_trait]
    impl LlmClient for CountingMockLlm {
        async fn generate(&self, _: &[Message]) -> Result<String> {
            Ok(self.response.clone())
        }

        async fn generate_structured_json(
            &self,
            _: &[Message],
            _schema: serde_json::Value,
        ) -> Result<String> {
            self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(format!(r#"{{"summary": "{}"}}"#, self.response))
        }
    }

    #[tokio::test]
    async fn empty_summaries_returns_empty_string() {
        let llm = CountingMockLlm::new("irrelevant");
        let result = summarize_community(&llm, &[]).await.unwrap();
        assert!(result.is_empty());
        assert_eq!(llm.calls(), 0, "no LLM call expected for empty input");
    }

    #[tokio::test]
    async fn single_entity_single_llm_call() {
        let llm = CountingMockLlm::new("A summary.");
        let summaries = vec!["Alice is a researcher.".to_string()];
        let result = summarize_community(&llm, &summaries).await.unwrap();
        assert_eq!(result, "A summary.");
        assert_eq!(llm.calls(), 1);
    }

    #[tokio::test]
    async fn within_chunk_size_single_llm_call() {
        let llm = CountingMockLlm::new("Combined summary.");
        let summaries: Vec<String> = (0..CHUNK_SIZE).map(|i| format!("Entity {i}.")).collect();
        let result = summarize_community(&llm, &summaries).await.unwrap();
        assert_eq!(result, "Combined summary.");
        assert_eq!(llm.calls(), 1, "exactly one LLM call for ≤ CHUNK_SIZE summaries");
    }

    #[tokio::test]
    async fn oversized_community_triggers_hierarchical_reduction() {
        // CHUNK_SIZE+1 summaries → 2 chunks → 2 intermediate calls → 1 final call = 3 calls.
        let llm = CountingMockLlm::new("chunk summary");
        let summaries: Vec<String> =
            (0..=CHUNK_SIZE).map(|i| format!("Entity {i}.")).collect();
        assert_eq!(summaries.len(), CHUNK_SIZE + 1);

        let _ = summarize_community(&llm, &summaries).await.unwrap();
        // Round 1: ceil(11/10) = 2 chunks → 2 LLM calls.
        // Round 2: 2 summaries ≤ 10 → 1 LLM call.
        // Total = 3.
        assert_eq!(llm.calls(), 3);
    }

    #[tokio::test]
    async fn large_community_reduces_correctly() {
        // 100 summaries → round 1: 10 chunks × 10 each = 10 calls → 10 summaries
        // round 2: 1 chunk of 10 = 1 call → total 11 calls.
        let llm = CountingMockLlm::new("reduced");
        let summaries: Vec<String> = (0..100).map(|i| format!("Entity {i}.")).collect();
        let _ = summarize_community(&llm, &summaries).await.unwrap();
        assert_eq!(llm.calls(), 11);
    }

    #[tokio::test]
    async fn summary_text_is_returned() {
        let llm = CountingMockLlm::new("AI researchers community.");
        let summaries = vec!["Alice.".to_string(), "Bob.".to_string()];
        let result = summarize_community(&llm, &summaries).await.unwrap();
        assert_eq!(result, "AI researchers community.");
    }
}
