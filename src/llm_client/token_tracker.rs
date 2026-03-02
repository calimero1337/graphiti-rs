//! Token usage tracking for LLM calls.
//!
//! [`TokenTracker`] accumulates prompt and completion token counts across all
//! LLM calls in a thread-safe manner using atomic operations.

use std::sync::atomic::{AtomicU64, Ordering};

use serde::Serialize;

/// A snapshot of token usage at a point in time.
#[derive(Debug, Clone, Serialize, Default)]
pub struct TokenUsage {
    /// Tokens consumed in the prompt (input).
    pub prompt_tokens: u64,
    /// Tokens generated in the completion (output).
    pub completion_tokens: u64,
    /// Total tokens (prompt + completion).
    pub total_tokens: u64,
}

/// Thread-safe accumulator for LLM token usage.
///
/// Uses [`AtomicU64`] counters so it can be shared across tasks without a
/// mutex.  All operations use [`Ordering::Relaxed`] — the counters are
/// independent scalars and do not need to synchronise with other memory
/// operations.
#[derive(Debug, Default)]
pub struct TokenTracker {
    prompt_tokens: AtomicU64,
    completion_tokens: AtomicU64,
}

impl TokenTracker {
    /// Create a new tracker with all counters at zero.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add `prompt` and `completion` tokens to the running totals.
    pub fn record(&self, prompt: u64, completion: u64) {
        self.prompt_tokens.fetch_add(prompt, Ordering::Relaxed);
        self.completion_tokens.fetch_add(completion, Ordering::Relaxed);
    }

    /// Return the current cumulative prompt token count.
    pub fn prompt_tokens(&self) -> u64 {
        self.prompt_tokens.load(Ordering::Relaxed)
    }

    /// Return the current cumulative completion token count.
    pub fn completion_tokens(&self) -> u64 {
        self.completion_tokens.load(Ordering::Relaxed)
    }

    /// Return the current cumulative total token count (prompt + completion).
    pub fn total_tokens(&self) -> u64 {
        self.prompt_tokens() + self.completion_tokens()
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        self.prompt_tokens.store(0, Ordering::Relaxed);
        self.completion_tokens.store(0, Ordering::Relaxed);
    }

    /// Return a [`TokenUsage`] snapshot of the current counts.
    ///
    /// # Approximation note
    ///
    /// The two atomic loads (prompt and completion) are performed separately, not
    /// as a single atomic operation. Under concurrent writes a snapshot may capture
    /// prompt and completion counts from different logical points in time — for
    /// example, the prompt count *before* a `record()` call and the completion count
    /// *after* it. The `total_tokens` field is always the sum of the two loaded
    /// values and is therefore internally consistent with the snapshot, but the
    /// snapshot as a whole may not correspond to any single instant. For monitoring
    /// and cost-tracking purposes this approximation is acceptable.
    pub fn snapshot(&self) -> TokenUsage {
        let prompt = self.prompt_tokens();
        let completion = self.completion_tokens();
        TokenUsage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_tracker_starts_at_zero() {
        let tracker = TokenTracker::new();
        assert_eq!(tracker.prompt_tokens(), 0);
        assert_eq!(tracker.completion_tokens(), 0);
        assert_eq!(tracker.total_tokens(), 0);
    }

    #[test]
    fn token_tracker_record_accumulates() {
        let tracker = TokenTracker::new();
        tracker.record(10, 20);
        assert_eq!(tracker.prompt_tokens(), 10);
        assert_eq!(tracker.completion_tokens(), 20);
        assert_eq!(tracker.total_tokens(), 30);

        tracker.record(5, 15);
        assert_eq!(tracker.prompt_tokens(), 15);
        assert_eq!(tracker.completion_tokens(), 35);
        assert_eq!(tracker.total_tokens(), 50);
    }

    #[test]
    fn token_tracker_reset_clears_counters() {
        let tracker = TokenTracker::new();
        tracker.record(100, 200);
        tracker.reset();
        assert_eq!(tracker.prompt_tokens(), 0);
        assert_eq!(tracker.completion_tokens(), 0);
        assert_eq!(tracker.total_tokens(), 0);
    }

    #[test]
    fn token_tracker_snapshot_reflects_current_state() {
        let tracker = TokenTracker::new();
        tracker.record(7, 13);
        let snap = tracker.snapshot();
        assert_eq!(snap.prompt_tokens, 7);
        assert_eq!(snap.completion_tokens, 13);
        assert_eq!(snap.total_tokens, 20);
    }

    #[test]
    fn token_usage_default_is_zero() {
        let usage = TokenUsage::default();
        assert_eq!(usage.prompt_tokens, 0);
        assert_eq!(usage.completion_tokens, 0);
        assert_eq!(usage.total_tokens, 0);
    }

    #[test]
    fn token_tracker_record_multiple_calls() {
        let tracker = TokenTracker::new();
        for _ in 0..5 {
            tracker.record(10, 20);
        }
        assert_eq!(tracker.prompt_tokens(), 50);
        assert_eq!(tracker.completion_tokens(), 100);
        assert_eq!(tracker.total_tokens(), 150);
    }
}
