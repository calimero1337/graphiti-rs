//! Delegated LLM client that submits tasks to [`LlmTaskQueue`] instead of
//! calling an LLM directly.
//!
//! The external orchestrator (e.g. a Claude agent) polls the queue, performs
//! the actual LLM work, and posts results back.  [`DelegatedLlmClient`]
//! bridges the [`LlmClient`] trait to this async handoff.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use super::task_queue::{LlmTaskQueue, LlmTaskType};
use super::{LlmClient, Message, Role, TokenUsage};
use crate::errors::{GraphitiError, LlmError};

/// An [`LlmClient`] implementation that delegates work to an [`LlmTaskQueue`].
///
/// Each call to [`generate`](LlmClient::generate) or
/// [`generate_structured_json`](LlmClient::generate_structured_json) submits a
/// task to the queue and blocks (async) until the task is completed by an
/// external worker or the configured timeout elapses.
pub struct DelegatedLlmClient {
    queue: Arc<LlmTaskQueue>,
    group_id: String,
    timeout: Duration,
    prompt_tokens: AtomicU64,
    completion_tokens: AtomicU64,
}

impl DelegatedLlmClient {
    /// Create a new delegated client.
    ///
    /// * `queue`    — shared task queue
    /// * `group_id` — identifier used to tag all tasks submitted by this client
    /// * `timeout`  — maximum time to wait for a result before returning an error
    pub fn new(queue: Arc<LlmTaskQueue>, group_id: String, timeout: Duration) -> Self {
        Self {
            queue,
            group_id,
            timeout,
            prompt_tokens: AtomicU64::new(0),
            completion_tokens: AtomicU64::new(0),
        }
    }
}

/// Extract the system and user portions from a slice of [`Message`]s.
///
/// Assistant messages are ignored — they are not meaningful when submitting
/// a fresh task to the queue.
fn messages_to_parts(messages: &[Message]) -> (String, String) {
    let mut system = String::new();
    let mut user = String::new();
    for msg in messages {
        match msg.role {
            Role::System => system.push_str(&msg.content),
            Role::User => user.push_str(&msg.content),
            Role::Assistant => {} // ignore assistant messages
        }
    }
    (system, user)
}

/// Infer the [`LlmTaskType`] from the system prompt content.
///
/// Uses keyword matching against the lowercased system prompt to determine
/// which pipeline step this request corresponds to.
fn infer_task_type(system_prompt: &str) -> LlmTaskType {
    let lower = system_prompt.to_lowercase();

    if lower.contains("extract") && lower.contains("entit") {
        LlmTaskType::ExtractEntities
    } else if lower.contains("extract") && (lower.contains("relation") || lower.contains("edge"))
    {
        LlmTaskType::ExtractEdges
    } else if (lower.contains("duplicate") && lower.contains("node"))
        || (lower.contains("entity") && lower.contains("resolution"))
    {
        LlmTaskType::DedupeNodes
    } else if lower.contains("duplicate") && lower.contains("edge") {
        LlmTaskType::DedupeEdges
    } else if lower.contains("contradict") {
        LlmTaskType::ResolveContradictions
    } else if lower.contains("summar") {
        LlmTaskType::Summarize
    } else {
        LlmTaskType::ExtractEntities
    }
}

#[async_trait::async_trait]
impl LlmClient for DelegatedLlmClient {
    async fn generate(&self, messages: &[Message]) -> crate::errors::Result<String> {
        let (system_prompt, user_prompt) = messages_to_parts(messages);
        let task_type = infer_task_type(&system_prompt);

        let id = self.queue.submit(
            task_type,
            system_prompt,
            user_prompt,
            serde_json::Value::Null,
            self.group_id.clone(),
        );

        let result = self
            .queue
            .wait_for_result(&id, self.timeout)
            .await
            .map_err(|e| GraphitiError::Llm(LlmError::Api { status: 0, message: e }))?;

        self.completion_tokens.fetch_add(1, Ordering::Relaxed);
        self.prompt_tokens.fetch_add(1, Ordering::Relaxed);

        Ok(result)
    }

    async fn generate_structured_json(
        &self,
        messages: &[Message],
        schema: serde_json::Value,
    ) -> crate::errors::Result<String> {
        let (system_prompt, user_prompt) = messages_to_parts(messages);
        let task_type = infer_task_type(&system_prompt);

        let id = self.queue.submit(
            task_type,
            system_prompt,
            user_prompt,
            schema,
            self.group_id.clone(),
        );

        let result = self
            .queue
            .wait_for_result(&id, self.timeout)
            .await
            .map_err(|e| GraphitiError::Llm(LlmError::Api { status: 0, message: e }))?;

        self.completion_tokens.fetch_add(1, Ordering::Relaxed);
        self.prompt_tokens.fetch_add(1, Ordering::Relaxed);

        Ok(result)
    }

    fn token_usage(&self) -> TokenUsage {
        let prompt = self.prompt_tokens.load(Ordering::Relaxed);
        let completion = self.completion_tokens.load(Ordering::Relaxed);
        TokenUsage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
        }
    }

    fn reset_token_usage(&self) {
        self.prompt_tokens.store(0, Ordering::Relaxed);
        self.completion_tokens.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_generate_submits_task_and_returns_result() {
        let queue = Arc::new(LlmTaskQueue::new());
        let client = DelegatedLlmClient::new(
            Arc::clone(&queue),
            "test-group".into(),
            Duration::from_secs(5),
        );

        let messages = vec![
            Message {
                role: Role::System,
                content: "extract entities from text".into(),
            },
            Message {
                role: Role::User,
                content: "Alice met Bob in Paris".into(),
            },
        ];

        // Spawn a task that will complete the queue entry
        let queue_clone = Arc::clone(&queue);
        let completer = tokio::spawn(async move {
            // Poll until the task appears
            loop {
                let pending = queue_clone.poll_pending(Some("test-group"), 10);
                if let Some(task) = pending.first() {
                    assert_eq!(task.task_type, LlmTaskType::ExtractEntities);
                    assert_eq!(task.system_prompt, "extract entities from text");
                    assert_eq!(task.user_prompt, "Alice met Bob in Paris");
                    queue_clone.claim(&task.id);
                    queue_clone.complete(&task.id, r#"{"entities":["Alice","Bob"]}"#.into());
                    break;
                }
                tokio::task::yield_now().await;
            }
        });

        let result = client.generate(&messages).await.unwrap();
        completer.await.unwrap();

        assert_eq!(result, r#"{"entities":["Alice","Bob"]}"#);

        // Verify token counters incremented
        let usage = client.token_usage();
        assert_eq!(usage.prompt_tokens, 1);
        assert_eq!(usage.completion_tokens, 1);
    }

    #[tokio::test]
    async fn test_generate_structured_json_submits_with_schema() {
        let queue = Arc::new(LlmTaskQueue::new());
        let client = DelegatedLlmClient::new(
            Arc::clone(&queue),
            "structured-group".into(),
            Duration::from_secs(5),
        );

        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            }
        });

        let messages = vec![Message {
            role: Role::User,
            content: "hello".into(),
        }];

        let expected_schema = schema.clone();
        let queue_clone = Arc::clone(&queue);
        let completer = tokio::spawn(async move {
            loop {
                let pending = queue_clone.poll_pending(Some("structured-group"), 10);
                if let Some(task) = pending.first() {
                    // Verify the schema was passed through
                    assert_eq!(task.response_schema, expected_schema);
                    queue_clone.claim(&task.id);
                    queue_clone.complete(&task.id, r#"{"name":"test"}"#.into());
                    break;
                }
                tokio::task::yield_now().await;
            }
        });

        let result = client
            .generate_structured_json(&messages, schema)
            .await
            .unwrap();
        completer.await.unwrap();

        assert_eq!(result, r#"{"name":"test"}"#);
    }

    #[tokio::test]
    async fn test_generate_timeout_returns_error() {
        let queue = Arc::new(LlmTaskQueue::new());
        let client = DelegatedLlmClient::new(
            Arc::clone(&queue),
            "timeout-group".into(),
            Duration::from_millis(50),
        );

        let messages = vec![Message {
            role: Role::User,
            content: "this will time out".into(),
        }];

        // Nobody completes the task, so it should time out
        let result = client.generate(&messages).await;
        assert!(result.is_err());

        let err = result.unwrap_err();
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("timed out"),
            "expected timeout error, got: {err_msg}"
        );
    }
}
