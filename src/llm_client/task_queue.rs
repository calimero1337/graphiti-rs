//! In-memory task queue for async LLM task delegation.
//!
//! Provides [`LlmTaskQueue`] backed by `DashMap<Uuid, TaskEntry>` where each
//! entry holds an [`LlmTask`] plus an `Arc<Notify>` for waking waiters.

use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::sync::Notify;
use uuid::Uuid;

/// The type of LLM task to be performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlmTaskType {
    ExtractEntities,
    ExtractEdges,
    DedupeNodes,
    DedupeEdges,
    ResolveContradictions,
    Summarize,
}

/// Current status of an LLM task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
}

/// An LLM task submitted to the queue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmTask {
    pub id: Uuid,
    pub task_type: LlmTaskType,
    pub status: TaskStatus,
    pub system_prompt: String,
    pub user_prompt: String,
    pub response_schema: serde_json::Value,
    pub result: Option<String>,
    pub group_id: String,
    pub created_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

/// Internal entry wrapping an [`LlmTask`] with a notification handle.
struct TaskEntry {
    task: LlmTask,
    notify: Arc<Notify>,
}

/// In-memory task queue for async LLM task delegation.
pub struct LlmTaskQueue {
    tasks: DashMap<Uuid, TaskEntry>,
}

impl LlmTaskQueue {
    /// Create a new empty task queue.
    pub fn new() -> Self {
        Self {
            tasks: DashMap::new(),
        }
    }

    /// Submit a new task to the queue. Returns the task's UUID.
    pub fn submit(
        &self,
        task_type: LlmTaskType,
        system_prompt: String,
        user_prompt: String,
        response_schema: serde_json::Value,
        group_id: String,
    ) -> Uuid {
        let id = Uuid::now_v7();
        let task = LlmTask {
            id,
            task_type,
            status: TaskStatus::Pending,
            system_prompt,
            user_prompt,
            response_schema,
            result: None,
            group_id,
            created_at: Utc::now(),
            completed_at: None,
        };
        let entry = TaskEntry {
            task,
            notify: Arc::new(Notify::new()),
        };
        self.tasks.insert(id, entry);
        id
    }

    /// Retrieve a clone of the task by ID, or `None` if not found.
    pub fn get(&self, id: &Uuid) -> Option<LlmTask> {
        self.tasks.get(id).map(|entry| entry.task.clone())
    }

    /// Return up to `limit` pending tasks, optionally filtered by `group_id`.
    pub fn poll_pending(&self, group_id: Option<&str>, limit: usize) -> Vec<LlmTask> {
        self.tasks
            .iter()
            .filter(|entry| entry.value().task.status == TaskStatus::Pending)
            .filter(|entry| match group_id {
                Some(gid) => entry.value().task.group_id == gid,
                None => true,
            })
            .take(limit)
            .map(|entry| entry.value().task.clone())
            .collect()
    }

    /// Atomically claim a task: transitions from `Pending` to `InProgress`.
    ///
    /// Returns `true` if the claim succeeded, `false` if the task was not found
    /// or was not in `Pending` status.
    pub fn claim(&self, id: &Uuid) -> bool {
        match self.tasks.get_mut(id) {
            Some(mut entry) if entry.task.status == TaskStatus::Pending => {
                entry.task.status = TaskStatus::InProgress;
                true
            }
            _ => false,
        }
    }

    /// Mark a task as completed with the given result and notify any waiters.
    pub fn complete(&self, id: &Uuid, result: String) {
        if let Some(mut entry) = self.tasks.get_mut(id) {
            entry.task.status = TaskStatus::Completed;
            entry.task.result = Some(result);
            entry.task.completed_at = Some(Utc::now());
            entry.notify.notify_waiters();
        }
    }

    /// Mark a task as failed with the given error message and notify any waiters.
    pub fn fail(&self, id: &Uuid, error: String) {
        if let Some(mut entry) = self.tasks.get_mut(id) {
            entry.task.status = TaskStatus::Failed;
            entry.task.result = Some(error);
            entry.task.completed_at = Some(Utc::now());
            entry.notify.notify_waiters();
        }
    }

    /// Wait for a task to produce a result, with a timeout.
    ///
    /// Returns `Ok(result)` if the task completed, or `Err(message)` if the
    /// task failed, timed out, or was not found.
    pub async fn wait_for_result(&self, id: &Uuid, timeout: Duration) -> Result<String, String> {
        // Grab the notify handle and check current status
        let notify = {
            let entry = self
                .tasks
                .get(id)
                .ok_or_else(|| format!("task {id} not found"))?;
            // If already completed or failed, return immediately
            match entry.task.status {
                TaskStatus::Completed => {
                    return entry
                        .task
                        .result
                        .clone()
                        .ok_or_else(|| "task completed but result is empty".to_string());
                }
                TaskStatus::Failed => {
                    return Err(entry
                        .task
                        .result
                        .clone()
                        .unwrap_or_else(|| "task failed with no error message".to_string()));
                }
                _ => {}
            }
            Arc::clone(&entry.notify)
        };

        // Wait for notification or timeout
        match tokio::time::timeout(timeout, notify.notified()).await {
            Ok(()) => {
                // Re-read the task after notification
                let entry = self
                    .tasks
                    .get(id)
                    .ok_or_else(|| format!("task {id} not found after notification"))?;
                match entry.task.status {
                    TaskStatus::Completed => entry
                        .task
                        .result
                        .clone()
                        .ok_or_else(|| "task completed but result is empty".to_string()),
                    TaskStatus::Failed => Err(entry
                        .task
                        .result
                        .clone()
                        .unwrap_or_else(|| "task failed with no error message".to_string())),
                    other => Err(format!("task in unexpected status after notification: {other:?}")),
                }
            }
            Err(_) => Err(format!("task {id} timed out after {timeout:?}")),
        }
    }
}

impl Default for LlmTaskQueue {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_submit_creates_pending_task() {
        let queue = LlmTaskQueue::new();
        let id = queue.submit(
            LlmTaskType::ExtractEntities,
            "system".into(),
            "user".into(),
            serde_json::json!({}),
            "group-1".into(),
        );
        let task = queue.get(&id).unwrap();
        assert_eq!(task.status, TaskStatus::Pending);
        assert_eq!(task.task_type, LlmTaskType::ExtractEntities);
        assert_eq!(task.group_id, "group-1");
        assert_eq!(task.system_prompt, "system");
        assert_eq!(task.user_prompt, "user");
        assert!(task.result.is_none());
        assert!(task.completed_at.is_none());
    }

    #[test]
    fn test_poll_returns_pending_tasks() {
        let queue = LlmTaskQueue::new();
        let id1 = queue.submit(
            LlmTaskType::ExtractEntities,
            "s".into(),
            "u".into(),
            serde_json::json!({}),
            "g".into(),
        );
        let id2 = queue.submit(
            LlmTaskType::ExtractEdges,
            "s".into(),
            "u".into(),
            serde_json::json!({}),
            "g".into(),
        );

        let pending = queue.poll_pending(None, 10);
        assert_eq!(pending.len(), 2);

        let ids: Vec<Uuid> = pending.iter().map(|t| t.id).collect();
        assert!(ids.contains(&id1));
        assert!(ids.contains(&id2));

        // All should be Pending
        for task in &pending {
            assert_eq!(task.status, TaskStatus::Pending);
        }
    }

    #[test]
    fn test_claim_transitions_to_in_progress() {
        let queue = LlmTaskQueue::new();
        let id = queue.submit(
            LlmTaskType::DedupeNodes,
            "s".into(),
            "u".into(),
            serde_json::json!({}),
            "g".into(),
        );

        assert!(queue.claim(&id));
        let task = queue.get(&id).unwrap();
        assert_eq!(task.status, TaskStatus::InProgress);
    }

    #[test]
    fn test_claim_fails_if_not_pending() {
        let queue = LlmTaskQueue::new();
        let id = queue.submit(
            LlmTaskType::Summarize,
            "s".into(),
            "u".into(),
            serde_json::json!({}),
            "g".into(),
        );

        // First claim succeeds
        assert!(queue.claim(&id));
        // Second claim fails (already InProgress)
        assert!(!queue.claim(&id));

        // Complete it, then try to claim again
        queue.complete(&id, "done".into());
        assert!(!queue.claim(&id));
    }

    #[tokio::test]
    async fn test_submit_and_complete_notifies_waiter() {
        let queue = Arc::new(LlmTaskQueue::new());
        let id = queue.submit(
            LlmTaskType::ExtractEntities,
            "s".into(),
            "u".into(),
            serde_json::json!({}),
            "g".into(),
        );

        let queue_clone = Arc::clone(&queue);
        let handle = tokio::spawn(async move {
            queue_clone.wait_for_result(&id, Duration::from_secs(5)).await
        });

        // Claim and complete in the main task
        assert!(queue.claim(&id));
        queue.complete(&id, "the answer".into());

        let result = handle.await.unwrap();
        assert_eq!(result, Ok("the answer".to_string()));
    }

    #[tokio::test]
    async fn test_wait_times_out() {
        let queue = Arc::new(LlmTaskQueue::new());
        let id = queue.submit(
            LlmTaskType::ExtractEdges,
            "s".into(),
            "u".into(),
            serde_json::json!({}),
            "g".into(),
        );

        let result = queue
            .wait_for_result(&id, Duration::from_millis(50))
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("timed out"));
    }

    #[test]
    fn test_poll_filters_by_group_id() {
        let queue = LlmTaskQueue::new();
        queue.submit(
            LlmTaskType::ExtractEntities,
            "s".into(),
            "u".into(),
            serde_json::json!({}),
            "alpha".into(),
        );
        queue.submit(
            LlmTaskType::ExtractEdges,
            "s".into(),
            "u".into(),
            serde_json::json!({}),
            "beta".into(),
        );
        queue.submit(
            LlmTaskType::DedupeNodes,
            "s".into(),
            "u".into(),
            serde_json::json!({}),
            "alpha".into(),
        );

        let alpha_tasks = queue.poll_pending(Some("alpha"), 10);
        assert_eq!(alpha_tasks.len(), 2);
        for task in &alpha_tasks {
            assert_eq!(task.group_id, "alpha");
        }

        let beta_tasks = queue.poll_pending(Some("beta"), 10);
        assert_eq!(beta_tasks.len(), 1);
        assert_eq!(beta_tasks[0].group_id, "beta");

        // Limit parameter
        let limited = queue.poll_pending(Some("alpha"), 1);
        assert_eq!(limited.len(), 1);
    }
}
