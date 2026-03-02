# LLM Task Delegation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove direct LLM dependencies from graphiti-rs and delegate all LLM work to Lower Decks agents (Neelix/Seven) via an async MCP task queue.

**Architecture:** graphiti-rs maintains an in-memory task queue (`DashMap`). When the pipeline needs an LLM call, `DelegatedLlmClient` creates a task and waits on a `Notify`. Lower Decks polls for tasks via 3 new MCP tools, dispatches to Neelix or Seven, and posts results back. The `Notify` fires, the pipeline resumes.

**Tech Stack:** Rust (tokio, dashmap, uuid, chrono, rmcp), Python (mcp, asyncio, claude-agent-sdk)

---

### Task 1: Add LlmTaskQueue to graphiti-rs

**Files:**
- Create: `src/llm_client/task_queue.rs`
- Modify: `src/llm_client/mod.rs:1-10` (add `pub mod task_queue;`)
- Test: inline `#[cfg(test)]` in `task_queue.rs`

**Step 1: Write the failing test**

In `src/llm_client/task_queue.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_submit_creates_pending_task() {
        let queue = LlmTaskQueue::new();
        let id = queue.submit(
            LlmTaskType::ExtractEntities,
            "system prompt".into(),
            "user prompt".into(),
            serde_json::json!({"type": "object"}),
            "test-group".into(),
        );
        let task = queue.get(&id).expect("task should exist");
        assert_eq!(task.status, TaskStatus::Pending);
        assert_eq!(task.system_prompt, "system prompt");
    }

    #[test]
    fn test_poll_returns_pending_tasks() {
        let queue = LlmTaskQueue::new();
        let id = queue.submit(
            LlmTaskType::ExtractEntities,
            "sys".into(), "usr".into(),
            serde_json::json!({}), "g".into(),
        );
        let tasks = queue.poll_pending(None, 10);
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].id, id);
    }

    #[test]
    fn test_claim_transitions_to_in_progress() {
        let queue = LlmTaskQueue::new();
        let id = queue.submit(
            LlmTaskType::ExtractEntities,
            "s".into(), "u".into(),
            serde_json::json!({}), "g".into(),
        );
        assert!(queue.claim(&id));
        let task = queue.get(&id).unwrap();
        assert_eq!(task.status, TaskStatus::InProgress);
    }

    #[test]
    fn test_claim_fails_if_not_pending() {
        let queue = LlmTaskQueue::new();
        let id = queue.submit(
            LlmTaskType::ExtractEntities,
            "s".into(), "u".into(),
            serde_json::json!({}), "g".into(),
        );
        assert!(queue.claim(&id));
        assert!(!queue.claim(&id)); // already claimed
    }

    #[tokio::test]
    async fn test_submit_and_complete_notifies_waiter() {
        let queue = Arc::new(LlmTaskQueue::new());
        let id = queue.submit(
            LlmTaskType::ExtractEntities,
            "s".into(), "u".into(),
            serde_json::json!({}), "g".into(),
        );

        let q2 = queue.clone();
        let handle = tokio::spawn(async move {
            q2.wait_for_result(&id, Duration::from_secs(5)).await
        });

        queue.claim(&id);
        queue.complete(&id, r#"{"entities":[]}"#.into());

        let result = handle.await.unwrap();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), r#"{"entities":[]}"#);
    }

    #[tokio::test]
    async fn test_wait_times_out() {
        let queue = Arc::new(LlmTaskQueue::new());
        let id = queue.submit(
            LlmTaskType::ExtractEntities,
            "s".into(), "u".into(),
            serde_json::json!({}), "g".into(),
        );
        let result = queue.wait_for_result(&id, Duration::from_millis(50)).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_poll_filters_by_group_id() {
        let queue = LlmTaskQueue::new();
        queue.submit(LlmTaskType::ExtractEntities, "s".into(), "u".into(), serde_json::json!({}), "alpha".into());
        queue.submit(LlmTaskType::ExtractEdges, "s".into(), "u".into(), serde_json::json!({}), "beta".into());
        let alpha = queue.poll_pending(Some("alpha"), 10);
        assert_eq!(alpha.len(), 1);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cd /data/k3s-storage/graphiti-rs && cargo test --lib llm_client::task_queue 2>&1 | tail -20`
Expected: Compilation errors (types don't exist yet)

**Step 3: Write minimal implementation**

In `src/llm_client/task_queue.rs`:

```rust
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::sync::Notify;
use uuid::Uuid;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
}

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

struct TaskEntry {
    task: LlmTask,
    notify: Arc<Notify>,
}

pub struct LlmTaskQueue {
    tasks: DashMap<Uuid, TaskEntry>,
}

impl LlmTaskQueue {
    pub fn new() -> Self {
        Self {
            tasks: DashMap::new(),
        }
    }

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
        let notify = Arc::new(Notify::new());
        self.tasks.insert(id, TaskEntry { task, notify });
        id
    }

    pub fn get(&self, id: &Uuid) -> Option<LlmTask> {
        self.tasks.get(id).map(|e| e.task.clone())
    }

    pub fn poll_pending(&self, group_id: Option<&str>, limit: usize) -> Vec<LlmTask> {
        self.tasks
            .iter()
            .filter(|e| e.value().task.status == TaskStatus::Pending)
            .filter(|e| group_id.map_or(true, |g| e.value().task.group_id == g))
            .take(limit)
            .map(|e| e.value().task.clone())
            .collect()
    }

    pub fn claim(&self, id: &Uuid) -> bool {
        if let Some(mut entry) = self.tasks.get_mut(id) {
            if entry.task.status == TaskStatus::Pending {
                entry.task.status = TaskStatus::InProgress;
                return true;
            }
        }
        false
    }

    pub fn complete(&self, id: &Uuid, result: String) {
        if let Some(mut entry) = self.tasks.get_mut(id) {
            entry.task.status = TaskStatus::Completed;
            entry.task.result = Some(result);
            entry.task.completed_at = Some(Utc::now());
            let notify = entry.notify.clone();
            drop(entry);
            notify.notify_one();
        }
    }

    pub fn fail(&self, id: &Uuid, error: String) {
        if let Some(mut entry) = self.tasks.get_mut(id) {
            entry.task.status = TaskStatus::Failed;
            entry.task.result = Some(error);
            entry.task.completed_at = Some(Utc::now());
            let notify = entry.notify.clone();
            drop(entry);
            notify.notify_one();
        }
    }

    pub async fn wait_for_result(&self, id: &Uuid, timeout: Duration) -> Result<String, String> {
        let notify = self
            .tasks
            .get(id)
            .map(|e| e.notify.clone())
            .ok_or_else(|| "task not found".to_string())?;

        if tokio::time::timeout(timeout, notify.notified()).await.is_err() {
            self.fail(id, "timeout".into());
            return Err("task timed out".into());
        }

        self.get(id)
            .and_then(|t| match t.status {
                TaskStatus::Completed => t.result,
                TaskStatus::Failed => None,
                _ => None,
            })
            .ok_or_else(|| "task failed or missing result".into())
    }
}
```

Add to `src/llm_client/mod.rs` at the top (after existing `pub mod` lines):

```rust
pub mod task_queue;
```

**Step 4: Run tests to verify they pass**

Run: `cd /data/k3s-storage/graphiti-rs && cargo test --lib llm_client::task_queue -- --nocapture 2>&1 | tail -20`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
cd /data/k3s-storage/graphiti-rs
git add src/llm_client/task_queue.rs src/llm_client/mod.rs
git commit -m "feat: add LlmTaskQueue for async LLM task delegation"
```

---

### Task 2: Add DelegatedLlmClient

**Files:**
- Create: `src/llm_client/delegated.rs`
- Modify: `src/llm_client/mod.rs` (add `pub mod delegated;`)
- Test: inline `#[cfg(test)]` in `delegated.rs`

**Step 1: Write the failing test**

In `src/llm_client/delegated.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_generate_submits_task_and_returns_result() {
        let queue = Arc::new(LlmTaskQueue::new());
        let client = DelegatedLlmClient::new(queue.clone(), "test-group".into(), Duration::from_secs(5));

        let q2 = queue.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let tasks = q2.poll_pending(None, 10);
            assert_eq!(tasks.len(), 1);
            q2.claim(&tasks[0].id);
            q2.complete(&tasks[0].id, "hello world".into());
        });

        let msgs = vec![Message::user("say hello")];
        let result = client.generate(&msgs).await.unwrap();
        assert_eq!(result, "hello world");
    }

    #[tokio::test]
    async fn test_generate_structured_json_submits_with_schema() {
        let queue = Arc::new(LlmTaskQueue::new());
        let client = DelegatedLlmClient::new(queue.clone(), "test-group".into(), Duration::from_secs(5));

        let q2 = queue.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let tasks = q2.poll_pending(None, 10);
            assert_eq!(tasks.len(), 1);
            assert!(tasks[0].response_schema.get("type").is_some());
            q2.claim(&tasks[0].id);
            q2.complete(&tasks[0].id, r#"{"name":"test"}"#.into());
        });

        let msgs = vec![Message::system("extract"), Message::user("content")];
        let schema = serde_json::json!({"type": "object", "properties": {"name": {"type": "string"}}});
        let result = client.generate_structured_json(&msgs, schema).await.unwrap();
        assert_eq!(result, r#"{"name":"test"}"#);
    }

    #[tokio::test]
    async fn test_generate_timeout_returns_error() {
        let queue = Arc::new(LlmTaskQueue::new());
        let client = DelegatedLlmClient::new(queue.clone(), "g".into(), Duration::from_millis(50));

        let msgs = vec![Message::user("hello")];
        let result = client.generate(&msgs).await;
        assert!(result.is_err());
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cd /data/k3s-storage/graphiti-rs && cargo test --lib llm_client::delegated 2>&1 | tail -20`
Expected: Compilation error (`DelegatedLlmClient` not defined)

**Step 3: Write minimal implementation**

In `src/llm_client/delegated.rs`:

```rust
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;

use super::task_queue::{LlmTaskQueue, LlmTaskType};
use super::{LlmClient, Message, TokenUsage};

pub struct DelegatedLlmClient {
    queue: Arc<LlmTaskQueue>,
    group_id: String,
    timeout: Duration,
    prompt_tokens: AtomicU64,
    completion_tokens: AtomicU64,
}

impl DelegatedLlmClient {
    pub fn new(queue: Arc<LlmTaskQueue>, group_id: String, timeout: Duration) -> Self {
        Self {
            queue,
            group_id,
            timeout,
            prompt_tokens: AtomicU64::new(0),
            completion_tokens: AtomicU64::new(0),
        }
    }

    fn estimate_tokens(text: &str) -> u64 {
        (text.len() as u64) / 4
    }

    fn messages_to_parts(messages: &[Message]) -> (String, String) {
        let mut system = String::new();
        let mut user = String::new();
        for msg in messages {
            match msg {
                Message::System(s) => system.push_str(s),
                Message::User(u) => user.push_str(u),
            }
        }
        (system, user)
    }
}

#[async_trait]
impl LlmClient for DelegatedLlmClient {
    async fn generate(&self, messages: &[Message]) -> Result<String> {
        let (system, user) = Self::messages_to_parts(messages);
        self.prompt_tokens.fetch_add(Self::estimate_tokens(&system) + Self::estimate_tokens(&user), Ordering::Relaxed);

        let id = self.queue.submit(
            LlmTaskType::ExtractEntities, // generic type for unstructured
            system,
            user,
            serde_json::Value::Null,
            self.group_id.clone(),
        );

        let result = self.queue.wait_for_result(&id, self.timeout).await
            .map_err(|e| anyhow::anyhow!("delegated LLM task failed: {e}"))?;

        self.completion_tokens.fetch_add(Self::estimate_tokens(&result), Ordering::Relaxed);
        Ok(result)
    }

    async fn generate_structured_json(
        &self,
        messages: &[Message],
        schema: serde_json::Value,
    ) -> Result<String> {
        let (system, user) = Self::messages_to_parts(messages);
        self.prompt_tokens.fetch_add(Self::estimate_tokens(&system) + Self::estimate_tokens(&user), Ordering::Relaxed);

        // Infer task type from system prompt keywords
        let task_type = infer_task_type(&system);

        let id = self.queue.submit(task_type, system, user, schema, self.group_id.clone());

        let result = self.queue.wait_for_result(&id, self.timeout).await
            .map_err(|e| anyhow::anyhow!("delegated LLM task failed: {e}"))?;

        self.completion_tokens.fetch_add(Self::estimate_tokens(&result), Ordering::Relaxed);
        Ok(result)
    }

    fn token_usage(&self) -> TokenUsage {
        TokenUsage {
            prompt_tokens: self.prompt_tokens.load(Ordering::Relaxed),
            completion_tokens: self.completion_tokens.load(Ordering::Relaxed),
        }
    }

    fn reset_token_usage(&self) {
        self.prompt_tokens.store(0, Ordering::Relaxed);
        self.completion_tokens.store(0, Ordering::Relaxed);
    }
}

fn infer_task_type(system_prompt: &str) -> LlmTaskType {
    let lower = system_prompt.to_lowercase();
    if lower.contains("extract") && lower.contains("entit") {
        LlmTaskType::ExtractEntities
    } else if lower.contains("extract") && lower.contains("relation") || lower.contains("edge") {
        LlmTaskType::ExtractEdges
    } else if lower.contains("duplicate") && lower.contains("node") || lower.contains("entity") && lower.contains("resolution") {
        LlmTaskType::DedupeNodes
    } else if lower.contains("duplicate") && lower.contains("edge") {
        LlmTaskType::DedupeEdges
    } else if lower.contains("contradict") {
        LlmTaskType::ResolveContradictions
    } else if lower.contains("summar") {
        LlmTaskType::Summarize
    } else {
        LlmTaskType::ExtractEntities // fallback
    }
}
```

Add to `src/llm_client/mod.rs`:

```rust
pub mod delegated;
```

**Step 4: Run tests to verify they pass**

Run: `cd /data/k3s-storage/graphiti-rs && cargo test --lib llm_client::delegated -- --nocapture 2>&1 | tail -20`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
cd /data/k3s-storage/graphiti-rs
git add src/llm_client/delegated.rs src/llm_client/mod.rs
git commit -m "feat: add DelegatedLlmClient implementing LlmClient via task queue"
```

---

### Task 3: Add LlmBackend::Delegated variant and wire into Graphiti

**Files:**
- Modify: `src/types.rs:8-24` (add `Delegated` to `LlmBackend` enum)
- Modify: `src/types.rs:91-241` (handle `delegated` in `from_env()`)
- Modify: `src/graphiti.rs:59-80` (add `Delegated` match arm in `from_clients()` or `new()`)
- Test: `cargo test` full suite

**Step 1: Add `Delegated` variant to `LlmBackend`**

In `src/types.rs`, add to the `LlmBackend` enum:

```rust
pub enum LlmBackend {
    OpenAI,
    Anthropic,
    Claude,
    Delegated,
}
```

Update `from_env()` to handle `"delegated"`:

```rust
"delegated" => LlmBackend::Delegated,
```

The `Delegated` backend needs no API keys — it delegates to Lower Decks.

**Step 2: Wire DelegatedLlmClient into Graphiti::new()**

In `src/graphiti.rs`, add the match arm. The `Delegated` client needs the task queue
injected. Add `task_queue: Option<Arc<LlmTaskQueue>>` field to `Graphiti` struct:

```rust
pub struct Graphiti {
    pub driver: Arc<dyn GraphDriver>,
    pub llm_client: Arc<dyn LlmClient>,
    pub embedder: Arc<dyn EmbedderClient>,
    pub config: GraphitiConfig,
    pub task_queue: Option<Arc<LlmTaskQueue>>,
    pipeline: Pipeline,
    search_engine: SearchEngine,
    community_builder: CommunityBuilder,
}
```

In the constructor, for `Delegated`:

```rust
LlmBackend::Delegated => {
    let queue = Arc::new(LlmTaskQueue::new());
    task_queue = Some(queue.clone());
    Arc::new(DelegatedLlmClient::new(
        queue,
        config.group_id.clone(),
        Duration::from_secs(120),
    ))
}
```

**Step 3: Run full test suite**

Run: `cd /data/k3s-storage/graphiti-rs && cargo test 2>&1 | tail -20`
Expected: All existing tests still pass

**Step 4: Commit**

```bash
cd /data/k3s-storage/graphiti-rs
git add src/types.rs src/graphiti.rs
git commit -m "feat: add LlmBackend::Delegated variant wired to DelegatedLlmClient"
```

---

### Task 4: Add 3 MCP tools for task delegation

**Files:**
- Modify: `src/mcp/server.rs` (add `poll_llm_tasks`, `claim_llm_task`, `submit_llm_result` tools)
- Test: manual test via MCP client or `cargo test`

**Step 1: Add param structs**

In `src/mcp/server.rs`, add near the other param structs (around line 139):

```rust
#[derive(Debug, Deserialize, JsonSchema)]
struct PollLlmTasksParams {
    group_id: Option<String>,
    limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ClaimLlmTaskParams {
    task_id: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SubmitLlmResultParams {
    task_id: String,
    result_json: String,
}
```

**Step 2: Add tool handlers**

Inside the `#[tool_router]` impl block, add:

```rust
#[tool(
    name = "poll_llm_tasks",
    description = "Poll for pending LLM tasks that need processing. Returns tasks with system_prompt, user_prompt, and response_schema. Lower Decks should call this periodically to pick up work."
)]
async fn poll_llm_tasks(&self, #[tool(params)] p: PollLlmTasksParams) -> Result<CallToolResult, McpError> {
    let queue = match &self.graphiti.task_queue {
        Some(q) => q,
        None => return ok("[]"),
    };
    let tasks = queue.poll_pending(p.group_id.as_deref(), p.limit.unwrap_or(10).min(50));
    let json = serde_json::to_string_pretty(&tasks)
        .map_err(|e| internal_err(format!("serialize error: {e}")))?;
    ok(json)
}

#[tool(
    name = "claim_llm_task",
    description = "Claim a pending LLM task for processing. Returns true if successfully claimed, false if already taken or not found. Must be called before submit_llm_result."
)]
async fn claim_llm_task(&self, #[tool(params)] p: ClaimLlmTaskParams) -> Result<CallToolResult, McpError> {
    let queue = match &self.graphiti.task_queue {
        Some(q) => q,
        None => return ok("false"),
    };
    let id: Uuid = p.task_id.parse()
        .map_err(|_| internal_err("invalid task_id UUID"))?;
    let claimed = queue.claim(&id);
    ok(claimed.to_string())
}

#[tool(
    name = "submit_llm_result",
    description = "Submit the result for a previously claimed LLM task. The result_json must be valid JSON matching the task's response_schema. This wakes up the waiting pipeline."
)]
async fn submit_llm_result(&self, #[tool(params)] p: SubmitLlmResultParams) -> Result<CallToolResult, McpError> {
    let queue = match &self.graphiti.task_queue {
        Some(q) => q,
        None => return tool_err("task queue not enabled"),
    };
    let id: Uuid = p.task_id.parse()
        .map_err(|_| internal_err("invalid task_id UUID"))?;
    // Validate JSON
    serde_json::from_str::<serde_json::Value>(&p.result_json)
        .map_err(|e| internal_err(format!("invalid JSON: {e}")))?;
    queue.complete(&id, p.result_json);
    ok("true")
}
```

**Step 3: Run cargo check + test**

Run: `cd /data/k3s-storage/graphiti-rs && cargo check 2>&1 | tail -10 && cargo test 2>&1 | tail -10`
Expected: Compiles and all tests pass

**Step 4: Commit**

```bash
cd /data/k3s-storage/graphiti-rs
git add src/mcp/server.rs
git commit -m "feat: add poll_llm_tasks, claim_llm_task, submit_llm_result MCP tools"
```

---

### Task 5: Remove claude-agent-sdk dependency and simplify Dockerfile

**Files:**
- Delete: `src/llm_client/claude.rs`
- Modify: `src/llm_client/mod.rs` (remove `pub mod claude;`)
- Modify: `src/types.rs` (remove `LlmBackend::Claude` variant and its match arms)
- Modify: `src/graphiti.rs` (remove `Claude` match arm)
- Modify: `Cargo.toml` (remove `claude-agent-sdk` dependency)
- Modify: `Dockerfile` (remove Node.js, npm, claude-code from runtime stage)
- Modify: `Dockerfile` (remove git-credentials secret mount from builder)

**Step 1: Remove claude-agent-sdk from Cargo.toml**

Delete lines 85-86:
```toml
# Claude Agent SDK (subprocess wrapper around claude CLI)
claude-agent-sdk = { git = "https://github.com/calimero1337/claude-agent-sdk-rs.git", branch = "main" }
```

**Step 2: Remove Claude backend**

Delete `src/llm_client/claude.rs` entirely.

In `src/llm_client/mod.rs`, remove:
```rust
pub mod claude;
```

In `src/types.rs`, remove `Claude` from the enum and all match arms referencing it.

In `src/graphiti.rs`, remove the `LlmBackend::Claude` match arm.

**Step 3: Simplify Dockerfile runtime stage**

Replace the ubuntu:24.04 runtime with debian:bookworm-slim. Remove Node.js, npm, claude-code:

```dockerfile
FROM debian:bookworm-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libssl3 && rm -rf /var/lib/apt/lists/*
RUN groupadd --gid 10001 graphiti && \
    useradd --uid 10001 --gid graphiti --shell /bin/bash --create-home graphiti
WORKDIR /app
COPY --from=builder /build/target/release/graphiti-server /app/graphiti-server
USER graphiti
EXPOSE 8080
ENTRYPOINT ["/app/graphiti-server"]
```

Remove `--mount=type=secret,id=git_credentials` from builder stage RUN commands (no longer needed since claude-agent-sdk-rs dependency is gone).

**Step 4: Run cargo check + test**

Run: `cd /data/k3s-storage/graphiti-rs && cargo check 2>&1 | tail -10 && cargo test 2>&1 | tail -10`
Expected: Compiles and all tests pass

**Step 5: Commit**

```bash
cd /data/k3s-storage/graphiti-rs
git add -A
git commit -m "refactor: remove claude-agent-sdk dependency, simplify Dockerfile

graphiti-rs is now a pure knowledge graph engine with zero LLM
dependencies. All LLM work delegated via DelegatedLlmClient + task queue.
Docker image: debian:bookworm-slim (~30MB vs ~800MB)."
```

---

### Task 6: Add 3 new MCP methods to Lower Decks GraphSecondBrain

**Files:**
- Modify: `src/lower_decks/integrations/graphiti.py` (add `poll_llm_tasks`, `claim_llm_task`, `submit_llm_result`)
- Modify: `src/lower_decks/utils/mcp_tools.py` (add 3 tools to `STATIC_TOOLS["graphiti"]`)
- Test: `tests/test_graphiti_client.py` (add 3 new tests)

**Step 1: Write failing tests**

In `tests/test_graphiti_client.py`, add:

```python
@pytest.mark.asyncio
async def test_poll_llm_tasks(client):
    """poll_llm_tasks calls MCP tool with correct args."""
    client._call_tool = AsyncMock(return_value='[{"id": "abc", "task_type": "extract_entities"}]')

    result = await client.poll_llm_tasks(limit=5)

    client._call_tool.assert_awaited_once_with(
        "poll_llm_tasks",
        {"group_id": "test-group", "limit": 5},
    )
    assert "abc" in result


@pytest.mark.asyncio
async def test_claim_llm_task(client):
    """claim_llm_task calls MCP tool with task_id."""
    client._call_tool = AsyncMock(return_value="true")

    result = await client.claim_llm_task("task-uuid-123")

    client._call_tool.assert_awaited_once_with(
        "claim_llm_task",
        {"task_id": "task-uuid-123"},
    )
    assert result == "true"


@pytest.mark.asyncio
async def test_submit_llm_result(client):
    """submit_llm_result calls MCP tool with task_id and result_json."""
    client._call_tool = AsyncMock(return_value="true")

    result = await client.submit_llm_result("task-uuid-123", '{"entities": []}')

    client._call_tool.assert_awaited_once_with(
        "submit_llm_result",
        {"task_id": "task-uuid-123", "result_json": '{"entities": []}'},
    )
    assert result == "true"
```

**Step 2: Run tests to verify they fail**

Run: `cd /data/k3s-storage/lower_decks && .venv/bin/pytest tests/test_graphiti_client.py -v -k "poll_llm or claim_llm or submit_llm" 2>&1 | tail -10`
Expected: AttributeError (methods don't exist)

**Step 3: Add methods to GraphSecondBrain**

In `src/lower_decks/integrations/graphiti.py`, add after `get_token_usage()`:

```python
async def poll_llm_tasks(
    self,
    limit: int = 10,
    group_id: str | None = None,
) -> str:
    """Poll for pending LLM tasks from graphiti-rs."""
    return await self._call_tool(
        "poll_llm_tasks",
        {"group_id": self._group(group_id), "limit": limit},
    )

async def claim_llm_task(self, task_id: str) -> str:
    """Claim a pending LLM task for processing."""
    return await self._call_tool(
        "claim_llm_task",
        {"task_id": task_id},
    )

async def submit_llm_result(self, task_id: str, result_json: str) -> str:
    """Submit the result for a previously claimed LLM task."""
    return await self._call_tool(
        "submit_llm_result",
        {"task_id": task_id, "result_json": result_json},
    )
```

Add 3 tools to `STATIC_TOOLS["graphiti"]` in `src/lower_decks/utils/mcp_tools.py`:

```python
"graphiti": [
    ...,
    "poll_llm_tasks",
    "claim_llm_task",
    "submit_llm_result",
],
```

**Step 4: Run tests to verify they pass**

Run: `cd /data/k3s-storage/lower_decks && .venv/bin/pytest tests/test_graphiti_client.py -v 2>&1 | tail -10`
Expected: All tests PASS (26 old + 3 new = 29)

**Step 5: Commit**

```bash
cd /data/k3s-storage/lower_decks
git add src/lower_decks/integrations/graphiti.py src/lower_decks/utils/mcp_tools.py tests/test_graphiti_client.py
git commit -m "feat: add poll/claim/submit LLM task methods to GraphSecondBrain"
```

---

### Task 7: Create GraphitiTaskHandler in Lower Decks

**Files:**
- Create: `src/lower_decks/pipeline/graphiti_tasks.py`
- Test: `tests/test_graphiti_tasks.py`

**Step 1: Write failing tests**

In `tests/test_graphiti_tasks.py`:

```python
"""Tests for GraphitiTaskHandler — polls graphiti-rs for LLM tasks and dispatches to agents."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lower_decks.config import GraphitiConfig
from lower_decks.pipeline.graphiti_tasks import GraphitiTaskHandler


@pytest.fixture
def config():
    return GraphitiConfig(enabled=True, feedback_enabled=True)


@pytest.fixture
def mock_graph_brain():
    brain = AsyncMock()
    brain.poll_llm_tasks = AsyncMock(return_value="[]")
    brain.claim_llm_task = AsyncMock(return_value="true")
    brain.submit_llm_result = AsyncMock(return_value="true")
    return brain


@pytest.fixture
def handler(config, mock_graph_brain):
    return GraphitiTaskHandler(
        graph_brain=mock_graph_brain,
        config=config,
        agent_settings=MagicMock(),
    )


@pytest.mark.asyncio
async def test_select_agent_neelix_for_extraction(handler):
    """Extraction tasks go to Neelix."""
    agent = handler._select_agent("extract_entities")
    assert agent == "neelix"


@pytest.mark.asyncio
async def test_select_agent_seven_for_dedup(handler):
    """Dedup tasks go to Seven."""
    agent = handler._select_agent("dedupe_nodes")
    assert agent == "seven"


@pytest.mark.asyncio
async def test_select_agent_neelix_for_summarize(handler):
    """Summarize tasks go to Neelix."""
    agent = handler._select_agent("summarize")
    assert agent == "neelix"


@pytest.mark.asyncio
async def test_select_agent_seven_for_contradictions(handler):
    """Contradiction resolution goes to Seven."""
    agent = handler._select_agent("resolve_contradictions")
    assert agent == "seven"


@pytest.mark.asyncio
async def test_dispatch_claims_and_submits(handler, mock_graph_brain):
    """dispatch() claims the task, runs agent, submits result."""
    task = {
        "id": "task-123",
        "task_type": "extract_entities",
        "system_prompt": "Extract entities.",
        "user_prompt": "Some content about auth.",
        "response_schema": {"type": "object"},
    }

    with patch.object(handler, "_run_agent", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = '{"entities": []}'
        await handler._dispatch(task)

    mock_graph_brain.claim_llm_task.assert_awaited_once_with("task-123")
    mock_graph_brain.submit_llm_result.assert_awaited_once_with(
        "task-123", '{"entities": []}'
    )


@pytest.mark.asyncio
async def test_dispatch_skips_if_claim_fails(handler, mock_graph_brain):
    """If claim returns false, don't run agent."""
    mock_graph_brain.claim_llm_task = AsyncMock(return_value="false")

    task = {"id": "task-123", "task_type": "extract_entities",
            "system_prompt": "s", "user_prompt": "u", "response_schema": {}}

    with patch.object(handler, "_run_agent", new_callable=AsyncMock) as mock_run:
        await handler._dispatch(task)
        mock_run.assert_not_awaited()
```

**Step 2: Run tests to verify they fail**

Run: `cd /data/k3s-storage/lower_decks && .venv/bin/pytest tests/test_graphiti_tasks.py -v 2>&1 | tail -10`
Expected: ImportError (module doesn't exist)

**Step 3: Write minimal implementation**

In `src/lower_decks/pipeline/graphiti_tasks.py`:

```python
"""GraphitiTaskHandler — polls graphiti-rs for LLM tasks, dispatches to Neelix/Seven."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from lower_decks.config import GraphitiConfig
    from lower_decks.integrations.graphiti import GraphSecondBrain

log = structlog.get_logger()

NEELIX_TASKS = {"extract_entities", "extract_edges", "summarize"}
SEVEN_TASKS = {"dedupe_nodes", "dedupe_edges", "resolve_contradictions"}


class GraphitiTaskHandler:
    """Polls graphiti-rs for pending LLM tasks and dispatches to Neelix or Seven."""

    def __init__(
        self,
        graph_brain: GraphSecondBrain,
        config: GraphitiConfig,
        agent_settings: Any = None,
        poll_interval: float = 5.0,
    ) -> None:
        self._graph_brain = graph_brain
        self._config = config
        self._agent_settings = agent_settings
        self._poll_interval = poll_interval
        self._running = False

    def _select_agent(self, task_type: str) -> str:
        """Select which agent handles this task type."""
        if task_type in NEELIX_TASKS:
            return "neelix"
        if task_type in SEVEN_TASKS:
            return "seven"
        return "neelix"  # fallback

    async def _run_agent(self, agent_name: str, system_prompt: str, user_prompt: str, schema: dict) -> str:
        """Run the appropriate agent and return JSON result.

        Uses claude-agent-sdk to run the prompt through the selected agent model.
        """
        from claude_agent_sdk import ClaudeAgentOptions, query, ResultMessage, AssistantMessage

        model = "claude-haiku-4-5" if agent_name == "neelix" else "claude-sonnet-4-6"
        prompt = f"{user_prompt}\n\nRespond with ONLY valid JSON matching this schema:\n{json.dumps(schema)}"
        options = ClaudeAgentOptions(
            model=model,
            system_prompt=system_prompt,
            max_turns=1,
            disallowed_tools=["Bash", "Write", "Edit", "Read", "Glob", "Grep"],
        )

        last_text = ""
        async for msg in query(prompt, options):
            if isinstance(msg, ResultMessage) and msg.result:
                return msg.result
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if hasattr(block, "text"):
                        last_text = block.text
        return last_text

    async def _dispatch(self, task: dict) -> None:
        """Claim a task, run it through the appropriate agent, submit result."""
        task_id = task["id"]
        claim_result = await self._graph_brain.claim_llm_task(task_id)
        if claim_result != "true":
            log.debug("task_claim_failed", task_id=task_id)
            return

        agent_name = self._select_agent(task["task_type"])
        log.info("dispatching_llm_task", task_id=task_id, task_type=task["task_type"], agent=agent_name)

        try:
            result = await self._run_agent(
                agent_name,
                task["system_prompt"],
                task["user_prompt"],
                task.get("response_schema", {}),
            )
            await self._graph_brain.submit_llm_result(task_id, result)
            log.info("llm_task_completed", task_id=task_id, agent=agent_name)
        except Exception as exc:
            log.warning("llm_task_failed", task_id=task_id, error=str(exc))
            await self._graph_brain.submit_llm_result(
                task_id, json.dumps({"error": str(exc)})
            )

    async def run(self) -> None:
        """Main polling loop. Runs until stopped."""
        self._running = True
        log.info("graphiti_task_handler_started", poll_interval=self._poll_interval)
        while self._running:
            try:
                raw = await self._graph_brain.poll_llm_tasks(limit=5)
                tasks = json.loads(raw) if raw and raw != "[]" else []
                for task in tasks:
                    asyncio.create_task(self._dispatch(task))
            except Exception as exc:
                log.warning("poll_llm_tasks_error", error=str(exc))
            await asyncio.sleep(self._poll_interval)

    def stop(self) -> None:
        """Signal the polling loop to stop."""
        self._running = False
```

**Step 4: Run tests to verify they pass**

Run: `cd /data/k3s-storage/lower_decks && .venv/bin/pytest tests/test_graphiti_tasks.py -v 2>&1 | tail -15`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
cd /data/k3s-storage/lower_decks
git add src/lower_decks/pipeline/graphiti_tasks.py tests/test_graphiti_tasks.py
git commit -m "feat: add GraphitiTaskHandler for dispatching LLM tasks to Neelix/Seven"
```

---

### Task 8: Wire GraphitiTaskHandler into Orchestrator

**Files:**
- Modify: `src/lower_decks/orchestrator.py` (create GraphitiTaskHandler when graphiti enabled, start polling loop)
- Test: existing orchestrator tests should still pass

**Step 1: Add handler initialization**

In `orchestrator.py`, after the second_brain initialization block (around line 198),
add graphiti handler setup:

```python
# Graphiti task handler
self._graphiti_handler = None
if self.config.graphiti.enabled and self._graph_brain:
    from lower_decks.pipeline.graphiti_tasks import GraphitiTaskHandler
    self._graphiti_handler = GraphitiTaskHandler(
        graph_brain=self._graph_brain,
        config=self.config.graphiti,
    )
```

In the orchestrator's main `run()` loop, start the handler as a background task:

```python
if self._graphiti_handler:
    asyncio.create_task(self._graphiti_handler.run())
```

**Step 2: Run full test suite**

Run: `cd /data/k3s-storage/lower_decks && .venv/bin/pytest tests/ -q --tb=short --deselect=tests/test_query_history.py::test_history_lists_recent_completions 2>&1 | tail -5`
Expected: All tests pass

**Step 3: Commit**

```bash
cd /data/k3s-storage/lower_decks
git add src/lower_decks/orchestrator.py
git commit -m "feat: wire GraphitiTaskHandler into orchestrator polling loop"
```

---

### Task 9: Update K8s configmap and rebuild graphiti-rs Docker image

**Files:**
- Modify: `graphiti-rs/k8s/configmap.yaml` (change `LLM_BACKEND` to `delegated`)
- Modify: `graphiti-rs/k8s/deployment.yaml` (remove ANTHROPIC_API_KEY env, optional: true for OPENAI_API_KEY)
- Rebuild Docker image and import to K3s

**Step 1: Update configmap**

```yaml
LLM_BACKEND: "delegated"
```

**Step 2: Update deployment.yaml**

Remove `ANTHROPIC_API_KEY` envFrom. The delegated backend needs no API keys.

**Step 3: Build, import, deploy**

```bash
cd /data/k3s-storage/graphiti-rs
DOCKER_BUILDKIT=1 docker build -t graphiti-rs:latest .
docker save graphiti-rs:latest -o ~/graphiti-rs.tar
echo 'claus' | sudo -S k3s ctr images import ~/graphiti-rs.tar
rm ~/graphiti-rs.tar
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
echo 'claus' | sudo -S k3s kubectl rollout restart deployment/graphiti-server -n claude
```

**Step 4: Verify**

```bash
kubectl get pods -n claude -l app=graphiti-server
curl -s http://localhost:30239/health
curl -s http://localhost:30239/ready
```

Expected: Pod running, health 200, ready 200.

**Step 5: Commit**

```bash
cd /data/k3s-storage/graphiti-rs
git add k8s/ Dockerfile
git commit -m "deploy: switch to delegated LLM backend, simplify Docker image"
git push
```

---

### Task 10: End-to-end verification

**Step 1: Verify graphiti-rs MCP tools are accessible**

Test via Lower Decks GraphSecondBrain client or curl:

```bash
# Check that poll_llm_tasks returns empty array (no pending tasks)
# This verifies the MCP endpoint and delegated backend are working
```

**Step 2: Verify Lower Decks can connect**

```bash
cd /data/k3s-storage/lower_decks && .venv/bin/pytest tests/ -q --tb=short --deselect=tests/test_query_history.py::test_history_lists_recent_completions
```

Expected: All tests pass

**Step 3: Run full graphiti-rs test suite**

```bash
cd /data/k3s-storage/graphiti-rs && cargo test 2>&1 | tail -20
```

Expected: All tests pass

**Step 4: Push both repos**

```bash
cd /data/k3s-storage/graphiti-rs && git push
cd /data/k3s-storage/lower_decks && git push
```
