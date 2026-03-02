# LLM Task Delegation via MCP

**Date:** 2026-03-02
**Status:** Approved

## Problem

graphiti-rs currently spawns `claude` CLI subprocesses via `claude-agent-sdk-rs` for 7 LLM
operations (entity extraction, edge extraction, dedup, contradiction resolution, summarization).
This requires Node.js + Claude CLI in the Docker image (~800MB runtime), tightly couples the
knowledge graph engine to a specific LLM backend, and wastes the potential of Lower Decks'
specialized agents (Neelix for curation, Seven for analysis).

## Decision

Remove all direct LLM dependencies from graphiti-rs. Instead, graphiti-rs maintains an
in-memory task queue. Lower Decks polls for pending tasks via MCP, dispatches them to
Neelix or Seven, and posts results back. graphiti-rs becomes a pure knowledge graph engine
with zero LLM dependencies.

## Architecture

```
Episode ingestion                     Lower Decks
─────────────────                     ───────────
POST /v1/episodes
  → pipeline starts
  → DelegatedLlmClient.generate_structured()
    → creates LlmTask(type, prompt, schema)
    → inserts into TaskQueue
    → waits on Notify (with timeout)
                                      GraphitiTaskHandler
                                        → poll_llm_tasks() via MCP
                                        → claim_llm_task(id)
                                        → dispatch to Neelix or Seven
                                        → agent processes prompt
                                        → submit_llm_result(id, json)
  ← Notify fires, pipeline resumes
  → deserialize JSON to T
  → continue extraction
```

## Task Queue

In-memory `DashMap<Uuid, LlmTask>` with `tokio::sync::Notify` per task.

```rust
struct LlmTask {
    id: Uuid,
    task_type: LlmTaskType,        // ExtractEntities, ExtractEdges, DedupeNodes,
                                    // DedupeEdges, ResolveContradictions, Summarize
    status: TaskStatus,             // Pending, InProgress, Completed, Failed
    system_prompt: String,
    user_prompt: String,
    response_schema: serde_json::Value,
    result: Option<String>,
    group_id: String,
    created_at: DateTime<Utc>,
    completed_at: Option<DateTime<Utc>>,
    timeout_secs: u64,              // default: 120
}
```

## New MCP Tools (graphiti-rs)

| Tool | Params | Returns | Purpose |
|------|--------|---------|---------|
| `poll_llm_tasks` | `group_id?, limit?` | `Vec<LlmTaskSummary>` | Get pending tasks |
| `claim_llm_task` | `task_id` | `bool` | Atomically claim a task |
| `submit_llm_result` | `task_id, result_json` | `bool` | Post result, wake pipeline |

## DelegatedLlmClient

Implements `LlmClient` trait. Instead of calling an LLM:

1. Builds `LlmTask` from messages + schema
2. Inserts into `TaskQueue`
3. Awaits `Notify` with timeout
4. Deserializes result JSON to expected type
5. Returns to caller

Zero changes needed to pipeline, prompts, or call sites — they all go through the
`LlmClient` trait abstraction.

## Lower Decks: GraphitiTaskHandler

```python
class GraphitiTaskHandler:
    async def run(self):
        while True:
            tasks = await self._graph_brain.poll_llm_tasks(limit=5)
            for task in tasks:
                claimed = await self._graph_brain.claim_llm_task(task["id"])
                if claimed:
                    asyncio.create_task(self._dispatch(task))
            await asyncio.sleep(self._poll_interval)

    async def _dispatch(self, task):
        agent = self._select_agent(task["task_type"])
        result = await agent.run(task["system_prompt"], task["user_prompt"])
        await self._graph_brain.submit_llm_result(task["id"], result)
```

## Agent Assignment

| Task Type | Agent | Rationale |
|-----------|-------|-----------|
| `extract_entities` | Neelix | Knowledge curation |
| `extract_edges` | Neelix | Relationship identification |
| `summarize` | Neelix | Information condensing |
| `dedupe_nodes` | Seven | Analytical comparison |
| `dedupe_edges` | Seven | Duplicate detection |
| `resolve_contradictions` | Seven | Temporal reasoning |

## Docker Image Simplification

**Before:** `ubuntu:24.04` + Node.js + npm + `@anthropic-ai/claude-code` (~800MB)
**After:** `debian:bookworm-slim` (~30MB)

No more git credentials secret needed at build time (claude-agent-sdk-rs removed).

## Timeout & Error Handling

- Default timeout: 120 seconds per task
- If not picked up within timeout: pipeline step fails with retriable error
- Failed tasks requeued up to 2 times
- If Lower Decks down: episode ingestion returns 503 Service Unavailable
- Stale task cleanup: tasks older than 10 minutes auto-expired

## Dependencies Removed from graphiti-rs

- `claude-agent-sdk` (git dependency)
- Node.js runtime
- npm / `@anthropic-ai/claude-code`

## Dependencies Added

- None (task queue uses existing `dashmap`, `tokio`, `uuid`, `chrono`)

## New Dependencies in Lower Decks

- 3 new methods on `GraphSecondBrain`: `poll_llm_tasks`, `claim_llm_task`, `submit_llm_result`
- New `GraphitiTaskHandler` class in orchestrator
