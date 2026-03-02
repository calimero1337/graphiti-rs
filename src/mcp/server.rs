//! Graphiti MCP server — exposes knowledge-graph operations via the Model Context Protocol.
//!
//! Follows the same rmcp patterns as obsidian-mcp-server, using `#[tool_router]` /
//! `#[tool]` / `#[tool_handler]` procedural macros.

use std::sync::Arc;

use rmcp::{
    handler::server::{tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, Content, Implementation, ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router, ErrorData as McpError, ServerHandler,
};
use schemars::JsonSchema;
use serde::Deserialize;
use tracing::debug;
use uuid::Uuid;

use crate::errors::GraphitiError;
use crate::graphiti::Graphiti;
use crate::nodes::episodic::EpisodeType;
use crate::search::SearchConfig;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Maximum number of search results an MCP caller may request.
const MAX_SEARCH_LIMIT: usize = 100;
/// Maximum number of episodes an MCP caller may request.
const MAX_EPISODES_LIMIT: usize = 200;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn ok(text: impl Into<String>) -> Result<CallToolResult, McpError> {
    Ok(CallToolResult::success(vec![Content::text(text.into())]))
}

fn tool_err(text: impl Into<String>) -> Result<CallToolResult, McpError> {
    Ok(CallToolResult::error(vec![Content::text(text.into())]))
}

fn graphiti_err(e: GraphitiError) -> Result<CallToolResult, McpError> {
    tool_err(e.to_string())
}

// ── Parameter types ───────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, JsonSchema)]
struct AddEpisodeParams {
    /// Name / title for the episode.
    name: String,
    /// Text content to ingest into the knowledge graph.
    content: String,
    /// Content type: "text", "message", or "json". Defaults to "text".
    source_type: Option<String>,
    /// Group ID to scope the episode (defaults to "default").
    group_id: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SearchParams {
    /// Natural-language query to search for.
    query: String,
    /// Maximum number of results to return (default: 10).
    limit: Option<usize>,
    /// Group ID to scope the search (defaults to "default").
    group_id: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct GetEntityParams {
    /// Entity name to look up (fuzzy-matched against the graph).
    name: Option<String>,
    /// Entity UUID for an exact lookup.
    uuid: Option<String>,
    /// Group ID to scope name-based lookup (defaults to "default").
    group_id: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ListEpisodesParams {
    /// Group ID to list episodes for (defaults to "default").
    group_id: Option<String>,
    /// Maximum number of episodes to return (default: 20).
    limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct BuildCommunitiesParams {
    /// Group ID to build communities for (defaults to "default").
    group_id: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ContextualizeParams {
    /// Natural-language query to find relevant knowledge.
    query: String,
    /// Maximum number of context entities to return (default: 10).
    limit: Option<usize>,
    /// Group ID to scope the search (defaults to "default").
    group_id: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct RecordOutcomeParams {
    /// Identifier of the task that produced this outcome.
    task_id: String,
    /// Agent that executed the task.
    agent: String,
    /// Entity names relevant to this outcome.
    entity_names: Vec<String>,
    /// Whether the task succeeded.
    success: bool,
    /// Free-text details about the outcome.
    details: String,
    /// Group ID to scope the outcome (defaults to "default").
    group_id: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct GetTimelineParams {
    /// Entity name to get the timeline for.
    entity_name: String,
    /// Group ID (defaults to "default").
    group_id: Option<String>,
    /// Maximum number of timeline entries to return.
    limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct RememberParams {
    /// The knowledge content to remember.
    content: String,
    /// Source context: task ID, agent name, or description.
    source: Option<String>,
    /// Group ID to scope the memory (defaults to "default").
    group_id: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct GetTokenUsageParams {}

#[derive(Debug, Deserialize, JsonSchema)]
struct PollLlmTasksParams {
    /// Group ID to filter tasks (optional — returns all groups if absent).
    group_id: Option<String>,
    /// Maximum number of tasks to return (default: 10, max: 50).
    limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ClaimLlmTaskParams {
    /// UUID of the task to claim.
    task_id: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SubmitLlmResultParams {
    /// UUID of the task to submit results for.
    task_id: String,
    /// JSON string containing the LLM result. Must be valid JSON matching the task's response_schema.
    result_json: String,
}

// ── Server struct ─────────────────────────────────────────────────────────────

/// MCP server that exposes Graphiti knowledge-graph operations as tools.
#[derive(Clone)]
pub struct GraphitiMcpServer {
    graphiti: Arc<Graphiti>,
    tool_router: ToolRouter<Self>,
}

impl GraphitiMcpServer {
    /// Create a new `GraphitiMcpServer` backed by the given `Graphiti` instance.
    pub fn new(graphiti: Arc<Graphiti>) -> Self {
        Self {
            graphiti,
            tool_router: Self::tool_router(),
        }
    }
}

// ── Tool implementations ──────────────────────────────────────────────────────

#[tool_router]
impl GraphitiMcpServer {
    #[tool(description = "Ingest text content into the knowledge graph as an episode. Entities and relationships are extracted automatically via LLM.")]
    async fn add_episode(
        &self,
        params: Parameters<AddEpisodeParams>,
    ) -> Result<CallToolResult, McpError> {
        let p = params.0;
        debug!(name = %p.name, "add_episode");

        let source_type = match p.source_type.as_deref().unwrap_or("text") {
            "text" => EpisodeType::Text,
            "message" => EpisodeType::Message,
            "json" => EpisodeType::Json,
            other => {
                return tool_err(format!(
                    "unknown source_type '{other}'; expected text, message, or json"
                ))
            }
        };

        let group_id = p.group_id.as_deref().unwrap_or("default");

        let result = match self
            .graphiti
            .add_episode(&p.name, &p.content, source_type, group_id, "mcp")
            .await
        {
            Ok(r) => r,
            Err(e) => return graphiti_err(e),
        };

        ok(format!(
            "Episode ingested successfully.\n\
             Episode ID: {}\n\
             Entities extracted: {}\n\
             Relationships extracted: {}",
            result.episode.uuid,
            result.nodes.len(),
            result.edges.len(),
        ))
    }

    #[tool(description = "Search the knowledge graph for relevant facts using hybrid BM25 + vector search. Returns ranked facts with scores.")]
    async fn search(
        &self,
        params: Parameters<SearchParams>,
    ) -> Result<CallToolResult, McpError> {
        let p = params.0;
        debug!(query = %p.query, "search");

        let group_id = p.group_id.unwrap_or_else(|| "default".to_string());
        let limit = p.limit.unwrap_or(10).min(MAX_SEARCH_LIMIT);

        let config = SearchConfig::default()
            .with_group_ids(vec![group_id])
            .with_limit(limit);

        let results = match self.graphiti.search(&p.query, &config).await {
            Ok(r) => r,
            Err(e) => return graphiti_err(e),
        };

        if results.edges.is_empty() && results.nodes.is_empty() {
            return ok("No results found.");
        }

        let mut lines: Vec<String> = Vec::new();

        if !results.edges.is_empty() {
            lines.push(format!("Found {} fact(s):", results.edges.len()));
            for (i, (edge, score)) in results.edges.iter().enumerate() {
                lines.push(format!("{}. [score: {:.3}] {}", i + 1, score, edge.fact));
                match (edge.valid_at, edge.invalid_at) {
                    (Some(v), Some(iv)) => lines.push(format!(
                        "   Valid: {} → {}",
                        v.format("%Y-%m-%d"),
                        iv.format("%Y-%m-%d")
                    )),
                    (Some(v), None) => {
                        lines.push(format!("   Valid from: {}", v.format("%Y-%m-%d")))
                    }
                    _ => {}
                }
            }
        }

        if !results.nodes.is_empty() {
            if !lines.is_empty() {
                lines.push(String::new());
            }
            lines.push(format!("Related entities ({}):", results.nodes.len()));
            for (node, score) in &results.nodes {
                lines.push(format!(
                    "- {} [score: {:.3}]: {}",
                    node.name, score, node.summary
                ));
            }
        }

        ok(lines.join("\n"))
    }

    #[tool(description = "Get details about a specific entity in the knowledge graph. Provide either 'name' (fuzzy search) or 'uuid' (exact lookup).")]
    async fn get_entity(
        &self,
        params: Parameters<GetEntityParams>,
    ) -> Result<CallToolResult, McpError> {
        let p = params.0;
        debug!(name = ?p.name, uuid = ?p.uuid, "get_entity");

        let group_id = p.group_id.as_deref().unwrap_or("default");

        let node = if let Some(uuid_str) = &p.uuid {
            let uuid = match uuid_str.parse::<Uuid>() {
                Ok(u) => u,
                Err(_) => return tool_err(format!("invalid UUID: '{uuid_str}'")),
            };
            match self.graphiti.driver.get_entity_node(&uuid).await {
                Ok(Some(n)) => n,
                Ok(None) => return ok(format!("No entity found with UUID '{uuid_str}'.")),
                Err(e) => return graphiti_err(e),
            }
        } else if let Some(name) = &p.name {
            let mut nodes = match self
                .graphiti
                .driver
                .search_entity_nodes_by_name(name, group_id, 1)
                .await
            {
                Ok(n) => n,
                Err(e) => return graphiti_err(e),
            };
            if nodes.is_empty() {
                return ok(format!("No entity found with name '{name}'."));
            }
            nodes.remove(0)
        } else {
            return tool_err("provide either 'name' or 'uuid'");
        };

        let mut lines = vec![
            format!("Entity: {}", node.name),
            format!("UUID:   {}", node.uuid),
            format!("Group:  {}", node.group_id),
        ];

        if !node.summary.is_empty() {
            lines.push(format!("Summary: {}", node.summary));
        }

        if !node.labels.is_empty() {
            lines.push(format!("Labels:  {}", node.labels.join(", ")));
        }

        lines.push(format!(
            "Created: {}",
            node.created_at.format("%Y-%m-%d %H:%M UTC")
        ));

        ok(lines.join("\n"))
    }

    #[tool(description = "List recent episodes ingested into the knowledge graph, ordered by ingestion time descending.")]
    async fn list_episodes(
        &self,
        params: Parameters<ListEpisodesParams>,
    ) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let group_id = p.group_id.as_deref().unwrap_or("default");
        let limit = p.limit.unwrap_or(20).min(MAX_EPISODES_LIMIT);
        debug!(group_id, limit, "list_episodes");

        let episodes = match self.graphiti.retrieve_episodes(&[group_id], limit).await {
            Ok(e) => e,
            Err(e) => return graphiti_err(e),
        };

        if episodes.is_empty() {
            return ok("No episodes found.");
        }

        let mut lines = vec![format!("Episodes ({}):", episodes.len())];
        for ep in &episodes {
            lines.push(format!(
                "- [{}] {} ({})",
                ep.valid_at.format("%Y-%m-%d"),
                ep.name,
                ep.uuid,
            ));
        }

        ok(lines.join("\n"))
    }

    #[tool(description = "Trigger community detection to cluster related entities in the knowledge graph. Returns the number of communities found.")]
    async fn build_communities(
        &self,
        params: Parameters<BuildCommunitiesParams>,
    ) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let group_id = p.group_id.as_deref().unwrap_or("default");
        debug!(group_id, "build_communities");

        let communities = match self.graphiti.build_communities(&[group_id]).await {
            Ok(c) => c,
            Err(e) => return graphiti_err(e),
        };

        ok(format!(
            "Community detection complete. Communities found: {}",
            communities.len()
        ))
    }

    #[tool(description = "Get relevant knowledge context for a task or question. Performs hybrid search and expands results with 1-hop graph relationships. Use this as the primary retrieval tool.")]
    async fn contextualize(
        &self,
        params: Parameters<ContextualizeParams>,
    ) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let group_id = p.group_id.unwrap_or_else(|| "default".to_string());
        let limit = p.limit.unwrap_or(10).min(MAX_SEARCH_LIMIT);
        debug!(query = %p.query, group_id, limit, "contextualize");

        let resp = match self
            .graphiti
            .contextualize(&p.query, &[group_id], limit)
            .await
        {
            Ok(r) => r,
            Err(e) => return graphiti_err(e),
        };

        if resp.entities.is_empty() {
            return ok("No relevant knowledge found.");
        }

        let mut lines = Vec::new();
        lines.push(format!("Found {} relevant entities:", resp.entities.len()));

        for entity in &resp.entities {
            lines.push(format!("\n## {}", entity.name));
            if !entity.summary.is_empty() {
                lines.push(entity.summary.clone());
            }
            for rel in &entity.relationships {
                let validity = if rel.valid { "current" } else { "expired" };
                let target = if rel.target.is_empty() {
                    String::new()
                } else {
                    format!(" → {}", rel.target)
                };
                lines.push(format!("  - [{}]{}: {}", validity, target, rel.fact));
            }
        }

        if !resp.warnings.is_empty() {
            lines.push("\n⚠ Warnings:".to_string());
            for w in &resp.warnings {
                lines.push(format!("  - {w}"));
            }
        }

        ok(lines.join("\n"))
    }

    #[tool(description = "Record whether retrieved knowledge helped or not. Creates a feedback episode that strengthens or weakens entity reliability over time.")]
    async fn record_outcome(
        &self,
        params: Parameters<RecordOutcomeParams>,
    ) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let group_id = p.group_id.as_deref().unwrap_or("default");
        debug!(task_id = %p.task_id, agent = %p.agent, success = p.success, "record_outcome");

        let resp = match self
            .graphiti
            .record_outcome(&p.task_id, &p.agent, &p.entity_names, p.success, &p.details, group_id)
            .await
        {
            Ok(r) => r,
            Err(e) => return graphiti_err(e),
        };

        let label = if p.success { "SUCCESS" } else { "FAILURE" };
        ok(format!(
            "Outcome recorded: {label}\n\
             Episode ID: {}\n\
             Task: {}\n\
             Agent: {}\n\
             Entities: {}",
            resp.episode_id,
            p.task_id,
            p.agent,
            if p.entity_names.is_empty() { "none".to_string() } else { p.entity_names.join(", ") },
        ))
    }

    #[tool(description = "Get the chronological timeline of facts related to an entity. Shows how knowledge about the entity has evolved over time.")]
    async fn get_timeline(
        &self,
        params: Parameters<GetTimelineParams>,
    ) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let group_id = p.group_id.as_deref().unwrap_or("default");
        debug!(entity_name = %p.entity_name, group_id, "get_timeline");

        let resp = match self
            .graphiti
            .get_timeline(&p.entity_name, group_id, p.limit)
            .await
        {
            Ok(r) => r,
            Err(e) => return graphiti_err(e),
        };

        if resp.entries.is_empty() {
            return ok(format!("No timeline entries found for '{}'.", p.entity_name));
        }

        let mut lines = vec![format!("Timeline for '{}':", resp.entity_name)];
        for entry in &resp.entries {
            let validity = if entry.still_valid { "✓" } else { "✗" };
            let date = entry.valid_at.as_deref().unwrap_or("unknown");
            lines.push(format!("  [{validity}] {date}: {}", entry.fact));
            if let Some(inv) = &entry.invalid_at {
                lines.push(format!("       Invalidated: {inv}"));
            }
        }

        ok(lines.join("\n"))
    }

    #[tool(description = "Store knowledge in the graph. Agent reports a finding and graphiti-rs ingests it as an episode, automatically extracting entities and relationships.")]
    async fn remember(
        &self,
        params: Parameters<RememberParams>,
    ) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let group_id = p.group_id.as_deref().unwrap_or("default");
        let source = p.source.as_deref().unwrap_or("mcp-remember");
        debug!(source, group_id, "remember");

        let name = format!("memory-{}", uuid::Uuid::new_v4().as_simple());

        let result = match self
            .graphiti
            .add_episode(&name, &p.content, EpisodeType::Text, group_id, source)
            .await
        {
            Ok(r) => r,
            Err(e) => return graphiti_err(e),
        };

        ok(format!(
            "Knowledge stored successfully.\n\
             Episode ID: {}\n\
             Entities extracted: {}\n\
             Relationships extracted: {}",
            result.episode.uuid,
            result.nodes.len(),
            result.edges.len(),
        ))
    }

    #[tool(description = "Check cumulative LLM token consumption for this server session.")]
    async fn get_token_usage(
        &self,
        _params: Parameters<GetTokenUsageParams>,
    ) -> Result<CallToolResult, McpError> {
        debug!("get_token_usage");
        let usage = self.graphiti.token_usage();
        ok(format!(
            "Token usage: prompt_tokens={}, completion_tokens={}, total_tokens={}",
            usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
        ))
    }

    #[tool(
        name = "poll_llm_tasks",
        description = "Poll for pending LLM tasks that need processing. Returns tasks with system_prompt, user_prompt, and response_schema. Lower Decks should call this periodically to pick up work."
    )]
    async fn poll_llm_tasks(
        &self,
        params: Parameters<PollLlmTasksParams>,
    ) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let queue = match &self.graphiti.task_queue {
            Some(q) => q,
            None => return ok("[]"),
        };
        let tasks = queue.poll_pending(p.group_id.as_deref(), p.limit.unwrap_or(10).min(50));
        let json = serde_json::to_string_pretty(&tasks)
            .map_err(|e| McpError::internal_error(format!("serialize error: {e}"), None))?;
        ok(json)
    }

    #[tool(
        name = "claim_llm_task",
        description = "Claim a pending LLM task for processing. Returns true if successfully claimed, false if already taken or not found. Must be called before submit_llm_result."
    )]
    async fn claim_llm_task(
        &self,
        params: Parameters<ClaimLlmTaskParams>,
    ) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let queue = match &self.graphiti.task_queue {
            Some(q) => q,
            None => return ok("false"),
        };
        let id: Uuid = p
            .task_id
            .parse()
            .map_err(|_| McpError::internal_error("invalid task_id UUID", None))?;
        let claimed = queue.claim(&id);
        ok(claimed.to_string())
    }

    #[tool(
        name = "submit_llm_result",
        description = "Submit the result for a previously claimed LLM task. The result_json must be valid JSON matching the task's response_schema. This wakes up the waiting pipeline."
    )]
    async fn submit_llm_result(
        &self,
        params: Parameters<SubmitLlmResultParams>,
    ) -> Result<CallToolResult, McpError> {
        let p = params.0;
        let queue = match &self.graphiti.task_queue {
            Some(q) => q,
            None => return tool_err("task queue not enabled"),
        };
        let id: Uuid = p
            .task_id
            .parse()
            .map_err(|_| McpError::internal_error("invalid task_id UUID", None))?;
        // Validate JSON
        serde_json::from_str::<serde_json::Value>(&p.result_json)
            .map_err(|e| McpError::internal_error(format!("invalid JSON: {e}"), None))?;
        queue.complete(&id, p.result_json);
        ok("true")
    }
}

// ── ServerHandler ─────────────────────────────────────────────────────────────

#[tool_handler]
impl ServerHandler for GraphitiMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: Default::default(),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: "graphiti-mcp-server".into(),
                version: env!("CARGO_PKG_VERSION").into(),
                title: None,
                description: Some(
                    "MCP server for Graphiti knowledge graph operations".into(),
                ),
                icons: None,
                website_url: None,
            },
            instructions: Some(
                "Graphiti knowledge graph MCP server. \
                Use 'contextualize' for natural-language knowledge retrieval (primary tool). \
                Use 'remember' to store new knowledge (entities/relationships extracted automatically). \
                Use 'record_outcome' to report whether knowledge helped (feedback loop). \
                Use 'get_timeline' for chronological entity history. \
                Use 'search' for raw hybrid BM25+vector search. \
                Use 'get_entity' to look up specific entities. \
                Use 'add_episode' for structured content ingestion. \
                Use 'list_episodes' to see recent ingestion history. \
                Use 'build_communities' to cluster related entities. \
                Use 'get_token_usage' to monitor LLM token consumption. \
                Use 'poll_llm_tasks' to poll for pending LLM tasks that need processing. \
                Use 'claim_llm_task' to claim a task before processing it. \
                Use 'submit_llm_result' to submit the LLM result for a claimed task."
                    .into(),
            ),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{
        AddEpisodeParams, BuildCommunitiesParams, ClaimLlmTaskParams, ContextualizeParams,
        GetEntityParams, GetTimelineParams, GetTokenUsageParams, GraphitiMcpServer,
        ListEpisodesParams, PollLlmTasksParams, RecordOutcomeParams, RememberParams, SearchParams,
        SubmitLlmResultParams,
    };
    use crate::graphiti::Graphiti;
    use crate::testutils::{MockDriver, MockEmbedder, MockLlmClient};
    use crate::types::GraphitiConfig;
    use rmcp::handler::server::wrapper::Parameters;
    use std::sync::Arc;

    // ── Helpers ────────────────────────────────────────────────────────────────

    fn make_graphiti() -> Arc<Graphiti> {
        Arc::new(Graphiti::from_clients(
            Arc::new(MockDriver),
            Arc::new(MockEmbedder::new()),
            Arc::new(MockLlmClient),
            GraphitiConfig::default(),
        ))
    }

    fn make_graphiti_with_llm(llm: Arc<dyn crate::llm_client::LlmClient>) -> Arc<Graphiti> {
        Arc::new(Graphiti::from_clients(
            Arc::new(MockDriver),
            Arc::new(MockEmbedder::new()),
            llm,
            GraphitiConfig::default(),
        ))
    }

    fn make_server() -> GraphitiMcpServer {
        GraphitiMcpServer::new(make_graphiti())
    }

    fn make_server_with_queue() -> (GraphitiMcpServer, Arc<crate::llm_client::task_queue::LlmTaskQueue>) {
        let queue = Arc::new(crate::llm_client::task_queue::LlmTaskQueue::new());
        let mut graphiti = Graphiti::from_clients(
            Arc::new(MockDriver),
            Arc::new(MockEmbedder::new()),
            Arc::new(MockLlmClient),
            GraphitiConfig::default(),
        );
        graphiti.task_queue = Some(Arc::clone(&queue));
        let server = GraphitiMcpServer::new(Arc::new(graphiti));
        (server, queue)
    }

    // ── Sync construction tests ────────────────────────────────────────────────

    #[test]
    fn graphiti_mcp_server_new_constructs_instance() {
        let _server = GraphitiMcpServer::new(make_graphiti());
    }

    #[test]
    fn graphiti_mcp_server_is_clone() {
        let server = GraphitiMcpServer::new(make_graphiti());
        let _cloned = server.clone();
    }

    /// Compile-time assertion: `GraphitiMcpServer` must be `Send + Sync` for
    /// use as a tower `Service` across async tasks.
    #[allow(dead_code)]
    fn assert_mcp_server_is_send_sync()
    where
        GraphitiMcpServer: Send + Sync,
    {
    }

    #[test]
    fn search_limit_is_clamped_to_max() {
        // Values above MAX_SEARCH_LIMIT must be silently clamped.
        let huge: usize = 999_999;
        let clamped = huge.min(super::MAX_SEARCH_LIMIT);
        assert_eq!(clamped, super::MAX_SEARCH_LIMIT);

        // Values at or below the max pass through unchanged.
        let small: usize = 5;
        assert_eq!(small.min(super::MAX_SEARCH_LIMIT), 5);

        // Default (10) is well within the cap.
        let default_limit: usize = 10;
        assert_eq!(default_limit.min(super::MAX_SEARCH_LIMIT), 10);
    }

    #[test]
    fn list_episodes_limit_is_clamped_to_max() {
        let huge: usize = 999_999;
        let clamped = huge.min(super::MAX_EPISODES_LIMIT);
        assert_eq!(clamped, super::MAX_EPISODES_LIMIT);

        let small: usize = 5;
        assert_eq!(small.min(super::MAX_EPISODES_LIMIT), 5);

        // Default (20) is well within the cap.
        let default_limit: usize = 20;
        assert_eq!(default_limit.min(super::MAX_EPISODES_LIMIT), 20);
    }

    // ── Async handler tests: add_episode ──────────────────────────────────────

    /// Calling `add_episode` with a valid "text" source type must ingest the
    /// episode and return a success result (`is_error = false`).
    ///
    /// # Failure (TDD)
    ///
    /// Currently FAILS because the pipeline calls `MockLlmClient::generate_structured`,
    /// which panics with "unimplemented". A queue-backed LLM mock is needed.
    #[tokio::test]
    async fn add_episode_text_source_type_returns_success() {
        let server = make_server();
        let result = server
            .add_episode(Parameters(AddEpisodeParams {
                name: "test-episode".to_string(),
                content: "Alice is a software engineer.".to_string(),
                source_type: Some("text".to_string()),
                group_id: Some("grp1".to_string()),
            }))
            .await
            .expect("add_episode handler must not return McpError");

        assert_eq!(
            result.is_error,
            Some(false),
            "successful ingestion must set is_error=false"
        );
        let text = result
            .content
            .first()
            .and_then(|c| c.as_text())
            .expect("result must contain at least one text content item");
        assert!(
            text.text.contains("Episode ingested successfully"),
            "success message must contain confirmation; got: {text:?}"
        );
    }

    /// `add_episode` with "message" source type must also report success.
    ///
    /// # Failure (TDD)
    ///
    /// Same root cause as `add_episode_text_source_type_returns_success`.
    #[tokio::test]
    async fn add_episode_message_source_type_returns_success() {
        let server = make_server();
        let result = server
            .add_episode(Parameters(AddEpisodeParams {
                name: "chat-ep".to_string(),
                content: "User: hi\nAssistant: hello".to_string(),
                source_type: Some("message".to_string()),
                group_id: None,
            }))
            .await
            .expect("add_episode handler must not return McpError");

        assert_eq!(result.is_error, Some(false));
    }

    /// An unrecognised `source_type` must produce a tool-level error without
    /// touching the ingestion pipeline.
    #[tokio::test]
    async fn add_episode_unknown_source_type_returns_tool_error() {
        let server = make_server();
        let result = server
            .add_episode(Parameters(AddEpisodeParams {
                name: "ep".to_string(),
                content: "some content".to_string(),
                source_type: Some("csv".to_string()),
                group_id: None,
            }))
            .await
            .expect("handler must not propagate McpError");

        assert_eq!(
            result.is_error,
            Some(true),
            "unknown source_type must produce is_error=true"
        );
        let text = result
            .content
            .first()
            .and_then(|c| c.as_text())
            .expect("error result must contain text");
        assert!(
            text.text.contains("unknown source_type"),
            "error message must name the bad source_type; got: {text:?}"
        );
    }

    /// When `group_id` is absent the handler must default to "default" and
    /// still complete successfully.
    ///
    /// # Failure (TDD)
    ///
    /// Same root cause as other `add_episode` success tests.
    #[tokio::test]
    async fn add_episode_missing_group_id_defaults_to_default() {
        let server = make_server();
        let result = server
            .add_episode(Parameters(AddEpisodeParams {
                name: "ep".to_string(),
                content: "some content".to_string(),
                source_type: None,
                group_id: None,
            }))
            .await
            .expect("handler must not return McpError");

        assert_eq!(result.is_error, Some(false));
    }

    // ── Async handler tests: search ───────────────────────────────────────────

    /// When the driver returns no edges or nodes, `search` must return the
    /// human-readable sentinel "No results found." without an error flag.
    #[tokio::test]
    async fn search_empty_results_returns_no_results_found_message() {
        let server = make_server();
        let result = server
            .search(Parameters(SearchParams {
                query: "Who is Alice?".to_string(),
                limit: None,
                group_id: Some("default".to_string()),
            }))
            .await
            .expect("search handler must not return McpError");

        assert_eq!(
            result.is_error,
            Some(false),
            "empty search must succeed (not error)"
        );
        let text = result
            .content
            .first()
            .and_then(|c| c.as_text())
            .expect("search result must contain text");
        assert_eq!(
            text.text, "No results found.",
            "empty search must return exactly 'No results found.'"
        );
    }

    /// When `group_id` is absent the handler must default to "default" and
    /// still complete without error.
    #[tokio::test]
    async fn search_missing_group_id_defaults_to_default() {
        let server = make_server();
        let result = server
            .search(Parameters(SearchParams {
                query: "test query".to_string(),
                limit: None,
                group_id: None,
            }))
            .await
            .expect("search handler must not return McpError");

        assert_eq!(result.is_error, Some(false));
    }

    /// When `limit` is absent the handler must apply the default (10) and
    /// complete without error.
    #[tokio::test]
    async fn search_missing_limit_uses_default() {
        let server = make_server();
        let result = server
            .search(Parameters(SearchParams {
                query: "anything".to_string(),
                limit: None,
                group_id: Some("g".to_string()),
            }))
            .await
            .expect("search handler must not return McpError");

        assert_eq!(result.is_error, Some(false));
    }

    // ── Async handler tests: get_entity ───────────────────────────────────────

    /// Providing neither `name` nor `uuid` must produce a tool-level error.
    #[tokio::test]
    async fn get_entity_neither_name_nor_uuid_returns_tool_error() {
        let server = make_server();
        let result = server
            .get_entity(Parameters(GetEntityParams {
                name: None,
                uuid: None,
                group_id: None,
            }))
            .await
            .expect("handler must not return McpError");

        assert_eq!(
            result.is_error,
            Some(true),
            "missing both name and uuid must be a tool error"
        );
        let text = result
            .content
            .first()
            .and_then(|c| c.as_text())
            .expect("error must contain text");
        assert!(
            text.text.contains("name") || text.text.contains("uuid"),
            "error must hint at the missing fields; got: {text:?}"
        );
    }

    /// A syntactically invalid UUID string must produce a tool-level error.
    #[tokio::test]
    async fn get_entity_invalid_uuid_string_returns_tool_error() {
        let server = make_server();
        let result = server
            .get_entity(Parameters(GetEntityParams {
                name: None,
                uuid: Some("not-a-real-uuid".to_string()),
                group_id: None,
            }))
            .await
            .expect("handler must not return McpError");

        assert_eq!(
            result.is_error,
            Some(true),
            "invalid UUID string must be a tool error"
        );
        let text = result
            .content
            .first()
            .and_then(|c| c.as_text())
            .expect("error must contain text");
        assert!(
            text.text.contains("invalid UUID"),
            "error must mention invalid UUID; got: {text:?}"
        );
    }

    /// A well-formed UUID that does not exist in the graph must return an
    /// informational "not found" success (`is_error = false`).
    #[tokio::test]
    async fn get_entity_valid_uuid_not_found_returns_ok_message() {
        let server = make_server();
        let result = server
            .get_entity(Parameters(GetEntityParams {
                name: None,
                uuid: Some("550e8400-e29b-41d4-a716-446655440000".to_string()),
                group_id: None,
            }))
            .await
            .expect("handler must not return McpError");

        // MockDriver::get_entity_node always returns Ok(None).
        assert_eq!(
            result.is_error,
            Some(false),
            "not-found UUID must be a success, not an error"
        );
        let text = result
            .content
            .first()
            .and_then(|c| c.as_text())
            .expect("result must contain text");
        assert!(
            text.text.contains("No entity found"),
            "text must report entity not found; got: {text:?}"
        );
    }

    /// A name that matches nothing must return an informational "not found"
    /// success (`is_error = false`).
    #[tokio::test]
    async fn get_entity_name_not_found_returns_ok_message() {
        let server = make_server();
        let result = server
            .get_entity(Parameters(GetEntityParams {
                name: Some("NonExistentEntity".to_string()),
                uuid: None,
                group_id: None,
            }))
            .await
            .expect("handler must not return McpError");

        // MockDriver::search_entity_nodes_by_name returns an empty Vec.
        assert_eq!(
            result.is_error,
            Some(false),
            "not-found name must be a success, not an error"
        );
        let text = result
            .content
            .first()
            .and_then(|c| c.as_text())
            .expect("result must contain text");
        assert!(
            text.text.contains("No entity found"),
            "text must report entity not found; got: {text:?}"
        );
    }

    // ── Async handler tests: list_episodes ────────────────────────────────────

    /// When the driver holds no episodes `list_episodes` must return the
    /// sentinel "No episodes found." without an error flag.
    #[tokio::test]
    async fn list_episodes_empty_returns_no_episodes_found_message() {
        let server = make_server();
        let result = server
            .list_episodes(Parameters(ListEpisodesParams {
                group_id: Some("default".to_string()),
                limit: None,
            }))
            .await
            .expect("list_episodes handler must not return McpError");

        assert_eq!(result.is_error, Some(false));
        let text = result
            .content
            .first()
            .and_then(|c| c.as_text())
            .expect("result must contain text");
        assert_eq!(
            text.text, "No episodes found.",
            "empty episode list must return exactly 'No episodes found.'"
        );
    }

    /// `list_episodes` must succeed when `group_id` is absent.
    #[tokio::test]
    async fn list_episodes_missing_group_id_defaults_to_default() {
        let server = make_server();
        let result = server
            .list_episodes(Parameters(ListEpisodesParams {
                group_id: None,
                limit: None,
            }))
            .await
            .expect("list_episodes handler must not return McpError");

        assert_eq!(result.is_error, Some(false));
    }

    // ── Async handler tests: build_communities ────────────────────────────────

    /// `build_communities` must report the community count (0 for the mock
    /// driver) without an error flag.
    #[tokio::test]
    async fn build_communities_returns_zero_communities_message() {
        let server = make_server();
        let result = server
            .build_communities(Parameters(BuildCommunitiesParams {
                group_id: Some("default".to_string()),
            }))
            .await
            .expect("build_communities handler must not return McpError");

        assert_eq!(result.is_error, Some(false));
        let text = result
            .content
            .first()
            .and_then(|c| c.as_text())
            .expect("result must contain text");
        assert!(
            text.text.contains("Community detection complete"),
            "text must confirm detection ran; got: {text:?}"
        );
        assert!(
            text.text.contains('0'),
            "text must report 0 communities; got: {text:?}"
        );
    }

    /// `build_communities` must succeed when `group_id` is absent.
    #[tokio::test]
    async fn build_communities_missing_group_id_defaults_to_default() {
        let server = make_server();
        let result = server
            .build_communities(Parameters(BuildCommunitiesParams { group_id: None }))
            .await
            .expect("build_communities handler must not return McpError");

        assert_eq!(result.is_error, Some(false));
    }

    // ── Async handler tests: get_token_usage ──────────────────────────────────

    /// `get_token_usage` must return zeroed counters and `is_error = false`
    /// when no LLM calls have been recorded.
    #[tokio::test]
    async fn get_token_usage_returns_zeroed_counter_message() {
        let server = make_server();
        let result = server
            .get_token_usage(Parameters(GetTokenUsageParams {}))
            .await
            .expect("get_token_usage handler must not return McpError");

        assert_eq!(result.is_error, Some(false));
        let text = result
            .content
            .first()
            .and_then(|c| c.as_text())
            .expect("result must contain text");
        assert!(
            text.text.contains("Token usage"),
            "text must contain 'Token usage'; got: {text:?}"
        );
        assert!(
            text.text.contains("prompt_tokens=0"),
            "text must contain zeroed prompt_tokens; got: {text:?}"
        );
        assert!(
            text.text.contains("completion_tokens=0"),
            "text must contain zeroed completion_tokens; got: {text:?}"
        );
        assert!(
            text.text.contains("total_tokens=0"),
            "text must contain zeroed total_tokens; got: {text:?}"
        );
    }

    /// `get_token_usage` must delegate to `graphiti.token_usage()` and return
    /// real non-zero counts when the underlying LLM client has recorded tokens.
    #[tokio::test]
    async fn get_token_usage_returns_real_counts_from_llm_client() {
        use crate::llm_client::token_tracker::TokenTracker;
        use crate::llm_client::{LlmClient, Message, TokenUsage};
        use std::sync::Arc;

        struct TrackingMock {
            tracker: TokenTracker,
        }

        #[async_trait::async_trait]
        impl LlmClient for TrackingMock {
            async fn generate(&self, _: &[Message]) -> crate::errors::Result<String> {
                Ok("mock".to_string())
            }
            async fn generate_structured_json(
                &self,
                _: &[Message],
                _schema: serde_json::Value,
            ) -> crate::errors::Result<String> {
                unimplemented!()
            }
            fn token_usage(&self) -> TokenUsage {
                self.tracker.snapshot()
            }
            fn reset_token_usage(&self) {
                self.tracker.reset();
            }
        }

        let mock = TrackingMock {
            tracker: TokenTracker::new(),
        };
        mock.tracker.record(200, 75);
        let llm: Arc<dyn LlmClient> = Arc::new(mock);

        let graphiti = make_graphiti_with_llm(llm);
        let server = GraphitiMcpServer::new(graphiti);

        let result = server
            .get_token_usage(Parameters(GetTokenUsageParams {}))
            .await
            .expect("get_token_usage handler must not return McpError");

        assert_eq!(result.is_error, Some(false));
        let text = result
            .content
            .first()
            .and_then(|c| c.as_text())
            .expect("result must contain text");
        assert!(
            text.text.contains("prompt_tokens=200"),
            "text must reflect real prompt count; got: {text:?}"
        );
        assert!(
            text.text.contains("completion_tokens=75"),
            "text must reflect real completion count; got: {text:?}"
        );
        assert!(
            text.text.contains("total_tokens=275"),
            "text must reflect real total count; got: {text:?}"
        );
    }

    // ── Async handler tests: contextualize ───────────────────────────────────

    #[tokio::test]
    async fn contextualize_empty_returns_no_knowledge_message() {
        let server = make_server();
        let result = server
            .contextualize(Parameters(ContextualizeParams {
                query: "error handling in Rust".to_string(),
                limit: None,
                group_id: Some("default".to_string()),
            }))
            .await
            .expect("contextualize handler must not return McpError");

        assert_eq!(result.is_error, Some(false));
        let text = result.content.first().and_then(|c| c.as_text()).expect("result must contain text");
        assert!(
            text.text.contains("No relevant knowledge found"),
            "empty contextualize must return sentinel; got: {text:?}"
        );
    }

    #[tokio::test]
    async fn contextualize_missing_group_id_defaults_to_default() {
        let server = make_server();
        let result = server
            .contextualize(Parameters(ContextualizeParams {
                query: "test".to_string(),
                limit: None,
                group_id: None,
            }))
            .await
            .expect("contextualize handler must not return McpError");

        assert_eq!(result.is_error, Some(false));
    }

    // ── Async handler tests: record_outcome ──────────────────────────────────

    #[tokio::test]
    async fn record_outcome_success_returns_confirmation() {
        let server = make_server();
        let result = server
            .record_outcome(Parameters(RecordOutcomeParams {
                task_id: "T-042".to_string(),
                agent: "coding-agent".to_string(),
                entity_names: vec!["circuit_breaker".to_string()],
                success: true,
                details: "Applied circuit breaker, all tests pass".to_string(),
                group_id: Some("default".to_string()),
            }))
            .await
            .expect("record_outcome handler must not return McpError");

        assert_eq!(result.is_error, Some(false));
        let text = result.content.first().and_then(|c| c.as_text()).expect("result must contain text");
        assert!(text.text.contains("SUCCESS"), "success outcome must say SUCCESS; got: {text:?}");
        assert!(text.text.contains("T-042"), "must include task_id; got: {text:?}");
    }

    #[tokio::test]
    async fn record_outcome_failure_returns_confirmation() {
        let server = make_server();
        let result = server
            .record_outcome(Parameters(RecordOutcomeParams {
                task_id: "T-043".to_string(),
                agent: "debug-agent".to_string(),
                entity_names: vec![],
                success: false,
                details: "Timeout despite retry".to_string(),
                group_id: None,
            }))
            .await
            .expect("record_outcome handler must not return McpError");

        assert_eq!(result.is_error, Some(false));
        let text = result.content.first().and_then(|c| c.as_text()).expect("result must contain text");
        assert!(text.text.contains("FAILURE"), "failure outcome must say FAILURE; got: {text:?}");
    }

    // ── Async handler tests: get_timeline ────────────────────────────────────

    #[tokio::test]
    async fn get_timeline_empty_returns_no_entries_message() {
        let server = make_server();
        let result = server
            .get_timeline(Parameters(GetTimelineParams {
                entity_name: "NonExistent".to_string(),
                group_id: Some("default".to_string()),
                limit: None,
            }))
            .await
            .expect("get_timeline handler must not return McpError");

        assert_eq!(result.is_error, Some(false));
        let text = result.content.first().and_then(|c| c.as_text()).expect("result must contain text");
        assert!(
            text.text.contains("No timeline entries found"),
            "empty timeline must return sentinel; got: {text:?}"
        );
    }

    #[tokio::test]
    async fn get_timeline_missing_group_id_defaults_to_default() {
        let server = make_server();
        let result = server
            .get_timeline(Parameters(GetTimelineParams {
                entity_name: "Alice".to_string(),
                group_id: None,
                limit: None,
            }))
            .await
            .expect("get_timeline handler must not return McpError");

        assert_eq!(result.is_error, Some(false));
    }

    // ── Async handler tests: remember ────────────────────────────────────────

    #[tokio::test]
    async fn remember_stores_knowledge_and_returns_success() {
        let server = make_server();
        let result = server
            .remember(Parameters(RememberParams {
                content: "Circuit breaker uses 3-retry with exponential backoff".to_string(),
                source: Some("T-042/coding-agent".to_string()),
                group_id: Some("lower_decks".to_string()),
            }))
            .await
            .expect("remember handler must not return McpError");

        assert_eq!(result.is_error, Some(false));
        let text = result.content.first().and_then(|c| c.as_text()).expect("result must contain text");
        assert!(
            text.text.contains("Knowledge stored successfully"),
            "remember must confirm storage; got: {text:?}"
        );
    }

    #[tokio::test]
    async fn remember_defaults_group_id_and_source() {
        let server = make_server();
        let result = server
            .remember(Parameters(RememberParams {
                content: "Some knowledge".to_string(),
                source: None,
                group_id: None,
            }))
            .await
            .expect("remember handler must not return McpError");

        assert_eq!(result.is_error, Some(false));
    }

    // ── Async handler tests: poll_llm_tasks ─────────────────────────────────

    #[tokio::test]
    async fn poll_llm_tasks_no_queue_returns_empty_array() {
        let server = make_server();
        let result = server
            .poll_llm_tasks(Parameters(PollLlmTasksParams {
                group_id: None,
                limit: None,
            }))
            .await
            .expect("poll_llm_tasks handler must not return McpError");

        assert_eq!(result.is_error, Some(false));
        let text = result
            .content
            .first()
            .and_then(|c| c.as_text())
            .expect("result must contain text");
        assert_eq!(text.text, "[]");
    }

    #[tokio::test]
    async fn poll_llm_tasks_with_queue_returns_pending_tasks() {
        use crate::llm_client::task_queue::LlmTaskType;

        let (server, queue) = make_server_with_queue();
        queue.submit(
            LlmTaskType::ExtractEntities,
            "system".into(),
            "user".into(),
            serde_json::json!({}),
            "grp".into(),
        );
        let result = server
            .poll_llm_tasks(Parameters(PollLlmTasksParams {
                group_id: None,
                limit: None,
            }))
            .await
            .expect("poll_llm_tasks handler must not return McpError");

        assert_eq!(result.is_error, Some(false));
        let text = result
            .content
            .first()
            .and_then(|c| c.as_text())
            .expect("result must contain text");
        let tasks: Vec<serde_json::Value> =
            serde_json::from_str(&text.text).expect("result must be valid JSON array");
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0]["task_type"], "extract_entities");
    }

    // ── Async handler tests: claim_llm_task ─────────────────────────────────

    #[tokio::test]
    async fn claim_llm_task_no_queue_returns_false() {
        let server = make_server();
        let result = server
            .claim_llm_task(Parameters(ClaimLlmTaskParams {
                task_id: "00000000-0000-0000-0000-000000000000".to_string(),
            }))
            .await
            .expect("claim_llm_task handler must not return McpError");

        assert_eq!(result.is_error, Some(false));
        let text = result
            .content
            .first()
            .and_then(|c| c.as_text())
            .expect("result must contain text");
        assert_eq!(text.text, "false");
    }

    #[tokio::test]
    async fn claim_llm_task_invalid_uuid_returns_error() {
        let (server, _queue) = make_server_with_queue();
        let result = server
            .claim_llm_task(Parameters(ClaimLlmTaskParams {
                task_id: "not-a-uuid".to_string(),
            }))
            .await;

        assert!(result.is_err(), "invalid UUID must return McpError");
    }

    // ── Async handler tests: submit_llm_result ──────────────────────────────

    #[tokio::test]
    async fn submit_llm_result_no_queue_returns_tool_error() {
        let server = make_server();
        let result = server
            .submit_llm_result(Parameters(SubmitLlmResultParams {
                task_id: "00000000-0000-0000-0000-000000000000".to_string(),
                result_json: "{}".to_string(),
            }))
            .await
            .expect("submit_llm_result handler must not return McpError");

        assert_eq!(
            result.is_error,
            Some(true),
            "submit without queue must be tool error"
        );
        let text = result
            .content
            .first()
            .and_then(|c| c.as_text())
            .expect("result must contain text");
        assert!(
            text.text.contains("task queue not enabled"),
            "error message must mention queue; got: {text:?}"
        );
    }

    #[tokio::test]
    async fn submit_llm_result_invalid_json_returns_error() {
        use crate::llm_client::task_queue::LlmTaskType;

        let (server, queue) = make_server_with_queue();
        let id = queue.submit(
            LlmTaskType::ExtractEntities,
            "system".into(),
            "user".into(),
            serde_json::json!({}),
            "grp".into(),
        );
        queue.claim(&id);
        let result = server
            .submit_llm_result(Parameters(SubmitLlmResultParams {
                task_id: id.to_string(),
                result_json: "not valid json{{{".to_string(),
            }))
            .await;

        assert!(result.is_err(), "invalid JSON must return McpError");
    }

    #[tokio::test]
    async fn poll_claim_submit_round_trip() {
        use crate::llm_client::task_queue::LlmTaskType;

        let (server, queue) = make_server_with_queue();
        let id = queue.submit(
            LlmTaskType::Summarize,
            "You are a summarizer.".into(),
            "Summarize this.".into(),
            serde_json::json!({"type": "object"}),
            "test-group".into(),
        );

        // 1. Poll — should see 1 task
        let poll = server
            .poll_llm_tasks(Parameters(PollLlmTasksParams {
                group_id: Some("test-group".to_string()),
                limit: None,
            }))
            .await
            .expect("poll must succeed");
        let text = poll
            .content
            .first()
            .and_then(|c| c.as_text())
            .expect("poll result text");
        let tasks: Vec<serde_json::Value> = serde_json::from_str(&text.text).expect("valid JSON");
        assert_eq!(tasks.len(), 1);

        // 2. Claim
        let claim = server
            .claim_llm_task(Parameters(ClaimLlmTaskParams {
                task_id: id.to_string(),
            }))
            .await
            .expect("claim must succeed");
        let claim_text = claim
            .content
            .first()
            .and_then(|c| c.as_text())
            .expect("claim result text");
        assert_eq!(claim_text.text, "true");

        // 3. Poll again — should be empty (claimed tasks are InProgress)
        let poll2 = server
            .poll_llm_tasks(Parameters(PollLlmTasksParams {
                group_id: Some("test-group".to_string()),
                limit: None,
            }))
            .await
            .expect("poll2 must succeed");
        let text2 = poll2
            .content
            .first()
            .and_then(|c| c.as_text())
            .expect("poll2 result text");
        let tasks2: Vec<serde_json::Value> =
            serde_json::from_str(&text2.text).expect("valid JSON");
        assert_eq!(tasks2.len(), 0);

        // 4. Submit result
        let submit = server
            .submit_llm_result(Parameters(SubmitLlmResultParams {
                task_id: id.to_string(),
                result_json: r#"{"summary": "done"}"#.to_string(),
            }))
            .await
            .expect("submit must succeed");
        let submit_text = submit
            .content
            .first()
            .and_then(|c| c.as_text())
            .expect("submit result text");
        assert_eq!(submit_text.text, "true");
    }
}
