//! Main Graphiti facade — top-level entry point for all graph operations.
//!
//! Wraps a [`crate::driver::GraphDriver`], [`crate::embedder::EmbedderClient`], and
//! [`crate::llm_client::LlmClient`] into a single ergonomic struct.
//!
//! # Constructors
//!
//! | Method | Description |
//! |--------|-------------|
//! | `Graphiti::new(config)` | Build everything from a [`crate::types::GraphitiConfig`] |
//! | `Graphiti::from_clients(driver, embedder, llm, config)` | Supply pre-built clients (ideal for testing) |

use std::sync::Arc;

use crate::community::CommunityBuilder;
use crate::driver::neo4j::Neo4jDriver;
use crate::driver::GraphDriver;
use crate::embedder::http::HttpEmbedder;
use crate::embedder::EmbedderClient;
use crate::errors::Result;
use crate::llm_client::anthropic::AnthropicClient;
use crate::llm_client::claude::ClaudeLlmClient;
use crate::llm_client::openai::{CacheConfig, OpenAiClient};
use crate::llm_client::{LlmClient, TokenUsage};
use crate::types::LlmBackend;
use crate::nodes::{CommunityNode, EntityNode, EpisodicNode};
use crate::pipeline::{AddEpisodeResult, Pipeline};
use crate::nodes::episodic::EpisodeType;
use crate::search::{SearchConfig, SearchEngine, SearchResults};
use crate::server::types::{
    ContextEntity, ContextRelationship, ContextualizeResponse, GetTimelineResponse,
    RecordOutcomeResponse, TimelineEntry,
};
use crate::types::GraphitiConfig;
use uuid::Uuid;

/// Top-level entry point for all Graphiti graph operations.
///
/// Owns shared references to the driver, LLM client, and embedder, and
/// delegates to the [`Pipeline`] and [`SearchEngine`] subsystems.
pub struct Graphiti {
    /// Shared graph database driver.
    pub driver: Arc<dyn GraphDriver>,
    /// Shared LLM client for entity/edge extraction.
    pub llm_client: Arc<dyn LlmClient>,
    /// Shared embedder for vectorising text.
    pub embedder: Arc<dyn EmbedderClient>,
    /// Configuration snapshot retained for informational access.
    pub config: GraphitiConfig,
    /// Ingestion pipeline (wraps driver + llm + embedder).
    pipeline: Pipeline,
    /// Hybrid search engine (wraps driver + embedder).
    search_engine: SearchEngine,
    /// Community detection and summarization (wraps driver + llm + embedder).
    community_builder: CommunityBuilder,
}

impl Graphiti {
    /// Construct a `Graphiti` instance from pre-built, shared clients.
    ///
    /// This is the synchronous, infallible constructor — ideal for tests and
    /// dependency-injection scenarios where the caller already holds the clients.
    pub fn from_clients(
        driver: Arc<dyn GraphDriver>,
        embedder: Arc<dyn EmbedderClient>,
        llm: Arc<dyn LlmClient>,
        config: GraphitiConfig,
    ) -> Self {
        let pipeline = Pipeline::new(
            Arc::clone(&driver),
            Arc::clone(&llm),
            Arc::clone(&embedder),
            config.ingestion.clone(),
        );
        let search_engine = SearchEngine::new(Arc::clone(&driver), Arc::clone(&embedder));
        let community_builder = CommunityBuilder {
            driver: Arc::clone(&driver),
            llm: Arc::clone(&llm),
            embedder: Arc::clone(&embedder),
        };
        Self {
            driver,
            llm_client: llm,
            embedder,
            config,
            pipeline,
            search_engine,
            community_builder,
        }
    }

    /// Connect to Neo4j and initialise all subsystems from `config`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::GraphitiError::Driver`] if the Neo4j connection fails.
    pub async fn new(config: GraphitiConfig) -> Result<Self> {
        let driver: Arc<dyn GraphDriver> = Arc::new(
            Neo4jDriver::new(&config.neo4j_uri, &config.neo4j_user, &config.neo4j_password)
                .await?,
        );

        let embedder: Arc<dyn EmbedderClient> = Arc::new(HttpEmbedder::new(
            &config.embedding_base_url,
            &config.embedding_model,
            config.embedding_dim,
        )?);

        let llm: Arc<dyn LlmClient> = match config.llm_backend {
            LlmBackend::Anthropic => Arc::new(AnthropicClient::new(
                &config.anthropic_api_key,
                &config.model_name,
                CacheConfig::default(),
            )),
            LlmBackend::Claude => Arc::new(ClaudeLlmClient::new(&config.model_name)),
            LlmBackend::OpenAI => Arc::new(OpenAiClient::new(
                &config.openai_api_key,
                &config.model_name,
                CacheConfig::default(),
            )),
        };

        Ok(Self::from_clients(driver, embedder, llm, config))
    }

    /// Create graph indexes and constraints.
    pub async fn build_indices(&self) -> Result<()> {
        self.driver.build_indices().await
    }

    /// Ingest an episode into the knowledge graph.
    pub async fn add_episode(
        &self,
        name: &str,
        content: &str,
        source_type: EpisodeType,
        group_id: &str,
        source_description: &str,
    ) -> Result<AddEpisodeResult> {
        self.pipeline
            .add_episode(name, content, source_type, group_id, source_description)
            .await
    }

    /// Search the knowledge graph with hybrid BM25 + vector search.
    pub async fn search(&self, query: &str, config: &SearchConfig) -> Result<SearchResults> {
        self.search_engine.search(query, config).await
    }

    /// Retrieve the most recent episodes for one or more groups, ordered by `valid_at` descending.
    ///
    /// Episodes from all requested groups are merged, deduplicated by UUID, sorted most-recent
    /// first, and then truncated to at most `limit` entries. Returns an empty `Vec` when
    /// `group_ids` is empty or `limit == 0`.
    pub async fn retrieve_episodes(
        &self,
        group_ids: &[&str],
        limit: usize,
    ) -> Result<Vec<EpisodicNode>> {
        if group_ids.is_empty() || limit == 0 {
            return Ok(vec![]);
        }

        // Fetch from all groups concurrently.
        let mut fetch_futures = Vec::with_capacity(group_ids.len());
        for &gid in group_ids {
            fetch_futures.push(self.driver.list_episodic_nodes(gid));
        }

        let mut merged: Vec<EpisodicNode> = Vec::new();
        let mut seen_uuids = std::collections::HashSet::new();
        for result in futures::future::join_all(fetch_futures).await {
            for ep in result? {
                if seen_uuids.insert(ep.uuid) {
                    merged.push(ep);
                }
            }
        }

        // Sort most-recent first, then cap to limit.
        merged.sort_by(|a, b| b.valid_at.cmp(&a.valid_at));
        merged.truncate(limit);
        Ok(merged)
    }

    /// Build or rebuild community structure for the given groups.
    ///
    /// Delegates to [`CommunityBuilder::build_communities`] which runs label
    /// propagation, LLM summarization, and persists the results.
    ///
    /// Returns the list of [`CommunityNode`]s that were created.
    pub async fn build_communities(&self, group_ids: &[&str]) -> Result<Vec<CommunityNode>> {
        let result = self.community_builder.build_communities(group_ids).await?;
        Ok(result.communities)
    }

    /// Retrieve a single [`EntityNode`] by its UUID.
    ///
    /// Returns `Ok(None)` when no entity with that UUID exists.
    pub async fn get_entity_by_uuid(&self, uuid: &Uuid) -> Result<Option<EntityNode>> {
        self.driver.get_entity_node(uuid).await
    }

    /// Search for [`EntityNode`]s whose name matches `query` within `group_id`.
    ///
    /// Returns at most `limit` results. Results are ordered by relevance (driver-defined).
    pub async fn search_entities_by_name(
        &self,
        query: &str,
        group_id: &str,
        limit: usize,
    ) -> Result<Vec<EntityNode>> {
        self.driver
            .search_entity_nodes_by_name(query, group_id, limit)
            .await
    }

    /// Return a snapshot of the cumulative LLM token usage since creation or
    /// the last call to [`reset_token_usage`].
    pub fn token_usage(&self) -> TokenUsage {
        self.llm_client.token_usage()
    }

    /// Reset the cumulative LLM token counters to zero.
    pub fn reset_token_usage(&self) {
        self.llm_client.reset_token_usage();
    }

    /// Contextualize a natural-language query by performing hybrid search and
    /// expanding matched entities with their 1-hop relationships.
    ///
    /// Returns [`ContextualizeResponse`] containing matched entities, their
    /// relationships, and any advisory warnings.
    pub async fn contextualize(
        &self,
        query: &str,
        group_ids: &[String],
        limit: usize,
    ) -> Result<ContextualizeResponse> {
        let config = SearchConfig::default()
            .with_group_ids(group_ids.to_vec())
            .with_limit(limit);
        let results = self.search(query, &config).await?;

        let mut entities = Vec::new();

        // For each matched entity node, fetch 1-hop edges.
        for (node, _score) in &results.nodes {
            let edges = self
                .driver
                .list_entity_edges(&node.group_id)
                .await
                .unwrap_or_default();

            // Filter edges that involve this entity (by UUID as source or target).
            let rels: Vec<ContextRelationship> = edges
                .iter()
                .filter(|e| {
                    e.source_node_uuid == node.uuid || e.target_node_uuid == node.uuid
                })
                .take(10)
                .map(|e| ContextRelationship {
                    target: String::new(),
                    fact: e.fact.clone(),
                    valid: e.invalid_at.is_none(),
                })
                .collect();

            entities.push(ContextEntity {
                name: node.name.clone(),
                uuid: node.uuid.to_string(),
                summary: node.summary.clone(),
                relationships: rels,
            });
        }

        // Include edge facts from search results that are not already represented.
        for (edge, _score) in &results.edges {
            let already_present = entities.iter().any(|e| e.name == edge.fact);
            if !already_present && entities.len() < limit {
                entities.push(ContextEntity {
                    name: format!("fact-{}", edge.uuid),
                    uuid: edge.uuid.to_string(),
                    summary: edge.fact.clone(),
                    relationships: vec![],
                });
            }
        }

        Ok(ContextualizeResponse {
            entities,
            warnings: vec![],
        })
    }

    /// Record a task outcome as an episode in the knowledge graph.
    ///
    /// Creates an episode containing the task result details, enabling the
    /// feedback loop for agent learning.
    pub async fn record_outcome(
        &self,
        task_id: &str,
        agent: &str,
        entity_names: &[String],
        success: bool,
        details: &str,
        group_id: &str,
    ) -> Result<RecordOutcomeResponse> {
        let outcome_label = if success { "SUCCESS" } else { "FAILURE" };
        let entities_str = if entity_names.is_empty() {
            "none".to_string()
        } else {
            entity_names.join(", ")
        };
        let content = format!(
            "Task outcome: {outcome_label}\n\
             Task ID: {task_id}\n\
             Agent: {agent}\n\
             Entities: {entities_str}\n\
             Details: {details}"
        );
        let episode_name = format!("outcome-{task_id}");

        let result = self
            .add_episode(&episode_name, &content, EpisodeType::Text, group_id, "outcome")
            .await?;

        Ok(RecordOutcomeResponse {
            episode_id: result.episode.uuid.to_string(),
            recorded: true,
        })
    }

    /// Retrieve a chronological timeline of facts related to an entity.
    ///
    /// Searches for edges involving the named entity, sorts them by `valid_at`,
    /// and returns them as [`GetTimelineResponse`].
    pub async fn get_timeline(
        &self,
        entity_name: &str,
        group_id: &str,
        limit: Option<usize>,
    ) -> Result<GetTimelineResponse> {
        let mut edges = self.driver.list_entity_edges(group_id).await?;

        // Find the entity UUID by name search.
        let entity_nodes = self
            .driver
            .search_entity_nodes_by_name(entity_name, group_id, 1)
            .await?;

        let entries = if let Some(entity) = entity_nodes.first() {
            // Filter edges that involve this entity.
            edges.retain(|e| {
                e.source_node_uuid == entity.uuid || e.target_node_uuid == entity.uuid
            });

            // Sort by valid_at ascending (None last).
            edges.sort_by(|a, b| a.valid_at.cmp(&b.valid_at));

            let limit = limit.unwrap_or(50);
            edges
                .iter()
                .take(limit)
                .map(|e| TimelineEntry {
                    fact: e.fact.clone(),
                    valid_at: e.valid_at.map(|dt| dt.to_rfc3339()),
                    invalid_at: e.invalid_at.map(|dt| dt.to_rfc3339()),
                    still_valid: e.invalid_at.is_none(),
                })
                .collect()
        } else {
            vec![]
        };

        Ok(GetTimelineResponse {
            entity_name: entity_name.to_string(),
            entries,
        })
    }

    /// Gracefully shut down the Neo4j connection pool.
    pub async fn close(&self) -> Result<()> {
        self.driver.close().await
    }
}

// ── TDD tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::Graphiti;

    use crate::driver::GraphDriver;
    use crate::embedder::EmbedderClient;
    use crate::llm_client::{LlmClient, TokenUsage};
    use crate::nodes::{CommunityNode, EntityNode, EpisodeType, EpisodicNode};
    use crate::search::SearchConfig;
    use crate::testutils::{MockDriver, MockEmbedder, MockLlmClient};
    use crate::types::GraphitiConfig;
    use std::sync::Arc;
    use uuid::Uuid;

    // ── Helper ─────────────────────────────────────────────────────────────

    fn mock_config() -> GraphitiConfig {
        GraphitiConfig::default()
    }

    fn make_graphiti() -> Graphiti {
        Graphiti::from_clients(
            Arc::new(MockDriver),
            Arc::new(MockEmbedder::new()),
            Arc::new(MockLlmClient),
            mock_config(),
        )
    }

    // ── Tests ──────────────────────────────────────────────────────────────

    #[test]
    fn graphiti_from_clients_constructs_instance() {
        let _g = make_graphiti();
    }

    #[test]
    fn graphiti_exposes_config() {
        let mut config = mock_config();
        config.group_id = Some("team-x".to_string());
        config.embedding_dim = 3072;

        let g = Graphiti::from_clients(
            Arc::new(MockDriver),
            Arc::new(MockEmbedder::new()),
            Arc::new(MockLlmClient),
            config,
        );

        assert_eq!(g.config.group_id, Some("team-x".to_string()));
        assert_eq!(g.config.embedding_dim, 3072);
    }

    #[test]
    fn graphiti_exposes_driver() {
        let g = make_graphiti();
        let _d: &Arc<dyn GraphDriver> = &g.driver;
    }

    #[test]
    fn graphiti_exposes_embedder() {
        let g = make_graphiti();
        let _e: &Arc<dyn EmbedderClient> = &g.embedder;
    }

    #[test]
    fn graphiti_exposes_llm_client() {
        let g = make_graphiti();
        let _l: &Arc<dyn LlmClient> = &g.llm_client;
    }

    /// Compile-time assertion: `Graphiti` must implement `Send + Sync`.
    #[allow(dead_code)]
    fn assert_graphiti_is_send_sync()
    where
        Graphiti: Send + Sync,
    {
    }

    /// `Graphiti::new(config)` is async, performs real I/O, and returns `Result<Graphiti>`.
    ///
    /// Ignored in CI because it requires a live Neo4j instance + valid API keys.
    #[tokio::test]
    #[ignore = "requires live Neo4j at bolt://localhost:7687 and valid OPENAI_API_KEY"]
    async fn graphiti_new_accepts_config() {
        let config = mock_config();
        let _g: Graphiti = Graphiti::new(config)
            .await
            .expect("Graphiti::new should succeed against a running Neo4j");
    }

    /// `Graphiti::new` with an unreachable Neo4j URI must return `Err`.
    #[tokio::test]
    async fn graphiti_new_bad_uri_returns_err() {
        let mut config = mock_config();
        config.neo4j_uri = "bolt://127.0.0.1:19999".to_string();
        config.neo4j_password = "irrelevant".to_string();
        config.openai_api_key = "sk-test".to_string();

        let result = Graphiti::new(config).await;
        assert!(
            result.is_err(),
            "Graphiti::new with an unreachable URI must return Err"
        );
    }

    #[tokio::test]
    async fn graphiti_search_returns_results() {
        let g = make_graphiti();
        let config = SearchConfig::default().with_group_ids(vec!["default".to_string()]);
        let results = g.search("Alice", &config).await.expect("search should succeed");
        assert!(results.edges.is_empty());
        assert!(results.nodes.is_empty());
    }

    #[tokio::test]
    async fn graphiti_search_empty_group_ids_returns_err() {
        let g = make_graphiti();
        let config = SearchConfig::default();
        let result = g.search("Alice", &config).await;
        assert!(result.is_err(), "search with empty group_ids must return Err");
    }

    #[tokio::test]
    async fn graphiti_add_episode_returns_episode() {
        let g = make_graphiti();
        let result = g
            .add_episode(
                "episode-1",
                "Alice is a software engineer.",
                EpisodeType::Text,
                "default",
                "unit test",
            )
            .await
            .expect("add_episode should succeed with mock driver");

        assert_eq!(result.episode.name, "episode-1");
        assert_eq!(result.episode.content, "Alice is a software engineer.");
        assert_eq!(result.episode.group_id, "default");
    }

    #[tokio::test]
    async fn graphiti_add_episode_result_has_nodes_and_edges() {
        let g = make_graphiti();
        let result = g
            .add_episode(
                "ep",
                "Some content",
                EpisodeType::Message,
                "grp",
                "test",
            )
            .await
            .expect("add_episode should succeed");

        let _nodes: &Vec<crate::nodes::EntityNode> = &result.nodes;
        let _edges: &Vec<crate::edges::EntityEdge> = &result.edges;
    }

    #[tokio::test]
    async fn graphiti_build_communities_returns_vec() {
        let g = make_graphiti();
        let communities: Vec<CommunityNode> = g
            .build_communities(&["default"])
            .await
            .expect("build_communities should succeed with empty mock graph");

        assert!(communities.is_empty());
    }

    #[tokio::test]
    async fn graphiti_build_communities_empty_groups_returns_empty() {
        let g = make_graphiti();
        let communities = g
            .build_communities(&[])
            .await
            .expect("build_communities with empty groups must succeed");

        assert!(
            communities.is_empty(),
            "empty group_ids must produce no communities"
        );
    }

    #[tokio::test]
    async fn graphiti_retrieve_episodes_returns_vec() {
        let g = make_graphiti();
        let episodes: Vec<EpisodicNode> = g
            .retrieve_episodes(&["default"], 10)
            .await
            .expect("retrieve_episodes should succeed with empty mock graph");

        assert!(episodes.is_empty());
    }

    #[tokio::test]
    async fn graphiti_retrieve_episodes_zero_limit_is_empty() {
        let g = make_graphiti();
        let episodes = g
            .retrieve_episodes(&["default"], 0)
            .await
            .expect("retrieve_episodes(limit=0) must succeed");

        assert!(episodes.is_empty(), "limit=0 must return an empty Vec");
    }

    #[tokio::test]
    async fn graphiti_retrieve_episodes_empty_group_ids_is_empty() {
        let g = make_graphiti();
        let episodes = g
            .retrieve_episodes(&[], 10)
            .await
            .expect("retrieve_episodes(&[]) must succeed");

        assert!(episodes.is_empty(), "empty group_ids must return an empty Vec");
    }

    #[tokio::test]
    async fn graphiti_get_entity_by_uuid_returns_none_for_unknown() {
        let g = make_graphiti();
        let result = g
            .get_entity_by_uuid(&Uuid::new_v4())
            .await
            .expect("get_entity_by_uuid must succeed with mock driver");

        assert!(result.is_none(), "mock driver returns None for any UUID");
    }

    #[tokio::test]
    async fn graphiti_search_entities_by_name_returns_vec() {
        let g = make_graphiti();
        let entities: Vec<EntityNode> = g
            .search_entities_by_name("Alice", "default", 10)
            .await
            .expect("search_entities_by_name must succeed with mock driver");

        assert!(entities.is_empty(), "mock driver returns no entities");
    }

    // ── Contextualize tests ────────────────────────────────────────────────

    #[tokio::test]
    async fn graphiti_contextualize_returns_response() {
        let g = make_graphiti();
        let resp = g
            .contextualize("Alice", &["default".to_string()], 10)
            .await
            .expect("contextualize should succeed with mock driver");

        assert!(resp.entities.is_empty());
        assert!(resp.warnings.is_empty());
    }

    // ── Record Outcome tests ─────────────────────────────────────────────────

    /// `Graphiti::record_outcome` creates an episode and returns a response.
    #[tokio::test]
    async fn graphiti_record_outcome_returns_response() {
        let g = make_graphiti();
        let resp = g
            .record_outcome(
                "task-1",
                "Sisko",
                &["Alice".to_string()],
                true,
                "completed successfully",
                "default",
            )
            .await
            .expect("record_outcome should succeed with mock driver");

        assert!(resp.recorded);
        assert!(!resp.episode_id.is_empty());
    }

    // ── Timeline tests ───────────────────────────────────────────────────────

    /// `Graphiti::get_timeline` returns an empty timeline when no entity exists.
    #[tokio::test]
    async fn graphiti_get_timeline_returns_empty_for_unknown_entity() {
        let g = make_graphiti();

        let resp = g
            .get_timeline("NonExistent", "default", None)
            .await
            .expect("get_timeline should succeed with mock driver");

        assert_eq!(resp.entity_name, "NonExistent");
        assert!(resp.entries.is_empty());
    }

    /// `Graphiti::search_entities_by_name` with `limit = 0` must return an empty `Vec`.
    #[tokio::test]
    async fn graphiti_search_entities_by_name_zero_limit() {
        let g = make_graphiti();

        let entities = g
            .search_entities_by_name("anything", "default", 0)
            .await
            .expect("search_entities_by_name(limit=0) must succeed");

        assert!(entities.is_empty());
    }

    /// `Graphiti::token_usage` must return a `TokenUsage` value (default zeros for the mock).
    #[test]
    fn graphiti_token_usage_returns_usage() {
        let g = make_graphiti();

        let usage: TokenUsage = g.token_usage();
        assert_eq!(usage.prompt_tokens, 0);
        assert_eq!(usage.completion_tokens, 0);
        assert_eq!(usage.total_tokens, 0);
    }

    /// `Graphiti::reset_token_usage` must be callable without panicking.
    #[test]
    fn graphiti_reset_token_usage_is_callable() {
        let g = make_graphiti();
        g.reset_token_usage(); // must not panic
    }

    /// `Graphiti::token_usage` must delegate to the inner `LlmClient` and return
    /// non-zero values when the client has recorded token usage.
    #[test]
    fn graphiti_token_usage_delegates_to_llm_client() {
        use crate::testutils::TokenTrackingMockLlmClient;

        let g = Graphiti::from_clients(
            Arc::new(MockDriver),
            Arc::new(MockEmbedder::new()),
            Arc::new(TokenTrackingMockLlmClient::with_usage(42, 17)),
            mock_config(),
        );

        let usage: TokenUsage = g.token_usage();
        assert_eq!(usage.prompt_tokens, 42);
        assert_eq!(usage.completion_tokens, 17);
        assert_eq!(usage.total_tokens, 59);
    }

    /// `Graphiti::reset_token_usage` must zero out a previously-populated tracker.
    #[test]
    fn graphiti_reset_token_usage_clears_counts() {
        use crate::testutils::TokenTrackingMockLlmClient;

        let g = Graphiti::from_clients(
            Arc::new(MockDriver),
            Arc::new(MockEmbedder::new()),
            Arc::new(TokenTrackingMockLlmClient::with_usage(100, 50)),
            mock_config(),
        );

        // Sanity-check that the pre-seeded counts are visible.
        let before = g.token_usage();
        assert_eq!(before.prompt_tokens, 100);
        assert_eq!(before.completion_tokens, 50);

        g.reset_token_usage();

        let after = g.token_usage();
        assert_eq!(after.prompt_tokens, 0);
        assert_eq!(after.completion_tokens, 0);
        assert_eq!(after.total_tokens, 0);
    }
}
