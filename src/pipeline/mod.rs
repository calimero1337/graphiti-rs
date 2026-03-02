//! Ingestion pipeline.
//!
//! Orchestrates the full write path: episode text → LLM extraction →
//! entity deduplication → edge deduplication → contradiction resolution →
//! graph persistence.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use futures::future::try_join_all;
use serde::Serialize;
use tokio::sync::Semaphore;
use uuid::Uuid;

use crate::driver::GraphDriver;
use crate::edges::{EntityEdge, EpisodicEdge};
use crate::embedder::EmbedderClient;
use crate::errors::{GraphitiError, Result};
use crate::llm_client::LlmClient;
use crate::nodes::episodic::EpisodeType;
use crate::nodes::{EntityNode, EpisodicNode};
use crate::prompts::dedupe_edges::{
    self, DedupeEdgesContext, EdgeDuplicateResult, ExistingEdgeStub as ExistingEdgeStubDedupe,
};
use crate::prompts::dedupe_nodes::{
    self, DedupeNodesContext, ExistingEntityStub, ExtractedNodeStub, NodeResolutions,
};
use crate::prompts::extract_edges::{self, ExtractedEdge, ExtractedEdges, ExtractEdgesContext};
use crate::prompts::extract_nodes::{self, ExtractedEntities, ExtractedEntity, ExtractNodesContext};
use crate::prompts::resolve_contradictions::{self, ContradictionResult, ResolveContradictionsContext};
use crate::types::IngestionConfig;

// ── Edge resolution outcome ───────────────────────────────────────────────────

/// The resolved outcome of edge deduplication for a single extracted edge.
enum EdgeResolution {
    /// A brand-new edge that was persisted.
    New(EntityEdge),
    /// An existing edge was identified as a duplicate; its UUID is returned.
    Duplicate(Uuid),
    /// The new fact contradicts an existing one; the old edge was invalidated.
    Contradiction {
        new_edge: EntityEdge,
        #[allow(dead_code)]
        invalidated_uuid: Uuid,
    },
}

// ── Public types ──────────────────────────────────────────────────────────────

/// Result returned by [`Pipeline::add_episode`].
#[derive(Debug, Clone, Serialize)]
pub struct AddEpisodeResult {
    /// The episode node that was saved.
    pub episode: EpisodicNode,
    /// All entity nodes resolved (existing or newly created) in this episode.
    pub nodes: Vec<EntityNode>,
    /// Newly-created entity edges (not counting episodes-list updates on existing edges).
    pub edges: Vec<EntityEdge>,
}

/// The ingestion pipeline that processes episodes into the knowledge graph.
pub struct Pipeline {
    driver: Arc<dyn GraphDriver>,
    llm: Arc<dyn LlmClient>,
    embedder: Arc<dyn EmbedderClient>,
    semaphore: Arc<Semaphore>,
    config: IngestionConfig,
}

impl Pipeline {
    /// Create a new pipeline with the given driver, LLM client, embedder, and configuration.
    pub fn new(
        driver: Arc<dyn GraphDriver>,
        llm: Arc<dyn LlmClient>,
        embedder: Arc<dyn EmbedderClient>,
        config: IngestionConfig,
    ) -> Self {
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent_llm_calls));
        Self { driver, llm, embedder, semaphore, config }
    }

    /// Process a single episode through the full pipeline.
    ///
    /// Steps:
    /// 1. Create and persist an [`EpisodicNode`].
    /// 2. Extract entities via LLM.
    /// 3. Deduplicate entities (parallel, semaphore-bounded).
    /// 4. Extract edges via LLM.
    /// 5. Deduplicate edges and resolve contradictions (sequential).
    /// 6. Create MENTIONS episodic edges.
    /// 7. Update the episode's `entity_edges` list and re-save.
    pub async fn add_episode(
        &self,
        name: &str,
        content: &str,
        source_type: EpisodeType,
        group_id: &str,
        source_description: &str,
    ) -> Result<AddEpisodeResult> {
        let now = Utc::now();

        // ── Step 1: Create and save EpisodicNode ─────────────────────────────
        let mut episode = EpisodicNode {
            uuid: Uuid::new_v4(),
            name: name.to_string(),
            group_id: group_id.to_string(),
            labels: vec!["EpisodicNode".to_string()],
            created_at: now,
            source: source_type.clone(),
            source_description: source_description.to_string(),
            content: content.to_string(),
            valid_at: now,
            entity_edges: vec![],
        };
        self.driver.save_episodic_node(&episode).await?;

        // ── Fetch previous episodes for co-reference context ──────────────────
        let previous_summaries: Vec<String> = if self.config.previous_episode_count > 0 {
            let mut recent = self.driver.list_episodic_nodes(group_id).await?;
            // Exclude the episode we just saved to avoid self-reference.
            recent.retain(|ep| ep.uuid != episode.uuid);
            recent.sort_by(|a, b| b.valid_at.cmp(&a.valid_at));
            recent.truncate(self.config.previous_episode_count);
            recent.into_iter().map(|ep| ep.content).collect()
        } else {
            vec![]
        };

        // ── Step 2: Extract entities ─────────────────────────────────────────
        let extract_ctx = ExtractNodesContext {
            episode_content: content,
            source_description,
            episode_type: &source_type,
            previous_episodes: &previous_summaries,
        };
        let extract_messages = extract_nodes::build_messages(&extract_ctx);
        let extracted_entities = {
            let _permit = self
                .semaphore
                .acquire()
                .await
                .map_err(|_| GraphitiError::Pipeline("semaphore closed".to_string()))?;
            crate::llm_client::generate_structured_via_dyn::<ExtractedEntities>(&*self.llm, &extract_messages).await?
        };

        if extracted_entities.entities.is_empty() {
            tracing::info!("No entities extracted from episode, returning early");
            return Ok(AddEpisodeResult { episode, nodes: vec![], edges: vec![] });
        }

        // Deduplicate extracted entity names (case-insensitive, keep first occurrence).
        let mut seen_names: HashSet<String> = HashSet::new();
        let unique_entities: Vec<ExtractedEntity> = extracted_entities
            .entities
            .into_iter()
            .filter(|e| seen_names.insert(e.name.to_lowercase()))
            .collect();

        tracing::info!(count = unique_entities.len(), "Entities extracted");

        // ── Step 3: Deduplicate entities (parallel, semaphore-bounded) ────────
        let dedup_futs: Vec<_> = unique_entities
            .into_iter()
            .map(|entity| {
                let driver = Arc::clone(&self.driver);
                let llm = Arc::clone(&self.llm);
                let embedder = Arc::clone(&self.embedder);
                let sem = Arc::clone(&self.semaphore);
                let limit = self.config.entity_search_limit;
                let gid = group_id.to_string();
                let ep_content = content.to_string();
                async move {
                    dedupe_entity(driver, llm, embedder, sem, entity, gid, ep_content, limit)
                        .await
                }
            })
            .collect();

        let resolved_nodes: Vec<EntityNode> = try_join_all(dedup_futs).await?;

        // Build lowercase-name → UUID map for edge resolution.
        let entity_map: HashMap<String, Uuid> = resolved_nodes
            .iter()
            .map(|n| (n.name.to_lowercase(), n.uuid))
            .collect();

        // ── Step 4: Extract edges ─────────────────────────────────────────────
        let entity_names: Vec<String> = resolved_nodes.iter().map(|n| n.name.clone()).collect();
        let edge_ctx = ExtractEdgesContext {
            episode_content: content,
            entities: &entity_names,
            reference_time: now,
            previous_episodes: &previous_summaries,
        };
        let edge_messages = extract_edges::build_messages(&edge_ctx);
        let extracted_edges = {
            let _permit = self
                .semaphore
                .acquire()
                .await
                .map_err(|_| GraphitiError::Pipeline("semaphore closed".to_string()))?;
            crate::llm_client::generate_structured_via_dyn::<ExtractedEdges>(&*self.llm, &edge_messages).await?
        };
        tracing::info!(count = extracted_edges.edges.len(), "Edges extracted");

        // ── Step 5: Deduplicate edges (sequential per source-target pair) ─────
        let mut new_edges: Vec<EntityEdge> = Vec::new();
        let mut all_edge_uuids: Vec<Uuid> = Vec::new();

        for extracted_edge in &extracted_edges.edges {
            let src_key = extracted_edge.source_node.to_lowercase();
            let tgt_key = extracted_edge.target_node.to_lowercase();

            let source_uuid = match entity_map.get(&src_key) {
                Some(u) => *u,
                None => {
                    tracing::warn!(
                        name = %extracted_edge.source_node,
                        "Source entity not resolved, skipping edge"
                    );
                    continue;
                }
            };
            let target_uuid = match entity_map.get(&tgt_key) {
                Some(u) => *u,
                None => {
                    tracing::warn!(
                        name = %extracted_edge.target_node,
                        "Target entity not resolved, skipping edge"
                    );
                    continue;
                }
            };

            if source_uuid == target_uuid {
                tracing::warn!(uuid = %source_uuid, "Self-referencing edge skipped");
                continue;
            }

            let resolution = dedupe_edge(
                &*self.driver,
                &*self.llm,
                &*self.embedder,
                &self.semaphore,
                extracted_edge,
                source_uuid,
                target_uuid,
                episode.uuid,
                group_id,
                now,
            )
            .await?;

            match resolution {
                EdgeResolution::New(edge) => {
                    all_edge_uuids.push(edge.uuid);
                    new_edges.push(edge);
                }
                EdgeResolution::Duplicate(existing_uuid) => {
                    all_edge_uuids.push(existing_uuid);
                }
                EdgeResolution::Contradiction { new_edge, invalidated_uuid: _ } => {
                    all_edge_uuids.push(new_edge.uuid);
                    new_edges.push(new_edge);
                }
            }
        }

        // ── Step 6: Create MENTIONS episodic edges ────────────────────────────
        for entity_node in &resolved_nodes {
            let ep_edge = EpisodicEdge {
                uuid: Uuid::new_v4(),
                source_node_uuid: episode.uuid,
                target_node_uuid: entity_node.uuid,
                created_at: now,
            };
            self.driver.save_episodic_edge(&ep_edge).await?;
        }

        // ── Step 7: Update episode entity_edges and re-save ───────────────────
        episode.entity_edges = all_edge_uuids.iter().map(|u| u.to_string()).collect();
        self.driver.save_episodic_node(&episode).await?;

        Ok(AddEpisodeResult { episode, nodes: resolved_nodes, edges: new_edges })
    }
}

// ── Private helper: entity deduplication ─────────────────────────────────────

/// Resolve a single extracted entity against the existing graph.
///
/// If a matching entity is found, returns the existing node. Otherwise creates,
/// embeds, and persists a new [`EntityNode`].
async fn dedupe_entity(
    driver: Arc<dyn GraphDriver>,
    llm: Arc<dyn LlmClient>,
    embedder: Arc<dyn EmbedderClient>,
    semaphore: Arc<Semaphore>,
    extracted: ExtractedEntity,
    group_id: String,
    episode_content: String,
    entity_search_limit: usize,
) -> Result<EntityNode> {
    // Search for existing candidates by name.
    let candidates = driver
        .search_entity_nodes_by_name(&extracted.name, &group_id, entity_search_limit)
        .await?;

    if !candidates.is_empty() {
        let extracted_stub =
            [ExtractedNodeStub { name: &extracted.name, summary: &extracted.summary }];
        let existing_stubs: Vec<ExistingEntityStub<'_>> = candidates
            .iter()
            .map(|c| ExistingEntityStub { uuid: c.uuid, name: &c.name, summary: &c.summary })
            .collect();

        let ctx = DedupeNodesContext {
            extracted_nodes: &extracted_stub,
            existing_nodes: &existing_stubs,
            episode_content: &episode_content,
        };
        let messages = dedupe_nodes::build_messages(&ctx);

        let resolutions: NodeResolutions = {
            let _permit = semaphore
                .acquire()
                .await
                .map_err(|_| GraphitiError::Pipeline("semaphore closed".to_string()))?;
            crate::llm_client::generate_structured_via_dyn::<NodeResolutions>(&*llm, &messages).await?
        };

        if let Some(resolution) = resolutions.resolutions.first() {
            if let Some(dup_uuid_str) = &resolution.duplicate_of {
                match Uuid::parse_str(dup_uuid_str) {
                    Ok(dup_uuid) => {
                        // Try the candidates list first (avoids an extra round-trip).
                        if let Some(existing) = candidates.iter().find(|c| c.uuid == dup_uuid) {
                            return Ok(existing.clone());
                        }
                        // Fall back to a direct driver lookup.
                        match driver.get_entity_node(&dup_uuid).await? {
                            Some(node) => return Ok(node),
                            None => {
                                tracing::warn!(
                                    uuid = %dup_uuid,
                                    "duplicate_of UUID not found in graph, creating new entity"
                                );
                            }
                        }
                    }
                    Err(_) => {
                        tracing::warn!(
                            value = %dup_uuid_str,
                            "LLM returned invalid UUID in duplicate_of, creating new entity"
                        );
                    }
                }
            }
        }
    }

    // Create, embed, and persist a new entity node.
    let embedding = embedder.embed(&extracted.name).await?;
    let node = EntityNode {
        uuid: Uuid::new_v4(),
        name: extracted.name,
        group_id,
        labels: vec!["EntityNode".to_string(), extracted.entity_type],
        summary: extracted.summary,
        name_embedding: Some(embedding),
        attributes: serde_json::Value::Object(Default::default()),
        created_at: Utc::now(),
    };
    driver.save_entity_node(&node).await?;
    Ok(node)
}

// ── Private helper: edge deduplication ───────────────────────────────────────

/// Resolve a single extracted edge against existing edges in the graph.
///
/// Returns an [`EdgeResolution`] describing whether the edge is new, a duplicate,
/// or contradicts an existing fact.
async fn dedupe_edge(
    driver: &dyn GraphDriver,
    llm: &dyn LlmClient,
    embedder: &dyn EmbedderClient,
    semaphore: &Semaphore,
    extracted: &ExtractedEdge,
    source_uuid: Uuid,
    target_uuid: Uuid,
    episode_uuid: Uuid,
    group_id: &str,
    reference_time: DateTime<Utc>,
) -> Result<EdgeResolution> {
    let existing = driver.get_entity_edges_between(&source_uuid, &target_uuid).await?;
    let valid_existing: Vec<&EntityEdge> =
        existing.iter().filter(|e| e.invalid_at.is_none()).collect();

    if valid_existing.is_empty() {
        let edge = build_new_edge(
            embedder,
            extracted,
            source_uuid,
            target_uuid,
            episode_uuid,
            group_id,
            reference_time,
        )
        .await?;
        driver.save_entity_edge(&edge).await?;
        return Ok(EdgeResolution::New(edge));
    }

    // Ask the LLM whether the new edge duplicates an existing one.
    let stubs: Vec<ExistingEdgeStubDedupe<'_>> = valid_existing
        .iter()
        .enumerate()
        .map(|(i, e)| ExistingEdgeStubDedupe { index: i, fact: &e.fact, relation_type: &e.name })
        .collect();

    let dedup_ctx = DedupeEdgesContext {
        new_edge_fact: &extracted.fact,
        new_edge_relation_type: &extracted.relation_type,
        existing_edges: &stubs,
    };
    let dedup_messages = dedupe_edges::build_messages(&dedup_ctx);

    let dedup_result: EdgeDuplicateResult = {
        let _permit = semaphore
            .acquire()
            .await
            .map_err(|_| GraphitiError::Pipeline("semaphore closed".to_string()))?;
        crate::llm_client::generate_structured_via_dyn::<EdgeDuplicateResult>(&*llm, &dedup_messages).await?
    };

    if let Some(idx) = dedup_result.duplicate_of_index {
        if let Some(existing_edge) = valid_existing.get(idx) {
            let mut updated = (*existing_edge).clone();
            if !updated.episodes.contains(&episode_uuid) {
                updated.episodes.push(episode_uuid);
            }
            driver.save_entity_edge(&updated).await?;
            return Ok(EdgeResolution::Duplicate(updated.uuid));
        }
        tracing::warn!(idx, "duplicate_of_index out of bounds, treating edge as new");
    }

    // Not a duplicate — check for contradictions with each valid existing edge.
    let ref_time_str = reference_time.format("%Y-%m-%dT%H:%M:%SZ").to_string();
    let mut invalidated_uuid: Option<Uuid> = None;

    for existing_edge in &valid_existing {
        let contra_ctx = ResolveContradictionsContext {
            new_fact: &extracted.fact,
            new_relation_type: &extracted.relation_type,
            existing_fact: &existing_edge.fact,
            existing_relation_type: &existing_edge.name,
            reference_time: &ref_time_str,
        };
        let contra_messages = resolve_contradictions::build_messages(&contra_ctx);

        let contra_result: ContradictionResult = {
            let _permit = semaphore
                .acquire()
                .await
                .map_err(|_| GraphitiError::Pipeline("semaphore closed".to_string()))?;
            crate::llm_client::generate_structured_via_dyn::<ContradictionResult>(&*llm, &contra_messages).await?
        };

        if contra_result.invalidates {
            driver.invalidate_edge(&existing_edge.uuid, reference_time).await?;
            invalidated_uuid = Some(existing_edge.uuid);
            break;
        }
    }

    let new_edge = build_new_edge(
        embedder,
        extracted,
        source_uuid,
        target_uuid,
        episode_uuid,
        group_id,
        reference_time,
    )
    .await?;
    driver.save_entity_edge(&new_edge).await?;

    if let Some(inv_uuid) = invalidated_uuid {
        Ok(EdgeResolution::Contradiction { new_edge, invalidated_uuid: inv_uuid })
    } else {
        Ok(EdgeResolution::New(new_edge))
    }
}

// ── Private helper: construct a new EntityEdge with fact embedding ────────────

async fn build_new_edge(
    embedder: &dyn EmbedderClient,
    extracted: &ExtractedEdge,
    source_uuid: Uuid,
    target_uuid: Uuid,
    episode_uuid: Uuid,
    group_id: &str,
    reference_time: DateTime<Utc>,
) -> Result<EntityEdge> {
    let valid_at = extracted.valid_at.as_deref().and_then(|s| {
        s.parse::<DateTime<Utc>>().map_err(|_| {
            tracing::warn!(datetime_str = s, "Could not parse valid_at datetime, using None");
        }).ok()
    });

    let fact_embedding = embedder.embed(&extracted.fact).await?;

    Ok(EntityEdge {
        uuid: Uuid::new_v4(),
        source_node_uuid: source_uuid,
        target_node_uuid: target_uuid,
        name: extracted.relation_type.clone(),
        fact: extracted.fact.clone(),
        fact_embedding: Some(fact_embedding),
        episodes: vec![episode_uuid],
        valid_at,
        invalid_at: None,
        created_at: reference_time,
        expired_at: None,
        weight: 1.0,
        attributes: serde_json::Value::Object(Default::default()),
        group_id: Some(group_id.to_string()),
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edges::CommunityEdge;
    use crate::llm_client::Message;
    use crate::nodes::CommunityNode;
    use crate::testutils::MockEmbedder;
    use dashmap::DashMap;
    use serde_json::json;
    use std::collections::VecDeque;
    use std::sync::Mutex;

    // ── Mock LLM client ───────────────────────────────────────────────────────

    struct MockLlmClient {
        responses: Mutex<VecDeque<serde_json::Value>>,
    }

    impl MockLlmClient {
        fn new(responses: Vec<serde_json::Value>) -> Self {
            Self { responses: Mutex::new(responses.into_iter().collect()) }
        }
    }

    #[async_trait::async_trait]
    impl LlmClient for MockLlmClient {
        async fn generate(&self, _messages: &[Message]) -> Result<String> {
            unimplemented!("MockLlmClient::generate not used in pipeline tests")
        }

        async fn generate_structured_json(
            &self,
            _messages: &[Message],
            _schema: serde_json::Value,
        ) -> Result<String> {
            let value = {
                let mut guard = self
                    .responses
                    .lock()
                    .map_err(|_| GraphitiError::Pipeline("mock mutex poisoned".to_string()))?;
                guard
                    .pop_front()
                    .ok_or_else(|| GraphitiError::Pipeline("no more mock responses".to_string()))?
            };
            serde_json::to_string(&value).map_err(Into::into)
        }
    }

    // ── Mock embedder ─────────────────────────────────────────────────────────

    struct MockEmbedderClient {
        dim: usize,
    }

    impl MockEmbedderClient {
        fn new(dim: usize) -> Self {
            Self { dim }
        }
    }

    #[async_trait::async_trait]
    impl EmbedderClient for MockEmbedderClient {
        async fn embed(&self, _text: &str) -> Result<Vec<f32>> {
            Ok(vec![0.0_f32; self.dim])
        }

        async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![0.0_f32; self.dim]).collect())
        }

        fn dim(&self) -> usize {
            self.dim
        }
    }

    // ── Mock graph driver ─────────────────────────────────────────────────────

    struct MockGraphDriver {
        entity_nodes: DashMap<Uuid, EntityNode>,
        episodic_nodes: DashMap<Uuid, EpisodicNode>,
        entity_edges: DashMap<Uuid, EntityEdge>,
        episodic_edges: DashMap<Uuid, EpisodicEdge>,
    }

    impl MockGraphDriver {
        fn new() -> Self {
            Self {
                entity_nodes: DashMap::new(),
                episodic_nodes: DashMap::new(),
                entity_edges: DashMap::new(),
                episodic_edges: DashMap::new(),
            }
        }
    }

    #[async_trait::async_trait]
    impl GraphDriver for MockGraphDriver {
        async fn ping(&self) -> Result<()> {
            Ok(())
        }
        async fn close(&self) -> Result<()> {
            Ok(())
        }
        async fn save_entity_node(&self, node: &EntityNode) -> Result<()> {
            self.entity_nodes.insert(node.uuid, node.clone());
            Ok(())
        }
        async fn get_entity_node(&self, uuid: &Uuid) -> Result<Option<EntityNode>> {
            Ok(self.entity_nodes.get(uuid).map(|r| r.clone()))
        }
        async fn delete_entity_node(&self, uuid: &Uuid) -> Result<()> {
            self.entity_nodes.remove(uuid);
            Ok(())
        }
        async fn save_episodic_node(&self, node: &EpisodicNode) -> Result<()> {
            self.episodic_nodes.insert(node.uuid, node.clone());
            Ok(())
        }
        async fn get_episodic_node(&self, uuid: &Uuid) -> Result<Option<EpisodicNode>> {
            Ok(self.episodic_nodes.get(uuid).map(|r| r.clone()))
        }
        async fn delete_episodic_node(&self, _uuid: &Uuid) -> Result<()> {
            Ok(())
        }
        async fn list_episodic_nodes(&self, group_id: &str) -> Result<Vec<EpisodicNode>> {
            Ok(self.episodic_nodes.iter()
                .filter(|r| r.group_id == group_id)
                .map(|r| r.clone())
                .collect())
        }
        async fn list_entity_nodes(&self, group_id: &str) -> Result<Vec<EntityNode>> {
            Ok(self.entity_nodes.iter()
                .filter(|r| r.group_id == group_id)
                .map(|r| r.clone())
                .collect())
        }
        async fn list_entity_edges(&self, group_id: &str) -> Result<Vec<EntityEdge>> {
            Ok(self.entity_edges.iter()
                .filter(|r| r.group_id.as_deref() == Some(group_id))
                .map(|r| r.clone())
                .collect())
        }
        async fn search_entity_nodes_by_embedding(
            &self,
            _embedding: &[f32],
            _group_id: &str,
            _limit: usize,
        ) -> Result<Vec<EntityNode>> {
            Ok(vec![])
        }
        async fn save_community_node(&self, _node: &CommunityNode) -> Result<()> {
            Ok(())
        }
        async fn save_entity_edge(&self, edge: &EntityEdge) -> Result<()> {
            self.entity_edges.insert(edge.uuid, edge.clone());
            Ok(())
        }
        async fn get_entity_edge(&self, uuid: &Uuid) -> Result<Option<EntityEdge>> {
            Ok(self.entity_edges.get(uuid).map(|r| r.clone()))
        }
        async fn save_episodic_edge(&self, edge: &EpisodicEdge) -> Result<()> {
            self.episodic_edges.insert(edge.uuid, edge.clone());
            Ok(())
        }
        async fn save_community_edge(&self, _edge: &CommunityEdge) -> Result<()> {
            Ok(())
        }
        async fn search_entity_nodes_by_name(
            &self,
            query: &str,
            group_id: &str,
            limit: usize,
        ) -> Result<Vec<EntityNode>> {
            let q = query.to_lowercase();
            let results: Vec<EntityNode> = self
                .entity_nodes
                .iter()
                .filter(|r| r.group_id == group_id && r.name.to_lowercase().contains(&q))
                .map(|r| r.clone())
                .take(limit)
                .collect();
            Ok(results)
        }
        async fn search_entity_edges_by_fact(
            &self,
            _embedding: &[f32],
            _group_id: &str,
            _limit: usize,
        ) -> Result<Vec<EntityEdge>> {
            Ok(vec![])
        }
        async fn bm25_search_edges(
            &self,
            _query: &str,
            _group_id: &str,
            _limit: usize,
        ) -> Result<Vec<EntityEdge>> {
            Ok(vec![])
        }
        async fn build_indices(&self) -> Result<()> {
            Ok(())
        }
        async fn get_entity_edges_between(
            &self,
            source: &Uuid,
            target: &Uuid,
        ) -> Result<Vec<EntityEdge>> {
            let results: Vec<EntityEdge> = self
                .entity_edges
                .iter()
                .filter(|r| {
                    (r.source_node_uuid == *source && r.target_node_uuid == *target)
                        || (r.source_node_uuid == *target && r.target_node_uuid == *source)
                })
                .map(|r| r.clone())
                .collect();
            Ok(results)
        }
        async fn invalidate_edge(&self, uuid: &Uuid, invalid_at: DateTime<Utc>) -> Result<()> {
            if let Some(mut edge) = self.entity_edges.get_mut(uuid) {
                edge.invalid_at = Some(invalid_at);
            }
            Ok(())
        }
    }

    // ── Test helpers ──────────────────────────────────────────────────────────

    fn make_pipeline(
        responses: Vec<serde_json::Value>,
    ) -> (Pipeline, Arc<MockGraphDriver>) {
        let driver = Arc::new(MockGraphDriver::new());
        let llm = Arc::new(MockLlmClient::new(responses));
        let embedder = Arc::new(MockEmbedderClient::new(4));
        let config = IngestionConfig::default();
        let pipeline = Pipeline::new(
            driver.clone() as Arc<dyn GraphDriver>,
            llm as Arc<dyn LlmClient>,
            embedder as Arc<dyn EmbedderClient>,
            config,
        );
        (pipeline, driver)
    }

    // ── Tests: entity extraction ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_extract_entities_success() {
        let responses = vec![
            // extract entities
            json!({ "entities": [
                { "name": "Alice", "entity_type": "Person", "summary": "A person named Alice" },
                { "name": "Acme Corp", "entity_type": "Organization", "summary": "A company" }
            ]}),
            // extract edges (empty)
            json!({ "edges": [] }),
        ];
        let (pipeline, driver) = make_pipeline(responses);
        let result = pipeline
            .add_episode("ep1", "Alice works at Acme Corp.", EpisodeType::Text, "grp", "test")
            .await
            .expect("add_episode should succeed");
        assert_eq!(result.nodes.len(), 2);
        assert_eq!(driver.entity_nodes.len(), 2);
    }

    #[tokio::test]
    async fn test_extract_entities_empty_returns_early() {
        let responses = vec![
            json!({ "entities": [] }),
        ];
        let (pipeline, driver) = make_pipeline(responses);
        let result = pipeline
            .add_episode("ep1", "No entities here.", EpisodeType::Text, "grp", "test")
            .await
            .expect("add_episode should succeed");
        assert!(result.nodes.is_empty());
        assert!(result.edges.is_empty());
        // Episode should still be saved.
        assert_eq!(driver.episodic_nodes.len(), 1);
    }

    #[tokio::test]
    async fn test_duplicate_extracted_names_deduped() {
        // LLM returns the same name twice — pipeline should only process it once.
        let responses = vec![
            json!({ "entities": [
                { "name": "Alice", "entity_type": "Person", "summary": "Alice" },
                { "name": "alice", "entity_type": "Person", "summary": "Same Alice" }
            ]}),
            json!({ "edges": [] }),
        ];
        let (pipeline, driver) = make_pipeline(responses);
        let result = pipeline
            .add_episode("ep1", "Alice, alice.", EpisodeType::Text, "grp", "test")
            .await
            .expect("add_episode should succeed");
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(driver.entity_nodes.len(), 1);
    }

    // ── Tests: entity deduplication ───────────────────────────────────────────

    #[tokio::test]
    async fn test_dedupe_entity_no_candidates() {
        // Fresh graph — no existing nodes, so a new node is created.
        let responses = vec![
            json!({ "entities": [
                { "name": "Bob", "entity_type": "Person", "summary": "Bob is a developer" }
            ]}),
            json!({ "edges": [] }),
        ];
        let (pipeline, driver) = make_pipeline(responses);
        let result = pipeline
            .add_episode("ep1", "Bob is a developer.", EpisodeType::Text, "grp", "test")
            .await
            .expect("add_episode should succeed");
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].name, "Bob");
        // New node must have an embedding.
        assert!(result.nodes[0].name_embedding.is_some());
        assert_eq!(driver.entity_nodes.len(), 1);
    }

    #[tokio::test]
    async fn test_dedupe_entity_match_found_reuses_existing() {
        // Pre-seed an existing entity.
        let existing_uuid = Uuid::new_v4();
        let existing = EntityNode {
            uuid: existing_uuid,
            name: "Bob".to_string(),
            group_id: "grp".to_string(),
            labels: vec!["EntityNode".to_string(), "Person".to_string()],
            summary: "Bob the developer".to_string(),
            name_embedding: Some(vec![1.0, 0.0, 0.0, 0.0]),
            attributes: serde_json::Value::Null,
            created_at: Utc::now(),
        };

        let responses = vec![
            json!({ "entities": [
                { "name": "Bob", "entity_type": "Person", "summary": "Bob" }
            ]}),
            // dedupe_nodes: says it's a duplicate of the existing node
            json!({ "resolutions": [
                { "extracted_name": "Bob", "duplicate_of": existing_uuid.to_string() }
            ]}),
            json!({ "edges": [] }),
        ];
        let (pipeline, driver) = make_pipeline(responses);
        // Pre-seed the driver.
        driver.entity_nodes.insert(existing_uuid, existing.clone());

        let result = pipeline
            .add_episode("ep1", "Bob fixed the bug.", EpisodeType::Text, "grp", "test")
            .await
            .expect("add_episode should succeed");

        // Should reuse the existing node, not create a new one.
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].uuid, existing_uuid);
        // Driver should still have exactly one entity node.
        assert_eq!(driver.entity_nodes.len(), 1);
    }

    #[tokio::test]
    async fn test_dedupe_entity_no_match_creates_new() {
        // Pre-seed an existing entity, but LLM says it's NOT a duplicate.
        let existing_uuid = Uuid::new_v4();
        let existing = EntityNode {
            uuid: existing_uuid,
            name: "Bobby Tables".to_string(),
            group_id: "grp".to_string(),
            labels: vec!["EntityNode".to_string()],
            summary: "A different person".to_string(),
            name_embedding: None,
            attributes: serde_json::Value::Null,
            created_at: Utc::now(),
        };

        let responses = vec![
            json!({ "entities": [
                { "name": "Bob", "entity_type": "Person", "summary": "Bob" }
            ]}),
            // dedupe_nodes: not a duplicate
            json!({ "resolutions": [
                { "extracted_name": "Bob", "duplicate_of": null }
            ]}),
            json!({ "edges": [] }),
        ];
        let (pipeline, driver) = make_pipeline(responses);
        driver.entity_nodes.insert(existing_uuid, existing);

        let result = pipeline
            .add_episode("ep1", "Bob is here.", EpisodeType::Text, "grp", "test")
            .await
            .expect("add_episode should succeed");

        // New entity created alongside the existing one.
        assert_eq!(result.nodes.len(), 1);
        assert_ne!(result.nodes[0].uuid, existing_uuid);
        assert_eq!(driver.entity_nodes.len(), 2);
    }

    // ── Tests: edge extraction ────────────────────────────────────────────────

    #[tokio::test]
    async fn test_extract_edges_success() {
        let responses = vec![
            json!({ "entities": [
                { "name": "Alice", "entity_type": "Person", "summary": "Alice" },
                { "name": "Acme Corp", "entity_type": "Organization", "summary": "Acme" }
            ]}),
            // extract edges
            json!({ "edges": [
                {
                    "source_node": "Alice",
                    "target_node": "Acme Corp",
                    "relation_type": "WORKS_AT",
                    "fact": "Alice works at Acme Corp.",
                    "valid_at": null
                }
            ]}),
        ];
        let (pipeline, _driver) = make_pipeline(responses);
        let result = pipeline
            .add_episode("ep1", "Alice works at Acme Corp.", EpisodeType::Text, "grp", "test")
            .await
            .expect("add_episode should succeed");
        assert_eq!(result.edges.len(), 1);
        assert_eq!(result.edges[0].name, "WORKS_AT");
    }

    // ── Tests: edge deduplication ─────────────────────────────────────────────

    #[tokio::test]
    async fn test_dedupe_edge_no_existing_creates_new() {
        let responses = vec![
            json!({ "entities": [
                { "name": "Alice", "entity_type": "Person", "summary": "Alice" },
                { "name": "Acme", "entity_type": "Organization", "summary": "Acme" }
            ]}),
            json!({ "edges": [
                {
                    "source_node": "Alice",
                    "target_node": "Acme",
                    "relation_type": "WORKS_AT",
                    "fact": "Alice works at Acme.",
                    "valid_at": null
                }
            ]}),
            // No existing edges → no dedup LLM call needed.
        ];
        let (pipeline, driver) = make_pipeline(responses);
        let result = pipeline
            .add_episode("ep1", "Alice works at Acme.", EpisodeType::Text, "grp", "test")
            .await
            .expect("add_episode should succeed");
        assert_eq!(result.edges.len(), 1);
        assert!(result.edges[0].fact_embedding.is_some());
        assert_eq!(driver.entity_edges.len(), 1);
    }

    #[tokio::test]
    async fn test_dedupe_edge_duplicate_extends_episodes() {
        // Pre-seed an existing edge.
        let alice_uuid = Uuid::new_v4();
        let acme_uuid = Uuid::new_v4();
        let existing_edge_uuid = Uuid::new_v4();
        let existing_edge = EntityEdge {
            uuid: existing_edge_uuid,
            source_node_uuid: alice_uuid,
            target_node_uuid: acme_uuid,
            name: "WORKS_AT".to_string(),
            fact: "Alice works at Acme.".to_string(),
            fact_embedding: Some(vec![0.0; 4]),
            episodes: vec![],
            valid_at: None,
            invalid_at: None,
            created_at: Utc::now(),
            expired_at: None,
            weight: 1.0,
            attributes: serde_json::Value::Null,
            group_id: Some("grp".to_string()),
        };
        let alice = EntityNode {
            uuid: alice_uuid,
            name: "Alice".to_string(),
            group_id: "grp".to_string(),
            labels: vec!["EntityNode".to_string()],
            summary: "Alice".to_string(),
            name_embedding: Some(vec![0.0; 4]),
            attributes: serde_json::Value::Null,
            created_at: Utc::now(),
        };
        let acme = EntityNode {
            uuid: acme_uuid,
            name: "Acme".to_string(),
            group_id: "grp".to_string(),
            labels: vec!["EntityNode".to_string()],
            summary: "Acme".to_string(),
            name_embedding: Some(vec![0.0; 4]),
            attributes: serde_json::Value::Null,
            created_at: Utc::now(),
        };

        let responses = vec![
            json!({ "entities": [
                { "name": "Alice", "entity_type": "Person", "summary": "Alice" },
                { "name": "Acme", "entity_type": "Organization", "summary": "Acme" }
            ]}),
            // Both are duplicates of existing nodes
            json!({ "resolutions": [
                { "extracted_name": "Alice", "duplicate_of": alice_uuid.to_string() }
            ]}),
            json!({ "resolutions": [
                { "extracted_name": "Acme", "duplicate_of": acme_uuid.to_string() }
            ]}),
            json!({ "edges": [
                {
                    "source_node": "Alice",
                    "target_node": "Acme",
                    "relation_type": "WORKS_AT",
                    "fact": "Alice is employed at Acme.",
                    "valid_at": null
                }
            ]}),
            // Edge dedup: it's a duplicate of index 0
            json!({ "duplicate_of_index": 0, "reason": "Same fact" }),
        ];
        let (pipeline, driver) = make_pipeline(responses);
        driver.entity_nodes.insert(alice_uuid, alice);
        driver.entity_nodes.insert(acme_uuid, acme);
        driver.entity_edges.insert(existing_edge_uuid, existing_edge);

        let result = pipeline
            .add_episode("ep2", "Alice is employed at Acme.", EpisodeType::Text, "grp", "test")
            .await
            .expect("add_episode should succeed");

        // No new edges — the existing edge was updated.
        assert!(result.edges.is_empty());
        // The existing edge should now have the episode in its list.
        let updated = driver.entity_edges.get(&existing_edge_uuid).unwrap();
        assert_eq!(updated.episodes.len(), 1);
    }

    #[tokio::test]
    async fn test_contradiction_detected_invalidates_old_edge() {
        let alice_uuid = Uuid::new_v4();
        let acme_uuid = Uuid::new_v4();
        let old_edge_uuid = Uuid::new_v4();
        let old_edge = EntityEdge {
            uuid: old_edge_uuid,
            source_node_uuid: alice_uuid,
            target_node_uuid: acme_uuid,
            name: "WORKS_AT".to_string(),
            fact: "Alice is an engineer at Acme.".to_string(),
            fact_embedding: Some(vec![0.0; 4]),
            episodes: vec![],
            valid_at: None,
            invalid_at: None,
            created_at: Utc::now(),
            expired_at: None,
            weight: 1.0,
            attributes: serde_json::Value::Null,
            group_id: Some("grp".to_string()),
        };
        let alice = EntityNode {
            uuid: alice_uuid,
            name: "Alice".to_string(),
            group_id: "grp".to_string(),
            labels: vec!["EntityNode".to_string()],
            summary: "Alice".to_string(),
            name_embedding: Some(vec![0.0; 4]),
            attributes: serde_json::Value::Null,
            created_at: Utc::now(),
        };
        let acme = EntityNode {
            uuid: acme_uuid,
            name: "Acme".to_string(),
            group_id: "grp".to_string(),
            labels: vec!["EntityNode".to_string()],
            summary: "Acme".to_string(),
            name_embedding: Some(vec![0.0; 4]),
            attributes: serde_json::Value::Null,
            created_at: Utc::now(),
        };

        let responses = vec![
            json!({ "entities": [
                { "name": "Alice", "entity_type": "Person", "summary": "Alice" },
                { "name": "Acme", "entity_type": "Organization", "summary": "Acme" }
            ]}),
            // Entity dedup: both are existing
            json!({ "resolutions": [{ "extracted_name": "Alice", "duplicate_of": alice_uuid.to_string() }] }),
            json!({ "resolutions": [{ "extracted_name": "Acme", "duplicate_of": acme_uuid.to_string() }] }),
            json!({ "edges": [
                {
                    "source_node": "Alice",
                    "target_node": "Acme",
                    "relation_type": "WORKS_AT",
                    "fact": "Alice is now VP at Acme.",
                    "valid_at": null
                }
            ]}),
            // Edge dedup: not a duplicate
            json!({ "duplicate_of_index": null, "reason": "Different role" }),
            // Contradiction check: new fact invalidates the old one
            json!({ "invalidates": true, "reason": "Job title changed" }),
        ];
        let (pipeline, driver) = make_pipeline(responses);
        driver.entity_nodes.insert(alice_uuid, alice);
        driver.entity_nodes.insert(acme_uuid, acme);
        driver.entity_edges.insert(old_edge_uuid, old_edge);

        let result = pipeline
            .add_episode("ep2", "Alice is now VP at Acme.", EpisodeType::Text, "grp", "test")
            .await
            .expect("add_episode should succeed");

        // One new edge was created.
        assert_eq!(result.edges.len(), 1);
        assert_eq!(result.edges[0].fact, "Alice is now VP at Acme.");
        // Old edge must be invalidated.
        let old = driver.entity_edges.get(&old_edge_uuid).unwrap();
        assert!(old.invalid_at.is_some(), "old edge should be invalidated");
    }

    #[tokio::test]
    async fn test_contradiction_not_detected_both_edges_remain() {
        let alice_uuid = Uuid::new_v4();
        let bob_uuid = Uuid::new_v4();
        let old_edge_uuid = Uuid::new_v4();
        let old_edge = EntityEdge {
            uuid: old_edge_uuid,
            source_node_uuid: alice_uuid,
            target_node_uuid: bob_uuid,
            name: "KNOWS".to_string(),
            fact: "Alice met Bob at a conference.".to_string(),
            fact_embedding: Some(vec![0.0; 4]),
            episodes: vec![],
            valid_at: None,
            invalid_at: None,
            created_at: Utc::now(),
            expired_at: None,
            weight: 1.0,
            attributes: serde_json::Value::Null,
            group_id: Some("grp".to_string()),
        };
        let alice = EntityNode {
            uuid: alice_uuid,
            name: "Alice".to_string(),
            group_id: "grp".to_string(),
            labels: vec!["EntityNode".to_string()],
            summary: "Alice".to_string(),
            name_embedding: Some(vec![0.0; 4]),
            attributes: serde_json::Value::Null,
            created_at: Utc::now(),
        };
        let bob = EntityNode {
            uuid: bob_uuid,
            name: "Bob".to_string(),
            group_id: "grp".to_string(),
            labels: vec!["EntityNode".to_string()],
            summary: "Bob".to_string(),
            name_embedding: Some(vec![0.0; 4]),
            attributes: serde_json::Value::Null,
            created_at: Utc::now(),
        };

        let responses = vec![
            json!({ "entities": [
                { "name": "Alice", "entity_type": "Person", "summary": "Alice" },
                { "name": "Bob", "entity_type": "Person", "summary": "Bob" }
            ]}),
            json!({ "resolutions": [{ "extracted_name": "Alice", "duplicate_of": alice_uuid.to_string() }] }),
            json!({ "resolutions": [{ "extracted_name": "Bob", "duplicate_of": bob_uuid.to_string() }] }),
            json!({ "edges": [
                {
                    "source_node": "Alice",
                    "target_node": "Bob",
                    "relation_type": "KNOWS",
                    "fact": "Alice and Bob collaborated on a project.",
                    "valid_at": null
                }
            ]}),
            // Edge dedup: not a duplicate
            json!({ "duplicate_of_index": null, "reason": "Different event" }),
            // Contradiction: both facts can coexist
            json!({ "invalidates": false, "reason": "Non-exclusive" }),
        ];
        let (pipeline, driver) = make_pipeline(responses);
        driver.entity_nodes.insert(alice_uuid, alice);
        driver.entity_nodes.insert(bob_uuid, bob);
        driver.entity_edges.insert(old_edge_uuid, old_edge);

        let result = pipeline
            .add_episode("ep2", "Alice and Bob collaborated.", EpisodeType::Text, "grp", "test")
            .await
            .expect("add_episode should succeed");

        // Both edges exist; old one not invalidated.
        assert_eq!(result.edges.len(), 1);
        let old = driver.entity_edges.get(&old_edge_uuid).unwrap();
        assert!(old.invalid_at.is_none(), "old edge should NOT be invalidated");
        assert_eq!(driver.entity_edges.len(), 2);
    }

    // ── Tests: episodic edges ─────────────────────────────────────────────────

    #[tokio::test]
    async fn test_episodic_edges_created_for_each_entity() {
        let responses = vec![
            json!({ "entities": [
                { "name": "Alice", "entity_type": "Person", "summary": "Alice" },
                { "name": "Bob", "entity_type": "Person", "summary": "Bob" }
            ]}),
            json!({ "edges": [] }),
        ];
        let (pipeline, driver) = make_pipeline(responses);
        let result = pipeline
            .add_episode("ep1", "Alice and Bob.", EpisodeType::Text, "grp", "test")
            .await
            .expect("add_episode should succeed");
        assert_eq!(result.nodes.len(), 2);
        // One MENTIONS edge per entity node.
        assert_eq!(driver.episodic_edges.len(), 2);
        // All MENTIONS edges point to the episode.
        for ep_edge in driver.episodic_edges.iter() {
            assert_eq!(ep_edge.source_node_uuid, result.episode.uuid);
        }
    }

    // ── Tests: edge skip conditions ───────────────────────────────────────────

    #[tokio::test]
    async fn test_unresolved_entity_name_skips_edge() {
        // Edge references "Unknown Corp" which is not in the entity list.
        let responses = vec![
            json!({ "entities": [
                { "name": "Alice", "entity_type": "Person", "summary": "Alice" }
            ]}),
            json!({ "edges": [
                {
                    "source_node": "Alice",
                    "target_node": "Unknown Corp",
                    "relation_type": "WORKS_AT",
                    "fact": "Alice works at Unknown Corp.",
                    "valid_at": null
                }
            ]}),
        ];
        let (pipeline, driver) = make_pipeline(responses);
        let result = pipeline
            .add_episode("ep1", "Alice works somewhere.", EpisodeType::Text, "grp", "test")
            .await
            .expect("add_episode should succeed");
        // Edge skipped because target not resolved.
        assert!(result.edges.is_empty());
        assert_eq!(driver.entity_edges.len(), 0);
    }

    #[tokio::test]
    async fn test_self_referencing_edge_skipped() {
        let responses = vec![
            json!({ "entities": [
                { "name": "Alice", "entity_type": "Person", "summary": "Alice" }
            ]}),
            // LLM creates an edge from Alice to Alice (self-reference).
            json!({ "edges": [
                {
                    "source_node": "Alice",
                    "target_node": "Alice",
                    "relation_type": "KNOWS",
                    "fact": "Alice knows herself.",
                    "valid_at": null
                }
            ]}),
        ];
        let (pipeline, driver) = make_pipeline(responses);
        let result = pipeline
            .add_episode("ep1", "Alice.", EpisodeType::Text, "grp", "test")
            .await
            .expect("add_episode should succeed");
        assert!(result.edges.is_empty());
        assert_eq!(driver.entity_edges.len(), 0);
    }

    // ── Test: pipeline construction ───────────────────────────────────────────

    #[tokio::test]
    async fn test_pipeline_construction() {
        let driver = Arc::new(MockGraphDriver::new());
        let llm = Arc::new(MockLlmClient::new(vec![]));
        let embedder = Arc::new(MockEmbedderClient::new(4));
        let config = IngestionConfig::default();
        let _pipeline = Pipeline::new(
            driver as Arc<dyn GraphDriver>,
            llm as Arc<dyn LlmClient>,
            embedder as Arc<dyn EmbedderClient>,
            config,
        );
    }

    // ── Test: full pipeline integration ───────────────────────────────────────

    #[tokio::test]
    async fn test_full_pipeline_integration() {
        let responses = vec![
            // extract entities: 2 entities
            json!({ "entities": [
                { "name": "Alice", "entity_type": "Person", "summary": "Alice the engineer" },
                { "name": "TechCorp", "entity_type": "Organization", "summary": "TechCorp" }
            ]}),
            // extract edges: 1 edge
            json!({ "edges": [
                {
                    "source_node": "Alice",
                    "target_node": "TechCorp",
                    "relation_type": "WORKS_AT",
                    "fact": "Alice works at TechCorp.",
                    "valid_at": null
                }
            ]}),
            // No existing edges → no dedup call needed.
        ];
        let (pipeline, driver) = make_pipeline(responses);

        let result = pipeline
            .add_episode(
                "Meeting notes",
                "Alice is an engineer at TechCorp.",
                EpisodeType::Text,
                "project-x",
                "Meeting transcript",
            )
            .await
            .expect("full pipeline should succeed");

        // Episode persisted.
        assert_eq!(driver.episodic_nodes.len(), 1);
        // 2 entity nodes.
        assert_eq!(result.nodes.len(), 2);
        assert_eq!(driver.entity_nodes.len(), 2);
        // 1 entity edge.
        assert_eq!(result.edges.len(), 1);
        assert_eq!(driver.entity_edges.len(), 1);
        // 2 MENTIONS episodic edges.
        assert_eq!(driver.episodic_edges.len(), 2);
        // Episode entity_edges updated.
        assert_eq!(result.episode.entity_edges.len(), 1);
        // Edge has an embedding.
        assert!(result.edges[0].fact_embedding.is_some());
        // All entity nodes have embeddings.
        for node in &result.nodes {
            assert!(node.name_embedding.is_some());
        }
    }
}
