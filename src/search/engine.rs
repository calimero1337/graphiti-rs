//! SearchEngine — orchestrates hybrid BM25 + vector search with RRF fusion.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::Utc;
use futures::future::join_all;
use uuid::Uuid;

use crate::cross_encoder::CrossEncoderClient;
use crate::driver::GraphDriver;
use crate::edges::EntityEdge;
use crate::embedder::EmbedderClient;
use crate::errors::{GraphitiError, Result};
use crate::nodes::EntityNode;
use crate::search::rrf::{reciprocal_rank_fusion, DEFAULT_K};
use crate::search::{SearchConfig, SearchResults};
use crate::utils::lucene_sanitize;

/// Orchestrates hybrid search over a graph database using BM25 + vector
/// similarity, fused via Reciprocal Rank Fusion (RRF).
///
/// Holds shared references to a [`GraphDriver`] (for database queries) and an
/// [`EmbedderClient`] (for query embedding).  All external I/O is async and
/// concurrent within each group search.
///
/// An optional [`CrossEncoderClient`] may be attached via [`SearchEngine::with_reranker`]
/// to enable post-RRF semantic reranking when [`SearchConfig::rerank`] is `true`.
pub struct SearchEngine {
    driver: Arc<dyn GraphDriver>,
    embedder: Arc<dyn EmbedderClient>,
    reranker: Option<Arc<dyn CrossEncoderClient>>,
}

impl SearchEngine {
    /// Create a new [`SearchEngine`] backed by `driver` and `embedder`.
    ///
    /// No reranker is attached by default.  Call [`SearchEngine::with_reranker`]
    /// to enable cross-encoder reranking.
    pub fn new(driver: Arc<dyn GraphDriver>, embedder: Arc<dyn EmbedderClient>) -> Self {
        Self {
            driver,
            embedder,
            reranker: None,
        }
    }

    /// Attach a cross-encoder reranker.
    ///
    /// When a reranker is attached, search results are re-scored by the
    /// cross-encoder after RRF fusion whenever [`SearchConfig::rerank`] is
    /// `true`.  The cross-encoder score replaces the RRF score and results
    /// are re-sorted before truncation.
    pub fn with_reranker(mut self, reranker: Arc<dyn CrossEncoderClient>) -> Self {
        self.reranker = Some(reranker);
        self
    }

    /// Perform hybrid search combining BM25 fulltext and vector similarity.
    ///
    /// # Pipeline
    ///
    /// 1. Validate `config.group_ids` is non-empty → [`GraphitiError::Validation`].
    /// 2. Embed `query` via the embedder.
    /// 3. Sanitize `query` with [`crate::utils::text::lucene_sanitize`] for BM25.
    /// 4. For each `group_id`, run edge search and node search concurrently.
    /// 5. All groups are searched concurrently via [`join_all`].
    /// 6. Fuse per-group results via RRF.
    /// 7. Merge across groups (dedup by UUID, keep highest score).
    /// 8. Sort descending by score, truncate to `config.limit`.
    ///
    /// # Errors
    ///
    /// - [`GraphitiError::Validation`] if `config.group_ids` is empty.
    /// - [`GraphitiError::Embedder`] if embedding fails.
    /// - [`GraphitiError::Driver`] if any driver search call fails.
    pub async fn search(&self, query: &str, config: &SearchConfig) -> Result<SearchResults> {
        if config.group_ids.is_empty() {
            return Err(GraphitiError::Validation(
                "group_ids must contain at least one entry".to_string(),
            ));
        }
        if config.bm25_weight < 0.0 {
            return Err(GraphitiError::Validation(
                "bm25_weight must be non-negative".to_string(),
            ));
        }
        if config.vector_weight < 0.0 {
            return Err(GraphitiError::Validation(
                "vector_weight must be non-negative".to_string(),
            ));
        }

        // Embed the query once; wrap in Arc so each group future gets a cheap
        // pointer clone instead of a full Vec<f32> heap copy.
        let embedding: Arc<Vec<f32>> = Arc::new(self.embedder.embed(query).await?);
        // Sanitise for BM25 / Lucene fulltext queries.
        let sanitised = lucene_sanitize(query);

        // Build one search future per group_id; all groups run concurrently.
        let group_futures = config.group_ids.iter().map(|group_id| {
            let driver = Arc::clone(&self.driver);
            // Arc::clone is O(1) — only increments a reference count, no heap copy.
            let embedding = Arc::clone(&embedding);
            let sanitised = sanitised.clone();
            let group_id = group_id.clone();
            let limit = config.limit;
            let search_edges = config.search_edges;
            let search_nodes = config.search_nodes;
            let bm25_weight = config.bm25_weight;
            let vector_weight = config.vector_weight;

            async move {
                let mut group_edges: HashMap<Uuid, (EntityEdge, f32)> = HashMap::new();
                let mut group_nodes: HashMap<Uuid, (EntityNode, f32)> = HashMap::new();

                // --- Edge search ---
                if search_edges {
                    let (bm25_edges, vec_edges) = tokio::join!(
                        driver.bm25_search_edges(&sanitised, &group_id, limit),
                        driver.search_entity_edges_by_fact(&*embedding, &group_id, limit),
                    );
                    let bm25_edges = bm25_edges?;
                    let vec_edges = vec_edges?;

                    // Build ranked UUID lists for RRF.
                    let bm25_ids: Vec<Uuid> = bm25_edges.iter().map(|e| e.uuid).collect();
                    let vec_ids: Vec<Uuid> = vec_edges.iter().map(|e| e.uuid).collect();

                    // Build a lookup map: uuid → edge.
                    let mut edge_map: HashMap<Uuid, EntityEdge> = HashMap::new();
                    for e in bm25_edges.into_iter().chain(vec_edges) {
                        edge_map.entry(e.uuid).or_insert(e);
                    }

                    let fused = reciprocal_rank_fusion(
                        &[bm25_ids, vec_ids],
                        &[bm25_weight, vector_weight],
                        DEFAULT_K,
                    );

                    // RRF guarantees each UUID appears at most once in `fused`,
                    // so `or_insert_with` always inserts into a `Vacant` entry and
                    // the `score > entry.1` branch is never reached within a single
                    // group.  The dedup pattern mirrors the cross-group merge below
                    // for consistency but is a no-op here.
                    for (uuid, score) in fused {
                        if let Some(edge) = edge_map.remove(&uuid) {
                            let entry = group_edges.entry(uuid).or_insert_with(|| (edge, score));
                            if score > entry.1 {
                                entry.1 = score;
                            }
                        }
                    }
                }

                // --- Node search ---
                if search_nodes {
                    let (text_nodes, vec_nodes) = tokio::join!(
                        driver.search_entity_nodes_by_name(&sanitised, &group_id, limit),
                        driver.search_entity_nodes_by_embedding(&*embedding, &group_id, limit),
                    );
                    let text_nodes = text_nodes?;
                    let vec_nodes = vec_nodes?;

                    let text_ids: Vec<Uuid> = text_nodes.iter().map(|n| n.uuid).collect();
                    let vec_ids: Vec<Uuid> = vec_nodes.iter().map(|n| n.uuid).collect();

                    let mut node_map: HashMap<Uuid, EntityNode> = HashMap::new();
                    for n in text_nodes.into_iter().chain(vec_nodes) {
                        node_map.entry(n.uuid).or_insert(n);
                    }

                    let fused = reciprocal_rank_fusion(
                        &[text_ids, vec_ids],
                        &[bm25_weight, vector_weight],
                        DEFAULT_K,
                    );

                    // Same reasoning as above: per-group dedup is a no-op because
                    // RRF produces each UUID at most once per group.
                    for (uuid, score) in fused {
                        if let Some(node) = node_map.remove(&uuid) {
                            let entry = group_nodes.entry(uuid).or_insert_with(|| (node, score));
                            if score > entry.1 {
                                entry.1 = score;
                            }
                        }
                    }
                }

                Ok::<_, GraphitiError>((group_edges, group_nodes))
            }
        });

        // Await all group searches concurrently.
        let group_results = join_all(group_futures).await;

        // Merge across groups: dedup by UUID, keeping the highest fused score.
        let mut all_edges: HashMap<Uuid, (EntityEdge, f32)> = HashMap::new();
        let mut all_nodes: HashMap<Uuid, (EntityNode, f32)> = HashMap::new();

        for result in group_results {
            let (group_edges, group_nodes) = result?;

            for (uuid, (edge, score)) in group_edges {
                match all_edges.entry(uuid) {
                    std::collections::hash_map::Entry::Occupied(mut occ) => {
                        if score > occ.get().1 {
                            occ.insert((edge, score));
                        }
                    }
                    std::collections::hash_map::Entry::Vacant(vac) => {
                        vac.insert((edge, score));
                    }
                }
            }

            for (uuid, (node, score)) in group_nodes {
                match all_nodes.entry(uuid) {
                    std::collections::hash_map::Entry::Occupied(mut occ) => {
                        if score > occ.get().1 {
                            occ.insert((node, score));
                        }
                    }
                    std::collections::hash_map::Entry::Vacant(vac) => {
                        vac.insert((node, score));
                    }
                }
            }
        }

        // Filter edges: remove expired (expired_at set) and invalid (invalid_at <= now).
        let now = Utc::now();
        let mut edges: Vec<(EntityEdge, f32)> = all_edges
            .into_values()
            .filter(|(e, _)| {
                e.expired_at.is_none()
                    && e.invalid_at.map_or(true, |t| t > now)
            })
            .collect();

        let mut nodes: Vec<(EntityNode, f32)> = all_nodes.into_values().collect();

        // Sort descending by fused score.
        edges.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        nodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Optional cross-encoder reranking step.
        //
        // When a reranker is attached and `config.rerank` is true, re-score
        // all candidates before the final truncation so the cross-encoder can
        // promote highly-relevant items that scored lower under RRF.
        if config.rerank {
            if let Some(reranker) = &self.reranker {
                if !edges.is_empty() {
                    let docs: Vec<&str> = edges.iter().map(|(e, _)| e.fact.as_str()).collect();
                    let scores = reranker.score(query, &docs).await?;
                    for ((_, score), new_score) in edges.iter_mut().zip(scores) {
                        *score = new_score;
                    }
                    edges.sort_by(|a, b| {
                        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                    });
                }

                if !nodes.is_empty() {
                    // Use `name: summary` as the document text for nodes, falling
                    // back to the name alone when the summary is empty.
                    let node_texts: Vec<String> = nodes
                        .iter()
                        .map(|(n, _)| {
                            if n.summary.is_empty() {
                                n.name.clone()
                            } else {
                                format!("{}: {}", n.name, n.summary)
                            }
                        })
                        .collect();
                    let docs: Vec<&str> = node_texts.iter().map(String::as_str).collect();
                    let scores = reranker.score(query, &docs).await?;
                    for ((_, score), new_score) in nodes.iter_mut().zip(scores) {
                        *score = new_score;
                    }
                    nodes.sort_by(|a, b| {
                        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
            }
        }

        // Truncate to configured limit.
        edges.truncate(config.limit);
        nodes.truncate(config.limit);

        Ok(SearchResults { edges, nodes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use chrono::{DateTime, Duration, Utc};
    use uuid::Uuid;

    use crate::cross_encoder::CrossEncoderClient;
    use crate::driver::GraphDriver;
    use crate::edges::{CommunityEdge, EntityEdge, EpisodicEdge};
    use crate::embedder::{EmbedderClient, Embedding};
    use crate::errors::{GraphitiError, Result};
    use crate::nodes::{CommunityNode, EntityNode, EpisodicNode};
    use crate::search::SearchConfig;

    // -------------------------------------------------------------------------
    // Helper constructors
    // -------------------------------------------------------------------------

    fn make_entity_node(name: &str, group_id: &str) -> EntityNode {
        EntityNode {
            uuid: Uuid::new_v4(),
            name: name.to_string(),
            group_id: group_id.to_string(),
            labels: vec![],
            summary: String::new(),
            name_embedding: None,
            attributes: serde_json::Value::Null,
            created_at: Utc::now(),
        }
    }

    fn make_entity_edge(
        fact: &str,
        group_id: &str,
        expired_at: Option<DateTime<Utc>>,
        invalid_at: Option<DateTime<Utc>>,
    ) -> EntityEdge {
        EntityEdge {
            uuid: Uuid::new_v4(),
            source_node_uuid: Uuid::new_v4(),
            target_node_uuid: Uuid::new_v4(),
            name: "RELATES".to_string(),
            fact: fact.to_string(),
            fact_embedding: None,
            episodes: vec![],
            valid_at: None,
            invalid_at,
            created_at: Utc::now(),
            expired_at,
            weight: 1.0,
            attributes: serde_json::Value::Null,
            group_id: Some(group_id.to_string()),
        }
    }

    // -------------------------------------------------------------------------
    // MockEmbedder — always returns a fixed embedding vector.
    // -------------------------------------------------------------------------

    struct MockEmbedder {
        embedding: Vec<f32>,
    }

    impl MockEmbedder {
        fn new() -> Self {
            Self {
                embedding: vec![0.1_f32; 8],
            }
        }
    }

    #[async_trait::async_trait]
    impl EmbedderClient for MockEmbedder {
        async fn embed(&self, _text: &str) -> Result<Embedding> {
            Ok(self.embedding.clone())
        }

        async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>> {
            Ok(texts.iter().map(|_| self.embedding.clone()).collect())
        }

        fn dim(&self) -> usize {
            self.embedding.len()
        }
    }

    // -------------------------------------------------------------------------
    // MockDriver — returns pre-configured results for all search methods.
    //
    // Non-search methods return trivial successes (Ok(()) / Ok(None)) since
    // they are never invoked during search tests.
    //
    // NOTE: when subtask 03 extends GraphDriver with
    // `search_entity_nodes_by_embedding`, MockDriver will need to implement it
    // (returning `self.node_vector_results.clone()`).  At that point this file
    // will fail to compile until the method is added here — that is the
    // expected TDD red signal.
    // -------------------------------------------------------------------------

    #[allow(dead_code)]
    struct MockDriver {
        /// Results for `search_entity_edges_by_fact` (vector similarity).
        edge_vector_results: Vec<EntityEdge>,
        /// Results for `bm25_search_edges` (BM25 fulltext).
        edge_bm25_results: Vec<EntityEdge>,
        /// Results for `search_entity_nodes_by_name` (text CONTAINS match).
        node_text_results: Vec<EntityNode>,
        /// Results for `search_entity_nodes_by_embedding` (vector similarity).
        /// Unused until subtask 03 adds the method to GraphDriver.
        node_vector_results: Vec<EntityNode>,
    }

    impl MockDriver {
        fn new() -> Self {
            Self {
                edge_vector_results: vec![],
                edge_bm25_results: vec![],
                node_text_results: vec![],
                node_vector_results: vec![],
            }
        }

        fn with_edge_results(
            mut self,
            bm25: Vec<EntityEdge>,
            vector: Vec<EntityEdge>,
        ) -> Self {
            self.edge_bm25_results = bm25;
            self.edge_vector_results = vector;
            self
        }

        fn with_node_results(
            mut self,
            text: Vec<EntityNode>,
            vector: Vec<EntityNode>,
        ) -> Self {
            self.node_text_results = text;
            self.node_vector_results = vector;
            self
        }
    }

    #[async_trait::async_trait]
    impl GraphDriver for MockDriver {
        async fn ping(&self) -> Result<()> {
            Ok(())
        }

        async fn close(&self) -> Result<()> {
            Ok(())
        }

        async fn save_entity_node(&self, _node: &EntityNode) -> Result<()> {
            Ok(())
        }

        async fn get_entity_node(&self, _uuid: &Uuid) -> Result<Option<EntityNode>> {
            Ok(None)
        }

        async fn delete_entity_node(&self, _uuid: &Uuid) -> Result<()> {
            Ok(())
        }

        async fn save_episodic_node(&self, _node: &EpisodicNode) -> Result<()> {
            Ok(())
        }

        async fn get_episodic_node(&self, _uuid: &Uuid) -> Result<Option<EpisodicNode>> {
            Ok(None)
        }

        async fn delete_episodic_node(&self, _uuid: &Uuid) -> Result<()> {
            Ok(())
        }

        async fn list_episodic_nodes(&self, _group_id: &str) -> Result<Vec<EpisodicNode>> {
            Ok(vec![])
        }

        async fn list_entity_nodes(&self, _group_id: &str) -> Result<Vec<EntityNode>> {
            Ok(vec![])
        }

        async fn list_entity_edges(&self, _group_id: &str) -> Result<Vec<EntityEdge>> {
            Ok(vec![])
        }

        async fn save_community_node(&self, _node: &CommunityNode) -> Result<()> {
            Ok(())
        }

        async fn save_entity_edge(&self, _edge: &EntityEdge) -> Result<()> {
            Ok(())
        }

        async fn get_entity_edge(&self, _uuid: &Uuid) -> Result<Option<EntityEdge>> {
            Ok(None)
        }

        async fn save_episodic_edge(&self, _edge: &EpisodicEdge) -> Result<()> {
            Ok(())
        }

        async fn save_community_edge(&self, _edge: &CommunityEdge) -> Result<()> {
            Ok(())
        }

        async fn search_entity_nodes_by_name(
            &self,
            _query: &str,
            _group_id: &str,
            _limit: usize,
        ) -> Result<Vec<EntityNode>> {
            Ok(self.node_text_results.clone())
        }

        async fn search_entity_nodes_by_embedding(
            &self,
            _embedding: &[f32],
            _group_id: &str,
            _limit: usize,
        ) -> Result<Vec<EntityNode>> {
            Ok(self.node_vector_results.clone())
        }

        async fn search_entity_edges_by_fact(
            &self,
            _embedding: &[f32],
            _group_id: &str,
            _limit: usize,
        ) -> Result<Vec<EntityEdge>> {
            Ok(self.edge_vector_results.clone())
        }

        async fn bm25_search_edges(
            &self,
            _query: &str,
            _group_id: &str,
            _limit: usize,
        ) -> Result<Vec<EntityEdge>> {
            Ok(self.edge_bm25_results.clone())
        }

        async fn build_indices(&self) -> Result<()> {
            Ok(())
        }

        async fn get_entity_edges_between(
            &self,
            _source: &Uuid,
            _target: &Uuid,
        ) -> Result<Vec<EntityEdge>> {
            Ok(vec![])
        }

        async fn invalidate_edge(
            &self,
            _uuid: &Uuid,
            _invalid_at: DateTime<Utc>,
        ) -> Result<()> {
            Ok(())
        }
    }

    // -------------------------------------------------------------------------
    // Helpers to build engines and configs for tests.
    // -------------------------------------------------------------------------

    // -------------------------------------------------------------------------
    // MockCrossEncoder — returns caller-supplied scores in document order.
    //
    // `scores` is a fixed mapping from the document text to a relevance score.
    // Documents not found in the map receive a score of 0.0.
    // -------------------------------------------------------------------------

    struct MockCrossEncoder {
        /// Maps document text → relevance score.
        scores: std::collections::HashMap<String, f32>,
    }

    impl MockCrossEncoder {
        fn new(scores: Vec<(&'static str, f32)>) -> Self {
            Self {
                scores: scores.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
            }
        }
    }

    #[async_trait::async_trait]
    impl CrossEncoderClient for MockCrossEncoder {
        async fn score(&self, _query: &str, documents: &[&str]) -> Result<Vec<f32>> {
            Ok(documents
                .iter()
                .map(|doc| self.scores.get(*doc).copied().unwrap_or(0.0))
                .collect())
        }
    }

    fn make_engine(driver: MockDriver) -> SearchEngine {
        SearchEngine::new(Arc::new(driver), Arc::new(MockEmbedder::new()))
    }

    fn make_engine_with_reranker(
        driver: MockDriver,
        reranker: MockCrossEncoder,
    ) -> SearchEngine {
        SearchEngine::new(Arc::new(driver), Arc::new(MockEmbedder::new()))
            .with_reranker(Arc::new(reranker))
    }

    fn single_group_config(group_id: &str) -> SearchConfig {
        SearchConfig::default().with_group_ids(vec![group_id.to_string()])
    }

    // =========================================================================
    // Test 1: Hybrid edge search — shared items must rank highest.
    //
    // BM25 returns [A, B, C]; vector returns [B, C, D].
    // B and C appear in both ranked lists, so after RRF fusion they accumulate
    // higher scores than A (BM25-only) or D (vector-only).
    // =========================================================================
    #[tokio::test]
    async fn test_hybrid_edge_fusion_shared_items_rank_highest() {
        let edge_a = make_entity_edge("fact-A", "g1", None, None);
        let edge_b = make_entity_edge("fact-B", "g1", None, None);
        let edge_c = make_entity_edge("fact-C", "g1", None, None);
        let edge_d = make_entity_edge("fact-D", "g1", None, None);

        // BM25: [A, B, C], Vector: [B, C, D]
        let driver = MockDriver::new().with_edge_results(
            vec![edge_a.clone(), edge_b.clone(), edge_c.clone()],
            vec![edge_b.clone(), edge_c.clone(), edge_d.clone()],
        );

        let engine = make_engine(driver);
        let config = single_group_config("g1").with_search_nodes(false);

        let results = engine.search("test query", &config).await.unwrap();

        // B and C must be in the top 2 positions after RRF fusion.
        let top2_uuids: Vec<Uuid> = results.edges[..2].iter().map(|(e, _)| e.uuid).collect();
        assert!(
            top2_uuids.contains(&edge_b.uuid),
            "edge B (in both lists) must rank in top 2"
        );
        assert!(
            top2_uuids.contains(&edge_c.uuid),
            "edge C (in both lists) must rank in top 2"
        );

        // Output must be sorted descending by fused score.
        let scores: Vec<f32> = results.edges.iter().map(|(_, s)| *s).collect();
        let mut expected = scores.clone();
        expected.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert_eq!(scores, expected, "edge scores must be in descending order");
    }

    // =========================================================================
    // Test 2: Edge-only search — nodes must be empty when search_nodes=false.
    // =========================================================================
    #[tokio::test]
    async fn test_edge_only_search_produces_no_nodes() {
        let driver = MockDriver::new().with_edge_results(
            vec![make_entity_edge("fact", "g1", None, None)],
            vec![],
        );
        let engine = make_engine(driver);
        let config = single_group_config("g1").with_search_nodes(false);

        let results = engine.search("query", &config).await.unwrap();

        assert!(
            results.nodes.is_empty(),
            "nodes must be empty when search_nodes=false"
        );
        assert!(!results.edges.is_empty(), "edges must be populated");
    }

    // =========================================================================
    // Test 3: Node-only search — edges must be empty when search_edges=false.
    // =========================================================================
    #[tokio::test]
    async fn test_node_only_search_produces_no_edges() {
        let node = make_entity_node("Alice", "g1");
        let driver = MockDriver::new().with_node_results(vec![node], vec![]);
        let engine = make_engine(driver);
        let config = single_group_config("g1").with_search_edges(false);

        let results = engine.search("query", &config).await.unwrap();

        assert!(
            results.edges.is_empty(),
            "edges must be empty when search_edges=false"
        );
        assert!(!results.nodes.is_empty(), "nodes must be populated");
    }

    // =========================================================================
    // Test 4: Empty driver results → empty SearchResults.
    // =========================================================================
    #[tokio::test]
    async fn test_empty_driver_results_produce_empty_search_results() {
        let driver = MockDriver::new(); // returns empty vecs for all searches
        let engine = make_engine(driver);
        let config = single_group_config("g1");

        let results = engine.search("query", &config).await.unwrap();

        assert!(results.edges.is_empty(), "expected no edges from empty driver");
        assert!(results.nodes.is_empty(), "expected no nodes from empty driver");
    }

    // =========================================================================
    // Test 5: Expired edges are filtered out of results.
    //
    // `expired_at = Some(t)` means the edge was superseded in the graph
    // (transaction-time end).  It must never surface in search results.
    // =========================================================================
    #[tokio::test]
    async fn test_expired_edge_is_excluded_from_results() {
        let live_edge = make_entity_edge("live fact", "g1", None, None);
        let expired_edge = make_entity_edge(
            "expired fact",
            "g1",
            Some(Utc::now() - Duration::hours(1)),
            None,
        );

        let driver = MockDriver::new()
            .with_edge_results(vec![live_edge.clone(), expired_edge.clone()], vec![]);
        let engine = make_engine(driver);
        let config = single_group_config("g1").with_search_nodes(false);

        let results = engine.search("query", &config).await.unwrap();
        let result_uuids: Vec<Uuid> = results.edges.iter().map(|(e, _)| e.uuid).collect();

        assert!(
            result_uuids.contains(&live_edge.uuid),
            "live edge must appear in results"
        );
        assert!(
            !result_uuids.contains(&expired_edge.uuid),
            "expired edge (expired_at set) must be excluded"
        );
    }

    // =========================================================================
    // Test 6: Edge with past invalid_at is excluded.
    //
    // `invalid_at = Some(t)` where `t <= Utc::now()` means the fact is no
    // longer true in the real world.
    // =========================================================================
    #[tokio::test]
    async fn test_past_invalid_at_edge_is_excluded() {
        let live_edge = make_entity_edge("live fact", "g1", None, None);
        let stale_edge = make_entity_edge(
            "stale fact",
            "g1",
            None,
            Some(Utc::now() - Duration::hours(2)), // became invalid 2 h ago
        );

        let driver = MockDriver::new()
            .with_edge_results(vec![live_edge.clone(), stale_edge.clone()], vec![]);
        let engine = make_engine(driver);
        let config = single_group_config("g1").with_search_nodes(false);

        let results = engine.search("query", &config).await.unwrap();
        let result_uuids: Vec<Uuid> = results.edges.iter().map(|(e, _)| e.uuid).collect();

        assert!(
            result_uuids.contains(&live_edge.uuid),
            "live edge must appear"
        );
        assert!(
            !result_uuids.contains(&stale_edge.uuid),
            "edge with past invalid_at must be excluded"
        );
    }

    // =========================================================================
    // Test 7: Edge with future invalid_at is kept.
    //
    // `invalid_at = Some(t)` where `t > Utc::now()` means the fact is still
    // currently true (the invalidation has not yet occurred).
    // =========================================================================
    #[tokio::test]
    async fn test_future_invalid_at_edge_is_retained() {
        let future_invalid_edge = make_entity_edge(
            "future invalid fact",
            "g1",
            None,
            Some(Utc::now() + Duration::days(30)), // valid for another 30 days
        );

        let driver = MockDriver::new()
            .with_edge_results(vec![future_invalid_edge.clone()], vec![]);
        let engine = make_engine(driver);
        let config = single_group_config("g1").with_search_nodes(false);

        let results = engine.search("query", &config).await.unwrap();
        let result_uuids: Vec<Uuid> = results.edges.iter().map(|(e, _)| e.uuid).collect();

        assert!(
            result_uuids.contains(&future_invalid_edge.uuid),
            "edge with future invalid_at must be retained (fact is still valid)"
        );
    }

    // =========================================================================
    // Test 8: Limit truncation.
    //
    // When the driver returns more edges than `config.limit`, the output is
    // truncated to at most `config.limit` entries.
    // =========================================================================
    #[tokio::test]
    async fn test_limit_truncates_results_to_configured_size() {
        // 7 distinct edges in each of BM25 and vector lists.
        let edges: Vec<EntityEdge> = (0..7)
            .map(|i| make_entity_edge(&format!("fact-{i}"), "g1", None, None))
            .collect();

        let driver = MockDriver::new()
            .with_edge_results(edges.clone(), edges.clone());
        let engine = make_engine(driver);
        let config = single_group_config("g1")
            .with_limit(3)
            .with_search_nodes(false);

        let results = engine.search("query", &config).await.unwrap();

        assert!(
            results.edges.len() <= 3,
            "expected at most 3 edges after truncation, got {}",
            results.edges.len()
        );
    }

    // =========================================================================
    // Test 9: Multi-group deduplication — same UUID across groups keeps one entry.
    //
    // When two group_ids each return the same edge, the final results must
    // deduplicate by UUID, retaining the entry with the highest fused score.
    // =========================================================================
    #[tokio::test]
    async fn test_multi_group_deduplicates_same_edge_uuid() {
        let shared_edge = make_entity_edge("shared fact", "g1", None, None);

        // Both group_ids hit the same MockDriver, so they both return the same
        // edge from BM25 and vector search.
        let driver = MockDriver::new().with_edge_results(
            vec![shared_edge.clone()], // bm25
            vec![shared_edge.clone()], // vector
        );
        let engine = make_engine(driver);
        let config = SearchConfig::default()
            .with_group_ids(vec!["g1".to_string(), "g2".to_string()])
            .with_search_nodes(false);

        let results = engine.search("query", &config).await.unwrap();

        let count = results
            .edges
            .iter()
            .filter(|(e, _)| e.uuid == shared_edge.uuid)
            .count();
        assert_eq!(
            count, 1,
            "shared edge UUID must appear exactly once after multi-group dedup"
        );
    }

    // =========================================================================
    // Test 10: Empty group_ids → GraphitiError::Validation.
    //
    // At least one group_id is required.  An empty slice must be rejected
    // immediately without touching the driver or embedder.
    // =========================================================================
    #[tokio::test]
    async fn test_empty_group_ids_returns_validation_error() {
        let driver = MockDriver::new();
        let engine = make_engine(driver);
        let config = SearchConfig::default(); // group_ids defaults to vec![]

        let result = engine.search("query", &config).await;

        assert!(result.is_err(), "expected an Err for empty group_ids");
        match result.unwrap_err() {
            GraphitiError::Validation(_) => {}
            other => panic!("expected GraphitiError::Validation, got: {other:?}"),
        }
    }

    // =========================================================================
    // Test 11: Negative bm25_weight → GraphitiError::Validation.
    //
    // Negative weights produce negative RRF scores, inverting rank order.
    // The search must reject them before any I/O.
    // =========================================================================
    #[tokio::test]
    async fn test_negative_bm25_weight_returns_validation_error() {
        let driver = MockDriver::new();
        let engine = make_engine(driver);
        let config = single_group_config("g1").with_bm25_weight(-1.0);

        let result = engine.search("query", &config).await;

        assert!(result.is_err(), "expected an Err for negative bm25_weight");
        match result.unwrap_err() {
            GraphitiError::Validation(_) => {}
            other => panic!("expected GraphitiError::Validation, got: {other:?}"),
        }
    }

    // =========================================================================
    // Test 12: Negative vector_weight → GraphitiError::Validation.
    // =========================================================================
    #[tokio::test]
    async fn test_negative_vector_weight_returns_validation_error() {
        let driver = MockDriver::new();
        let engine = make_engine(driver);
        let config = single_group_config("g1").with_vector_weight(-0.5);

        let result = engine.search("query", &config).await;

        assert!(result.is_err(), "expected an Err for negative vector_weight");
        match result.unwrap_err() {
            GraphitiError::Validation(_) => {}
            other => panic!("expected GraphitiError::Validation, got: {other:?}"),
        }
    }

    // =========================================================================
    // Test 13: Zero weights are valid (zero-weight lists contribute no score).
    // =========================================================================
    #[tokio::test]
    async fn test_zero_weights_are_accepted() {
        let driver = MockDriver::new();
        let engine = make_engine(driver);
        let config = single_group_config("g1")
            .with_bm25_weight(0.0)
            .with_vector_weight(0.0);

        let result = engine.search("query", &config).await;

        assert!(result.is_ok(), "zero weights must be accepted");
    }

    // =========================================================================
    // Test 14: Multi-group concurrent search — both edge UUIDs appear in
    //          merged results exactly once after cross-group deduplication.
    //
    // NOTE: MockDriver is intentionally NOT group-aware — it returns the same
    // pre-configured edge list regardless of the `group_id` argument passed to
    // each driver search method.  Concretely, both edge_a and edge_b are
    // returned for BOTH group "g1" and group "g2", producing two copies of
    // each UUID before the cross-group merge step.
    //
    // What this test DOES validate:
    //   • The `join_all` fan-out runs all group searches concurrently.
    //   • The cross-group merge deduplicates by UUID, keeping each at most once.
    //
    // What this test does NOT validate:
    //   • Group isolation — that a "g1" search returns only edges tagged "g1".
    //     Group isolation is enforced by the real Neo4j driver's Cypher WHERE
    //     clauses, not by the engine layer, and is exercised by integration
    //     tests against a live database.
    // =========================================================================
    #[tokio::test]
    async fn test_multi_group_concurrent_search_merges_distinct_results() {
        let edge_a = make_entity_edge("fact-A", "g1", None, None);
        let edge_b = make_entity_edge("fact-B", "g2", None, None);

        // MockDriver is not group-aware: both edges are returned for every
        // group_id.  After the cross-group merge each UUID must appear once.
        let driver = MockDriver::new().with_edge_results(
            vec![edge_a.clone(), edge_b.clone()],
            vec![],
        );
        let engine = make_engine(driver);
        let config = SearchConfig::default()
            .with_group_ids(vec!["g1".to_string(), "g2".to_string()])
            .with_search_nodes(false);

        let results = engine.search("query", &config).await.unwrap();
        let result_uuids: Vec<Uuid> = results.edges.iter().map(|(e, _)| e.uuid).collect();

        assert!(
            result_uuids.contains(&edge_a.uuid),
            "edge A must appear in merged multi-group results"
        );
        assert!(
            result_uuids.contains(&edge_b.uuid),
            "edge B must appear in merged multi-group results"
        );
        // Each UUID must appear exactly once (deduplication).
        let count_a = result_uuids.iter().filter(|&&u| u == edge_a.uuid).count();
        let count_b = result_uuids.iter().filter(|&&u| u == edge_b.uuid).count();
        assert_eq!(count_a, 1, "edge A must appear exactly once");
        assert_eq!(count_b, 1, "edge B must appear exactly once");
    }
}
