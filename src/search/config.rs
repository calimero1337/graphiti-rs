//! SearchConfig — configuration for hybrid search queries.

use serde::{Deserialize, Serialize};

/// Configuration for a hybrid search query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Maximum number of results to return per type (edges, nodes).
    pub limit: usize,
    /// Tenant/partition filter. At least one entry required.
    pub group_ids: Vec<String>,
    /// Whether to search entity edges.
    pub search_edges: bool,
    /// Whether to search entity nodes.
    pub search_nodes: bool,
    /// Weight applied to BM25 fulltext rankings in RRF fusion.
    pub bm25_weight: f32,
    /// Weight applied to vector similarity rankings in RRF fusion.
    pub vector_weight: f32,
    /// Apply cross-encoder reranking after RRF fusion when a reranker is
    /// configured on the [`crate::search::SearchEngine`].  Has no effect
    /// when the engine has no reranker attached.  Defaults to `false`.
    pub rerank: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            limit: 10,
            group_ids: vec![],
            search_edges: true,
            search_nodes: true,
            bm25_weight: 1.0,
            vector_weight: 1.0,
            rerank: false,
        }
    }
}

impl SearchConfig {
    /// Set the result limit.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Set the group IDs for tenant filtering.
    pub fn with_group_ids(mut self, group_ids: Vec<String>) -> Self {
        self.group_ids = group_ids;
        self
    }

    /// Enable or disable edge search.
    pub fn with_search_edges(mut self, search_edges: bool) -> Self {
        self.search_edges = search_edges;
        self
    }

    /// Enable or disable node search.
    pub fn with_search_nodes(mut self, search_nodes: bool) -> Self {
        self.search_nodes = search_nodes;
        self
    }

    /// Set BM25 weight for RRF fusion.
    pub fn with_bm25_weight(mut self, bm25_weight: f32) -> Self {
        self.bm25_weight = bm25_weight;
        self
    }

    /// Set vector similarity weight for RRF fusion.
    pub fn with_vector_weight(mut self, vector_weight: f32) -> Self {
        self.vector_weight = vector_weight;
        self
    }

    /// Enable or disable cross-encoder reranking.
    ///
    /// Reranking only takes effect when a [`crate::cross_encoder::CrossEncoderClient`]
    /// has been attached to the [`crate::search::SearchEngine`] via
    /// [`crate::search::SearchEngine::with_reranker`].
    pub fn with_rerank(mut self, rerank: bool) -> Self {
        self.rerank = rerank;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_config_defaults() {
        let config = SearchConfig::default();
        assert_eq!(config.limit, 10);
        assert!(config.group_ids.is_empty());
        assert!(config.search_edges);
        assert!(config.search_nodes);
        assert_eq!(config.bm25_weight, 1.0);
        assert_eq!(config.vector_weight, 1.0);
    }

    #[test]
    fn test_search_config_builder_methods() {
        let config = SearchConfig::default()
            .with_limit(5)
            .with_group_ids(vec!["g1".to_string(), "g2".to_string()])
            .with_search_edges(false)
            .with_search_nodes(true)
            .with_bm25_weight(2.0)
            .with_vector_weight(0.5);

        assert_eq!(config.limit, 5);
        assert_eq!(config.group_ids, vec!["g1", "g2"]);
        assert!(!config.search_edges);
        assert!(config.search_nodes);
        assert_eq!(config.bm25_weight, 2.0);
        assert_eq!(config.vector_weight, 0.5);
    }

    #[test]
    fn test_search_config_serde_roundtrip() {
        let config = SearchConfig::default()
            .with_limit(20)
            .with_group_ids(vec!["tenant-a".to_string()]);

        let json = serde_json::to_string(&config).expect("serialization failed");
        let restored: SearchConfig = serde_json::from_str(&json).expect("deserialization failed");

        assert_eq!(restored.limit, 20);
        assert_eq!(restored.group_ids, vec!["tenant-a"]);
    }
}
