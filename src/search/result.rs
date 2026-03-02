//! SearchResults — container for hybrid search output.

use serde::Serialize;

use crate::edges::entity::EntityEdge;
use crate::nodes::entity::EntityNode;

/// Container for hybrid search results.
///
/// Each entry pairs the matched record with its RRF-fused relevance score.
#[derive(Debug, Clone, Default, Serialize)]
pub struct SearchResults {
    /// Matched entity edges with fused scores (score descending).
    pub edges: Vec<(EntityEdge, f32)>,
    /// Matched entity nodes with fused scores (score descending).
    pub nodes: Vec<(EntityNode, f32)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_results_default() {
        let results = SearchResults::default();
        assert!(results.edges.is_empty());
        assert!(results.nodes.is_empty());
    }

    #[test]
    fn test_search_results_clone() {
        let results = SearchResults::default();
        let cloned = results.clone();
        assert!(cloned.edges.is_empty());
        assert!(cloned.nodes.is_empty());
    }
}
