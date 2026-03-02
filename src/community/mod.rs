//! Community detection and summarization.
//!
//! Uses label propagation over the entity graph (see [`detection`]) followed by
//! LLM-based hierarchical summarization (see [`summarize`]) of each discovered
//! community, then persists the results to the graph database.
//!
//! # Usage
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use graphiti_rs::community::CommunityBuilder;
//! # async fn run(
//! #     driver: Arc<dyn graphiti_rs::driver::GraphDriver>,
//! #     llm: Arc<dyn graphiti_rs::llm_client::LlmClient>,
//! #     embedder: Arc<dyn graphiti_rs::embedder::EmbedderClient>,
//! # ) -> graphiti_rs::Result<()> {
//! let builder = CommunityBuilder { driver, llm, embedder };
//! let result = builder.build_communities(&["group-1", "group-2"]).await?;
//! println!("Detected {} communities", result.communities.len());
//! # Ok(())
//! # }
//! ```

pub mod detection;
pub mod summarize;

use std::collections::HashMap;
use std::sync::Arc;

use chrono::Utc;
use uuid::Uuid;

use crate::driver::GraphDriver;
use crate::edges::CommunityEdge;
use crate::embedder::EmbedderClient;
use crate::errors::Result;
use crate::llm_client::LlmClient;
use crate::nodes::{CommunityNode, EntityNode};

use detection::EdgeSpec;

// ── Public types ──────────────────────────────────────────────────────────────

/// The outcome of a [`CommunityBuilder::build_communities`] call.
#[derive(Debug)]
pub struct CommunityResult {
    /// Every `CommunityNode` that was detected and persisted.
    pub communities: Vec<CommunityNode>,
    /// Every `CommunityEdge` (HAS_MEMBER) that was persisted.
    pub edges: Vec<CommunityEdge>,
}

/// Detects communities in the entity graph and persists them to the graph database.
pub struct CommunityBuilder {
    /// Graph database driver.
    pub driver: Arc<dyn GraphDriver>,
    /// LLM client used to summarize each community.
    pub llm: Arc<dyn LlmClient>,
    /// Embedder used to produce `name_embedding` for each community node.
    pub embedder: Arc<dyn EmbedderClient>,
}

impl CommunityBuilder {
    /// Detect communities across the entity graphs of all `group_ids`, summarize
    /// them with the LLM, generate name embeddings, and persist the results.
    ///
    /// # Steps
    ///
    /// 1. Load all `EntityNode`s and active `EntityEdge`s for the requested groups.
    /// 2. Run label propagation to discover community clusters.
    /// 3. For each cluster, collect member entity summaries and call the LLM
    ///    (with hierarchical reduction for large communities) to produce a summary.
    /// 4. Derive a human-readable community name from the summary.
    /// 5. Embed the community name.
    /// 6. Persist `CommunityNode` and `CommunityEdge` records to the database.
    pub async fn build_communities(&self, group_ids: &[&str]) -> Result<CommunityResult> {
        // ── 1. Load entity nodes and edges ────────────────────────────────────
        let mut all_nodes: Vec<EntityNode> = Vec::new();
        let mut all_edge_specs: Vec<EdgeSpec> = Vec::new();

        for &group_id in group_ids {
            let nodes = self.driver.list_entity_nodes(group_id).await?;
            let edges = self.driver.list_entity_edges(group_id).await?;

            all_nodes.extend(nodes);
            for edge in edges {
                all_edge_specs
                    .push(EdgeSpec { source: edge.source_node_uuid, target: edge.target_node_uuid });
            }
        }

        // UUID → EntityNode lookup (for accessing summaries during step 3).
        let node_map: HashMap<Uuid, &EntityNode> =
            all_nodes.iter().map(|n| (n.uuid, n)).collect();

        let node_uuids: Vec<Uuid> = all_nodes.iter().map(|n| n.uuid).collect();

        // ── 2. Label propagation ──────────────────────────────────────────────
        let clusters = detection::detect_communities(&node_uuids, &all_edge_specs, 30);

        // ── 3–6. Summarize, embed, and persist each community ─────────────────
        let mut community_nodes: Vec<CommunityNode> = Vec::new();
        let mut community_edges: Vec<CommunityEdge> = Vec::new();

        for (_label, member_uuids) in &clusters {
            // Collect non-empty entity summaries for this community.
            let entity_summaries: Vec<String> = member_uuids
                .iter()
                .filter_map(|uuid| node_map.get(uuid))
                .map(|n| n.summary.clone())
                .filter(|s| !s.is_empty())
                .collect();

            // Generate community summary (hierarchical if large).
            let summary =
                summarize::summarize_community(self.llm.as_ref(), &entity_summaries).await?;

            // Derive a short name from the summary.
            let name = derive_community_name(&summary);

            // Embed the community name.
            let name_embedding = self.embedder.embed(&name).await?;

            let community_node = CommunityNode {
                uuid: Uuid::new_v4(),
                name,
                name_embedding: Some(name_embedding),
                summary,
                created_at: Utc::now(),
            };

            // Persist the community node.
            self.driver.save_community_node(&community_node).await?;

            // Create and persist HAS_MEMBER edges.
            for &member_uuid in member_uuids {
                let edge = CommunityEdge {
                    uuid: Uuid::new_v4(),
                    source_node_uuid: community_node.uuid,
                    target_node_uuid: member_uuid,
                    created_at: Utc::now(),
                };
                self.driver.save_community_edge(&edge).await?;
                community_edges.push(edge);
            }

            community_nodes.push(community_node);
        }

        Ok(CommunityResult { communities: community_nodes, edges: community_edges })
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Derive a short community name from its LLM-generated summary.
///
/// Extracts the first sentence (split on `.`) and truncates to 60 characters,
/// adding `…` if truncated. Falls back to `"Community"` for empty summaries.
fn derive_community_name(summary: &str) -> String {
    let first = summary.split('.').next().unwrap_or(summary).trim();
    if first.is_empty() {
        return "Community".to_string();
    }
    // Truncate to 60 chars on a character boundary, not a byte boundary.
    let mut chars = first.chars();
    let truncated: String = chars.by_ref().take(60).collect();
    if chars.next().is_some() {
        // There were characters beyond the 60-char limit.
        format!("{truncated}…")
    } else {
        truncated
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutils::{MockDriver, MockEmbedder, MockLlmClient};

    // ── derive_community_name ─────────────────────────────────────────────────

    #[test]
    fn derive_name_uses_first_sentence() {
        let name = derive_community_name("AI researchers. Other details.");
        assert_eq!(name, "AI researchers");
    }

    #[test]
    fn derive_name_trims_whitespace() {
        let name = derive_community_name("  Leading spaces. ");
        assert_eq!(name, "Leading spaces");
    }

    #[test]
    fn derive_name_truncates_long_sentence() {
        let long = "A".repeat(70);
        let name = derive_community_name(&long);
        assert!(name.ends_with('…'), "long name must end with ellipsis");
        // The content before '…' must be exactly 60 chars.
        let content: String = name.chars().take_while(|&c| c != '…').collect();
        assert_eq!(content.chars().count(), 60);
    }

    #[test]
    fn derive_name_empty_summary_returns_fallback() {
        assert_eq!(derive_community_name(""), "Community");
        assert_eq!(derive_community_name("."), "Community");
    }

    #[test]
    fn derive_name_short_summary_unchanged() {
        let name = derive_community_name("Short.");
        assert_eq!(name, "Short");
    }

    // ── CommunityBuilder (integration-style with mocks) ───────────────────────

    #[tokio::test]
    async fn build_communities_empty_groups_returns_empty_result() {
        let builder = CommunityBuilder {
            driver: Arc::new(MockDriver),
            llm: Arc::new(MockLlmClient),
            embedder: Arc::new(MockEmbedder::new()),
        };
        let result = builder.build_communities(&[]).await.unwrap();
        assert!(result.communities.is_empty());
        assert!(result.edges.is_empty());
    }

    #[tokio::test]
    async fn build_communities_no_entities_returns_empty_result() {
        // MockDriver returns empty lists for list_entity_nodes / list_entity_edges.
        let builder = CommunityBuilder {
            driver: Arc::new(MockDriver),
            llm: Arc::new(MockLlmClient),
            embedder: Arc::new(MockEmbedder::new()),
        };
        let result = builder.build_communities(&["group-1"]).await.unwrap();
        assert!(result.communities.is_empty());
        assert!(result.edges.is_empty());
    }

    #[test]
    fn community_result_is_debug() {
        // Ensure the `Debug` derive compiles and produces some output.
        let r = CommunityResult { communities: vec![], edges: vec![] };
        let dbg = format!("{r:?}");
        assert!(dbg.contains("CommunityResult"));
    }
}
