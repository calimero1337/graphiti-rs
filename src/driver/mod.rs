//! Graph database driver abstraction.
//!
//! Defines the [`GraphDriver`] trait that all backend implementations must satisfy,
//! plus the Neo4j implementation.

pub mod neo4j;

use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::edges::{CommunityEdge, EntityEdge, EpisodicEdge};
use crate::errors::Result;
use crate::nodes::{CommunityNode, EntityNode, EpisodicNode};

/// Trait representing a graph database backend.
///
/// Covers node CRUD, edge CRUD, search, and maintenance operations.
/// Phase 1 implements only `neo4j::Neo4jDriver`.
#[async_trait::async_trait]
pub trait GraphDriver: Send + Sync {
    // ── Connection ────────────────────────────────────────────────────────────

    /// Health check — verify connectivity to the database.
    async fn ping(&self) -> Result<()>;

    /// Close the connection pool / session.
    async fn close(&self) -> Result<()>;

    // ── Node CRUD ─────────────────────────────────────────────────────────────

    /// Upsert an [`EntityNode`] by UUID.
    async fn save_entity_node(&self, node: &EntityNode) -> Result<()>;

    /// Retrieve an [`EntityNode`] by UUID, returning `None` if not found.
    async fn get_entity_node(&self, uuid: &Uuid) -> Result<Option<EntityNode>>;

    /// Delete an [`EntityNode`] and all its relationships.
    async fn delete_entity_node(&self, uuid: &Uuid) -> Result<()>;

    /// Upsert an [`EpisodicNode`] by UUID.
    async fn save_episodic_node(&self, node: &EpisodicNode) -> Result<()>;

    /// Retrieve an [`EpisodicNode`] by UUID, returning `None` if not found.
    async fn get_episodic_node(&self, uuid: &Uuid) -> Result<Option<EpisodicNode>>;

    /// Delete an [`EpisodicNode`] and all its relationships.
    ///
    /// Idempotent: deleting a node that does not exist must return `Ok(())`.
    async fn delete_episodic_node(&self, uuid: &Uuid) -> Result<()>;

    /// Return every [`EpisodicNode`] that belongs to `group_id`, in no
    /// guaranteed order.
    async fn list_episodic_nodes(&self, group_id: &str) -> Result<Vec<EpisodicNode>>;

    /// Return every [`EntityNode`] that belongs to `group_id`, in no guaranteed order.
    async fn list_entity_nodes(&self, group_id: &str) -> Result<Vec<EntityNode>>;

    /// Return every active (non-invalidated) [`EntityEdge`] that belongs to `group_id`.
    async fn list_entity_edges(&self, group_id: &str) -> Result<Vec<EntityEdge>>;

    /// Upsert a [`CommunityNode`] by UUID.
    async fn save_community_node(&self, node: &CommunityNode) -> Result<()>;

    // ── Edge CRUD ─────────────────────────────────────────────────────────────

    /// Upsert an [`EntityEdge`] between two nodes.
    async fn save_entity_edge(&self, edge: &EntityEdge) -> Result<()>;

    /// Retrieve an [`EntityEdge`] by UUID, returning `None` if not found.
    async fn get_entity_edge(&self, uuid: &Uuid) -> Result<Option<EntityEdge>>;

    /// Upsert an [`EpisodicEdge`] (MENTIONS) between two nodes.
    async fn save_episodic_edge(&self, edge: &EpisodicEdge) -> Result<()>;

    /// Upsert a [`CommunityEdge`] (HAS_MEMBER) between two nodes.
    async fn save_community_edge(&self, edge: &CommunityEdge) -> Result<()>;

    // ── Search ────────────────────────────────────────────────────────────────

    /// Text search for entity nodes whose name contains `query` within `group_id`.
    async fn search_entity_nodes_by_name(
        &self,
        query: &str,
        group_id: &str,
        limit: usize,
    ) -> Result<Vec<EntityNode>>;

    /// Vector similarity search for entity nodes by name embedding.
    async fn search_entity_nodes_by_embedding(
        &self,
        embedding: &[f32],
        group_id: &str,
        limit: usize,
    ) -> Result<Vec<EntityNode>>;

    /// Vector similarity search for entity edges by fact embedding.
    async fn search_entity_edges_by_fact(
        &self,
        embedding: &[f32],
        group_id: &str,
        limit: usize,
    ) -> Result<Vec<EntityEdge>>;

    /// BM25 fulltext search for entity edges by fact text.
    async fn bm25_search_edges(
        &self,
        query: &str,
        group_id: &str,
        limit: usize,
    ) -> Result<Vec<EntityEdge>>;

    // ── Maintenance ───────────────────────────────────────────────────────────

    /// Create vector and fulltext indexes if they do not already exist.
    async fn build_indices(&self) -> Result<()>;

    /// Retrieve all [`EntityEdge`]s between two nodes (in either direction).
    async fn get_entity_edges_between(
        &self,
        source: &Uuid,
        target: &Uuid,
    ) -> Result<Vec<EntityEdge>>;

    /// Set `invalid_at` on an edge, marking the fact as no longer true.
    async fn invalidate_edge(&self, uuid: &Uuid, invalid_at: DateTime<Utc>) -> Result<()>;
}
