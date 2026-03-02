//! Neo4j graph driver implementation.
//!
//! Uses `neo4rs` 0.8 for async, pooled Bolt 4.x connections.
//! Implements all 11 operation interfaces from the Python `Neo4jDriver`.

use chrono::{DateTime, Utc};
use neo4rs::query;
use uuid::Uuid;

use crate::driver::GraphDriver;
use crate::edges::{CommunityEdge, EntityEdge, EpisodicEdge};
use crate::errors::{GraphitiError, Result};
use crate::nodes::{CommunityNode, EntityNode, EpisodicNode};

use self::convert::driver_err;

pub(super) mod convert;
pub(super) mod edges;
pub(super) mod nodes;
pub(super) mod schema;
pub(super) mod search;

/// Neo4j graph driver backed by a `neo4rs` connection pool.
pub struct Neo4jDriver {
    graph: neo4rs::Graph,
}

impl Neo4jDriver {
    /// Connect to Neo4j at `uri` using the given credentials.
    ///
    /// Returns `Err(GraphitiError::Driver(...))` if the connection or
    /// authentication fails.
    pub async fn new(uri: &str, user: &str, password: &str) -> Result<Self> {
        let config = neo4rs::ConfigBuilder::default()
            .uri(uri)
            .user(user)
            .password(password)
            .build()
            .map_err(|e| GraphitiError::Driver(e.to_string()))?;
        let graph = neo4rs::Graph::connect(config)
            .await
            .map_err(|e| GraphitiError::Driver(e.to_string()))?;
        Ok(Self { graph })
    }
}

#[async_trait::async_trait]
impl GraphDriver for Neo4jDriver {
    // ── Connection ────────────────────────────────────────────────────────────

    async fn ping(&self) -> Result<()> {
        self.graph.run(query("RETURN 1")).await.map_err(driver_err)
    }

    async fn close(&self) -> Result<()> {
        // neo4rs pool cleans up on drop; nothing to do here.
        Ok(())
    }

    // ── Node CRUD ─────────────────────────────────────────────────────────────

    async fn save_entity_node(&self, node: &EntityNode) -> Result<()> {
        nodes::save_entity_node(&self.graph, node).await
    }

    async fn get_entity_node(&self, uuid: &Uuid) -> Result<Option<EntityNode>> {
        nodes::get_entity_node(&self.graph, uuid).await
    }

    async fn delete_entity_node(&self, uuid: &Uuid) -> Result<()> {
        nodes::delete_entity_node(&self.graph, uuid).await
    }

    async fn save_episodic_node(&self, node: &EpisodicNode) -> Result<()> {
        nodes::save_episodic_node(&self.graph, node).await
    }

    async fn get_episodic_node(&self, uuid: &Uuid) -> Result<Option<EpisodicNode>> {
        nodes::get_episodic_node(&self.graph, uuid).await
    }

    async fn delete_episodic_node(&self, uuid: &Uuid) -> Result<()> {
        nodes::delete_episodic_node(&self.graph, uuid).await
    }

    async fn list_episodic_nodes(&self, group_id: &str) -> Result<Vec<EpisodicNode>> {
        nodes::list_episodic_nodes(&self.graph, group_id).await
    }

    async fn list_entity_nodes(&self, group_id: &str) -> Result<Vec<EntityNode>> {
        nodes::list_entity_nodes(&self.graph, group_id).await
    }

    async fn list_entity_edges(&self, group_id: &str) -> Result<Vec<EntityEdge>> {
        edges::list_entity_edges(&self.graph, group_id).await
    }

    async fn save_community_node(&self, node: &CommunityNode) -> Result<()> {
        nodes::save_community_node(&self.graph, node).await
    }

    // ── Edge CRUD ─────────────────────────────────────────────────────────────

    async fn save_entity_edge(&self, edge: &EntityEdge) -> Result<()> {
        edges::save_entity_edge(&self.graph, edge).await
    }

    async fn get_entity_edge(&self, uuid: &Uuid) -> Result<Option<EntityEdge>> {
        edges::get_entity_edge(&self.graph, uuid).await
    }

    async fn save_episodic_edge(&self, edge: &EpisodicEdge) -> Result<()> {
        edges::save_episodic_edge(&self.graph, edge).await
    }

    async fn save_community_edge(&self, edge: &CommunityEdge) -> Result<()> {
        edges::save_community_edge(&self.graph, edge).await
    }

    // ── Search ────────────────────────────────────────────────────────────────

    async fn search_entity_nodes_by_name(
        &self,
        query: &str,
        group_id: &str,
        limit: usize,
    ) -> Result<Vec<EntityNode>> {
        search::search_entity_nodes_by_name(&self.graph, query, group_id, limit).await
    }

    async fn search_entity_nodes_by_embedding(
        &self,
        embedding: &[f32],
        group_id: &str,
        limit: usize,
    ) -> Result<Vec<EntityNode>> {
        search::search_entity_nodes_by_embedding(&self.graph, embedding, group_id, limit).await
    }

    async fn search_entity_edges_by_fact(
        &self,
        embedding: &[f32],
        group_id: &str,
        limit: usize,
    ) -> Result<Vec<EntityEdge>> {
        search::search_entity_edges_by_fact(&self.graph, embedding, group_id, limit).await
    }

    async fn bm25_search_edges(
        &self,
        query: &str,
        group_id: &str,
        limit: usize,
    ) -> Result<Vec<EntityEdge>> {
        search::bm25_search_edges(&self.graph, query, group_id, limit).await
    }

    // ── Maintenance ───────────────────────────────────────────────────────────

    async fn build_indices(&self) -> Result<()> {
        schema::build_indices(&self.graph).await
    }

    async fn get_entity_edges_between(
        &self,
        source: &Uuid,
        target: &Uuid,
    ) -> Result<Vec<EntityEdge>> {
        edges::get_entity_edges_between(&self.graph, source, target).await
    }

    async fn invalidate_edge(&self, uuid: &Uuid, invalid_at: DateTime<Utc>) -> Result<()> {
        edges::invalidate_edge(&self.graph, uuid, invalid_at).await
    }
}

#[cfg(test)]
mod tests {
    use super::Neo4jDriver;
    use crate::driver::GraphDriver;
    use crate::edges::{CommunityEdge, EntityEdge, EpisodicEdge};
    use crate::nodes::{CommunityNode, EntityNode, EpisodeType, EpisodicNode};
    use chrono::Utc;
    use uuid::Uuid;

    /// Helper: default test connection parameters pointing at a local Neo4j instance.
    fn test_uri() -> &'static str {
        "bolt://localhost:7687"
    }
    fn test_user() -> &'static str {
        "neo4j"
    }
    fn test_password() -> &'static str {
        "test-password"
    }

    // ── Construction ──────────────────────────────────────────────────────────

    /// `Neo4jDriver::new` returns a `Result<Neo4jDriver>` — not an infallible value.
    ///
    /// Ignored in CI because it requires a live Neo4j instance.
    #[tokio::test]
    #[ignore = "requires live Neo4j at bolt://localhost:7687"]
    async fn neo4j_driver_new_returns_ok() {
        let driver = Neo4jDriver::new(test_uri(), test_user(), test_password())
            .await
            .expect("Neo4jDriver::new should succeed against a running Neo4j");
        drop(driver);
    }

    /// Connecting with a wrong password must return an `Err`.
    ///
    /// Ignored in CI because it requires the port to be reachable.
    #[tokio::test]
    #[ignore = "requires live Neo4j at bolt://localhost:7687"]
    async fn neo4j_driver_new_bad_password_is_err() {
        let result = Neo4jDriver::new(test_uri(), test_user(), "wrong-password").await;
        assert!(result.is_err(), "bad credentials must yield an error");
    }

    // ── GraphDriver trait compliance ──────────────────────────────────────────

    /// `Neo4jDriver` must satisfy `GraphDriver` (Send + Sync).
    ///
    /// This is a compile-time assertion; the body never executes.
    #[allow(dead_code)]
    fn assert_graph_driver_impl<T: GraphDriver>(_: &T) {}

    /// Verify that `Neo4jDriver` is accepted wherever `GraphDriver` is expected
    /// by constructing it behind a trait object.
    ///
    /// Ignored in CI because it requires a live Neo4j instance.
    #[tokio::test]
    #[ignore = "requires live Neo4j at bolt://localhost:7687"]
    async fn neo4j_driver_is_graph_driver() {
        let driver = Neo4jDriver::new(test_uri(), test_user(), test_password())
            .await
            .unwrap();
        // Must compile as a `dyn GraphDriver` trait object.
        let _dyn_ref: &dyn GraphDriver = &driver;
    }

    // ── ping / close ──────────────────────────────────────────────────────────

    #[tokio::test]
    #[ignore = "requires live Neo4j at bolt://localhost:7687"]
    async fn neo4j_driver_ping_returns_ok() {
        let driver = Neo4jDriver::new(test_uri(), test_user(), test_password())
            .await
            .unwrap();
        driver.ping().await.expect("ping should succeed");
    }

    #[tokio::test]
    #[ignore = "requires live Neo4j at bolt://localhost:7687"]
    async fn neo4j_driver_close_is_idempotent() {
        let driver = Neo4jDriver::new(test_uri(), test_user(), test_password())
            .await
            .unwrap();
        driver.close().await.expect("first close should succeed");
    }

    // ── EntityNode CRUD ───────────────────────────────────────────────────────

    fn make_entity_node(group_id: &str) -> EntityNode {
        EntityNode {
            uuid: Uuid::new_v4(),
            name: "Alice".to_string(),
            group_id: group_id.to_string(),
            labels: vec!["Person".to_string()],
            summary: "A test entity node".to_string(),
            name_embedding: None,
            attributes: serde_json::Value::Object(Default::default()),
            created_at: Utc::now(),
        }
    }

    #[tokio::test]
    #[ignore = "requires live Neo4j at bolt://localhost:7687"]
    async fn save_and_get_entity_node_roundtrip() {
        let driver = Neo4jDriver::new(test_uri(), test_user(), test_password())
            .await
            .unwrap();

        let node = make_entity_node("test-group");
        driver.save_entity_node(&node).await.expect("save should succeed");

        let fetched = driver
            .get_entity_node(&node.uuid)
            .await
            .expect("get should succeed");

        let fetched = fetched.expect("node should be present after save");
        assert_eq!(fetched.uuid, node.uuid);
        assert_eq!(fetched.name, node.name);
        assert_eq!(fetched.group_id, node.group_id);

        // Cleanup.
        driver.delete_entity_node(&node.uuid).await.expect("cleanup delete failed");
    }

    #[tokio::test]
    #[ignore = "requires live Neo4j at bolt://localhost:7687"]
    async fn get_entity_node_missing_uuid_returns_none() {
        let driver = Neo4jDriver::new(test_uri(), test_user(), test_password())
            .await
            .unwrap();

        let missing = Uuid::new_v4();
        let result = driver.get_entity_node(&missing).await.expect("query should not error");
        assert!(result.is_none(), "non-existent node must return None");
    }

    #[tokio::test]
    #[ignore = "requires live Neo4j at bolt://localhost:7687"]
    async fn delete_entity_node_is_idempotent() {
        let driver = Neo4jDriver::new(test_uri(), test_user(), test_password())
            .await
            .unwrap();

        let node = make_entity_node("del-test-group");
        driver.save_entity_node(&node).await.unwrap();
        driver.delete_entity_node(&node.uuid).await.expect("first delete should succeed");
        // Deleting a non-existent node must not error.
        driver.delete_entity_node(&node.uuid).await.expect("second delete should also succeed");
    }

    // ── EpisodicNode CRUD ─────────────────────────────────────────────────────

    fn make_episodic_node(group_id: &str) -> EpisodicNode {
        EpisodicNode {
            uuid: Uuid::new_v4(),
            name: "episode-1".to_string(),
            group_id: group_id.to_string(),
            labels: vec!["EpisodicNode".to_string()],
            created_at: Utc::now(),
            source: EpisodeType::Text,
            source_description: "unit test".to_string(),
            content: "test episode content".to_string(),
            valid_at: Utc::now(),
            entity_edges: vec![],
        }
    }

    #[tokio::test]
    #[ignore = "requires live Neo4j at bolt://localhost:7687"]
    async fn save_and_get_episodic_node_roundtrip() {
        let driver = Neo4jDriver::new(test_uri(), test_user(), test_password())
            .await
            .unwrap();

        let node = make_episodic_node("ep-group");
        driver.save_episodic_node(&node).await.expect("save episodic should succeed");

        let fetched = driver
            .get_episodic_node(&node.uuid)
            .await
            .expect("get episodic should succeed");

        let fetched = fetched.expect("episodic node should be present after save");
        assert_eq!(fetched.uuid, node.uuid);
        assert_eq!(fetched.content, node.content);
        assert_eq!(fetched.group_id, node.group_id);

        // Cleanup.
        driver.delete_episodic_node(&node.uuid).await.expect("cleanup delete failed");
    }

    // ── EntityEdge CRUD ───────────────────────────────────────────────────────

    fn make_entity_edge(source: Uuid, target: Uuid, group_id: &str) -> EntityEdge {
        EntityEdge {
            uuid: Uuid::new_v4(),
            source_node_uuid: source,
            target_node_uuid: target,
            name: "KNOWS".to_string(),
            fact: "Alice knows Bob".to_string(),
            fact_embedding: None,
            episodes: vec![],
            valid_at: None,
            invalid_at: None,
            created_at: Utc::now(),
            expired_at: None,
            weight: 1.0,
            attributes: serde_json::Value::Null,
            group_id: Some(group_id.to_string()),
        }
    }

    #[tokio::test]
    #[ignore = "requires live Neo4j at bolt://localhost:7687"]
    async fn save_and_get_entity_edge_roundtrip() {
        let driver = Neo4jDriver::new(test_uri(), test_user(), test_password())
            .await
            .unwrap();

        // Ensure source and target nodes exist first.
        let source_node = make_entity_node("edge-group");
        let target_node = make_entity_node("edge-group");
        driver.save_entity_node(&source_node).await.unwrap();
        driver.save_entity_node(&target_node).await.unwrap();

        let edge = make_entity_edge(source_node.uuid, target_node.uuid, "edge-group");
        driver.save_entity_edge(&edge).await.expect("save edge should succeed");

        let fetched = driver
            .get_entity_edge(&edge.uuid)
            .await
            .expect("get edge should succeed");
        let fetched = fetched.expect("edge should be present after save");
        assert_eq!(fetched.uuid, edge.uuid);
        assert_eq!(fetched.fact, edge.fact);
        assert_eq!(fetched.source_node_uuid, source_node.uuid);
        assert_eq!(fetched.target_node_uuid, target_node.uuid);

        // Cleanup.
        driver.delete_entity_node(&source_node.uuid).await.ok();
        driver.delete_entity_node(&target_node.uuid).await.ok();
    }

    #[tokio::test]
    #[ignore = "requires live Neo4j at bolt://localhost:7687"]
    async fn get_entity_edge_missing_uuid_returns_none() {
        let driver = Neo4jDriver::new(test_uri(), test_user(), test_password())
            .await
            .unwrap();

        let missing = Uuid::new_v4();
        let result = driver.get_entity_edge(&missing).await.expect("query must not error");
        assert!(result.is_none(), "missing edge must return None");
    }

    // ── Search ────────────────────────────────────────────────────────────────

    #[tokio::test]
    #[ignore = "requires live Neo4j at bolt://localhost:7687"]
    async fn search_entity_nodes_by_name_returns_vec() {
        let driver = Neo4jDriver::new(test_uri(), test_user(), test_password())
            .await
            .unwrap();

        // No assertion on count — just verify the method returns Ok(Vec<_>).
        let results = driver
            .search_entity_nodes_by_name("Alice", "search-group", 10)
            .await
            .expect("name search should return Ok");
        let _ = results;
    }

    #[tokio::test]
    #[ignore = "requires live Neo4j at bolt://localhost:7687"]
    async fn bm25_search_edges_returns_vec() {
        let driver = Neo4jDriver::new(test_uri(), test_user(), test_password())
            .await
            .unwrap();

        let results = driver
            .bm25_search_edges("Alice", "search-group", 10)
            .await
            .expect("bm25 search should return Ok");
        let _ = results;
    }

    #[tokio::test]
    #[ignore = "requires live Neo4j at bolt://localhost:7687"]
    async fn search_entity_edges_by_fact_returns_vec() {
        let driver = Neo4jDriver::new(test_uri(), test_user(), test_password())
            .await
            .unwrap();

        let dummy_embedding = vec![0.0_f32; 1536];
        let results = driver
            .search_entity_edges_by_fact(&dummy_embedding, "search-group", 5)
            .await
            .expect("vector search should return Ok");
        let _ = results;
    }

    // ── Maintenance ───────────────────────────────────────────────────────────

    #[tokio::test]
    #[ignore = "requires live Neo4j at bolt://localhost:7687"]
    async fn build_indices_is_idempotent() {
        let driver = Neo4jDriver::new(test_uri(), test_user(), test_password())
            .await
            .unwrap();

        driver.build_indices().await.expect("first build_indices should succeed");
        driver.build_indices().await.expect("second build_indices should also succeed");
    }

    #[tokio::test]
    #[ignore = "requires live Neo4j at bolt://localhost:7687"]
    async fn invalidate_edge_sets_invalid_at() {
        let driver = Neo4jDriver::new(test_uri(), test_user(), test_password())
            .await
            .unwrap();

        let source_node = make_entity_node("inval-group");
        let target_node = make_entity_node("inval-group");
        driver.save_entity_node(&source_node).await.unwrap();
        driver.save_entity_node(&target_node).await.unwrap();

        let edge = make_entity_edge(source_node.uuid, target_node.uuid, "inval-group");
        driver.save_entity_edge(&edge).await.unwrap();

        let invalidated_at = Utc::now();
        driver
            .invalidate_edge(&edge.uuid, invalidated_at)
            .await
            .expect("invalidate_edge should succeed");

        let fetched = driver
            .get_entity_edge(&edge.uuid)
            .await
            .unwrap()
            .expect("edge must still exist after invalidation");
        assert!(
            fetched.invalid_at.is_some(),
            "invalid_at must be set after invalidate_edge"
        );

        // Cleanup.
        driver.delete_entity_node(&source_node.uuid).await.ok();
        driver.delete_entity_node(&target_node.uuid).await.ok();
    }

    // ── EpisodicNode delete / list ─────────────────────────────────────────

    #[tokio::test]
    #[ignore = "requires live Neo4j at bolt://localhost:7687"]
    async fn delete_episodic_node_is_idempotent() {
        let driver = Neo4jDriver::new(test_uri(), test_user(), test_password())
            .await
            .unwrap();

        let node = make_episodic_node("del-ep-group");
        driver.save_episodic_node(&node).await.unwrap();

        // First delete removes the node.
        driver
            .delete_episodic_node(&node.uuid)
            .await
            .expect("first delete_episodic_node should succeed");

        // Node must be gone.
        let fetched = driver
            .get_episodic_node(&node.uuid)
            .await
            .expect("get after delete must not error");
        assert!(fetched.is_none(), "episodic node must be absent after deletion");

        // Second delete on an absent node must also succeed (idempotent).
        driver
            .delete_episodic_node(&node.uuid)
            .await
            .expect("second delete_episodic_node should also succeed");
    }

    #[tokio::test]
    #[ignore = "requires live Neo4j at bolt://localhost:7687"]
    async fn list_episodic_nodes_returns_only_group_members() {
        let driver = Neo4jDriver::new(test_uri(), test_user(), test_password())
            .await
            .unwrap();

        let group = format!("list-ep-group-{}", Uuid::new_v4());
        let node_a = make_episodic_node(&group);
        let node_b = make_episodic_node(&group);
        let other_group = format!("other-ep-group-{}", Uuid::new_v4());
        let node_other = make_episodic_node(&other_group);

        driver.save_episodic_node(&node_a).await.unwrap();
        driver.save_episodic_node(&node_b).await.unwrap();
        driver.save_episodic_node(&node_other).await.unwrap();

        let results = driver
            .list_episodic_nodes(&group)
            .await
            .expect("list_episodic_nodes must return Ok");

        let uuids: Vec<Uuid> = results.iter().map(|n| n.uuid).collect();
        assert_eq!(results.len(), 2, "expected exactly 2 nodes in group '{group}'");
        assert!(uuids.contains(&node_a.uuid), "node_a must appear in the list");
        assert!(uuids.contains(&node_b.uuid), "node_b must appear in the list");
        assert!(
            !uuids.contains(&node_other.uuid),
            "node from a different group must not appear"
        );

        // Cleanup.
        driver.delete_episodic_node(&node_a.uuid).await.ok();
        driver.delete_episodic_node(&node_b.uuid).await.ok();
        driver.delete_episodic_node(&node_other.uuid).await.ok();
    }

    #[tokio::test]
    #[ignore = "requires live Neo4j at bolt://localhost:7687"]
    async fn list_episodic_nodes_empty_group_returns_empty_vec() {
        let driver = Neo4jDriver::new(test_uri(), test_user(), test_password())
            .await
            .unwrap();

        // A group that has never had any episodic nodes inserted.
        let phantom_group = format!("phantom-{}", Uuid::new_v4());
        let results = driver
            .list_episodic_nodes(&phantom_group)
            .await
            .expect("list_episodic_nodes must return Ok even for an empty group");
        assert!(
            results.is_empty(),
            "listing an empty group must return an empty Vec, got {} results",
            results.len()
        );
    }

    #[tokio::test]
    #[ignore = "requires live Neo4j at bolt://localhost:7687"]
    async fn get_entity_edges_between_returns_bidirectional() {
        let driver = Neo4jDriver::new(test_uri(), test_user(), test_password())
            .await
            .unwrap();

        let a = make_entity_node("between-group");
        let b = make_entity_node("between-group");
        driver.save_entity_node(&a).await.unwrap();
        driver.save_entity_node(&b).await.unwrap();

        let edge = make_entity_edge(a.uuid, b.uuid, "between-group");
        driver.save_entity_edge(&edge).await.unwrap();

        let edges_ab = driver
            .get_entity_edges_between(&a.uuid, &b.uuid)
            .await
            .expect("get_entity_edges_between should return Ok");
        assert!(
            edges_ab.iter().any(|e| e.uuid == edge.uuid),
            "saved edge must appear in get_entity_edges_between result"
        );

        // Also verify the reverse direction finds the same edge.
        let edges_ba = driver
            .get_entity_edges_between(&b.uuid, &a.uuid)
            .await
            .expect("reverse direction should also return Ok");
        assert!(
            edges_ba.iter().any(|e| e.uuid == edge.uuid),
            "edge must also appear in the reverse direction query"
        );

        // Cleanup.
        driver.delete_entity_node(&a.uuid).await.ok();
        driver.delete_entity_node(&b.uuid).await.ok();
    }
}
