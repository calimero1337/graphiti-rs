//! EntityEdge — bi-temporal factual relationship between EntityNodes.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A factual relationship between two entity nodes, with bi-temporal metadata.
///
/// - **Valid time** (`valid_at` / `invalid_at`): when the fact was true in the real world.
/// - **Transaction time** (`created_at` / `expired_at`): when the edge exists in the graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityEdge {
    /// Unique identifier for this edge.
    pub uuid: Uuid,
    /// UUID of the source EntityNode.
    pub source_node_uuid: Uuid,
    /// UUID of the target EntityNode.
    pub target_node_uuid: Uuid,
    /// Relationship label (e.g. "KNOWS", "WORKS_AT").
    pub name: String,
    /// Human-readable fact string.
    pub fact: String,
    /// Optional embedding vector for the fact.
    pub fact_embedding: Option<Vec<f32>>,
    /// Episode UUIDs that mention this fact.
    pub episodes: Vec<Uuid>,
    /// When the fact became true in the real world (valid-time start).
    pub valid_at: Option<DateTime<Utc>>,
    /// When the fact ceased to be true in the real world (valid-time end).
    pub invalid_at: Option<DateTime<Utc>>,
    /// When this edge was created in the graph (transaction-time start).
    pub created_at: DateTime<Utc>,
    /// When this edge was superseded in the graph (transaction-time end).
    pub expired_at: Option<DateTime<Utc>>,
    /// Relevance weight (default 1.0).
    pub weight: f64,
    /// Arbitrary JSON attributes.
    pub attributes: serde_json::Value,
    /// Optional group / partition identifier.
    pub group_id: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use uuid::Uuid;

    fn make_edge() -> EntityEdge {
        EntityEdge {
            uuid: Uuid::new_v4(),
            source_node_uuid: Uuid::new_v4(),
            target_node_uuid: Uuid::new_v4(),
            name: "KNOWS".into(),
            fact: "Alice knows Bob".into(),
            fact_embedding: None,
            episodes: vec![],
            valid_at: None,
            invalid_at: None,
            created_at: "2026-01-01T00:00:00Z".parse().unwrap(),
            expired_at: None,
            weight: 1.0,
            attributes: serde_json::Value::Null,
            group_id: None,
        }
    }

    #[test]
    fn entity_edge_can_be_constructed() {
        let source = Uuid::new_v4();
        let target = Uuid::new_v4();
        let ep1 = Uuid::new_v4();
        let edge = EntityEdge {
            uuid: Uuid::new_v4(),
            source_node_uuid: source,
            target_node_uuid: target,
            name: "WORKS_AT".into(),
            fact: "Alice works at Acme".into(),
            fact_embedding: Some(vec![0.1, 0.2, 0.3]),
            episodes: vec![ep1],
            valid_at: Some(Utc::now()),
            invalid_at: None,
            created_at: Utc::now(),
            expired_at: None,
            weight: 1.0,
            attributes: serde_json::json!({"confidence": 0.9}),
            group_id: Some("g1".into()),
        };
        assert_eq!(edge.source_node_uuid, source);
        assert_eq!(edge.target_node_uuid, target);
        assert_eq!(edge.episodes, vec![ep1]);
        assert_eq!(edge.name, "WORKS_AT");
    }

    #[test]
    fn entity_edge_all_optional_fields_are_none() {
        let edge = make_edge();
        assert!(edge.fact_embedding.is_none());
        assert!(edge.valid_at.is_none());
        assert!(edge.invalid_at.is_none());
        assert!(edge.expired_at.is_none());
        assert!(edge.group_id.is_none());
        assert!(edge.episodes.is_empty());
    }

    #[test]
    fn entity_edge_episodes_stores_multiple_uuids() {
        let ep1 = Uuid::new_v4();
        let ep2 = Uuid::new_v4();
        let ep3 = Uuid::new_v4();
        let mut edge = make_edge();
        edge.episodes = vec![ep1, ep2, ep3];
        assert_eq!(edge.episodes.len(), 3);
        assert!(edge.episodes.contains(&ep1));
        assert!(edge.episodes.contains(&ep2));
        assert!(edge.episodes.contains(&ep3));
    }

    #[test]
    fn entity_edge_serializes_to_json() {
        let edge = make_edge();
        let json = serde_json::to_string(&edge).unwrap();
        assert!(json.contains("source_node_uuid"));
        assert!(json.contains("target_node_uuid"));
        assert!(json.contains("created_at"));
        assert!(json.contains("episodes"));
        assert!(json.contains("weight"));
    }

    #[test]
    fn entity_edge_roundtrip() {
        let ep1 = Uuid::new_v4();
        let original = EntityEdge {
            uuid: Uuid::new_v4(),
            source_node_uuid: Uuid::new_v4(),
            target_node_uuid: Uuid::new_v4(),
            name: "KNOWS".into(),
            fact: "Alice knows Bob".into(),
            fact_embedding: Some(vec![0.1_f32, 0.5_f32, 0.9_f32]),
            episodes: vec![ep1],
            valid_at: Some("2025-01-01T00:00:00Z".parse().unwrap()),
            invalid_at: Some("2025-06-01T00:00:00Z".parse().unwrap()),
            created_at: "2026-01-01T00:00:00Z".parse().unwrap(),
            expired_at: None,
            weight: 0.75,
            attributes: serde_json::json!({"key": "value"}),
            group_id: Some("grp-1".into()),
        };
        let json = serde_json::to_string(&original).unwrap();
        let restored: EntityEdge = serde_json::from_str(&json).unwrap();
        assert_eq!(original, restored);
    }

    // --- Bi-temporal invariant tests ---

    #[test]
    fn expired_edge_preserves_all_fields_through_roundtrip() {
        // An edge with expired_at = Some(_) must never be deleted — only marked expired.
        // Verify that serde roundtrip preserves every field including expired_at.
        let ep = Uuid::new_v4();
        let original = EntityEdge {
            uuid: Uuid::new_v4(),
            source_node_uuid: Uuid::new_v4(),
            target_node_uuid: Uuid::new_v4(),
            name: "LIKES".into(),
            fact: "Carol likes coffee".into(),
            fact_embedding: None,
            episodes: vec![ep],
            valid_at: Some("2024-01-01T00:00:00Z".parse().unwrap()),
            invalid_at: Some("2024-12-31T00:00:00Z".parse().unwrap()),
            created_at: "2024-01-01T00:00:00Z".parse().unwrap(),
            expired_at: Some("2025-01-01T00:00:00Z".parse().unwrap()),
            weight: 1.0,
            attributes: serde_json::Value::Null,
            group_id: None,
        };
        let json = serde_json::to_string(&original).unwrap();
        let restored: EntityEdge = serde_json::from_str(&json).unwrap();
        // Edge is preserved — not deleted
        assert_eq!(original, restored);
        // expired_at is still set
        assert!(restored.expired_at.is_some());
        // valid_at and invalid_at are also preserved independently
        assert!(restored.valid_at.is_some());
        assert!(restored.invalid_at.is_some());
        // fact content is preserved
        assert_eq!(restored.fact, "Carol likes coffee");
    }

    #[test]
    fn setting_expired_at_does_not_clear_valid_time_fields() {
        // The bi-temporal model requires valid_at/invalid_at and expired_at to be
        // independent axes. Expiring an edge must not affect valid-time data.
        let mut edge = make_edge();
        edge.valid_at = Some("2024-06-01T00:00:00Z".parse().unwrap());
        edge.invalid_at = Some("2024-12-01T00:00:00Z".parse().unwrap());
        let valid_at_before = edge.valid_at;
        let invalid_at_before = edge.invalid_at;

        // Expire the edge (transaction-time end)
        edge.expired_at = Some("2025-01-01T00:00:00Z".parse().unwrap());

        // Valid-time fields must be untouched
        assert_eq!(edge.valid_at, valid_at_before);
        assert_eq!(edge.invalid_at, invalid_at_before);
        // Transaction-time end is set
        assert!(edge.expired_at.is_some());
        // fact is unchanged
        assert_eq!(edge.fact, "Alice knows Bob");
    }
}
