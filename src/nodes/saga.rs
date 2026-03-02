//! SagaNode — represents a grouping/sequence of episodes.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A grouping or sequence of related episodes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SagaNode {
    pub uuid: Uuid,
    pub name: String,
    pub group_id: String,
    pub labels: Vec<String>,
    pub created_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::SagaNode;
    use chrono::Utc;
    use uuid::Uuid;

    #[test]
    fn test_saga_node_fields() {
        let now = Utc::now();
        let uuid = Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap();
        let node = SagaNode {
            uuid,
            name: "Test Saga".to_string(),
            group_id: "group-abc".to_string(),
            labels: vec!["Saga".to_string(), "Episode".to_string()],
            created_at: now,
        };

        assert_eq!(node.uuid, uuid);
        assert_eq!(node.name, "Test Saga");
        assert_eq!(node.group_id, "group-abc");
        assert_eq!(node.labels.len(), 2);
        assert_eq!(node.labels[0], "Saga");
    }

    #[test]
    fn test_saga_node_clone() {
        let now = Utc::now();
        let node = SagaNode {
            uuid: Uuid::new_v4(),
            name: "My Saga".to_string(),
            group_id: "grp-1".to_string(),
            labels: vec!["Saga".to_string()],
            created_at: now,
        };
        let cloned = node.clone();
        assert_eq!(cloned.uuid, node.uuid);
        assert_eq!(cloned.name, node.name);
    }

    #[test]
    fn test_saga_node_debug() {
        let now = Utc::now();
        let node = SagaNode {
            uuid: Uuid::new_v4(),
            name: "Debug Saga".to_string(),
            group_id: "grp-debug".to_string(),
            labels: vec![],
            created_at: now,
        };
        let debug_str = format!("{node:?}");
        assert!(debug_str.contains("Debug Saga"));
    }

    #[test]
    fn test_saga_node_serde_roundtrip() {
        let now = Utc::now();
        let node = SagaNode {
            uuid: Uuid::new_v4(),
            name: "Serde Saga".to_string(),
            group_id: "grp-serde".to_string(),
            labels: vec!["Saga".to_string()],
            created_at: now,
        };

        let json = serde_json::to_string(&node).expect("serialization failed");
        assert!(json.contains("Serde Saga"));

        let deserialized: SagaNode =
            serde_json::from_str(&json).expect("deserialization failed");
        assert_eq!(deserialized.uuid, node.uuid);
        assert_eq!(deserialized.name, node.name);
        assert_eq!(deserialized.group_id, node.group_id);
        assert_eq!(deserialized.labels, node.labels);
    }

    #[test]
    fn test_saga_node_empty_labels() {
        let now = Utc::now();
        let node = SagaNode {
            uuid: Uuid::new_v4(),
            name: "Empty Labels Saga".to_string(),
            group_id: "grp-empty".to_string(),
            labels: vec![],
            created_at: now,
        };
        assert!(node.labels.is_empty());
    }
}
