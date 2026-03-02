//! CommunityEdge — HAS_MEMBER relationship (CommunityNode → EntityNode).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// An edge representing a HAS_MEMBER relationship from a community node to an entity node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommunityEdge {
    /// Unique identifier for this edge.
    pub uuid: Uuid,
    /// UUID of the source CommunityNode.
    pub source_node_uuid: Uuid,
    /// UUID of the target EntityNode.
    pub target_node_uuid: Uuid,
    /// When this edge was created in the graph.
    pub created_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use uuid::Uuid;

    #[test]
    fn community_edge_can_be_constructed() {
        let source = Uuid::new_v4();
        let target = Uuid::new_v4();
        let edge = CommunityEdge {
            uuid: Uuid::new_v4(),
            source_node_uuid: source,
            target_node_uuid: target,
            created_at: Utc::now(),
        };
        assert_eq!(edge.source_node_uuid, source);
        assert_eq!(edge.target_node_uuid, target);
    }

    #[test]
    fn community_edge_serializes_to_json() {
        let edge = CommunityEdge {
            uuid: Uuid::new_v4(),
            source_node_uuid: Uuid::new_v4(),
            target_node_uuid: Uuid::new_v4(),
            created_at: Utc::now(),
        };
        let json = serde_json::to_string(&edge).unwrap();
        assert!(json.contains("source_node_uuid"));
        assert!(json.contains("target_node_uuid"));
        assert!(json.contains("created_at"));
    }

    #[test]
    fn community_edge_deserializes_from_json() {
        let uuid = Uuid::new_v4();
        let source = Uuid::new_v4();
        let target = Uuid::new_v4();
        let json = format!(
            r#"{{
                "uuid": "{uuid}",
                "source_node_uuid": "{source}",
                "target_node_uuid": "{target}",
                "created_at": "2026-01-01T00:00:00Z"
            }}"#
        );
        let edge: CommunityEdge = serde_json::from_str(&json).unwrap();
        assert_eq!(edge.uuid, uuid);
        assert_eq!(edge.source_node_uuid, source);
        assert_eq!(edge.target_node_uuid, target);
    }

    #[test]
    fn community_edge_roundtrip() {
        let original = CommunityEdge {
            uuid: Uuid::new_v4(),
            source_node_uuid: Uuid::new_v4(),
            target_node_uuid: Uuid::new_v4(),
            created_at: "2026-01-01T00:00:00Z".parse().unwrap(),
        };
        let json = serde_json::to_string(&original).unwrap();
        let restored: CommunityEdge = serde_json::from_str(&json).unwrap();
        assert_eq!(original, restored);
    }
}
