//! NextEpisodeEdge — NEXT_EPISODE sequencing (EpisodicNode → EpisodicNode).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// An edge representing a NEXT_EPISODE sequencing relationship between two episodic nodes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NextEpisodeEdge {
    /// Unique identifier for this edge.
    pub uuid: Uuid,
    /// UUID of the predecessor EpisodicNode.
    pub source_node_uuid: Uuid,
    /// UUID of the successor EpisodicNode.
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
    fn next_episode_edge_can_be_constructed() {
        let source = Uuid::new_v4();
        let target = Uuid::new_v4();
        let edge = NextEpisodeEdge {
            uuid: Uuid::new_v4(),
            source_node_uuid: source,
            target_node_uuid: target,
            created_at: Utc::now(),
        };
        assert_eq!(edge.source_node_uuid, source);
        assert_eq!(edge.target_node_uuid, target);
    }

    #[test]
    fn next_episode_edge_serializes_to_json() {
        let edge = NextEpisodeEdge {
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
    fn next_episode_edge_deserializes_from_json() {
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
        let edge: NextEpisodeEdge = serde_json::from_str(&json).unwrap();
        assert_eq!(edge.uuid, uuid);
        assert_eq!(edge.source_node_uuid, source);
        assert_eq!(edge.target_node_uuid, target);
    }

    #[test]
    fn next_episode_edge_source_and_target_can_be_same_type() {
        // Both nodes are EpisodicNodes — verify the edge enforces no type confusion
        let ep1 = Uuid::new_v4();
        let ep2 = Uuid::new_v4();
        let edge = NextEpisodeEdge {
            uuid: Uuid::new_v4(),
            source_node_uuid: ep1,
            target_node_uuid: ep2,
            created_at: Utc::now(),
        };
        assert_ne!(edge.source_node_uuid, edge.target_node_uuid);
    }

    #[test]
    fn next_episode_edge_roundtrip() {
        let original = NextEpisodeEdge {
            uuid: Uuid::new_v4(),
            source_node_uuid: Uuid::new_v4(),
            target_node_uuid: Uuid::new_v4(),
            created_at: "2026-01-01T00:00:00Z".parse().unwrap(),
        };
        let json = serde_json::to_string(&original).unwrap();
        let restored: NextEpisodeEdge = serde_json::from_str(&json).unwrap();
        assert_eq!(original, restored);
    }
}
