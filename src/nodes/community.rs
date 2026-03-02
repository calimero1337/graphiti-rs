//! CommunityNode — represents a detected community cluster.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A detected community cluster of related entities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommunityNode {
    pub uuid: Uuid,
    pub name: String,
    pub name_embedding: Option<Vec<f32>>,
    pub summary: String,
    pub created_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use uuid::Uuid;

    #[test]
    fn community_node_can_be_constructed() {
        let node = CommunityNode {
            uuid: Uuid::new_v4(),
            name: "AI Researchers".to_string(),
            name_embedding: Some(vec![0.1_f32, 0.2, 0.3]),
            summary: "A cluster of AI researchers".to_string(),
            created_at: Utc::now(),
        };
        assert_eq!(node.name, "AI Researchers");
        assert_eq!(node.summary, "A cluster of AI researchers");
        assert!(node.name_embedding.is_some());
    }

    #[test]
    fn community_node_serializes_to_json() {
        let uuid = Uuid::new_v4();
        let node = CommunityNode {
            uuid,
            name: "Test Community".to_string(),
            name_embedding: None,
            summary: "Summary text".to_string(),
            created_at: Utc::now(),
        };
        let json = serde_json::to_string(&node).unwrap();
        assert!(json.contains("Test Community"));
        assert!(json.contains("Summary text"));
    }

    #[test]
    fn community_node_deserializes_from_json() {
        let uuid = Uuid::new_v4();
        let json = format!(
            r#"{{
                "uuid": "{uuid}",
                "name": "Deserialized Community",
                "name_embedding": [0.5, 0.6],
                "summary": "Deserialized summary",
                "created_at": "2026-01-01T00:00:00Z"
            }}"#
        );
        let node: CommunityNode = serde_json::from_str(&json).unwrap();
        assert_eq!(node.uuid, uuid);
        assert_eq!(node.name, "Deserialized Community");
        assert_eq!(node.name_embedding, Some(vec![0.5_f32, 0.6]));
    }

    #[test]
    fn community_node_name_embedding_optional() {
        let node = CommunityNode {
            uuid: Uuid::new_v4(),
            name: "No Embedding".to_string(),
            name_embedding: None,
            summary: String::new(),
            created_at: Utc::now(),
        };
        assert!(node.name_embedding.is_none());
    }
}
