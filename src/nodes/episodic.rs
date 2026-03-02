//! EpisodicNode — represents an ingested data episode.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// The source type of an episode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EpisodeType {
    Message,
    Json,
    Text,
}

/// An ingested data episode (message, document, JSON record).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpisodicNode {
    pub uuid: Uuid,
    pub name: String,
    pub group_id: String,
    pub labels: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub source: EpisodeType,
    pub source_description: String,
    pub content: String,
    pub valid_at: DateTime<Utc>,
    pub entity_edges: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::{EpisodeType, EpisodicNode};
    use uuid::Uuid;

    /// EpisodeType::Message round-trips through JSON.
    #[test]
    fn test_episode_type_message_serde_roundtrip() {
        let original = EpisodeType::Message;
        let json = serde_json::to_string(&original).expect("serialize EpisodeType::Message");
        let deserialized: EpisodeType =
            serde_json::from_str(&json).expect("deserialize EpisodeType::Message");
        // Use Debug equality via format — PartialEq will be derived
        assert_eq!(format!("{original:?}"), format!("{deserialized:?}"));
    }

    /// EpisodeType::Json round-trips through JSON.
    #[test]
    fn test_episode_type_json_serde_roundtrip() {
        let original = EpisodeType::Json;
        let json = serde_json::to_string(&original).expect("serialize EpisodeType::Json");
        let deserialized: EpisodeType =
            serde_json::from_str(&json).expect("deserialize EpisodeType::Json");
        assert_eq!(format!("{original:?}"), format!("{deserialized:?}"));
    }

    /// EpisodeType::Text round-trips through JSON.
    #[test]
    fn test_episode_type_text_serde_roundtrip() {
        let original = EpisodeType::Text;
        let json = serde_json::to_string(&original).expect("serialize EpisodeType::Text");
        let deserialized: EpisodeType =
            serde_json::from_str(&json).expect("deserialize EpisodeType::Text");
        assert_eq!(format!("{original:?}"), format!("{deserialized:?}"));
    }

    /// EpisodicNode serializes and deserializes without data loss.
    #[test]
    fn test_episodic_node_serde_roundtrip() {
        let node = EpisodicNode {
            uuid: Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap(),
            name: "test episode".to_string(),
            group_id: "group-1".to_string(),
            labels: vec!["EpisodicNode".to_string()],
            created_at: chrono::DateTime::parse_from_rfc3339("2024-01-15T10:30:00Z")
                .expect("parse created_at")
                .with_timezone(&chrono::Utc),
            source: EpisodeType::Message,
            source_description: "user chat message".to_string(),
            content: "Hello, world!".to_string(),
            valid_at: chrono::DateTime::parse_from_rfc3339("2024-01-15T10:30:00Z")
                .expect("parse valid_at")
                .with_timezone(&chrono::Utc),
            entity_edges: vec!["edge-uuid-1".to_string()],
        };

        let json = serde_json::to_string(&node).expect("serialize EpisodicNode");
        let restored: EpisodicNode =
            serde_json::from_str(&json).expect("deserialize EpisodicNode");

        assert_eq!(node.uuid, restored.uuid);
        assert_eq!(node.name, restored.name);
        assert_eq!(node.group_id, restored.group_id);
        assert_eq!(node.content, restored.content);
        assert_eq!(node.source_description, restored.source_description);
        assert_eq!(node.entity_edges, restored.entity_edges);
        assert_eq!(node.created_at, restored.created_at);
        assert_eq!(node.valid_at, restored.valid_at);
    }

    /// EpisodicNode with EpisodeType::Json source serializes correctly.
    #[test]
    fn test_episodic_node_json_source() {
        let node = EpisodicNode {
            uuid: Uuid::new_v4(),
            name: "json doc".to_string(),
            group_id: "grp".to_string(),
            labels: vec![],
            created_at: chrono::Utc::now(),
            source: EpisodeType::Json,
            source_description: "API response".to_string(),
            content: r#"{"key": "value"}"#.to_string(),
            valid_at: chrono::Utc::now(),
            entity_edges: vec![],
        };

        let json = serde_json::to_string(&node).expect("serialize");
        // The serialized form must contain the source type discriminant
        assert!(json.contains("Json") || json.contains("json"));
    }

    /// EpisodicNode with empty entity_edges serializes to an empty array.
    #[test]
    fn test_episodic_node_empty_entity_edges() {
        let node = EpisodicNode {
            uuid: Uuid::new_v4(),
            name: "episode".to_string(),
            group_id: "grp".to_string(),
            labels: vec![],
            created_at: chrono::Utc::now(),
            source: EpisodeType::Text,
            source_description: "plain text".to_string(),
            content: "Some content".to_string(),
            valid_at: chrono::Utc::now(),
            entity_edges: vec![],
        };

        let json = serde_json::to_value(&node).expect("serialize to Value");
        assert!(
            json["entity_edges"].as_array().expect("entity_edges is array").is_empty(),
            "entity_edges should be an empty array"
        );
    }

    /// EpisodeType variants are distinct — Message != Json != Text.
    #[test]
    fn test_episode_type_variants_distinct() {
        let message_json =
            serde_json::to_string(&EpisodeType::Message).expect("serialize Message");
        let json_json = serde_json::to_string(&EpisodeType::Json).expect("serialize Json");
        let text_json = serde_json::to_string(&EpisodeType::Text).expect("serialize Text");

        assert_ne!(message_json, json_json);
        assert_ne!(json_json, text_json);
        assert_ne!(message_json, text_json);
    }
}
