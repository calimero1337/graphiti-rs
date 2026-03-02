//! EntityNode — represents a real-world entity extracted from episodes.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A real-world entity (person, place, concept) extracted from episodes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityNode {
    pub uuid: Uuid,
    pub name: String,
    pub group_id: String,
    pub labels: Vec<String>,
    pub summary: String,
    pub name_embedding: Option<Vec<f32>>,
    pub attributes: serde_json::Value,
    pub created_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use serde_json::json;
    use uuid::Uuid;

    /// Verify that EntityNode can be constructed with all required fields.
    #[test]
    fn test_entity_node_construction() {
        let node = EntityNode {
            uuid: Uuid::new_v4(),
            name: "Alice".to_string(),
            group_id: "test-group".to_string(),
            labels: vec!["Person".to_string()],
            summary: "Alice is a software engineer.".to_string(),
            name_embedding: None,
            attributes: serde_json::Value::Object(Default::default()),
            created_at: Utc::now(),
        };
        assert_eq!(node.name, "Alice");
        assert_eq!(node.group_id, "test-group");
        assert_eq!(node.summary, "Alice is a software engineer.");
        assert!(node.name_embedding.is_none());
    }

    /// Verify that EntityNode derives Clone and Debug.
    #[test]
    fn test_entity_node_clone_debug() {
        let node = EntityNode {
            uuid: Uuid::nil(),
            name: "Bob".to_string(),
            group_id: "g1".to_string(),
            labels: vec!["Organization".to_string()],
            summary: "Bob Corp".to_string(),
            name_embedding: Some(vec![0.1_f32, 0.2, 0.3]),
            attributes: json!({"founded": 2020}),
            created_at: Utc::now(),
        };
        let cloned = node.clone();
        assert_eq!(node.name, cloned.name);
        // Debug must compile
        let _ = format!("{:?}", &node);
    }

    /// Verify round-trip JSON serialization / deserialization.
    #[test]
    fn test_entity_node_serde_roundtrip() {
        let uuid = Uuid::new_v4();
        let now = chrono::DateTime::parse_from_rfc3339("2024-06-01T12:00:00Z")
            .unwrap()
            .with_timezone(&Utc);

        let node = EntityNode {
            uuid,
            name: "Acme Corp".to_string(),
            group_id: "corp-group".to_string(),
            labels: vec!["Organization".to_string(), "Company".to_string()],
            summary: "A fictional company.".to_string(),
            name_embedding: Some(vec![0.5_f32, 0.5]),
            attributes: json!({"industry": "technology", "employees": 500}),
            created_at: now,
        };

        let serialized = serde_json::to_string(&node).expect("serialization failed");
        let deserialized: EntityNode =
            serde_json::from_str(&serialized).expect("deserialization failed");

        assert_eq!(deserialized.uuid, node.uuid);
        assert_eq!(deserialized.name, node.name);
        assert_eq!(deserialized.group_id, node.group_id);
        assert_eq!(deserialized.labels, node.labels);
        assert_eq!(deserialized.summary, node.summary);
        assert_eq!(deserialized.name_embedding, node.name_embedding);
        assert_eq!(deserialized.created_at, node.created_at);
    }

    /// Verify that EntityNode deserializes from a raw JSON literal.
    #[test]
    fn test_entity_node_deserialize_from_json() {
        let raw = json!({
            "uuid": "00000000-0000-0000-0000-000000000001",
            "name": "Eve",
            "group_id": "grp",
            "labels": ["Person"],
            "summary": "Eve is a cryptographer.",
            "name_embedding": null,
            "attributes": {},
            "created_at": "2024-01-01T00:00:00Z"
        });

        let node: EntityNode =
            serde_json::from_value(raw).expect("deserialization from JSON value failed");
        assert_eq!(node.name, "Eve");
        assert!(node.name_embedding.is_none());
    }

    /// Verify that name_embedding with a populated vector survives serialization.
    #[test]
    fn test_entity_node_name_embedding_roundtrip() {
        let embedding = vec![0.1_f32, 0.2, 0.3, 0.4, 0.5];
        let node = EntityNode {
            uuid: Uuid::new_v4(),
            name: "Concept".to_string(),
            group_id: "g2".to_string(),
            labels: vec![],
            summary: "An abstract concept.".to_string(),
            name_embedding: Some(embedding.clone()),
            attributes: serde_json::Value::Null,
            created_at: Utc::now(),
        };

        let json_str = serde_json::to_string(&node).expect("serialization failed");
        let recovered: EntityNode =
            serde_json::from_str(&json_str).expect("deserialization failed");

        assert_eq!(recovered.name_embedding, Some(embedding));
    }

    /// Verify that PartialEq compares EntityNodes by uuid.
    #[test]
    fn test_entity_node_partial_eq() {
        let uuid = Uuid::new_v4();
        let now = Utc::now();

        let a = EntityNode {
            uuid,
            name: "Same UUID node v1".to_string(),
            group_id: "g".to_string(),
            labels: vec![],
            summary: String::new(),
            name_embedding: None,
            attributes: serde_json::Value::Null,
            created_at: now,
        };
        let b = EntityNode {
            uuid,
            name: "Same UUID node v2".to_string(),
            group_id: "g".to_string(),
            labels: vec![],
            summary: String::new(),
            name_embedding: None,
            attributes: serde_json::Value::Null,
            created_at: now,
        };
        // Two nodes with the same UUID should compare equal (field-by-field PartialEq).
        // This also exercises that PartialEq is derived.
        let c = a.clone();
        assert_eq!(a, c);
        // b has the same uuid but different name, so they are NOT equal under derived PartialEq.
        assert_ne!(a, b);
    }
}
