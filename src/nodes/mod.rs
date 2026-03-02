//! Node types for the knowledge graph.
//!
//! Mirrors the Python `nodes.py` module. Four node types:
//! - [`EntityNode`] — real-world entities (people, places, concepts)
//! - [`EpisodicNode`] — ingested data episodes (messages, documents, JSON records)
//! - [`CommunityNode`] — detected community clusters
//! - [`SagaNode`] — episode groupings / sequences

pub mod community;
pub mod entity;
pub mod episodic;
pub mod saga;

pub use community::CommunityNode;
pub use entity::EntityNode;
pub use episodic::{EpisodeType, EpisodicNode};
pub use saga::SagaNode;

use std::fmt;

/// A discriminant enum that wraps any of the four node types.
#[derive(Debug, Clone, PartialEq)]
pub enum NodeKind {
    Entity(EntityNode),
    Episodic(EpisodicNode),
    Community(CommunityNode),
    Saga(SagaNode),
}

impl fmt::Display for NodeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NodeKind::Entity(_) => write!(f, "Entity"),
            NodeKind::Episodic(_) => write!(f, "Episodic"),
            NodeKind::Community(_) => write!(f, "Community"),
            NodeKind::Saga(_) => write!(f, "Saga"),
        }
    }
}

impl From<EntityNode> for NodeKind {
    fn from(node: EntityNode) -> Self {
        NodeKind::Entity(node)
    }
}

impl From<EpisodicNode> for NodeKind {
    fn from(node: EpisodicNode) -> Self {
        NodeKind::Episodic(node)
    }
}

impl From<CommunityNode> for NodeKind {
    fn from(node: CommunityNode) -> Self {
        NodeKind::Community(node)
    }
}

impl From<SagaNode> for NodeKind {
    fn from(node: SagaNode) -> Self {
        NodeKind::Saga(node)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use uuid::Uuid;

    fn make_entity() -> EntityNode {
        EntityNode {
            uuid: Uuid::nil(),
            name: "Alice".to_string(),
            group_id: "g1".to_string(),
            labels: vec![],
            summary: String::new(),
            name_embedding: None,
            attributes: serde_json::Value::Null,
            created_at: Utc::now(),
        }
    }

    fn make_episodic() -> EpisodicNode {
        EpisodicNode {
            uuid: Uuid::new_v4(),
            name: "ep".to_string(),
            group_id: "g1".to_string(),
            labels: vec![],
            created_at: Utc::now(),
            source: EpisodeType::Text,
            source_description: String::new(),
            content: String::new(),
            valid_at: Utc::now(),
            entity_edges: vec![],
        }
    }

    fn make_community() -> CommunityNode {
        CommunityNode {
            uuid: Uuid::nil(),
            name: "Cluster".to_string(),
            name_embedding: None,
            summary: String::new(),
            created_at: Utc::now(),
        }
    }

    fn make_saga() -> SagaNode {
        SagaNode {
            uuid: Uuid::new_v4(),
            name: "MySaga".to_string(),
            group_id: "g1".to_string(),
            labels: vec![],
            created_at: Utc::now(),
        }
    }

    #[test]
    fn node_kind_display_entity() {
        let kind = NodeKind::from(make_entity());
        assert_eq!(kind.to_string(), "Entity");
    }

    #[test]
    fn node_kind_display_episodic() {
        let kind = NodeKind::from(make_episodic());
        assert_eq!(kind.to_string(), "Episodic");
    }

    #[test]
    fn node_kind_display_community() {
        let kind = NodeKind::from(make_community());
        assert_eq!(kind.to_string(), "Community");
    }

    #[test]
    fn node_kind_display_saga() {
        let kind = NodeKind::from(make_saga());
        assert_eq!(kind.to_string(), "Saga");
    }

    #[test]
    fn node_kind_from_entity() {
        let node = make_entity();
        let kind: NodeKind = node.clone().into();
        assert!(matches!(kind, NodeKind::Entity(_)));
    }

    #[test]
    fn node_kind_from_episodic() {
        let node = make_episodic();
        let kind: NodeKind = node.into();
        assert!(matches!(kind, NodeKind::Episodic(_)));
    }

    #[test]
    fn node_kind_from_community() {
        let node = make_community();
        let kind: NodeKind = node.into();
        assert!(matches!(kind, NodeKind::Community(_)));
    }

    #[test]
    fn node_kind_from_saga() {
        let node = make_saga();
        let kind: NodeKind = node.into();
        assert!(matches!(kind, NodeKind::Saga(_)));
    }

    #[test]
    fn node_kind_debug_contains_variant_name() {
        let kind = NodeKind::from(make_saga());
        let debug = format!("{kind:?}");
        assert!(debug.contains("Saga"));
    }
}
