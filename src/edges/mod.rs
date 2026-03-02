//! Edge types for the knowledge graph.
//!
//! Mirrors the Python `edges.py` module. Five edge types:
//! - [`EntityEdge`] — factual relationships between entities (bi-temporal)
//! - [`EpisodicEdge`] — MENTIONS relationships (episode → entity)
//! - [`CommunityEdge`] — HAS_MEMBER relationships (community → entity)
//! - [`HasEpisodeEdge`] — HAS_EPISODE relationships (saga → episode)
//! - [`NextEpisodeEdge`] — NEXT_EPISODE sequencing (episode → episode)

pub mod community;
pub mod entity;
pub mod episodic;
pub mod has_episode;
pub mod next_episode;

pub use community::CommunityEdge;
pub use entity::EntityEdge;
pub use episodic::EpisodicEdge;
pub use has_episode::HasEpisodeEdge;
pub use next_episode::NextEpisodeEdge;

use std::fmt;

/// A discriminant enum that wraps any of the five edge types.
#[derive(Debug, Clone, PartialEq)]
pub enum EdgeKind {
    Entity(EntityEdge),
    Episodic(EpisodicEdge),
    Community(CommunityEdge),
    HasEpisode(HasEpisodeEdge),
    NextEpisode(NextEpisodeEdge),
}

impl fmt::Display for EdgeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EdgeKind::Entity(_) => write!(f, "Entity"),
            EdgeKind::Episodic(_) => write!(f, "Episodic"),
            EdgeKind::Community(_) => write!(f, "Community"),
            EdgeKind::HasEpisode(_) => write!(f, "HasEpisode"),
            EdgeKind::NextEpisode(_) => write!(f, "NextEpisode"),
        }
    }
}

impl From<EntityEdge> for EdgeKind {
    fn from(edge: EntityEdge) -> Self {
        EdgeKind::Entity(edge)
    }
}

impl From<EpisodicEdge> for EdgeKind {
    fn from(edge: EpisodicEdge) -> Self {
        EdgeKind::Episodic(edge)
    }
}

impl From<CommunityEdge> for EdgeKind {
    fn from(edge: CommunityEdge) -> Self {
        EdgeKind::Community(edge)
    }
}

impl From<HasEpisodeEdge> for EdgeKind {
    fn from(edge: HasEpisodeEdge) -> Self {
        EdgeKind::HasEpisode(edge)
    }
}

impl From<NextEpisodeEdge> for EdgeKind {
    fn from(edge: NextEpisodeEdge) -> Self {
        EdgeKind::NextEpisode(edge)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------------------
    // Re-export surface: all five concrete types must be accessible via `edges::*`
    // ---------------------------------------------------------------------------
    #[test]
    fn reexport_entity_edge() {
        // `EntityEdge` must be re-exported at the `edges` crate level.
        let _ = std::any::TypeId::of::<entity::EntityEdge>();
        // This line requires `EntityEdge` to be available without the sub-module path.
        let _: &str = std::any::type_name::<EntityEdge>();
    }

    #[test]
    fn reexport_episodic_edge() {
        let _: &str = std::any::type_name::<EpisodicEdge>();
    }

    #[test]
    fn reexport_community_edge() {
        let _: &str = std::any::type_name::<CommunityEdge>();
    }

    #[test]
    fn reexport_has_episode_edge() {
        let _: &str = std::any::type_name::<HasEpisodeEdge>();
    }

    #[test]
    fn reexport_next_episode_edge() {
        let _: &str = std::any::type_name::<NextEpisodeEdge>();
    }

    // ---------------------------------------------------------------------------
    // EdgeKind::from — each concrete edge type must convert Into EdgeKind
    // ---------------------------------------------------------------------------
    #[test]
    fn edge_kind_from_entity_edge() {
        use chrono::Utc;
        use uuid::Uuid;
        let edge = entity::EntityEdge {
            uuid: Uuid::new_v4(),
            source_node_uuid: Uuid::new_v4(),
            target_node_uuid: Uuid::new_v4(),
            name: "KNOWS".into(),
            fact: "Alice knows Bob".into(),
            fact_embedding: None,
            episodes: vec![],
            valid_at: None,
            invalid_at: None,
            created_at: Utc::now(),
            expired_at: None,
            weight: 1.0,
            attributes: serde_json::Value::Null,
            group_id: None,
        };
        let kind = EdgeKind::from(edge);
        assert!(matches!(kind, EdgeKind::Entity(_)));
    }

    #[test]
    fn edge_kind_from_episodic_edge() {
        use chrono::Utc;
        use uuid::Uuid;
        let edge = episodic::EpisodicEdge {
            uuid: Uuid::new_v4(),
            source_node_uuid: Uuid::new_v4(),
            target_node_uuid: Uuid::new_v4(),
            created_at: Utc::now(),
        };
        let kind: EdgeKind = edge.into();
        assert!(matches!(kind, EdgeKind::Episodic(_)));
    }

    #[test]
    fn edge_kind_from_community_edge() {
        use chrono::Utc;
        use uuid::Uuid;
        let edge = community::CommunityEdge {
            uuid: Uuid::new_v4(),
            source_node_uuid: Uuid::new_v4(),
            target_node_uuid: Uuid::new_v4(),
            created_at: Utc::now(),
        };
        let kind: EdgeKind = edge.into();
        assert!(matches!(kind, EdgeKind::Community(_)));
    }

    #[test]
    fn edge_kind_from_has_episode_edge() {
        use chrono::Utc;
        use uuid::Uuid;
        let edge = has_episode::HasEpisodeEdge {
            uuid: Uuid::new_v4(),
            source_node_uuid: Uuid::new_v4(),
            target_node_uuid: Uuid::new_v4(),
            created_at: Utc::now(),
        };
        let kind: EdgeKind = edge.into();
        assert!(matches!(kind, EdgeKind::HasEpisode(_)));
    }

    #[test]
    fn edge_kind_from_next_episode_edge() {
        use chrono::Utc;
        use uuid::Uuid;
        let edge = next_episode::NextEpisodeEdge {
            uuid: Uuid::new_v4(),
            source_node_uuid: Uuid::new_v4(),
            target_node_uuid: Uuid::new_v4(),
            created_at: Utc::now(),
        };
        let kind: EdgeKind = edge.into();
        assert!(matches!(kind, EdgeKind::NextEpisode(_)));
    }

    // ---------------------------------------------------------------------------
    // Display: each variant must produce a stable, human-readable string
    // ---------------------------------------------------------------------------
    #[test]
    fn edge_kind_display_entity() {
        use chrono::Utc;
        use uuid::Uuid;
        let edge = entity::EntityEdge {
            uuid: Uuid::new_v4(),
            source_node_uuid: Uuid::new_v4(),
            target_node_uuid: Uuid::new_v4(),
            name: "KNOWS".into(),
            fact: "test".into(),
            fact_embedding: None,
            episodes: vec![],
            valid_at: None,
            invalid_at: None,
            created_at: Utc::now(),
            expired_at: None,
            weight: 1.0,
            attributes: serde_json::Value::Null,
            group_id: None,
        };
        let kind = EdgeKind::from(edge);
        let s = kind.to_string();
        assert_eq!(s, "Entity");
    }

    #[test]
    fn edge_kind_display_episodic() {
        use chrono::Utc;
        use uuid::Uuid;
        let edge = episodic::EpisodicEdge {
            uuid: Uuid::new_v4(),
            source_node_uuid: Uuid::new_v4(),
            target_node_uuid: Uuid::new_v4(),
            created_at: Utc::now(),
        };
        let kind: EdgeKind = edge.into();
        assert_eq!(kind.to_string(), "Episodic");
    }

    #[test]
    fn edge_kind_display_community() {
        use chrono::Utc;
        use uuid::Uuid;
        let edge = community::CommunityEdge {
            uuid: Uuid::new_v4(),
            source_node_uuid: Uuid::new_v4(),
            target_node_uuid: Uuid::new_v4(),
            created_at: Utc::now(),
        };
        let kind: EdgeKind = edge.into();
        assert_eq!(kind.to_string(), "Community");
    }

    #[test]
    fn edge_kind_display_has_episode() {
        use chrono::Utc;
        use uuid::Uuid;
        let edge = has_episode::HasEpisodeEdge {
            uuid: Uuid::new_v4(),
            source_node_uuid: Uuid::new_v4(),
            target_node_uuid: Uuid::new_v4(),
            created_at: Utc::now(),
        };
        let kind: EdgeKind = edge.into();
        assert_eq!(kind.to_string(), "HasEpisode");
    }

    #[test]
    fn edge_kind_display_next_episode() {
        use chrono::Utc;
        use uuid::Uuid;
        let edge = next_episode::NextEpisodeEdge {
            uuid: Uuid::new_v4(),
            source_node_uuid: Uuid::new_v4(),
            target_node_uuid: Uuid::new_v4(),
            created_at: Utc::now(),
        };
        let kind: EdgeKind = edge.into();
        assert_eq!(kind.to_string(), "NextEpisode");
    }
}
