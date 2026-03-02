//! Request and response types for the REST API.

use serde::{Deserialize, Serialize};

/// Request body for `POST /v1/episodes`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddEpisodeRequest {
    /// Human-readable episode name.
    pub name: String,
    /// Episode text content.
    pub content: String,
    /// Source type: one of `"text"`, `"message"`, or `"json"`.
    pub source_type: String,
    /// Tenant/partition group identifier.
    pub group_id: String,
    /// Human-readable description of the content source.
    pub source_description: String,
}

/// Response body for `POST /v1/episodes`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddEpisodeResponse {
    /// UUID of the newly created episode node.
    pub episode_id: String,
    /// Number of entity nodes created during this ingestion.
    pub nodes_created: usize,
    /// Number of entity edges created during this ingestion.
    pub edges_created: usize,
}

/// Request body for `POST /v1/search`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    /// Free-text search query.
    pub query: String,
    /// Tenant/partition filter — must contain at least one entry.
    pub group_ids: Vec<String>,
    /// Maximum number of results (defaults to 10 when absent).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
}

/// Query parameters for `GET /v1/episodes`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListEpisodesQuery {
    /// Tenant/partition group identifier (defaults to `"default"`).
    pub group_id: Option<String>,
    /// Maximum number of episodes to return (defaults to 20).
    pub limit: Option<usize>,
}

/// Request body for `POST /v1/communities/build`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildCommunitiesRequest {
    /// Group IDs to build community structure for.
    pub group_ids: Vec<String>,
}

/// Response body for `POST /v1/communities/build`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityResult {
    /// Number of communities built or updated.
    pub communities_built: usize,
}

/// Response body for `GET /v1/token-usage`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TokenUsageResponse {
    /// Total prompt tokens consumed.
    pub prompt_tokens: u64,
    /// Total completion tokens consumed.
    pub completion_tokens: u64,
    /// Total tokens (prompt + completion).
    pub total_tokens: u64,
}

// ── Contextualize types ──────────────────────────────────────────────────────

/// Request body for `POST /v1/contextualize`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualizeRequest {
    /// Natural-language query.
    pub query: String,
    /// Tenant/partition group identifiers.
    pub group_ids: Vec<String>,
    /// Maximum number of context entities to return (defaults to 10).
    #[serde(default = "default_context_limit")]
    pub limit: usize,
}

fn default_context_limit() -> usize {
    10
}

/// Response body for `POST /v1/contextualize`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualizeResponse {
    /// Matched entities with 1-hop relationships.
    pub entities: Vec<ContextEntity>,
    /// Warnings or advisory messages (e.g. feedback-related).
    pub warnings: Vec<String>,
}

/// A single entity in the contextualize response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextEntity {
    /// Entity name.
    pub name: String,
    /// Entity UUID.
    pub uuid: String,
    /// Entity summary.
    pub summary: String,
    /// 1-hop relationships from this entity.
    pub relationships: Vec<ContextRelationship>,
}

/// A relationship (edge) attached to a context entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextRelationship {
    /// Target entity name (may be empty if unavailable).
    pub target: String,
    /// Human-readable fact.
    pub fact: String,
    /// Whether the fact is still valid (invalid_at is None).
    pub valid: bool,
}

// ── Record Outcome types ─────────────────────────────────────────────────────

/// Request body for `POST /v1/outcomes`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordOutcomeRequest {
    /// Identifier of the task that produced this outcome.
    pub task_id: String,
    /// Agent that executed the task.
    pub agent: String,
    /// Entity names relevant to this outcome.
    pub entity_names: Vec<String>,
    /// Whether the task succeeded.
    pub success: bool,
    /// Free-text details about the outcome.
    pub details: String,
    /// Group ID to scope the outcome (defaults to "default").
    #[serde(default = "default_group")]
    pub group_id: String,
}

fn default_group() -> String {
    "default".into()
}

/// Response body for `POST /v1/outcomes`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordOutcomeResponse {
    /// UUID of the created episode.
    pub episode_id: String,
    /// Whether the outcome was successfully recorded.
    pub recorded: bool,
}

// ── Timeline types ───────────────────────────────────────────────────────────

/// Request body for `POST /v1/timeline`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetTimelineRequest {
    /// Entity name to get the timeline for.
    pub entity_name: String,
    /// Group ID (defaults to "default").
    #[serde(default = "default_group")]
    pub group_id: String,
    /// Maximum number of timeline entries to return.
    pub limit: Option<usize>,
}

/// A single entry in an entity timeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEntry {
    /// Human-readable fact.
    pub fact: String,
    /// When the fact became valid (ISO 8601).
    pub valid_at: Option<String>,
    /// When the fact became invalid (ISO 8601).
    pub invalid_at: Option<String>,
    /// Whether the fact is still valid.
    pub still_valid: bool,
}

/// Response body for `POST /v1/timeline`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetTimelineResponse {
    /// The entity name queried.
    pub entity_name: String,
    /// Chronological timeline entries.
    pub entries: Vec<TimelineEntry>,
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::{AddEpisodeRequest, AddEpisodeResponse, SearchRequest};

    #[test]
    fn add_episode_request_deserializes_full() {
        let json = r#"{
            "name": "ep-1",
            "content": "Alice is a software engineer.",
            "source_type": "text",
            "group_id": "default",
            "source_description": "unit test"
        }"#;
        let req: AddEpisodeRequest =
            serde_json::from_str(json).expect("should deserialize");
        assert_eq!(req.name, "ep-1");
        assert_eq!(req.content, "Alice is a software engineer.");
        assert_eq!(req.group_id, "default");
        assert_eq!(req.source_description, "unit test");
    }

    #[test]
    fn add_episode_request_missing_name_is_error() {
        let json = r#"{
            "content": "some content",
            "source_type": "text",
            "group_id": "default",
            "source_description": "test"
        }"#;
        let result: Result<AddEpisodeRequest, _> = serde_json::from_str(json);
        assert!(result.is_err(), "name is a required field");
    }

    #[test]
    fn add_episode_request_missing_content_is_error() {
        let json = r#"{
            "name": "ep-2",
            "source_type": "text",
            "group_id": "default",
            "source_description": "test"
        }"#;
        let result: Result<AddEpisodeRequest, _> = serde_json::from_str(json);
        assert!(result.is_err(), "content is a required field");
    }

    #[test]
    fn search_request_deserializes() {
        let json = r#"{
            "query": "Alice",
            "group_ids": ["default"],
            "limit": 10
        }"#;
        let req: SearchRequest = serde_json::from_str(json).expect("should deserialize");
        assert_eq!(req.query, "Alice");
        assert_eq!(req.group_ids, vec!["default".to_string()]);
        assert_eq!(req.limit, Some(10));
    }

    #[test]
    fn search_request_limit_is_optional() {
        let json = r#"{
            "query": "Bob",
            "group_ids": ["grp"]
        }"#;
        let req: SearchRequest = serde_json::from_str(json).expect("should deserialize");
        assert_eq!(req.query, "Bob");
        assert!(req.limit.is_none(), "limit should default to None");
    }

    #[test]
    fn search_request_missing_query_is_error() {
        let json = r#"{
            "group_ids": ["default"]
        }"#;
        let result: Result<SearchRequest, _> = serde_json::from_str(json);
        assert!(result.is_err(), "query is a required field");
    }

    #[test]
    fn add_episode_response_serializes_to_json() {
        let resp = AddEpisodeResponse {
            episode_id: uuid::Uuid::nil().to_string(),
            nodes_created: 3,
            edges_created: 2,
        };
        let json = serde_json::to_string(&resp).expect("should serialize");
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(v.get("episode_id").is_some(), "must have episode_id");
        assert_eq!(v["nodes_created"], 3);
        assert_eq!(v["edges_created"], 2);
    }

    #[test]
    fn add_episode_response_debug_and_clone() {
        let resp = AddEpisodeResponse {
            episode_id: "some-id".to_string(),
            nodes_created: 0,
            edges_created: 0,
        };
        let cloned = resp.clone();
        assert_eq!(cloned.episode_id, "some-id");
        let _ = format!("{:?}", &resp);
    }

    #[test]
    fn add_episode_request_source_type_roundtrip() {
        for source_type in &["text", "message", "json"] {
            let json = format!(
                r#"{{"name":"ep","content":"c","source_type":"{source_type}","group_id":"g","source_description":"d"}}"#
            );
            let req: AddEpisodeRequest =
                serde_json::from_str(&json).expect("should deserialize");
            let reser = serde_json::to_value(&req).expect("should re-serialize");
            assert_eq!(
                reser["source_type"].as_str().unwrap(),
                *source_type,
                "source_type must round-trip for {source_type}"
            );
        }
    }

    // ── Contextualize types tests ────────────────────────────────────────────

    #[test]
    fn contextualize_request_deserializes_full() {
        let json = r#"{
            "query": "What does Alice do?",
            "group_ids": ["default"],
            "limit": 5
        }"#;
        let req: ContextualizeRequest =
            serde_json::from_str(json).expect("should deserialize");
        assert_eq!(req.query, "What does Alice do?");
        assert_eq!(req.group_ids, vec!["default"]);
        assert_eq!(req.limit, 5);
    }

    #[test]
    fn contextualize_request_limit_defaults_to_10() {
        let json = r#"{
            "query": "test",
            "group_ids": ["g1"]
        }"#;
        let req: ContextualizeRequest =
            serde_json::from_str(json).expect("should deserialize");
        assert_eq!(req.limit, 10);
    }

    #[test]
    fn contextualize_response_serializes() {
        let resp = ContextualizeResponse {
            entities: vec![ContextEntity {
                name: "Alice".to_string(),
                uuid: "some-uuid".to_string(),
                summary: "A software engineer".to_string(),
                relationships: vec![ContextRelationship {
                    target: "Bob".to_string(),
                    fact: "Alice knows Bob".to_string(),
                    valid: true,
                }],
            }],
            warnings: vec![],
        };
        let json = serde_json::to_string(&resp).expect("should serialize");
        assert!(json.contains("Alice"));
        assert!(json.contains("Alice knows Bob"));
    }

    // ── Record Outcome types tests ───────────────────────────────────────────

    #[test]
    fn record_outcome_request_deserializes_full() {
        let json = r#"{
            "task_id": "task-123",
            "agent": "Sisko",
            "entity_names": ["Alice", "Bob"],
            "success": true,
            "details": "Task completed successfully",
            "group_id": "team-x"
        }"#;
        let req: RecordOutcomeRequest =
            serde_json::from_str(json).expect("should deserialize");
        assert_eq!(req.task_id, "task-123");
        assert_eq!(req.agent, "Sisko");
        assert!(req.success);
        assert_eq!(req.group_id, "team-x");
    }

    #[test]
    fn record_outcome_request_group_id_defaults_to_default() {
        let json = r#"{
            "task_id": "t",
            "agent": "a",
            "entity_names": [],
            "success": false,
            "details": "failed"
        }"#;
        let req: RecordOutcomeRequest =
            serde_json::from_str(json).expect("should deserialize");
        assert_eq!(req.group_id, "default");
    }

    #[test]
    fn record_outcome_response_serializes() {
        let resp = RecordOutcomeResponse {
            episode_id: "ep-uuid".to_string(),
            recorded: true,
        };
        let json = serde_json::to_string(&resp).expect("should serialize");
        assert!(json.contains("ep-uuid"));
        assert!(json.contains("true"));
    }

    // ── Timeline types tests ─────────────────────────────────────────────────

    #[test]
    fn get_timeline_request_deserializes_full() {
        let json = r#"{
            "entity_name": "Alice",
            "group_id": "default",
            "limit": 20
        }"#;
        let req: GetTimelineRequest =
            serde_json::from_str(json).expect("should deserialize");
        assert_eq!(req.entity_name, "Alice");
        assert_eq!(req.group_id, "default");
        assert_eq!(req.limit, Some(20));
    }

    #[test]
    fn get_timeline_request_group_id_defaults_to_default() {
        let json = r#"{
            "entity_name": "Alice"
        }"#;
        let req: GetTimelineRequest =
            serde_json::from_str(json).expect("should deserialize");
        assert_eq!(req.group_id, "default");
        assert!(req.limit.is_none());
    }

    #[test]
    fn timeline_entry_serializes() {
        let entry = TimelineEntry {
            fact: "Alice works at Acme".to_string(),
            valid_at: Some("2024-01-01T00:00:00Z".to_string()),
            invalid_at: None,
            still_valid: true,
        };
        let json = serde_json::to_string(&entry).expect("should serialize");
        assert!(json.contains("Alice works at Acme"));
        assert!(json.contains("still_valid"));
    }

    #[test]
    fn get_timeline_response_serializes() {
        let resp = GetTimelineResponse {
            entity_name: "Alice".to_string(),
            entries: vec![TimelineEntry {
                fact: "Alice joined Acme".to_string(),
                valid_at: Some("2024-06-01".to_string()),
                invalid_at: None,
                still_valid: true,
            }],
        };
        let json = serde_json::to_string(&resp).expect("should serialize");
        assert!(json.contains("Alice"));
        assert!(json.contains("Alice joined Acme"));
    }
}
