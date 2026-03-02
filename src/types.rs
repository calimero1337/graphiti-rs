//! Shared configuration and client container types.

use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::embedder::http::{DEFAULT_BASE_URL, DEFAULT_DIM, DEFAULT_MODEL};

/// Which LLM backend to use.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LlmBackend {
    /// OpenAI API (requires `openai_api_key`).
    OpenAI,
    /// Anthropic Messages API directly (requires `anthropic_api_key`).
    Anthropic,
    /// Claude CLI via Agent SDK (requires `claude` in PATH, local dev only).
    Claude,
    /// Delegated to external worker via task queue (zero LLM dependencies).
    Delegated,
}

impl Default for LlmBackend {
    fn default() -> Self {
        Self::OpenAI
    }
}

/// Central configuration loaded from environment variables.
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct GraphitiConfig {
    /// Neo4j connection URI (e.g. `bolt://localhost:7687`).
    #[validate(length(min = 1))]
    pub neo4j_uri: String,

    /// Neo4j username.
    pub neo4j_user: String,

    /// Neo4j password.
    #[validate(length(min = 1))]
    pub neo4j_password: String,

    /// OpenAI API key (required when `llm_backend` is `OpenAI`).
    pub openai_api_key: String,

    /// Anthropic API key (required when `llm_backend` is `Anthropic`).
    pub anthropic_api_key: String,

    /// Which LLM backend to use.
    pub llm_backend: LlmBackend,

    /// Base URL of the HTTP embedding server.
    pub embedding_base_url: String,

    /// Embedding model name sent in requests to the embedding server.
    pub embedding_model: String,

    /// Embedding vector dimension (must be > 0).
    pub embedding_dim: usize,

    /// Default LLM model name.
    pub model_name: String,

    /// Smaller/cheaper LLM model name.
    pub small_model_name: String,

    /// Optional group ID for partitioning graph data.
    pub group_id: Option<String>,

    /// Ingestion pipeline configuration.
    pub ingestion: IngestionConfig,
}

impl Default for GraphitiConfig {
    fn default() -> Self {
        Self {
            neo4j_uri: "bolt://localhost:7687".to_string(),
            neo4j_user: "neo4j".to_string(),
            neo4j_password: String::new(),
            openai_api_key: String::new(),
            anthropic_api_key: String::new(),
            llm_backend: LlmBackend::default(),
            embedding_base_url: DEFAULT_BASE_URL.to_string(),
            embedding_model: DEFAULT_MODEL.to_string(),
            embedding_dim: DEFAULT_DIM,
            model_name: "gpt-4o".to_string(),
            small_model_name: "gpt-4.1-nano".to_string(),
            group_id: None,
            ingestion: IngestionConfig::default(),
        }
    }
}

impl GraphitiConfig {
    /// Load configuration from environment variables.
    ///
    /// Calls `dotenvy::dotenv().ok()` first (non-fatal if `.env` is absent),
    /// then reads each variable from the process environment. Required variables
    /// (`NEO4J_PASSWORD`, `OPENAI_API_KEY`) return a [`crate::GraphitiError::Validation`]
    /// error when absent or empty.
    pub fn from_env() -> crate::Result<Self> {
        dotenvy::dotenv().ok();

        let neo4j_uri = std::env::var("NEO4J_URI")
            .unwrap_or_else(|_| "bolt://localhost:7687".to_string());

        let neo4j_user = std::env::var("NEO4J_USER")
            .unwrap_or_else(|_| "neo4j".to_string());

        let neo4j_password = std::env::var("NEO4J_PASSWORD").map_err(|_| {
            crate::GraphitiError::Validation("NEO4J_PASSWORD is required".to_string())
        })?;

        let llm_backend = match std::env::var("LLM_BACKEND") {
            Ok(val) if val.eq_ignore_ascii_case("anthropic") => LlmBackend::Anthropic,
            Ok(val) if val.eq_ignore_ascii_case("claude") => LlmBackend::Claude,
            Ok(val) if val.eq_ignore_ascii_case("delegated") => LlmBackend::Delegated,
            _ => LlmBackend::OpenAI,
        };

        let openai_api_key = match llm_backend {
            LlmBackend::OpenAI => std::env::var("OPENAI_API_KEY").map_err(|_| {
                crate::GraphitiError::Validation("OPENAI_API_KEY is required when LLM_BACKEND=openai".to_string())
            })?,
            LlmBackend::Anthropic | LlmBackend::Claude | LlmBackend::Delegated => std::env::var("OPENAI_API_KEY").unwrap_or_default(),
        };

        let anthropic_api_key = match llm_backend {
            LlmBackend::Anthropic => std::env::var("ANTHROPIC_API_KEY").map_err(|_| {
                crate::GraphitiError::Validation("ANTHROPIC_API_KEY is required when LLM_BACKEND=anthropic".to_string())
            })?,
            _ => std::env::var("ANTHROPIC_API_KEY").unwrap_or_default(),
        };

        let embedding_base_url = std::env::var("EMBEDDING_BASE_URL")
            .unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());

        let embedding_model = std::env::var("EMBEDDING_MODEL")
            .unwrap_or_else(|_| DEFAULT_MODEL.to_string());

        let embedding_dim = match std::env::var("EMBEDDING_DIM") {
            Ok(val) => val.parse::<usize>().map_err(|_| {
                crate::GraphitiError::Validation(
                    "EMBEDDING_DIM must be a positive integer".to_string(),
                )
            })?,
            Err(_) => DEFAULT_DIM,
        };

        if embedding_dim == 0 {
            return Err(crate::GraphitiError::Validation(
                "EMBEDDING_DIM must be > 0".to_string(),
            ));
        }

        let model_name = std::env::var("MODEL_NAME").unwrap_or_else(|_| "gpt-4o".to_string());

        let small_model_name = std::env::var("SMALL_MODEL_NAME")
            .unwrap_or_else(|_| "gpt-4.1-nano".to_string());

        let group_id = std::env::var("GROUP_ID").ok();

        let max_concurrent_llm_calls = match std::env::var("MAX_CONCURRENT_LLM_CALLS") {
            Ok(val) => val.parse::<usize>().map_err(|_| {
                crate::GraphitiError::Validation(
                    "MAX_CONCURRENT_LLM_CALLS must be a positive integer".to_string(),
                )
            })?,
            Err(_) => 5,
        };

        if max_concurrent_llm_calls == 0 {
            return Err(crate::GraphitiError::Validation(
                "MAX_CONCURRENT_LLM_CALLS must be > 0".to_string(),
            ));
        }

        let entity_search_limit = match std::env::var("ENTITY_SEARCH_LIMIT") {
            Ok(val) => val.parse::<usize>().map_err(|_| {
                crate::GraphitiError::Validation(
                    "ENTITY_SEARCH_LIMIT must be a positive integer".to_string(),
                )
            })?,
            Err(_) => 10,
        };

        if entity_search_limit == 0 {
            return Err(crate::GraphitiError::Validation(
                "ENTITY_SEARCH_LIMIT must be > 0".to_string(),
            ));
        }

        let edge_search_limit = match std::env::var("EDGE_SEARCH_LIMIT") {
            Ok(val) => val.parse::<usize>().map_err(|_| {
                crate::GraphitiError::Validation(
                    "EDGE_SEARCH_LIMIT must be a positive integer".to_string(),
                )
            })?,
            Err(_) => 10,
        };

        if edge_search_limit == 0 {
            return Err(crate::GraphitiError::Validation(
                "EDGE_SEARCH_LIMIT must be > 0".to_string(),
            ));
        }

        let previous_episode_count = match std::env::var("PREVIOUS_EPISODE_COUNT") {
            Ok(val) => val.parse::<usize>().map_err(|_| {
                crate::GraphitiError::Validation(
                    "PREVIOUS_EPISODE_COUNT must be a non-negative integer".to_string(),
                )
            })?,
            Err(_) => 3,
        };

        let ingestion = IngestionConfig {
            max_concurrent_llm_calls,
            entity_search_limit,
            edge_search_limit,
            previous_episode_count,
        };

        let config = Self {
            neo4j_uri,
            neo4j_user,
            neo4j_password,
            openai_api_key,
            anthropic_api_key,
            llm_backend,
            embedding_base_url,
            embedding_model,
            embedding_dim,
            model_name,
            small_model_name,
            group_id,
            ingestion,
        };

        config.validate().map_err(|e| {
            crate::GraphitiError::Validation(e.to_string())
        })?;

        Ok(config)
    }
}

/// Configuration for the ingestion pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionConfig {
    /// Maximum number of concurrent LLM calls (semaphore permits).
    pub max_concurrent_llm_calls: usize,
    /// Maximum number of candidate entity nodes returned by name search.
    pub entity_search_limit: usize,
    /// Maximum number of candidate edges returned by edge search.
    pub edge_search_limit: usize,
    /// Number of recent episodes to fetch for co-reference resolution context (0 to disable).
    pub previous_episode_count: usize,
}

impl Default for IngestionConfig {
    fn default() -> Self {
        Self {
            max_concurrent_llm_calls: 5,
            entity_search_limit: 10,
            edge_search_limit: 10,
            previous_episode_count: 3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::AddEpisodeResult;
    use serial_test::serial;
    use std::env;

    /// Temporarily sets env vars for a test, restoring originals afterward.
    fn with_env<F, R>(vars: &[(&str, &str)], f: F) -> R
    where
        F: FnOnce() -> R,
    {
        // Save originals.
        let originals: Vec<(&str, Option<String>)> =
            vars.iter().map(|(k, _)| (*k, env::var(k).ok())).collect();

        // Set test values.
        for (k, v) in vars {
            env::set_var(k, v);
        }

        let result = f();

        // Restore originals.
        for (k, original) in &originals {
            match original {
                Some(v) => env::set_var(k, v),
                None => env::remove_var(k),
            }
        }

        result
    }

    #[test]
    #[serial]
    fn test_config_defaults() {
        with_env(
            &[
                ("NEO4J_PASSWORD", "secret"),
                ("OPENAI_API_KEY", "sk-test"),
            ],
            || {
                // Remove optional vars in case they're set in the process env.
                env::remove_var("NEO4J_URI");
                env::remove_var("NEO4J_USER");
                env::remove_var("EMBEDDING_BASE_URL");
                env::remove_var("EMBEDDING_MODEL");
                env::remove_var("EMBEDDING_DIM");
                env::remove_var("MODEL_NAME");
                env::remove_var("SMALL_MODEL_NAME");
                env::remove_var("GROUP_ID");

                let config = GraphitiConfig::from_env().expect("config should load");
                assert_eq!(config.neo4j_uri, "bolt://localhost:7687");
                assert_eq!(config.neo4j_user, "neo4j");
                assert_eq!(config.embedding_base_url, DEFAULT_BASE_URL);
                assert_eq!(config.embedding_model, DEFAULT_MODEL);
                assert_eq!(config.embedding_dim, DEFAULT_DIM);
                assert_eq!(config.model_name, "gpt-4o");
                assert_eq!(config.small_model_name, "gpt-4.1-nano");
                assert!(config.group_id.is_none());
            },
        );
    }

    #[test]
    #[serial]
    fn test_config_custom_values() {
        with_env(
            &[
                ("NEO4J_URI", "bolt://db.example.com:7687"),
                ("NEO4J_USER", "admin"),
                ("NEO4J_PASSWORD", "mysecret"),
                ("OPENAI_API_KEY", "sk-real-key"),
                ("EMBEDDING_DIM", "3072"),
                ("MODEL_NAME", "gpt-4o-mini"),
                ("SMALL_MODEL_NAME", "gpt-3.5-turbo"),
                ("GROUP_ID", "team-alpha"),
            ],
            || {
                let config = GraphitiConfig::from_env().expect("config should load");
                assert_eq!(config.neo4j_uri, "bolt://db.example.com:7687");
                assert_eq!(config.neo4j_user, "admin");
                assert_eq!(config.neo4j_password, "mysecret");
                assert_eq!(config.openai_api_key, "sk-real-key");
                assert_eq!(config.embedding_dim, 3072);
                assert_eq!(config.model_name, "gpt-4o-mini");
                assert_eq!(config.small_model_name, "gpt-3.5-turbo");
                assert_eq!(config.group_id, Some("team-alpha".to_string()));
            },
        );
    }

    #[test]
    #[serial]
    fn test_config_missing_password() {
        // Save and clear both required vars.
        let saved_pw = env::var("NEO4J_PASSWORD").ok();
        let saved_key = env::var("OPENAI_API_KEY").ok();
        env::remove_var("NEO4J_PASSWORD");
        env::remove_var("OPENAI_API_KEY");

        let result = GraphitiConfig::from_env();

        // Restore.
        if let Some(v) = saved_pw { env::set_var("NEO4J_PASSWORD", v); }
        if let Some(v) = saved_key { env::set_var("OPENAI_API_KEY", v); }

        assert!(result.is_err());
        match result.unwrap_err() {
            crate::GraphitiError::Validation(msg) => {
                assert!(msg.contains("NEO4J_PASSWORD"));
            }
            e => panic!("expected Validation error, got {:?}", e),
        }
    }

    #[test]
    #[serial]
    fn test_config_missing_api_key() {
        let saved_key = env::var("OPENAI_API_KEY").ok();
        env::remove_var("OPENAI_API_KEY");

        // Make sure password is present.
        let saved_pw = env::var("NEO4J_PASSWORD").ok();
        env::set_var("NEO4J_PASSWORD", "secret");

        let result = GraphitiConfig::from_env();

        if let Some(v) = saved_key { env::set_var("OPENAI_API_KEY", v); } else { env::remove_var("OPENAI_API_KEY"); }
        if let Some(v) = saved_pw { env::set_var("NEO4J_PASSWORD", v); } else { env::remove_var("NEO4J_PASSWORD"); }

        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn test_config_invalid_embedding_dim() {
        with_env(
            &[
                ("NEO4J_PASSWORD", "secret"),
                ("OPENAI_API_KEY", "sk-test"),
                ("EMBEDDING_DIM", "not-a-number"),
            ],
            || {
                let result = GraphitiConfig::from_env();
                assert!(result.is_err());
                match result.unwrap_err() {
                    crate::GraphitiError::Validation(msg) => {
                        assert!(msg.contains("EMBEDDING_DIM"));
                    }
                    e => panic!("expected Validation error, got {:?}", e),
                }
            },
        );
    }

    #[test]
    #[serial]
    fn test_config_zero_embedding_dim() {
        with_env(
            &[
                ("NEO4J_PASSWORD", "secret"),
                ("OPENAI_API_KEY", "sk-test"),
                ("EMBEDDING_DIM", "0"),
            ],
            || {
                let result = GraphitiConfig::from_env();
                assert!(result.is_err());
            },
        );
    }

    #[test]
    #[serial]
    fn test_config_zero_entity_search_limit() {
        with_env(
            &[
                ("NEO4J_PASSWORD", "secret"),
                ("OPENAI_API_KEY", "sk-test"),
                ("ENTITY_SEARCH_LIMIT", "0"),
            ],
            || {
                let result = GraphitiConfig::from_env();
                assert!(result.is_err());
                match result.unwrap_err() {
                    crate::GraphitiError::Validation(msg) => {
                        assert!(msg.contains("ENTITY_SEARCH_LIMIT"));
                    }
                    e => panic!("expected Validation error, got {:?}", e),
                }
            },
        );
    }

    #[test]
    #[serial]
    fn test_config_zero_edge_search_limit() {
        with_env(
            &[
                ("NEO4J_PASSWORD", "secret"),
                ("OPENAI_API_KEY", "sk-test"),
                ("EDGE_SEARCH_LIMIT", "0"),
            ],
            || {
                let result = GraphitiConfig::from_env();
                assert!(result.is_err());
                match result.unwrap_err() {
                    crate::GraphitiError::Validation(msg) => {
                        assert!(msg.contains("EDGE_SEARCH_LIMIT"));
                    }
                    e => panic!("expected Validation error, got {:?}", e),
                }
            },
        );
    }

    #[test]
    fn test_search_config_default() {
        let _ = crate::search::SearchConfig::default();
    }

    #[test]
    fn test_ingestion_config_default() {
        let config = IngestionConfig::default();
        assert_eq!(config.previous_episode_count, 3);
    }

    #[test]
    #[serial]
    fn test_graphiti_config_from_env_ingestion_defaults() {
        with_env(
            &[
                ("NEO4J_PASSWORD", "secret"),
                ("OPENAI_API_KEY", "sk-test"),
            ],
            || {
                env::remove_var("MAX_CONCURRENT_LLM_CALLS");
                env::remove_var("ENTITY_SEARCH_LIMIT");
                env::remove_var("EDGE_SEARCH_LIMIT");
                env::remove_var("PREVIOUS_EPISODE_COUNT");
                env::remove_var("NEO4J_URI");
                env::remove_var("NEO4J_USER");
                env::remove_var("EMBEDDING_DIM");
                env::remove_var("MODEL_NAME");
                env::remove_var("SMALL_MODEL_NAME");
                env::remove_var("GROUP_ID");

                let config = GraphitiConfig::from_env().expect("config should load");
                assert_eq!(config.ingestion.max_concurrent_llm_calls, 5);
                assert_eq!(config.ingestion.entity_search_limit, 10);
                assert_eq!(config.ingestion.edge_search_limit, 10);
                assert_eq!(config.ingestion.previous_episode_count, 3);
            },
        );
    }

    #[test]
    #[serial]
    fn test_config_custom_previous_episode_count() {
        with_env(
            &[
                ("NEO4J_PASSWORD", "secret"),
                ("OPENAI_API_KEY", "sk-test"),
                ("PREVIOUS_EPISODE_COUNT", "5"),
            ],
            || {
                let config = GraphitiConfig::from_env().expect("config should load");
                assert_eq!(config.ingestion.previous_episode_count, 5);
            },
        );
    }

    #[test]
    #[serial]
    fn test_config_zero_previous_episode_count_is_valid() {
        with_env(
            &[
                ("NEO4J_PASSWORD", "secret"),
                ("OPENAI_API_KEY", "sk-test"),
                ("PREVIOUS_EPISODE_COUNT", "0"),
            ],
            || {
                // 0 is explicitly valid — it disables the feature.
                let result = GraphitiConfig::from_env();
                assert!(result.is_ok());
                assert_eq!(result.unwrap().ingestion.previous_episode_count, 0);
            },
        );
    }

    // --- Failing tests for TASK-GR-016.02 ---
    // These tests reference types and fields that do not yet exist.
    // They are intentionally left failing (TDD red phase).

    /// AddEpisodeResult must expose `episode`, `nodes`, and `edges` fields.
    #[test]
    fn test_add_episode_result_fields() {
        use crate::edges::entity::EntityEdge;
        use crate::nodes::entity::EntityNode;
        use crate::nodes::episodic::{EpisodeType, EpisodicNode};
        use uuid::Uuid;

        let episode = EpisodicNode {
            uuid: Uuid::nil(),
            name: "ep".to_string(),
            group_id: "g".to_string(),
            labels: vec!["EpisodicNode".to_string()],
            created_at: chrono::Utc::now(),
            source: EpisodeType::Text,
            source_description: "test".to_string(),
            content: "hello".to_string(),
            valid_at: chrono::Utc::now(),
            entity_edges: vec![],
        };

        let result = AddEpisodeResult {
            episode,
            nodes: Vec::<EntityNode>::new(),
            edges: Vec::<EntityEdge>::new(),
        };

        assert!(result.nodes.is_empty());
        assert!(result.edges.is_empty());
        assert_eq!(result.episode.group_id, "g");
    }

    /// AddEpisodeResult must implement Debug and Clone.
    #[test]
    fn test_add_episode_result_debug_clone() {
        use crate::nodes::episodic::{EpisodeType, EpisodicNode};
        use uuid::Uuid;

        let episode = EpisodicNode {
            uuid: Uuid::nil(),
            name: "ep".to_string(),
            group_id: "grp".to_string(),
            labels: vec![],
            created_at: chrono::Utc::now(),
            source: EpisodeType::Message,
            source_description: "msg".to_string(),
            content: "content".to_string(),
            valid_at: chrono::Utc::now(),
            entity_edges: vec![],
        };

        let result = AddEpisodeResult {
            episode,
            nodes: vec![],
            edges: vec![],
        };

        let cloned = result.clone();
        assert_eq!(cloned.episode.group_id, "grp");
        // Debug must compile
        let _ = format!("{:?}", &result);
    }

    /// GraphitiConfig must carry an embedded IngestionConfig field named `ingestion`.
    #[test]
    fn test_graphiti_config_has_ingestion_field() {
        let config = GraphitiConfig::default();
        // The `ingestion` field must exist and carry correct defaults.
        assert_eq!(config.ingestion.max_concurrent_llm_calls, 5);
        assert_eq!(config.ingestion.entity_search_limit, 10);
        assert_eq!(config.ingestion.edge_search_limit, 10);
    }
}
