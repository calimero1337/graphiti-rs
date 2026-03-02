//! REST API server module.
//!
//! Exposes the Graphiti knowledge-graph engine over HTTP using [axum].
//!
//! # Sub-modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`errors`] | `ServerError` — maps [`crate::GraphitiError`] to HTTP responses |
//! | [`types`]  | Request / response DTOs (`AddEpisodeRequest`, `SearchRequest`, …) |
//!
//! `AppState` (shared across all handlers) and the axum `Router` live in this
//! module.  Individual handler functions are in the [`handlers`] sub-module and
//! route registration is in [`routes`].

pub mod errors;
pub mod types;

use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use tower_http::timeout::TimeoutLayer;
use uuid::Uuid;

use crate::graphiti::Graphiti;
use crate::nodes::episodic::EpisodeType;
use crate::search::SearchConfig;

use errors::ServerError;
use types::{
    AddEpisodeRequest, AddEpisodeResponse, BuildCommunitiesRequest, CommunityResult,
    ContextualizeRequest, GetTimelineRequest, ListEpisodesQuery, RecordOutcomeRequest,
    SearchRequest, TokenUsageResponse,
};

// ── AppState ──────────────────────────────────────────────────────────────────

/// Shared state injected into every request handler via axum's `State` extractor.
///
/// Wraps an `Arc<Graphiti>` so it can be cheaply cloned across handler tasks.
#[derive(Clone)]
pub struct AppState {
    /// The Graphiti facade providing all graph operations.
    pub graphiti: Arc<Graphiti>,
}

impl AppState {
    /// Create a new `AppState` from a shared `Graphiti` instance.
    pub fn new(graphiti: Arc<Graphiti>) -> Self {
        Self { graphiti }
    }
}

// ── Router ────────────────────────────────────────────────────────────────────

/// Default request timeout applied to every route.
///
/// 30 seconds is a conservative ceiling.  Long-running operations (e.g.
/// community building) may legitimately exceed this, but keeping a hard upper
/// bound prevents runaway requests from tying up server resources indefinitely.
const REQUEST_TIMEOUT_SECS: u64 = 30;

/// Build the axum `Router` with all API routes attached to `state`.
///
/// A [`TimeoutLayer`] is applied globally: any request that has not produced a
/// complete response within [`REQUEST_TIMEOUT_SECS`] seconds will receive a
/// `408 Request Timeout` reply.
#[allow(deprecated)] // TimeoutLayer::new is deprecated in favour of ::with_status_code
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route(
            "/v1/episodes",
            post(handle_add_episode).get(handle_list_episodes),
        )
        .route("/v1/episodes/{uuid}", delete(handle_remove_episode))
        .route("/v1/search", post(handle_search))
        .route("/v1/communities/build", post(handle_build_communities))
        .route("/v1/contextualize", post(handle_contextualize))
        .route("/v1/outcomes", post(handle_record_outcome))
        .route("/v1/timeline", post(handle_get_timeline))
        .route("/v1/token-usage", get(handle_token_usage))
        .route("/v1/token-usage/reset", post(handle_token_usage_reset))
        .route("/health", get(handle_health))
        .route("/ready", get(handle_ready))
        .layer(TimeoutLayer::new(Duration::from_secs(REQUEST_TIMEOUT_SECS)))
        .with_state(state)
}

// ── Handlers ──────────────────────────────────────────────────────────────────

/// `POST /v1/episodes` — ingest an episode into the knowledge graph.
async fn handle_add_episode(
    State(state): State<AppState>,
    Json(req): Json<AddEpisodeRequest>,
) -> Result<impl IntoResponse, ServerError> {
    let source_type = parse_source_type(&req.source_type)?;

    let result = state
        .graphiti
        .add_episode(
            &req.name,
            &req.content,
            source_type,
            &req.group_id,
            &req.source_description,
        )
        .await
        .map_err(ServerError::from)?;

    let response = AddEpisodeResponse {
        episode_id: result.episode.uuid.to_string(),
        nodes_created: result.nodes.len(),
        edges_created: result.edges.len(),
    };

    Ok((StatusCode::CREATED, Json(response)))
}

/// `GET /v1/episodes` — retrieve recent episodes for a group.
async fn handle_list_episodes(
    State(state): State<AppState>,
    Query(params): Query<ListEpisodesQuery>,
) -> Result<impl IntoResponse, ServerError> {
    let group_id = params.group_id.as_deref().unwrap_or("default");
    let limit = params.limit.unwrap_or(20);

    let episodes = state
        .graphiti
        .retrieve_episodes(&[group_id], limit)
        .await
        .map_err(ServerError::from)?;

    Ok(Json(episodes))
}

/// `DELETE /v1/episodes/:uuid` — remove an episode from the graph.
async fn handle_remove_episode(
    State(state): State<AppState>,
    Path(uuid): Path<Uuid>,
) -> Result<impl IntoResponse, ServerError> {
    state
        .graphiti
        .driver
        .delete_episodic_node(&uuid)
        .await
        .map_err(ServerError::from)?;

    Ok(StatusCode::NO_CONTENT)
}

/// `POST /v1/search` — hybrid search the knowledge graph.
async fn handle_search(
    State(state): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> Result<impl IntoResponse, ServerError> {
    let config = SearchConfig::default()
        .with_group_ids(req.group_ids)
        .with_limit(req.limit.unwrap_or(10));

    let results = state
        .graphiti
        .search(&req.query, &config)
        .await
        .map_err(ServerError::from)?;

    Ok(Json(results))
}

/// `POST /v1/communities/build` — build or rebuild community structure.
async fn handle_build_communities(
    State(state): State<AppState>,
    Json(req): Json<BuildCommunitiesRequest>,
) -> Result<impl IntoResponse, ServerError> {
    let refs: Vec<&str> = req.group_ids.iter().map(|s| s.as_str()).collect();
    let communities = state
        .graphiti
        .build_communities(&refs)
        .await
        .map_err(ServerError::from)?;

    Ok(Json(CommunityResult {
        communities_built: communities.len(),
    }))
}

/// `GET /v1/token-usage` — return cumulative LLM token usage.
///
/// Returns the prompt, completion, and total token counts accumulated since
/// the server started (or since the last reset).
async fn handle_token_usage(State(state): State<AppState>) -> impl IntoResponse {
    let usage = state.graphiti.token_usage();
    Json(TokenUsageResponse {
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        total_tokens: usage.total_tokens,
    })
}

/// `POST /v1/token-usage/reset` — reset cumulative LLM token counters to zero.
///
/// Resets the prompt and completion token counts accumulated since server start
/// (or the last reset). Returns `204 No Content` on success.
async fn handle_token_usage_reset(State(state): State<AppState>) -> StatusCode {
    state.graphiti.reset_token_usage();
    StatusCode::NO_CONTENT
}

/// `GET /health` — liveness probe; always returns 200.
async fn handle_health() -> StatusCode {
    StatusCode::OK
}

/// `GET /ready` — readiness probe; returns 200 if Neo4j is reachable.
async fn handle_ready(State(state): State<AppState>) -> impl IntoResponse {
    match state.graphiti.driver.ping().await {
        Ok(()) => StatusCode::OK,
        Err(_) => StatusCode::SERVICE_UNAVAILABLE,
    }
}

/// `POST /v1/contextualize` — natural language query with 1-hop graph expansion.
async fn handle_contextualize(
    State(state): State<AppState>,
    Json(req): Json<ContextualizeRequest>,
) -> Result<impl IntoResponse, ServerError> {
    let resp = state
        .graphiti
        .contextualize(&req.query, &req.group_ids, req.limit)
        .await
        .map_err(ServerError::from)?;

    Ok(Json(resp))
}

/// `POST /v1/outcomes` — record a task outcome (success/failure) for the feedback loop.
async fn handle_record_outcome(
    State(state): State<AppState>,
    Json(req): Json<RecordOutcomeRequest>,
) -> Result<impl IntoResponse, ServerError> {
    let resp = state
        .graphiti
        .record_outcome(
            &req.task_id,
            &req.agent,
            &req.entity_names,
            req.success,
            &req.details,
            &req.group_id,
        )
        .await
        .map_err(ServerError::from)?;

    Ok((StatusCode::CREATED, Json(resp)))
}

/// `POST /v1/timeline` — get chronological timeline of facts for an entity.
async fn handle_get_timeline(
    State(state): State<AppState>,
    Json(req): Json<GetTimelineRequest>,
) -> Result<impl IntoResponse, ServerError> {
    let resp = state
        .graphiti
        .get_timeline(&req.entity_name, &req.group_id, req.limit)
        .await
        .map_err(ServerError::from)?;

    Ok(Json(resp))
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn parse_source_type(s: &str) -> Result<EpisodeType, ServerError> {
    match s {
        "text" => Ok(EpisodeType::Text),
        "message" => Ok(EpisodeType::Message),
        "json" => Ok(EpisodeType::Json),
        other => Err(ServerError::Validation(format!(
            "unknown source_type: '{other}'; expected one of text, message, json"
        ))),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::AppState;
    use crate::testutils::{MockDriver, MockEmbedder, MockLlmClient, TokenTrackingMockLlmClient};
    use crate::llm_client::LlmClient;
    use std::sync::Arc;

    #[allow(dead_code)]
    fn assert_app_state_is_send_sync()
    where
        AppState: Send + Sync,
    {
    }

    fn make_graphiti() -> Arc<crate::graphiti::Graphiti> {
        Arc::new(crate::graphiti::Graphiti::from_clients(
            Arc::new(MockDriver),
            Arc::new(MockEmbedder::new()),
            Arc::new(MockLlmClient),
            crate::types::GraphitiConfig::default(),
        ))
    }

    #[test]
    fn app_state_new_accepts_graphiti_arc() {
        let _state = AppState::new(make_graphiti());
    }

    #[test]
    fn app_state_exposes_graphiti_field() {
        let g = make_graphiti();
        let state = AppState::new(Arc::clone(&g));
        let _: &Arc<crate::graphiti::Graphiti> = &state.graphiti;
    }

    #[test]
    fn app_state_is_clone() {
        let state = AppState::new(make_graphiti());
        let _cloned = state.clone();
    }

    #[test]
    fn build_router_applies_timeout_layer() {
        let state = AppState::new(make_graphiti());
        let _router = super::build_router(state);
    }

    #[test]
    fn request_timeout_constant_is_positive() {
        assert!(super::REQUEST_TIMEOUT_SECS > 0);
    }

    fn make_graphiti_with_tokens(prompt: u64, completion: u64) -> Arc<crate::graphiti::Graphiti> {
        let llm: Arc<dyn LlmClient> = Arc::new(TokenTrackingMockLlmClient::with_usage(prompt, completion));
        Arc::new(crate::graphiti::Graphiti::from_clients(
            Arc::new(MockDriver),
            Arc::new(MockEmbedder::new()),
            llm,
            crate::types::GraphitiConfig::default(),
        ))
    }

    /// `handle_token_usage` must return the actual cumulative token counts from
    /// the underlying `LlmClient`, not hard-coded zeros.
    #[tokio::test]
    async fn handle_token_usage_returns_real_counts() {
        use axum::body::to_bytes;
        use axum::http::Request;
        use tower::ServiceExt;
        use crate::server::types::TokenUsageResponse;

        let g = make_graphiti_with_tokens(100, 50);
        let state = AppState::new(g);
        let router = super::build_router(state);

        let request = Request::builder()
            .method("GET")
            .uri("/v1/token-usage")
            .body(axum::body::Body::empty())
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);

        let body_bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let usage: TokenUsageResponse = serde_json::from_slice(&body_bytes).unwrap();

        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 50);
        assert_eq!(usage.total_tokens, 150);
    }

    /// `POST /v1/token-usage/reset` must return 204 and clear the counters so a
    /// subsequent `GET /v1/token-usage` reports zeros.
    #[tokio::test]
    async fn handle_token_usage_reset_clears_counters() {
        use axum::body::to_bytes;
        use axum::http::Request;
        use tower::ServiceExt;
        use crate::server::types::TokenUsageResponse;

        let g = make_graphiti_with_tokens(80, 40);

        // Verify counts are non-zero before reset.
        let state = AppState::new(Arc::clone(&g));
        let pre_usage = g.token_usage();
        assert_eq!(pre_usage.prompt_tokens, 80);

        // Issue POST /v1/token-usage/reset and expect 204.
        let router = super::build_router(state);
        let reset_req = Request::builder()
            .method("POST")
            .uri("/v1/token-usage/reset")
            .body(axum::body::Body::empty())
            .expect("reset request must build");

        let reset_resp = router.oneshot(reset_req).await.expect("reset must not fail");
        assert_eq!(reset_resp.status(), axum::http::StatusCode::NO_CONTENT);

        // Verify the underlying tracker is now zero.
        let post_usage = g.token_usage();
        assert_eq!(post_usage.prompt_tokens, 0);
        assert_eq!(post_usage.completion_tokens, 0);
        assert_eq!(post_usage.total_tokens, 0);

        // Also verify GET /v1/token-usage reports zeros on a fresh router (same state).
        let state2 = AppState::new(Arc::clone(&g));
        let router2 = super::build_router(state2);
        let get_req = Request::builder()
            .method("GET")
            .uri("/v1/token-usage")
            .body(axum::body::Body::empty())
            .expect("get request must build");

        let get_resp = router2.oneshot(get_req).await.expect("get must not fail");
        assert_eq!(get_resp.status(), axum::http::StatusCode::OK);
        let body = to_bytes(get_resp.into_body(), usize::MAX).await.expect("body read");
        let usage: TokenUsageResponse = serde_json::from_slice(&body).expect("JSON parse");
        assert_eq!(usage.prompt_tokens, 0);
        assert_eq!(usage.completion_tokens, 0);
        assert_eq!(usage.total_tokens, 0);
    }

    /// `handle_token_usage` returns zeroed counts when no calls have been made.
    #[tokio::test]
    async fn handle_token_usage_returns_zeros_when_no_calls_made() {
        use axum::body::to_bytes;
        use axum::http::Request;
        use tower::ServiceExt;
        use crate::server::types::TokenUsageResponse;

        let state = AppState::new(make_graphiti());
        let router = super::build_router(state);

        let request = Request::builder()
            .method("GET")
            .uri("/v1/token-usage")
            .body(axum::body::Body::empty())
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);

        let body_bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let usage: TokenUsageResponse = serde_json::from_slice(&body_bytes).unwrap();

        assert_eq!(usage.prompt_tokens, 0);
        assert_eq!(usage.completion_tokens, 0);
        assert_eq!(usage.total_tokens, 0);
    }

    // ── Contextualize handler tests ──────────────────────────────────────────

    #[tokio::test]
    async fn handle_contextualize_returns_ok() {
        use axum::body::to_bytes;
        use axum::http::Request;
        use tower::ServiceExt;
        use crate::server::types::ContextualizeResponse;

        let state = AppState::new(make_graphiti());
        let router = super::build_router(state);

        let body = serde_json::json!({
            "query": "error handling in Rust",
            "group_ids": ["default"],
            "limit": 5
        });

        let request = Request::builder()
            .method("POST")
            .uri("/v1/contextualize")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);

        let body_bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let resp: ContextualizeResponse = serde_json::from_slice(&body_bytes).unwrap();
        assert!(resp.entities.is_empty());
        assert!(resp.warnings.is_empty());
    }

    // ── Record Outcome handler tests ─────────────────────────────────────────

    #[tokio::test]
    async fn handle_record_outcome_returns_created() {
        use axum::body::to_bytes;
        use axum::http::Request;
        use tower::ServiceExt;
        use crate::server::types::RecordOutcomeResponse;

        let state = AppState::new(make_graphiti());
        let router = super::build_router(state);

        let body = serde_json::json!({
            "task_id": "T-042",
            "agent": "coding-agent",
            "entity_names": ["circuit_breaker"],
            "success": true,
            "details": "Applied circuit breaker, tests pass"
        });

        let request = Request::builder()
            .method("POST")
            .uri("/v1/outcomes")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::CREATED);

        let body_bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let resp: RecordOutcomeResponse = serde_json::from_slice(&body_bytes).unwrap();
        assert!(resp.recorded);
        assert!(!resp.episode_id.is_empty());
    }

    // ── Timeline handler tests ───────────────────────────────────────────────

    #[tokio::test]
    async fn handle_get_timeline_returns_ok() {
        use axum::body::to_bytes;
        use axum::http::Request;
        use tower::ServiceExt;
        use crate::server::types::GetTimelineResponse;

        let state = AppState::new(make_graphiti());
        let router = super::build_router(state);

        let body = serde_json::json!({
            "entity_name": "circuit_breaker",
            "group_id": "default"
        });

        let request = Request::builder()
            .method("POST")
            .uri("/v1/timeline")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);

        let body_bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let resp: GetTimelineResponse = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(resp.entity_name, "circuit_breaker");
        assert!(resp.entries.is_empty());
    }
}
