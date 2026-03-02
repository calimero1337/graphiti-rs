//! MCP (Model Context Protocol) server for graphiti-rs.
//!
//! Exposes Graphiti knowledge-graph operations as MCP tools via the
//! streamable HTTP transport (MCP spec 2025-03-26).
//!
//! # Usage
//!
//! Mount the MCP router alongside the REST API in `server.rs`:
//!
//! ```ignore
//! let app = build_router(state).merge(graphiti_rs::mcp::build_mcp_router(graphiti));
//! ```

pub mod server;
pub use server::GraphitiMcpServer;

use std::sync::Arc;

use axum::Router;
use rmcp::transport::streamable_http_server::{
    session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
};

use crate::graphiti::Graphiti;

/// Build an Axum [`Router`] with the MCP streamable-HTTP service mounted at `/mcp`.
///
/// The returned router has no Axum state (`Router<()>`) and can be merged
/// directly with the REST API router produced by [`crate::server::build_router`].
pub fn build_mcp_router(graphiti: Arc<Graphiti>) -> Router {
    let session_manager = Arc::new(LocalSessionManager::default());
    let mcp_service = StreamableHttpService::new(
        move || Ok(GraphitiMcpServer::new(Arc::clone(&graphiti))),
        session_manager,
        StreamableHttpServerConfig::default(),
    );
    Router::new().route("/mcp", axum::routing::any_service(mcp_service))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::build_mcp_router;
    use crate::graphiti::Graphiti;
    use crate::testutils::{MockDriver, MockEmbedder, MockLlmClient};
    use crate::types::GraphitiConfig;
    use std::sync::Arc;

    fn make_graphiti() -> Arc<Graphiti> {
        Arc::new(Graphiti::from_clients(
            Arc::new(MockDriver),
            Arc::new(MockEmbedder::new()),
            Arc::new(MockLlmClient),
            GraphitiConfig::default(),
        ))
    }

    #[test]
    fn build_mcp_router_returns_router() {
        // Smoke test: building the router must not panic.
        let _router = build_mcp_router(make_graphiti());
    }
}
