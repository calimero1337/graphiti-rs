//! graphiti-server — REST API entry point.
//!
//! Reads configuration from environment variables, connects to Neo4j,
//! and serves the Graphiti REST API on `0.0.0.0:3000`.

use std::net::SocketAddr;
use std::sync::Arc;

use graphiti_rs::graphiti::Graphiti;
use graphiti_rs::mcp::build_mcp_router;
use graphiti_rs::server::{build_router, AppState};
use graphiti_rs::types::GraphitiConfig;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Structured JSON logging.
    tracing_subscriber::fmt()
        .json()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    info!("graphiti-server starting");

    // 2. Load config from environment.
    let config = GraphitiConfig::from_env()
        .map_err(|e| anyhow::anyhow!("config error: {e}"))?;

    let listen_addr: SocketAddr = std::env::var("LISTEN_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:3000".to_string())
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid LISTEN_ADDR: {e}"))?;

    // 3. Connect to Neo4j and build Graphiti.
    info!("connecting to Neo4j at {}", config.neo4j_uri);
    let graphiti = Graphiti::new(config)
        .await
        .map_err(|e| anyhow::anyhow!("failed to initialise Graphiti: {e}"))?;

    // 4. Build graph indices on first run (idempotent).
    graphiti
        .build_indices()
        .await
        .map_err(|e| anyhow::anyhow!("failed to build indices: {e}"))?;

    // 5. Build router: REST API + MCP service on the same port.
    let graphiti = Arc::new(graphiti);
    let state = AppState::new(Arc::clone(&graphiti));
    let app = build_router(state).merge(build_mcp_router(Arc::clone(&graphiti)));

    // 6. Serve with graceful shutdown on SIGTERM / Ctrl-C.
    info!("listening on {listen_addr}");
    let listener = tokio::net::TcpListener::bind(listen_addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("graphiti-server stopped");
    Ok(())
}

/// Resolves when SIGTERM or Ctrl-C is received.
async fn shutdown_signal() {
    use tokio::signal;

    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl-C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    info!("shutdown signal received");
}
