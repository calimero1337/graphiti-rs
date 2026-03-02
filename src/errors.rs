//! Error types for graphiti-rs.

/// Alias for Results returning [`GraphitiError`].
pub type Result<T> = std::result::Result<T, GraphitiError>;

/// Top-level error type for graphiti-rs.
#[derive(Debug, thiserror::Error)]
pub enum GraphitiError {
    #[error("Driver error: {0}")]
    Driver(String),

    #[error("LLM error: {0}")]
    Llm(#[from] LlmError),

    #[error("Embedder error: {0}")]
    Embedder(String),

    #[error("Search error: {0}")]
    Search(String),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Node not found: {0}")]
    NodeNotFound(String),

    #[error("Edge not found: {0}")]
    EdgeNotFound(String),

    #[error("Pipeline error: {0}")]
    Pipeline(String),

    #[error("Reranker error: {0}")]
    Reranker(String),
}

/// LLM-specific errors.
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("Rate limited")]
    RateLimit,

    #[error("Model refused to respond")]
    Refusal,

    #[error("Empty response from LLM")]
    EmptyResponse,

    #[error("Authentication failed")]
    Authentication,

    #[error("API error: HTTP {status} — {message}")]
    Api { status: u16, message: String },
}
