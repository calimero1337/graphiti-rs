//! Server-specific error type that maps [`crate::GraphitiError`] to HTTP responses.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

use crate::GraphitiError;

/// HTTP-layer error that maps domain errors to appropriate status codes.
#[derive(Debug)]
pub enum ServerError {
    /// 404 Not Found.
    NotFound(String),
    /// 422 Unprocessable Entity.
    Validation(String),
    /// 500 Internal Server Error.
    Internal(String),
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            Self::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
            Self::Validation(msg) => (StatusCode::UNPROCESSABLE_ENTITY, msg),
            Self::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };
        (status, Json(json!({"error": message}))).into_response()
    }
}

impl From<GraphitiError> for ServerError {
    fn from(e: GraphitiError) -> Self {
        match e {
            GraphitiError::NodeNotFound(msg) | GraphitiError::EdgeNotFound(msg) => {
                Self::NotFound(msg)
            }
            GraphitiError::Validation(msg) => Self::Validation(msg),
            other => Self::Internal(other.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ServerError;
    use axum::http::StatusCode;
    use axum::response::IntoResponse;

    #[test]
    fn server_error_not_found_returns_404() {
        let err = ServerError::NotFound("episode not found".to_string());
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn server_error_validation_returns_422() {
        let err = ServerError::Validation("invalid field value".to_string());
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[test]
    fn server_error_internal_returns_500() {
        let err = ServerError::Internal("unexpected failure".to_string());
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn graphiti_node_not_found_converts_to_404() {
        let ge = crate::GraphitiError::NodeNotFound("uuid-xyz".to_string());
        let se: ServerError = ge.into();
        let response = se.into_response();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn graphiti_edge_not_found_converts_to_404() {
        let ge = crate::GraphitiError::EdgeNotFound("uuid-abc".to_string());
        let se: ServerError = ge.into();
        let response = se.into_response();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn graphiti_validation_error_converts_to_422() {
        let ge = crate::GraphitiError::Validation("bad input".to_string());
        let se: ServerError = ge.into();
        let response = se.into_response();
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[test]
    fn graphiti_driver_error_converts_to_500() {
        let ge = crate::GraphitiError::Driver("connection refused".to_string());
        let se: ServerError = ge.into();
        let response = se.into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn server_error_body_contains_message() {
        use axum::body::to_bytes;
        use axum::response::IntoResponse;

        let err = ServerError::NotFound("missing resource".to_string());
        let response = err.into_response();
        let body_bytes = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should be readable");
        let body: serde_json::Value =
            serde_json::from_slice(&body_bytes).expect("body should be JSON");
        assert!(
            body["error"]
                .as_str()
                .unwrap_or("")
                .contains("missing resource"),
            "error body must echo the message: {:?}",
            body
        );
    }
}
