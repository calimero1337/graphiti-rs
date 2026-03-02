//! Shared utilities.
//!
//! Includes:
//! - Date/time helpers (bi-temporal timestamp formatting for Cypher)
//! - String normalization helpers
//! - Concurrency helpers (semaphore-bounded `join_all`)
//! - Cache key generation (MD5 of content for embedding caches)

pub mod datetime;
pub mod similarity;
pub mod text;

pub use datetime::{format_neo4j_datetime, parse_flexible_datetime};
pub use similarity::{cosine_similarity, normalize_l2};
pub use text::{
    extract_json_from_response, lucene_sanitize, normalize_whitespace, truncate_with_ellipsis,
};
