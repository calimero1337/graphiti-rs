//! Search subsystem — hybrid BM25 + vector retrieval with RRF fusion.

pub mod config;
pub mod engine;
pub mod result;
pub mod rrf;

pub use config::SearchConfig;
pub use engine::SearchEngine;
pub use result::SearchResults;
pub use rrf::{reciprocal_rank_fusion, DEFAULT_K};
