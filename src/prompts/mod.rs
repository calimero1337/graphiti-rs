//! Prompt templates for LLM interactions.
//!
//! Mirrors the Python `graphiti_core.prompts` package.
//! Each submodule handles prompts for a specific pipeline stage.
//!
//! Each module exposes:
//! - A `build_messages(ctx)` function returning `Vec<Message>`.
//!   The [`summarize`] module is an exception: it exposes two build functions,
//!   [`summarize::build_entity_messages`] and [`summarize::build_community_messages`],
//!   one for each summarisation task.
//! - A response struct that derives `Deserialize + schemars::JsonSchema`
//!   for use with [`crate::llm_client::LlmClient::generate_structured`].
//!
//! Prompts are stored as Rust string literals (not external files) for
//! compile-time inclusion and zero-cost access.

pub mod dedupe_edges;
pub mod dedupe_nodes;
pub mod extract_edges;
pub mod extract_nodes;
pub mod resolve_contradictions;
pub mod summarize;
