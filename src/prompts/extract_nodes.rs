//! Prompt for extracting entity nodes from episode content.
//!
//! Given raw episode content and metadata, instructs the LLM to identify
//! named entities (people, organisations, places, concepts, etc.) and
//! return them as structured JSON.

use crate::llm_client::{Message, Role};
use crate::nodes::episodic::EpisodeType;
use serde::Deserialize;

/// A single entity extracted from episode content.
#[derive(Debug, Clone, Deserialize, schemars::JsonSchema)]
pub struct ExtractedEntity {
    /// Canonical name for the entity (e.g. "Alice Smith", "Acme Corp").
    pub name: String,
    /// Broad category label (e.g. "Person", "Organization", "Location", "Concept").
    pub entity_type: String,
    /// One-sentence description of the entity as mentioned in the episode.
    pub summary: String,
}

/// Top-level response schema returned by the LLM.
#[derive(Debug, Clone, Deserialize, schemars::JsonSchema)]
pub struct ExtractedEntities {
    /// All entities found in the episode. May be empty if none are present.
    #[serde(default)]
    pub entities: Vec<ExtractedEntity>,
}

/// Parameters for building the extract-nodes prompt.
pub struct ExtractNodesContext<'a> {
    /// Raw content of the current episode.
    pub episode_content: &'a str,
    /// Human-readable description of the episode source (e.g. "Slack message from Alice").
    pub source_description: &'a str,
    /// Format of the episode: conversational message, plain text, or JSON record.
    pub episode_type: &'a EpisodeType,
    /// Optional summaries of immediately preceding episodes for co-reference resolution.
    pub previous_episodes: &'a [String],
}

/// Build the `[system, user]` messages for entity extraction.
pub fn build_messages(ctx: &ExtractNodesContext<'_>) -> Vec<Message> {
    let type_hint = episode_type_hint(ctx.episode_type);

    let system_content = format!(
        "You are a knowledge-graph construction assistant. \
Your task is to extract named entities from {type_hint}.\n\
\n\
Guidelines:\n\
- Extract only concrete, nameable entities: people, organisations, locations, \
products, events, and well-defined concepts.\n\
- Use the most specific, canonical form of each name (e.g. prefer \
\"United Kingdom\" over \"UK\" if the full form appears).\n\
- Assign a short entity_type label such as Person, Organization, Location, \
Product, Event, or Concept.\n\
- Write a one-sentence summary that captures what the episode says about \
the entity — do not add outside knowledge.\n\
- If no entities are present, return an empty list.\n\
- Return valid JSON that conforms exactly to the requested schema."
    );

    let previous_context = build_previous_context(ctx.previous_episodes);

    let user_content = format!(
        "Source: {source_description}\n\
\n\
{previous_context}\
Episode content:\n\
{episode_content}\n\
\n\
Extract all named entities from the episode content above.",
        source_description = ctx.source_description,
        episode_content = ctx.episode_content,
    );

    vec![
        Message { role: Role::System, content: system_content },
        Message { role: Role::User, content: user_content },
    ]
}

fn episode_type_hint(episode_type: &EpisodeType) -> &'static str {
    match episode_type {
        EpisodeType::Message => "a conversational message",
        EpisodeType::Text => "a text document",
        EpisodeType::Json => "a structured JSON record",
    }
}

fn build_previous_context(previous_episodes: &[String]) -> String {
    if previous_episodes.is_empty() {
        return String::new();
    }
    let mut out = String::from("Previous episodes (for context):\n");
    for (i, ep) in previous_episodes.iter().enumerate() {
        out.push_str(&format!("  {}. {ep}\n", i + 1));
    }
    out.push('\n');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ctx<'a>(
        content: &'a str,
        source: &'a str,
        ep_type: &'a EpisodeType,
        prev: &'a [String],
    ) -> ExtractNodesContext<'a> {
        ExtractNodesContext {
            episode_content: content,
            source_description: source,
            episode_type: ep_type,
            previous_episodes: prev,
        }
    }

    #[test]
    fn returns_two_messages() {
        let ctx = make_ctx("Alice met Bob at Acme.", "chat", &EpisodeType::Message, &[]);
        let msgs = build_messages(&ctx);
        assert_eq!(msgs.len(), 2);
    }

    #[test]
    fn first_message_is_system() {
        let ctx = make_ctx("Alice met Bob.", "chat", &EpisodeType::Message, &[]);
        let msgs = build_messages(&ctx);
        assert!(matches!(msgs[0].role, Role::System));
    }

    #[test]
    fn second_message_is_user() {
        let ctx = make_ctx("Alice met Bob.", "chat", &EpisodeType::Message, &[]);
        let msgs = build_messages(&ctx);
        assert!(matches!(msgs[1].role, Role::User));
    }

    #[test]
    fn episode_content_appears_in_user_message() {
        let content = "CEO Jane Doe signed the contract with TechCorp.";
        let ctx = make_ctx(content, "email", &EpisodeType::Text, &[]);
        let msgs = build_messages(&ctx);
        assert!(msgs[1].content.contains(content));
    }

    #[test]
    fn source_description_appears_in_user_message() {
        let ctx = make_ctx("foo", "Slack DM from Alice", &EpisodeType::Message, &[]);
        let msgs = build_messages(&ctx);
        assert!(msgs[1].content.contains("Slack DM from Alice"));
    }

    #[test]
    fn type_hint_message_appears_in_system() {
        let ctx = make_ctx("foo", "chat", &EpisodeType::Message, &[]);
        let msgs = build_messages(&ctx);
        assert!(msgs[0].content.contains("conversational message"));
    }

    #[test]
    fn type_hint_text_appears_in_system() {
        let ctx = make_ctx("foo", "doc", &EpisodeType::Text, &[]);
        let msgs = build_messages(&ctx);
        assert!(msgs[0].content.contains("text document"));
    }

    #[test]
    fn type_hint_json_appears_in_system() {
        let ctx = make_ctx("{}", "api", &EpisodeType::Json, &[]);
        let msgs = build_messages(&ctx);
        assert!(msgs[0].content.contains("JSON record"));
    }

    #[test]
    fn previous_episodes_appear_in_user_message() {
        let prev = vec!["Episode 1 summary".to_string()];
        let ctx = make_ctx("foo", "chat", &EpisodeType::Message, &prev);
        let msgs = build_messages(&ctx);
        assert!(msgs[1].content.contains("Episode 1 summary"));
    }

    #[test]
    fn no_previous_episodes_omits_section() {
        let ctx = make_ctx("foo", "chat", &EpisodeType::Message, &[]);
        let msgs = build_messages(&ctx);
        assert!(!msgs[1].content.contains("Previous episodes"));
    }

    #[test]
    fn system_and_user_messages_non_empty() {
        let ctx = make_ctx("content", "source", &EpisodeType::Text, &[]);
        let msgs = build_messages(&ctx);
        assert!(!msgs[0].content.is_empty());
        assert!(!msgs[1].content.is_empty());
    }
}
