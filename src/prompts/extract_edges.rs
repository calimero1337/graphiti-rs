//! Prompt for extracting relational edges between entity nodes.
//!
//! Given episode content and a list of already-extracted entities, instructs
//! the LLM to identify directed relationships between those entities and
//! return them as structured JSON with optional temporal metadata.

use crate::llm_client::{Message, Role};
use chrono::{DateTime, Utc};
use serde::Deserialize;

/// A single relationship extracted from episode content.
#[derive(Debug, Clone, Deserialize, schemars::JsonSchema)]
pub struct ExtractedEdge {
    /// Name of the source entity (must match one of the provided entity names).
    pub source_node: String,
    /// Name of the target entity (must match one of the provided entity names).
    pub target_node: String,
    /// Short, uppercase relation type label (e.g. "WORKS_AT", "KNOWS", "OWNS").
    pub relation_type: String,
    /// A complete, self-contained factual sentence describing the relationship.
    pub fact: String,
    /// ISO 8601 date/time when the relationship became true, if stated or implied.
    /// Use null when no temporal information is available.
    pub valid_at: Option<String>,
}

/// Top-level response schema returned by the LLM.
#[derive(Debug, Clone, Deserialize, schemars::JsonSchema)]
pub struct ExtractedEdges {
    /// All relationships found between the provided entities. May be empty.
    #[serde(default)]
    pub edges: Vec<ExtractedEdge>,
}

/// Parameters for building the extract-edges prompt.
pub struct ExtractEdgesContext<'a> {
    /// Raw content of the current episode.
    pub episode_content: &'a str,
    /// Canonical entity names extracted in the previous step.
    pub entities: &'a [String],
    /// Wall-clock time at which the episode was recorded (used as fallback for `valid_at`).
    pub reference_time: DateTime<Utc>,
    /// Optional summaries of immediately preceding episodes for co-reference resolution.
    pub previous_episodes: &'a [String],
}

/// Build the `[system, user]` messages for relationship extraction.
pub fn build_messages(ctx: &ExtractEdgesContext<'_>) -> Vec<Message> {
    let reference_time_str = ctx.reference_time.format("%Y-%m-%dT%H:%M:%SZ").to_string();

    let system_content = format!(
        "You are a knowledge-graph construction assistant. \
Your task is to extract directed relationships between entities from episode content.\n\
\n\
Guidelines:\n\
- Only extract relationships between entities in the provided entity list.\n\
- Each relationship must have a clear, directional meaning (source → target).\n\
- Use a short, uppercase SNAKE_CASE label for relation_type \
(e.g. WORKS_AT, LOCATED_IN, FOUNDED_BY, PART_OF).\n\
- Write a complete, self-contained fact sentence — a reader with no other \
context should understand the relationship.\n\
- For valid_at: use an ISO 8601 string if the episode states or clearly implies \
when the relationship became true; otherwise use null. \
The reference time for this episode is {reference_time_str}.\n\
- Do not fabricate relationships; only extract what the episode content supports.\n\
- Return valid JSON that conforms exactly to the requested schema."
    );

    let previous_context = build_previous_context(ctx.previous_episodes);
    let entity_list = build_entity_list(ctx.entities);

    let user_content = format!(
        "{previous_context}\
Known entities:\n\
{entity_list}\n\
Episode content:\n\
{episode_content}\n\
\n\
Extract all relationships between the known entities from the episode content above.",
        episode_content = ctx.episode_content,
    );

    vec![
        Message { role: Role::System, content: system_content },
        Message { role: Role::User, content: user_content },
    ]
}

fn build_entity_list(entities: &[String]) -> String {
    if entities.is_empty() {
        return "  (none)\n".to_string();
    }
    entities.iter().map(|e| format!("  - {e}\n")).collect()
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
    use chrono::TimeZone;

    fn reference_time() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2024, 6, 15, 12, 0, 0).unwrap()
    }

    fn make_ctx<'a>(
        content: &'a str,
        entities: &'a [String],
        prev: &'a [String],
    ) -> ExtractEdgesContext<'a> {
        ExtractEdgesContext {
            episode_content: content,
            entities,
            reference_time: reference_time(),
            previous_episodes: prev,
        }
    }

    #[test]
    fn returns_two_messages() {
        let entities = vec!["Alice".to_string(), "Acme Corp".to_string()];
        let ctx = make_ctx("Alice works at Acme Corp.", &entities, &[]);
        let msgs = build_messages(&ctx);
        assert_eq!(msgs.len(), 2);
    }

    #[test]
    fn first_message_is_system() {
        let ctx = make_ctx("foo", &[], &[]);
        let msgs = build_messages(&ctx);
        assert!(matches!(msgs[0].role, Role::System));
    }

    #[test]
    fn second_message_is_user() {
        let ctx = make_ctx("foo", &[], &[]);
        let msgs = build_messages(&ctx);
        assert!(matches!(msgs[1].role, Role::User));
    }

    #[test]
    fn episode_content_in_user_message() {
        let content = "Alice founded Acme Corp in 2005.";
        let entities = vec!["Alice".to_string(), "Acme Corp".to_string()];
        let ctx = make_ctx(content, &entities, &[]);
        let msgs = build_messages(&ctx);
        assert!(msgs[1].content.contains(content));
    }

    #[test]
    fn entities_listed_in_user_message() {
        let entities = vec!["Alice".to_string(), "Bob".to_string()];
        let ctx = make_ctx("...", &entities, &[]);
        let msgs = build_messages(&ctx);
        assert!(msgs[1].content.contains("Alice"));
        assert!(msgs[1].content.contains("Bob"));
    }

    #[test]
    fn reference_time_in_system_message() {
        let ctx = make_ctx("foo", &[], &[]);
        let msgs = build_messages(&ctx);
        assert!(msgs[0].content.contains("2024-06-15"));
    }

    #[test]
    fn previous_episodes_in_user_message() {
        let prev = vec!["Earlier summary".to_string()];
        let ctx = make_ctx("foo", &[], &prev);
        let msgs = build_messages(&ctx);
        assert!(msgs[1].content.contains("Earlier summary"));
    }

    #[test]
    fn no_previous_episodes_omits_section() {
        let ctx = make_ctx("foo", &[], &[]);
        let msgs = build_messages(&ctx);
        assert!(!msgs[1].content.contains("Previous episodes"));
    }

    #[test]
    fn empty_entity_list_shows_none() {
        let ctx = make_ctx("foo", &[], &[]);
        let msgs = build_messages(&ctx);
        assert!(msgs[1].content.contains("(none)"));
    }

    #[test]
    fn messages_non_empty() {
        let ctx = make_ctx("content", &[], &[]);
        let msgs = build_messages(&ctx);
        assert!(!msgs[0].content.is_empty());
        assert!(!msgs[1].content.is_empty());
    }
}
