//! Prompt for deduplicating entity nodes against an existing graph.
//!
//! Given a batch of newly-extracted entities and a list of existing entities
//! already in the graph, instructs the LLM to decide — for each extracted
//! entity — whether it refers to the same real-world thing as an existing one.

use crate::llm_client::{Message, Role};
use serde::Deserialize;
use uuid::Uuid;

/// Resolution decision for a single extracted entity.
#[derive(Debug, Clone, Deserialize, schemars::JsonSchema)]
pub struct NodeResolution {
    /// The extracted entity name being evaluated.
    pub extracted_name: String,
    /// UUID of the existing entity this one duplicates, or null if it is new.
    pub duplicate_of: Option<String>,
}

/// Top-level response schema returned by the LLM.
#[derive(Debug, Clone, Deserialize, schemars::JsonSchema)]
pub struct NodeResolutions {
    /// One resolution per extracted entity, in the same order as the input list.
    #[serde(default)]
    pub resolutions: Vec<NodeResolution>,
}

/// A stub of an existing entity used for deduplication lookup.
#[derive(Debug, Clone)]
pub struct ExistingEntityStub<'a> {
    pub uuid: Uuid,
    pub name: &'a str,
    pub summary: &'a str,
}

/// Parameters for building the dedupe-nodes prompt.
pub struct DedupeNodesContext<'a> {
    /// Names (and optionally summaries) of newly-extracted entities to evaluate.
    pub extracted_nodes: &'a [ExtractedNodeStub<'a>],
    /// Existing entities from the graph to compare against.
    pub existing_nodes: &'a [ExistingEntityStub<'a>],
    /// The episode content, for additional co-reference context.
    pub episode_content: &'a str,
}

/// A lightweight representation of a freshly-extracted entity.
#[derive(Debug, Clone)]
pub struct ExtractedNodeStub<'a> {
    pub name: &'a str,
    pub summary: &'a str,
}

/// Build the `[system, user]` messages for node deduplication.
pub fn build_messages(ctx: &DedupeNodesContext<'_>) -> Vec<Message> {
    let system_content = "\
You are a knowledge-graph curation assistant. \
Your task is to decide whether newly-extracted entities are duplicates of \
entities that already exist in the knowledge graph.\n\
\n\
Guidelines:\n\
- Compare each extracted entity against the full list of existing entities.\n\
- Two entities are duplicates only if they refer to the same real-world object \
(same person, organisation, location, concept, etc.).\n\
- Minor name variations count as duplicates (e.g. \"IBM\" and \"International Business Machines\").\n\
- Different people with the same name are NOT duplicates unless the episode \
confirms they are the same individual.\n\
- If an extracted entity is a duplicate, set duplicate_of to the UUID string of \
the matching existing entity.\n\
- If it is a genuinely new entity, set duplicate_of to null.\n\
- Return one resolution per extracted entity, in the same order as the input list.\n\
- Return valid JSON that conforms exactly to the requested schema."
        .to_string();

    let extracted_list = build_extracted_list(ctx.extracted_nodes);
    let existing_list = build_existing_list(ctx.existing_nodes);

    let user_content = format!(
        "Episode content (for context):\n\
{episode_content}\n\
\n\
Newly-extracted entities to evaluate:\n\
{extracted_list}\n\
Existing entities in the knowledge graph:\n\
{existing_list}\n\
For each extracted entity, decide if it is a duplicate of an existing entity.",
        episode_content = ctx.episode_content,
    );

    vec![
        Message { role: Role::System, content: system_content },
        Message { role: Role::User, content: user_content },
    ]
}

fn build_extracted_list(nodes: &[ExtractedNodeStub<'_>]) -> String {
    if nodes.is_empty() {
        return "  (none)\n".to_string();
    }
    nodes
        .iter()
        .enumerate()
        .map(|(i, n)| format!("  {}. {} — {}\n", i + 1, n.name, n.summary))
        .collect()
}

fn build_existing_list(nodes: &[ExistingEntityStub<'_>]) -> String {
    if nodes.is_empty() {
        return "  (none — this is a fresh graph)\n".to_string();
    }
    nodes
        .iter()
        .map(|n| format!("  [{}] {} — {}\n", n.uuid, n.name, n.summary))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uuid_a() -> Uuid {
        Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap()
    }

    fn make_ctx<'a>(
        extracted: &'a [ExtractedNodeStub<'a>],
        existing: &'a [ExistingEntityStub<'a>],
        content: &'a str,
    ) -> DedupeNodesContext<'a> {
        DedupeNodesContext {
            extracted_nodes: extracted,
            existing_nodes: existing,
            episode_content: content,
        }
    }

    #[test]
    fn returns_two_messages() {
        let ctx = make_ctx(&[], &[], "foo");
        assert_eq!(build_messages(&ctx).len(), 2);
    }

    #[test]
    fn first_message_is_system() {
        let ctx = make_ctx(&[], &[], "foo");
        let msgs = build_messages(&ctx);
        assert!(matches!(msgs[0].role, Role::System));
    }

    #[test]
    fn second_message_is_user() {
        let ctx = make_ctx(&[], &[], "foo");
        let msgs = build_messages(&ctx);
        assert!(matches!(msgs[1].role, Role::User));
    }

    #[test]
    fn extracted_names_in_user_message() {
        let extracted = [ExtractedNodeStub { name: "Alice", summary: "A person" }];
        let ctx = make_ctx(&extracted, &[], "Alice said hello.");
        let msgs = build_messages(&ctx);
        assert!(msgs[1].content.contains("Alice"));
    }

    #[test]
    fn existing_uuid_in_user_message() {
        let id = uuid_a();
        let existing = [ExistingEntityStub { uuid: id, name: "Bob", summary: "A person" }];
        let ctx = make_ctx(&[], &existing, "foo");
        let msgs = build_messages(&ctx);
        assert!(msgs[1].content.contains(&id.to_string()));
    }

    #[test]
    fn episode_content_in_user_message() {
        let content = "Alice joined Acme Corp last Monday.";
        let ctx = make_ctx(&[], &[], content);
        let msgs = build_messages(&ctx);
        assert!(msgs[1].content.contains(content));
    }

    #[test]
    fn empty_extracted_shows_none() {
        let ctx = make_ctx(&[], &[], "foo");
        let msgs = build_messages(&ctx);
        assert!(msgs[1].content.contains("(none)"));
    }

    #[test]
    fn empty_existing_shows_fresh_graph_note() {
        let ctx = make_ctx(&[], &[], "foo");
        let msgs = build_messages(&ctx);
        assert!(msgs[1].content.contains("fresh graph"));
    }

    #[test]
    fn messages_non_empty() {
        let ctx = make_ctx(&[], &[], "content");
        let msgs = build_messages(&ctx);
        assert!(!msgs[0].content.is_empty());
        assert!(!msgs[1].content.is_empty());
    }
}
