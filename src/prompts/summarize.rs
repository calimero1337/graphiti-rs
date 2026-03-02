//! Prompts for generating entity and community summaries.
//!
//! Provides two `build_messages` functions:
//! - [`build_entity_messages`] — summarise a single entity from its related facts.
//! - [`build_community_messages`] — summarise a community from its member entity summaries.

use crate::llm_client::{Message, Role};
use serde::Deserialize;

/// A natural-language summary produced by the LLM.
#[derive(Debug, Clone, Deserialize, schemars::JsonSchema)]
pub struct Summary {
    /// Concise, factual summary (2–5 sentences) covering the key attributes
    /// and relationships of the entity or community.
    pub summary: String,
}

// ---------------------------------------------------------------------------
// Entity summarisation
// ---------------------------------------------------------------------------

/// Parameters for building an entity-summary prompt.
pub struct EntitySummaryContext<'a> {
    /// Canonical name of the entity being summarised.
    pub entity_name: &'a str,
    /// The entity's current summary (may be empty for new entities).
    pub existing_summary: &'a str,
    /// Fact sentences drawn from the entity's related edges.
    pub facts: &'a [String],
}

/// Build the `[system, user]` messages for entity summarisation.
pub fn build_entity_messages(ctx: &EntitySummaryContext<'_>) -> Vec<Message> {
    let system_content = "\
You are a knowledge-graph documentation assistant. \
Your task is to write a concise, factual summary for a named entity based on \
the facts known about it in the knowledge graph.\n\
\n\
Guidelines:\n\
- Write 2–5 sentences that capture the most important attributes and \
relationships of the entity.\n\
- Prefer specific, verifiable facts over vague generalisations.\n\
- If an existing summary is provided, use it as a starting point and update it \
with any new information from the fact list.\n\
- Do not speculate or add information that is not supported by the provided facts.\n\
- Return valid JSON that conforms exactly to the requested schema."
        .to_string();

    let existing_section = if ctx.existing_summary.is_empty() {
        String::from("Existing summary: (none)\n")
    } else {
        format!("Existing summary:\n{}\n", ctx.existing_summary)
    };

    let facts_section = build_facts_section(ctx.facts);

    let user_content = format!(
        "Entity: {entity_name}\n\
\n\
{existing_section}\n\
Known facts:\n\
{facts_section}\n\
Write a concise summary for this entity.",
        entity_name = ctx.entity_name,
    );

    vec![
        Message { role: Role::System, content: system_content },
        Message { role: Role::User, content: user_content },
    ]
}

// ---------------------------------------------------------------------------
// Community summarisation
// ---------------------------------------------------------------------------

/// Parameters for building a community-summary prompt.
pub struct CommunitySummaryContext<'a> {
    /// Summaries of the individual entities that belong to this community.
    pub entity_summaries: &'a [String],
}

/// Build the `[system, user]` messages for community summarisation.
pub fn build_community_messages(ctx: &CommunitySummaryContext<'_>) -> Vec<Message> {
    let system_content = "\
You are a knowledge-graph documentation assistant. \
Your task is to write a concise, coherent summary for a community of related \
entities in a knowledge graph.\n\
\n\
Guidelines:\n\
- Synthesise the individual entity summaries into a single overview that \
describes the community's shared theme, domain, or purpose.\n\
- Write 2–5 sentences. Do not repeat every entity; identify common threads.\n\
- Do not speculate or add information beyond what the entity summaries contain.\n\
- Return valid JSON that conforms exactly to the requested schema."
        .to_string();

    let summaries_section = build_entity_summaries_section(ctx.entity_summaries);

    let user_content = format!(
        "Entity summaries for this community:\n\
{summaries_section}\n\
Write a concise summary that describes the community as a whole."
    );

    vec![
        Message { role: Role::System, content: system_content },
        Message { role: Role::User, content: user_content },
    ]
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_facts_section(facts: &[String]) -> String {
    if facts.is_empty() {
        return "  (none)\n".to_string();
    }
    facts.iter().enumerate().map(|(i, f)| format!("  {}. {f}\n", i + 1)).collect()
}

fn build_entity_summaries_section(summaries: &[String]) -> String {
    if summaries.is_empty() {
        return "  (none)\n".to_string();
    }
    summaries.iter().enumerate().map(|(i, s)| format!("  {}. {s}\n", i + 1)).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- entity ---

    fn make_entity_ctx<'a>(
        name: &'a str,
        existing: &'a str,
        facts: &'a [String],
    ) -> EntitySummaryContext<'a> {
        EntitySummaryContext { entity_name: name, existing_summary: existing, facts }
    }

    #[test]
    fn entity_returns_two_messages() {
        let ctx = make_entity_ctx("Alice", "", &[]);
        assert_eq!(build_entity_messages(&ctx).len(), 2);
    }

    #[test]
    fn entity_first_message_is_system() {
        let ctx = make_entity_ctx("Alice", "", &[]);
        let msgs = build_entity_messages(&ctx);
        assert!(matches!(msgs[0].role, Role::System));
    }

    #[test]
    fn entity_second_message_is_user() {
        let ctx = make_entity_ctx("Alice", "", &[]);
        let msgs = build_entity_messages(&ctx);
        assert!(matches!(msgs[1].role, Role::User));
    }

    #[test]
    fn entity_name_in_user_message() {
        let ctx = make_entity_ctx("Alice Smith", "", &[]);
        let msgs = build_entity_messages(&ctx);
        assert!(msgs[1].content.contains("Alice Smith"));
    }

    #[test]
    fn entity_facts_in_user_message() {
        let facts = vec!["Alice founded Acme in 2010.".to_string()];
        let ctx = make_entity_ctx("Alice", "", &facts);
        let msgs = build_entity_messages(&ctx);
        assert!(msgs[1].content.contains("Alice founded Acme in 2010."));
    }

    #[test]
    fn entity_existing_summary_in_user_message() {
        let ctx = make_entity_ctx("Alice", "Alice is a CEO.", &[]);
        let msgs = build_entity_messages(&ctx);
        assert!(msgs[1].content.contains("Alice is a CEO."));
    }

    #[test]
    fn entity_no_existing_summary_shows_none() {
        let ctx = make_entity_ctx("Alice", "", &[]);
        let msgs = build_entity_messages(&ctx);
        assert!(msgs[1].content.contains("(none)"));
    }

    #[test]
    fn entity_messages_non_empty() {
        let ctx = make_entity_ctx("Ent", "sum", &[]);
        let msgs = build_entity_messages(&ctx);
        assert!(!msgs[0].content.is_empty());
        assert!(!msgs[1].content.is_empty());
    }

    // --- community ---

    fn make_community_ctx(summaries: &[String]) -> CommunitySummaryContext<'_> {
        CommunitySummaryContext { entity_summaries: summaries }
    }

    #[test]
    fn community_returns_two_messages() {
        let ctx = make_community_ctx(&[]);
        assert_eq!(build_community_messages(&ctx).len(), 2);
    }

    #[test]
    fn community_first_message_is_system() {
        let ctx = make_community_ctx(&[]);
        let msgs = build_community_messages(&ctx);
        assert!(matches!(msgs[0].role, Role::System));
    }

    #[test]
    fn community_second_message_is_user() {
        let ctx = make_community_ctx(&[]);
        let msgs = build_community_messages(&ctx);
        assert!(matches!(msgs[1].role, Role::User));
    }

    #[test]
    fn community_summaries_in_user_message() {
        let summaries = vec!["Alice is a CEO.".to_string(), "Acme is a tech company.".to_string()];
        let ctx = make_community_ctx(&summaries);
        let msgs = build_community_messages(&ctx);
        assert!(msgs[1].content.contains("Alice is a CEO."));
        assert!(msgs[1].content.contains("Acme is a tech company."));
    }

    #[test]
    fn community_empty_summaries_shows_none() {
        let ctx = make_community_ctx(&[]);
        let msgs = build_community_messages(&ctx);
        assert!(msgs[1].content.contains("(none)"));
    }

    #[test]
    fn community_messages_non_empty() {
        let ctx = make_community_ctx(&[]);
        let msgs = build_community_messages(&ctx);
        assert!(!msgs[0].content.is_empty());
        assert!(!msgs[1].content.is_empty());
    }
}
