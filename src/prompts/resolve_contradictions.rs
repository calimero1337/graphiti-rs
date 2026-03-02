//! Prompt for detecting temporal contradictions between edges.
//!
//! When a newly-extracted fact potentially conflicts with an existing fact
//! in the knowledge graph (same entity pair, same or similar relation type),
//! this prompt asks the LLM whether the new fact logically invalidates the old
//! one — enabling bi-temporal expiry of superseded edges.

use crate::llm_client::{Message, Role};
use serde::Deserialize;

/// The LLM's contradiction decision for a pair of facts.
#[derive(Debug, Clone, Deserialize, schemars::JsonSchema)]
pub struct ContradictionResult {
    /// True if the new fact logically supersedes (invalidates) the existing fact.
    pub invalidates: bool,
    /// Explanation of why the existing fact is or is not invalidated.
    pub reason: String,
}

/// Parameters for building the contradiction-resolution prompt.
pub struct ResolveContradictionsContext<'a> {
    /// The new fact that was just extracted from the latest episode.
    pub new_fact: &'a str,
    /// The relation type of the new edge (e.g. "WORKS_AT").
    pub new_relation_type: &'a str,
    /// The existing fact already stored in the knowledge graph.
    pub existing_fact: &'a str,
    /// The relation type of the existing edge.
    pub existing_relation_type: &'a str,
    /// Human-readable timestamp when the new episode was recorded (context only).
    pub reference_time: &'a str,
}

/// Build the `[system, user]` messages for contradiction resolution.
pub fn build_messages(ctx: &ResolveContradictionsContext<'_>) -> Vec<Message> {
    let system_content = "\
You are a knowledge-graph temporal-reasoning assistant. \
Your task is to decide whether a new fact invalidates (supersedes) an existing \
fact in the knowledge graph.\n\
\n\
Guidelines:\n\
- Facts are contradictory when they cannot both be true at the same time \
(e.g. a person cannot hold two different job titles simultaneously).\n\
- If the new fact represents a change of state that makes the existing fact \
no longer true, set invalidates to true.\n\
- If both facts can be simultaneously true, or they refer to different time \
periods that do not conflict, set invalidates to false.\n\
- An addition is NOT a contradiction (e.g. a new role at a second employer does \
not invalidate an existing role at a first employer).\n\
- Always provide a concise, specific reason for your decision.\n\
- Return valid JSON that conforms exactly to the requested schema."
        .to_string();

    let user_content = format!(
        "Reference time (when the new episode was recorded): {reference_time}\n\
\n\
Existing fact in the knowledge graph:\n\
  relation_type: {existing_relation_type}\n\
  fact: {existing_fact}\n\
\n\
New fact just extracted:\n\
  relation_type: {new_relation_type}\n\
  fact: {new_fact}\n\
\n\
Does the new fact invalidate (supersede) the existing fact?",
        reference_time = ctx.reference_time,
        existing_relation_type = ctx.existing_relation_type,
        existing_fact = ctx.existing_fact,
        new_relation_type = ctx.new_relation_type,
        new_fact = ctx.new_fact,
    );

    vec![
        Message { role: Role::System, content: system_content },
        Message { role: Role::User, content: user_content },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ctx<'a>(
        new_fact: &'a str,
        new_rel: &'a str,
        existing_fact: &'a str,
        existing_rel: &'a str,
    ) -> ResolveContradictionsContext<'a> {
        ResolveContradictionsContext {
            new_fact,
            new_relation_type: new_rel,
            existing_fact,
            existing_relation_type: existing_rel,
            reference_time: "2024-06-15T12:00:00Z",
        }
    }

    #[test]
    fn returns_two_messages() {
        let ctx = make_ctx("Alice is VP at Acme.", "WORKS_AT", "Alice is engineer at Acme.", "WORKS_AT");
        assert_eq!(build_messages(&ctx).len(), 2);
    }

    #[test]
    fn first_message_is_system() {
        let ctx = make_ctx("a", "R", "b", "R");
        let msgs = build_messages(&ctx);
        assert!(matches!(msgs[0].role, Role::System));
    }

    #[test]
    fn second_message_is_user() {
        let ctx = make_ctx("a", "R", "b", "R");
        let msgs = build_messages(&ctx);
        assert!(matches!(msgs[1].role, Role::User));
    }

    #[test]
    fn new_fact_in_user_message() {
        let new_fact = "Alice is now CTO of Acme.";
        let ctx = make_ctx(new_fact, "LEADS", "Alice is VP at Acme.", "WORKS_AT");
        let msgs = build_messages(&ctx);
        assert!(msgs[1].content.contains(new_fact));
    }

    #[test]
    fn existing_fact_in_user_message() {
        let existing = "Alice was VP at Acme until 2023.";
        let ctx = make_ctx("Alice is CTO.", "LEADS", existing, "WORKS_AT");
        let msgs = build_messages(&ctx);
        assert!(msgs[1].content.contains(existing));
    }

    #[test]
    fn reference_time_in_user_message() {
        let ctx = make_ctx("a", "R", "b", "R");
        let msgs = build_messages(&ctx);
        assert!(msgs[1].content.contains("2024-06-15T12:00:00Z"));
    }

    #[test]
    fn relation_types_in_user_message() {
        let ctx = make_ctx("a", "NEW_REL", "b", "OLD_REL");
        let msgs = build_messages(&ctx);
        assert!(msgs[1].content.contains("NEW_REL"));
        assert!(msgs[1].content.contains("OLD_REL"));
    }

    #[test]
    fn messages_non_empty() {
        let ctx = make_ctx("new", "R1", "old", "R2");
        let msgs = build_messages(&ctx);
        assert!(!msgs[0].content.is_empty());
        assert!(!msgs[1].content.is_empty());
    }
}
