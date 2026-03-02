//! Prompt for deduplicating relational edges against existing edges.
//!
//! Given a newly-extracted edge (as a fact sentence) and the existing edges
//! between the same pair of nodes, instructs the LLM to decide whether the
//! new edge duplicates one of the existing ones.

use crate::llm_client::{Message, Role};
use serde::Deserialize;

/// The LLM's deduplication decision for an incoming edge.
#[derive(Debug, Clone, Deserialize, schemars::JsonSchema)]
pub struct EdgeDuplicateResult {
    /// 0-based index of the existing edge this duplicates, or null if it is new.
    ///
    /// Corresponds to the index shown in the "Existing edges" list in the prompt.
    pub duplicate_of_index: Option<usize>,
    /// Short explanation of the decision (useful for auditing).
    pub reason: String,
}

/// A concise representation of an existing edge used for comparison.
#[derive(Debug, Clone)]
pub struct ExistingEdgeStub<'a> {
    /// Position in the list (0-based), echoed to the LLM as a label.
    pub index: usize,
    /// The human-readable fact sentence for this edge.
    pub fact: &'a str,
    /// The relation type label (e.g. "WORKS_AT").
    pub relation_type: &'a str,
}

/// Parameters for building the dedupe-edges prompt.
pub struct DedupeEdgesContext<'a> {
    /// The fact sentence for the newly-extracted edge.
    pub new_edge_fact: &'a str,
    /// The relation type of the new edge.
    pub new_edge_relation_type: &'a str,
    /// Existing edges between the same source and target nodes.
    pub existing_edges: &'a [ExistingEdgeStub<'a>],
}

/// Build the `[system, user]` messages for edge deduplication.
pub fn build_messages(ctx: &DedupeEdgesContext<'_>) -> Vec<Message> {
    let system_content = "\
You are a knowledge-graph curation assistant. \
Your task is to decide whether a newly-extracted relationship is a duplicate \
of an existing relationship between the same pair of entities.\n\
\n\
Guidelines:\n\
- Two edges are duplicates if they express the same underlying fact, even if \
worded differently (e.g. \"Alice is CEO of Acme\" and \"Alice leads Acme Corp\" \
are duplicates if Acme and Acme Corp are the same organisation).\n\
- Do NOT flag an edge as a duplicate merely because it shares a relation type; \
the specific fact must be the same.\n\
- If the new edge is a duplicate, set duplicate_of_index to the 0-based index \
of the matching existing edge.\n\
- If it is a genuinely new fact, set duplicate_of_index to null.\n\
- Always provide a concise reason for your decision.\n\
- Return valid JSON that conforms exactly to the requested schema."
        .to_string();

    let existing_list = build_existing_list(ctx.existing_edges);

    let user_content = format!(
        "New edge to evaluate:\n\
  relation_type: {new_relation_type}\n\
  fact: {new_fact}\n\
\n\
Existing edges between the same node pair:\n\
{existing_list}\n\
Is the new edge a duplicate of any existing edge?",
        new_relation_type = ctx.new_edge_relation_type,
        new_fact = ctx.new_edge_fact,
    );

    vec![
        Message { role: Role::System, content: system_content },
        Message { role: Role::User, content: user_content },
    ]
}

fn build_existing_list(edges: &[ExistingEdgeStub<'_>]) -> String {
    if edges.is_empty() {
        return "  (none)\n".to_string();
    }
    edges
        .iter()
        .map(|e| format!("  [{}] ({}) {}\n", e.index, e.relation_type, e.fact))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ctx<'a>(
        new_fact: &'a str,
        new_relation: &'a str,
        existing: &'a [ExistingEdgeStub<'a>],
    ) -> DedupeEdgesContext<'a> {
        DedupeEdgesContext {
            new_edge_fact: new_fact,
            new_edge_relation_type: new_relation,
            existing_edges: existing,
        }
    }

    #[test]
    fn returns_two_messages() {
        let ctx = make_ctx("Alice works at Acme.", "WORKS_AT", &[]);
        assert_eq!(build_messages(&ctx).len(), 2);
    }

    #[test]
    fn first_message_is_system() {
        let ctx = make_ctx("Alice works at Acme.", "WORKS_AT", &[]);
        let msgs = build_messages(&ctx);
        assert!(matches!(msgs[0].role, Role::System));
    }

    #[test]
    fn second_message_is_user() {
        let ctx = make_ctx("Alice works at Acme.", "WORKS_AT", &[]);
        let msgs = build_messages(&ctx);
        assert!(matches!(msgs[1].role, Role::User));
    }

    #[test]
    fn new_fact_in_user_message() {
        let fact = "Alice is the CEO of Acme Corp.";
        let ctx = make_ctx(fact, "LEADS", &[]);
        let msgs = build_messages(&ctx);
        assert!(msgs[1].content.contains(fact));
    }

    #[test]
    fn new_relation_type_in_user_message() {
        let ctx = make_ctx("foo", "PART_OF", &[]);
        let msgs = build_messages(&ctx);
        assert!(msgs[1].content.contains("PART_OF"));
    }

    #[test]
    fn existing_edges_in_user_message() {
        let existing = [ExistingEdgeStub {
            index: 0,
            fact: "Alice founded Acme.",
            relation_type: "FOUNDED",
        }];
        let ctx = make_ctx("foo", "BAR", &existing);
        let msgs = build_messages(&ctx);
        assert!(msgs[1].content.contains("Alice founded Acme."));
    }

    #[test]
    fn empty_existing_shows_none() {
        let ctx = make_ctx("foo", "BAR", &[]);
        let msgs = build_messages(&ctx);
        assert!(msgs[1].content.contains("(none)"));
    }

    #[test]
    fn messages_non_empty() {
        let ctx = make_ctx("fact", "REL", &[]);
        let msgs = build_messages(&ctx);
        assert!(!msgs[0].content.is_empty());
        assert!(!msgs[1].content.is_empty());
    }
}
