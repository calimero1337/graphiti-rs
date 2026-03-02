//! Label propagation community detection on the entity graph.
//!
//! # Algorithm
//!
//! 1. Build an undirected [`petgraph::graph::UnGraph`] from the supplied node UUIDs
//!    and edge pairs.
//! 2. Initialise each node's label to its own UUID.
//! 3. Iterate (up to `max_iterations`): for each node (in deterministic, sorted order)
//!    adopt the most frequent label among its neighbours; break ties by choosing the
//!    lexicographically smallest label UUID.
//! 4. Stop early when no label changes in a full pass (convergence).
//! 5. Return a map from *winning label UUID* → *member node UUIDs*.

use std::collections::HashMap;

use petgraph::graph::NodeIndex;
use petgraph::prelude::UnGraph;
use uuid::Uuid;

/// An undirected connection between two entity nodes, identified by their UUIDs.
pub struct EdgeSpec {
    pub source: Uuid,
    pub target: Uuid,
}

/// Run label propagation and return a map of `community_label → member node UUIDs`.
///
/// * `node_ids`       — all entity node UUIDs to include (duplicates are ignored).
/// * `edges`          — connectivity; endpoints not present in `node_ids` are silently skipped.
/// * `max_iterations` — upper bound on propagation rounds (convergence may stop it sooner).
pub fn detect_communities(
    node_ids: &[Uuid],
    edges: &[EdgeSpec],
    max_iterations: usize,
) -> HashMap<Uuid, Vec<Uuid>> {
    if node_ids.is_empty() {
        return HashMap::new();
    }

    // ── Build petgraph ────────────────────────────────────────────────────────

    let mut graph: UnGraph<Uuid, ()> = UnGraph::new_undirected();

    // UUID → petgraph NodeIndex
    let mut uuid_to_idx: HashMap<Uuid, NodeIndex> = HashMap::with_capacity(node_ids.len());
    // Ordered list of (NodeIndex, UUID) for deterministic iteration
    let mut idx_to_uuid: Vec<(NodeIndex, Uuid)> = Vec::with_capacity(node_ids.len());

    for &id in node_ids {
        // Deduplicate: only add each UUID once.
        if uuid_to_idx.contains_key(&id) {
            continue;
        }
        let idx = graph.add_node(id);
        uuid_to_idx.insert(id, idx);
        idx_to_uuid.push((idx, id));
    }

    for edge in edges {
        match (uuid_to_idx.get(&edge.source), uuid_to_idx.get(&edge.target)) {
            (Some(&a), Some(&b)) if a != b => {
                graph.add_edge(a, b, ());
            }
            _ => {} // Skip self-loops or edges to unknown nodes.
        }
    }

    // ── Initialise labels ────────────────────────────────────────────────────
    // labels[NodeIndex.index()] = current community label
    let n = graph.node_count();
    let mut labels: Vec<Uuid> = vec![Uuid::nil(); n];
    for (idx, uuid) in &idx_to_uuid {
        labels[idx.index()] = *uuid;
    }

    // Deterministic processing order: sort by NodeIndex (equivalent to insertion order).
    let mut sorted_indices: Vec<NodeIndex> = graph.node_indices().collect();
    sorted_indices.sort_unstable();

    // ── Label propagation ────────────────────────────────────────────────────
    for _ in 0..max_iterations {
        let mut changed = false;

        for &idx in &sorted_indices {
            let neighbor_labels: Vec<Uuid> =
                graph.neighbors(idx).map(|n| labels[n.index()]).collect();

            if neighbor_labels.is_empty() {
                continue; // Isolated node keeps its own label.
            }

            // Count label frequencies among neighbours.
            let mut freq: HashMap<Uuid, usize> = HashMap::new();
            for &lbl in &neighbor_labels {
                *freq.entry(lbl).or_insert(0) += 1;
            }

            let max_count = freq.values().copied().max().unwrap_or(0);

            // Collect candidates with the highest frequency; break ties by
            // choosing the lexicographically smallest UUID for determinism.
            let mut candidates: Vec<Uuid> = freq
                .into_iter()
                .filter(|(_, c)| *c == max_count)
                .map(|(l, _)| l)
                .collect();
            candidates.sort_unstable();

            if let Some(best) = candidates.first().copied() {
                if labels[idx.index()] != best {
                    labels[idx.index()] = best;
                    changed = true;
                }
            }
        }

        if !changed {
            break; // Converged.
        }
    }

    // ── Collect results ──────────────────────────────────────────────────────
    let mut communities: HashMap<Uuid, Vec<Uuid>> = HashMap::new();
    for (idx, uuid) in &idx_to_uuid {
        let label = labels[idx.index()];
        communities.entry(label).or_default().push(*uuid);
    }

    communities
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn uuid(n: u8) -> Uuid {
        // Construct a UUID with the byte value `n` in the last position for easy ordering.
        let bytes = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, n];
        Uuid::from_bytes(bytes)
    }

    fn edge(src: u8, tgt: u8) -> EdgeSpec {
        EdgeSpec { source: uuid(src), target: uuid(tgt) }
    }

    // ── Basic structural tests ────────────────────────────────────────────────

    #[test]
    fn empty_input_returns_empty_map() {
        let result = detect_communities(&[], &[], 10);
        assert!(result.is_empty());
    }

    #[test]
    fn single_isolated_node_forms_its_own_community() {
        let id = uuid(1);
        let result = detect_communities(&[id], &[], 10);
        assert_eq!(result.len(), 1);
        let members: Vec<Uuid> = result.into_values().next().unwrap();
        assert_eq!(members, vec![id]);
    }

    #[test]
    fn two_connected_nodes_share_a_community() {
        let a = uuid(1);
        let b = uuid(2);
        let result = detect_communities(&[a, b], &[EdgeSpec { source: a, target: b }], 30);
        assert_eq!(result.len(), 1, "connected pair must share one community");
        let members = result.into_values().next().unwrap();
        assert!(members.contains(&a));
        assert!(members.contains(&b));
    }

    #[test]
    fn triangle_forms_single_community() {
        // A – B – C – A
        let a = uuid(1);
        let b = uuid(2);
        let c = uuid(3);
        let result = detect_communities(
            &[a, b, c],
            &[edge(1, 2), edge(2, 3), edge(3, 1)],
            30,
        );
        assert_eq!(result.len(), 1, "triangle must form a single community; got {result:?}");
        let members = result.into_values().next().unwrap();
        assert_eq!(members.len(), 3);
    }

    #[test]
    fn disconnected_pair_and_triangle_form_two_communities() {
        // Triangle: A-B-C
        // Pair: D-E
        // Isolated: F
        let a = uuid(1);
        let b = uuid(2);
        let c = uuid(3);
        let d = uuid(4);
        let e = uuid(5);
        let f = uuid(6);

        let result = detect_communities(
            &[a, b, c, d, e, f],
            &[edge(1, 2), edge(2, 3), edge(3, 1), edge(4, 5)],
            30,
        );

        // Should get exactly 3 communities: {A,B,C}, {D,E}, {F}
        assert_eq!(result.len(), 3, "expected 3 communities, got {result:?}");

        // Verify sizes
        let mut sizes: Vec<usize> = result.values().map(|v| v.len()).collect();
        sizes.sort_unstable();
        assert_eq!(sizes, vec![1, 2, 3]);
    }

    #[test]
    fn duplicate_node_ids_are_deduplicated() {
        let a = uuid(1);
        let result = detect_communities(&[a, a, a], &[], 10);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn edge_to_unknown_node_is_silently_skipped() {
        let a = uuid(1);
        let unknown = uuid(99);
        // Edge references a node not in `node_ids` — must not panic.
        let result =
            detect_communities(&[a], &[EdgeSpec { source: a, target: unknown }], 10);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn self_loop_is_ignored() {
        let a = uuid(1);
        let b = uuid(2);
        let result = detect_communities(
            &[a, b],
            &[EdgeSpec { source: a, target: a }, EdgeSpec { source: a, target: b }],
            30,
        );
        // Self-loop on A should not affect propagation; A and B should still merge.
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn zero_max_iterations_returns_singleton_communities() {
        // With 0 iterations no propagation occurs; each node keeps its own label.
        let a = uuid(1);
        let b = uuid(2);
        let result = detect_communities(&[a, b], &[EdgeSpec { source: a, target: b }], 0);
        assert_eq!(result.len(), 2, "0 iterations → each node is its own community");
    }

    #[test]
    fn convergence_stops_early() {
        // A tiny graph should converge well before `max_iterations`.
        // This test is structural: detect_communities must not run more iterations than needed.
        let a = uuid(1);
        let b = uuid(2);
        // Using 1000 max iterations — convergence should stop after 1–2 rounds.
        let result =
            detect_communities(&[a, b], &[EdgeSpec { source: a, target: b }], 1_000);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn all_community_members_are_present_in_result() {
        let nodes: Vec<Uuid> = (1..=5).map(uuid).collect();
        let edges: Vec<EdgeSpec> = vec![edge(1, 2), edge(2, 3), edge(4, 5)];
        let result = detect_communities(&nodes, &edges, 30);

        let all_members: Vec<Uuid> = result.into_values().flatten().collect();
        assert_eq!(all_members.len(), 5, "every input node must appear exactly once");
        for &id in &nodes {
            assert!(all_members.contains(&id), "node {id} missing from result");
        }
    }
}
