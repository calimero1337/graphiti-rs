//! Reciprocal Rank Fusion (RRF) algorithm for hybrid search reranking.
//!
//! RRF fuses multiple ranked lists into a single ranking by assigning each
//! item a score based on its reciprocal rank in each list. Items appearing in
//! multiple lists accumulate scores from each appearance.
//!
//! Reference: Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet and
//! individual Rank Learning Methods" (SIGIR 2009).

use std::collections::HashMap;

use uuid::Uuid;

/// Standard smoothing constant for RRF (prevents very high scores for rank 1).
pub const DEFAULT_K: f32 = 60.0;

/// Fuse multiple ranked lists using Reciprocal Rank Fusion.
///
/// Each inner `Vec<Uuid>` is ordered best-first (index 0 = rank 1).
/// `weights[i]` scales the contribution of `ranked_lists[i]`.
/// `k` is the smoothing constant (standard value: [`DEFAULT_K`] = 60.0).
///
/// # Panics
///
/// Panics if `ranked_lists.len() != weights.len()`.
///
/// # Returns
///
/// `(id, fused_score)` pairs sorted by score descending (stable sort).
pub fn reciprocal_rank_fusion(
    ranked_lists: &[Vec<Uuid>],
    weights: &[f32],
    k: f32,
) -> Vec<(Uuid, f32)> {
    assert_eq!(
        ranked_lists.len(),
        weights.len(),
        "ranked_lists and weights must have the same length"
    );

    // Track first-appearance order separately so stable sort on tied scores
    // preserves the order in which items were first seen across the lists.
    let mut order: Vec<Uuid> = Vec::new();
    let mut scores: HashMap<Uuid, f32> = HashMap::new();

    for (list, &weight) in ranked_lists.iter().zip(weights.iter()) {
        for (rank_0, &id) in list.iter().enumerate() {
            let rank = (rank_0 + 1) as f32;
            if !scores.contains_key(&id) {
                order.push(id);
            }
            *scores.entry(id).or_insert(0.0) += weight / (k + rank);
        }
    }

    let mut result: Vec<(Uuid, f32)> = order.into_iter().map(|id| (id, scores[&id])).collect();
    // Stable sort preserves first-appearance order among items with equal scores.
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper UUIDs — deterministic for reproducible test assertions.
    // -----------------------------------------------------------------------
    fn uuid_a() -> Uuid {
        Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap()
    }
    fn uuid_b() -> Uuid {
        Uuid::parse_str("00000000-0000-0000-0000-000000000002").unwrap()
    }
    fn uuid_c() -> Uuid {
        Uuid::parse_str("00000000-0000-0000-0000-000000000003").unwrap()
    }
    fn uuid_d() -> Uuid {
        Uuid::parse_str("00000000-0000-0000-0000-000000000004").unwrap()
    }
    fn uuid_e() -> Uuid {
        Uuid::parse_str("00000000-0000-0000-0000-000000000005").unwrap()
    }
    fn uuid_f() -> Uuid {
        Uuid::parse_str("00000000-0000-0000-0000-000000000006").unwrap()
    }

    /// Approximate float comparison.
    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-6
    }

    // -----------------------------------------------------------------------
    // Test 1: Two lists with NO overlap → each item gets a single-source score.
    // -----------------------------------------------------------------------
    #[test]
    fn test_no_overlap_single_source_scores() {
        // list1: [A, B], list2: [C, D]  weights=[1.0, 1.0], k=60.0
        // Expected scores:
        //   A = 1.0 / (60 + 1) = 1/61 ≈ 0.016393
        //   B = 1.0 / (60 + 2) = 1/62 ≈ 0.016129
        //   C = 1.0 / (60 + 1) = 1/61 ≈ 0.016393
        //   D = 1.0 / (60 + 2) = 1/62 ≈ 0.016129
        let list1 = vec![uuid_a(), uuid_b()];
        let list2 = vec![uuid_c(), uuid_d()];
        let result = reciprocal_rank_fusion(&[list1, list2], &[1.0, 1.0], 60.0);

        assert_eq!(result.len(), 4, "must return all 4 distinct items");

        // Collect into map for easy lookup.
        let scores: HashMap<Uuid, f32> = result.iter().cloned().collect();

        let expected_rank1 = 1.0_f32 / 61.0;
        let expected_rank2 = 1.0_f32 / 62.0;

        assert!(
            approx_eq(scores[&uuid_a()], expected_rank1),
            "A score wrong"
        );
        assert!(
            approx_eq(scores[&uuid_b()], expected_rank2),
            "B score wrong"
        );
        assert!(
            approx_eq(scores[&uuid_c()], expected_rank1),
            "C score wrong"
        );
        assert!(
            approx_eq(scores[&uuid_d()], expected_rank2),
            "D score wrong"
        );

        // Output must be sorted descending.
        let output_scores: Vec<f32> = result.iter().map(|(_, s)| *s).collect();
        let mut sorted = output_scores.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert_eq!(output_scores, sorted, "output must be sorted descending");
    }

    // -----------------------------------------------------------------------
    // Test 2: Two lists with FULL overlap, SAME order → scores double.
    // -----------------------------------------------------------------------
    #[test]
    fn test_full_overlap_same_order_doubles_scores() {
        // list1 = list2 = [A, B, C], weights=[1.0, 1.0], k=60.0
        // A: 1/61 + 1/61 = 2/61
        // B: 1/62 + 1/62 = 2/62
        // C: 1/63 + 1/63 = 2/63
        let list = vec![uuid_a(), uuid_b(), uuid_c()];
        let result = reciprocal_rank_fusion(&[list.clone(), list], &[1.0, 1.0], 60.0);

        assert_eq!(result.len(), 3);

        let scores: HashMap<Uuid, f32> = result.iter().cloned().collect();
        let single_a = 1.0_f32 / 61.0;
        let single_b = 1.0_f32 / 62.0;
        let single_c = 1.0_f32 / 63.0;

        assert!(
            approx_eq(scores[&uuid_a()], 2.0 * single_a),
            "A should double"
        );
        assert!(
            approx_eq(scores[&uuid_b()], 2.0 * single_b),
            "B should double"
        );
        assert!(
            approx_eq(scores[&uuid_c()], 2.0 * single_c),
            "C should double"
        );

        // Order: A > B > C.
        assert_eq!(result[0].0, uuid_a());
        assert_eq!(result[1].0, uuid_b());
        assert_eq!(result[2].0, uuid_c());
    }

    // -----------------------------------------------------------------------
    // Test 3: Two lists with FULL overlap, REVERSED order → reranking occurs.
    // -----------------------------------------------------------------------
    #[test]
    fn test_full_overlap_reversed_order_reranks() {
        // list1: [A, B, C], list2: [C, B, A], weights=[1.0, 1.0], k=60.0
        // A: 1/61 (rank1 in list1) + 1/63 (rank3 in list2)
        // B: 1/62 (rank2 in list1) + 1/62 (rank2 in list2) = 2/62
        // C: 1/63 (rank3 in list1) + 1/61 (rank1 in list2)
        // A_score = C_score; B_score = 2/62 ≈ 0.032258; A_score ≈ 1/61+1/63 ≈ 0.031984
        // So B has the highest score.
        let list1 = vec![uuid_a(), uuid_b(), uuid_c()];
        let list2 = vec![uuid_c(), uuid_b(), uuid_a()];
        let result = reciprocal_rank_fusion(&[list1, list2], &[1.0, 1.0], 60.0);

        assert_eq!(result.len(), 3);

        let scores: HashMap<Uuid, f32> = result.iter().cloned().collect();

        // B appears at rank 2 in both → highest combined score.
        let score_b = 2.0_f32 / 62.0;
        let score_a = 1.0_f32 / 61.0 + 1.0_f32 / 63.0;
        let score_c = 1.0_f32 / 63.0 + 1.0_f32 / 61.0; // same as A

        assert!(approx_eq(scores[&uuid_b()], score_b), "B score wrong");
        assert!(approx_eq(scores[&uuid_a()], score_a), "A score wrong");
        assert!(approx_eq(scores[&uuid_c()], score_c), "C score wrong");

        // B should be ranked first (highest score).
        assert_eq!(result[0].0, uuid_b(), "B should rank first after reranking");
    }

    // -----------------------------------------------------------------------
    // Test 4: Two lists with PARTIAL overlap → overlapping items rank higher.
    // -----------------------------------------------------------------------
    #[test]
    fn test_partial_overlap_boosts_shared_items() {
        // list1: [A, B, C], list2: [B, D, E], weights=[1.0, 1.0], k=60.0
        // A: 1/61 (only list1 rank1)
        // B: 1/62 (list1 rank2) + 1/61 (list2 rank1) = large
        // C: 1/63 (only list1 rank3)
        // D: 1/62 (only list2 rank2)
        // E: 1/63 (only list2 rank3)
        // B should outrank everything.
        let list1 = vec![uuid_a(), uuid_b(), uuid_c()];
        let list2 = vec![uuid_b(), uuid_d(), uuid_e()];
        let result = reciprocal_rank_fusion(&[list1, list2], &[1.0, 1.0], 60.0);

        assert_eq!(result.len(), 5);

        // B must rank first.
        assert_eq!(result[0].0, uuid_b(), "shared item B must rank first");

        let scores: HashMap<Uuid, f32> = result.iter().cloned().collect();
        let score_b = 1.0_f32 / 62.0 + 1.0_f32 / 61.0;
        let score_a = 1.0_f32 / 61.0;

        assert!(scores[&uuid_b()] > scores[&uuid_a()], "B must outscore A");
        assert!(approx_eq(scores[&uuid_b()], score_b), "B score wrong");
        assert!(approx_eq(scores[&uuid_a()], score_a), "A score wrong");
    }

    // -----------------------------------------------------------------------
    // Test 5: ASYMMETRIC weights → higher-weight list items are boosted.
    // -----------------------------------------------------------------------
    #[test]
    fn test_asymmetric_weights_boost_higher_weight_list() {
        // list1 (BM25, weight=2.0): [A, B], list2 (vector, weight=1.0): [C, D]
        // k=60.0
        // A (from list1): 2.0/61
        // C (from list2): 1.0/61
        // A should outscore C even at same rank.
        let list1 = vec![uuid_a(), uuid_b()];
        let list2 = vec![uuid_c(), uuid_d()];
        let result = reciprocal_rank_fusion(&[list1, list2], &[2.0, 1.0], 60.0);

        assert_eq!(result.len(), 4);

        let scores: HashMap<Uuid, f32> = result.iter().cloned().collect();

        let score_a = 2.0_f32 / 61.0;
        let score_b = 2.0_f32 / 62.0;
        let score_c = 1.0_f32 / 61.0;
        let score_d = 1.0_f32 / 62.0;

        assert!(approx_eq(scores[&uuid_a()], score_a), "A score wrong");
        assert!(approx_eq(scores[&uuid_b()], score_b), "B score wrong");
        assert!(approx_eq(scores[&uuid_c()], score_c), "C score wrong");
        assert!(approx_eq(scores[&uuid_d()], score_d), "D score wrong");

        // list1 items (A, B) should outscore list2 items (C, D) at same rank.
        assert!(
            scores[&uuid_a()] > scores[&uuid_c()],
            "A must outscore C (weight effect)"
        );
        assert!(
            scores[&uuid_b()] > scores[&uuid_d()],
            "B must outscore D (weight effect)"
        );
    }

    // -----------------------------------------------------------------------
    // Test 6: Single list → degenerates to weighted 1/(k+rank) scoring.
    // -----------------------------------------------------------------------
    #[test]
    fn test_single_list_degenerates_to_weighted_reciprocal() {
        // Single list: [A, B, C], weight=1.5, k=60.0
        // A: 1.5/61, B: 1.5/62, C: 1.5/63
        let list = vec![uuid_a(), uuid_b(), uuid_c()];
        let result = reciprocal_rank_fusion(&[list], &[1.5], 60.0);

        assert_eq!(result.len(), 3);

        let scores: HashMap<Uuid, f32> = result.iter().cloned().collect();
        assert!(
            approx_eq(scores[&uuid_a()], 1.5_f32 / 61.0),
            "A score wrong"
        );
        assert!(
            approx_eq(scores[&uuid_b()], 1.5_f32 / 62.0),
            "B score wrong"
        );
        assert!(
            approx_eq(scores[&uuid_c()], 1.5_f32 / 63.0),
            "C score wrong"
        );

        // Order preserved from single list.
        assert_eq!(result[0].0, uuid_a());
        assert_eq!(result[1].0, uuid_b());
        assert_eq!(result[2].0, uuid_c());
    }

    // -----------------------------------------------------------------------
    // Test 7: Empty input → empty output.
    // -----------------------------------------------------------------------
    #[test]
    fn test_empty_input_returns_empty() {
        let result = reciprocal_rank_fusion(&[], &[], 60.0);
        assert!(result.is_empty(), "empty input must yield empty output");
    }

    // -----------------------------------------------------------------------
    // Test 8: K=0 → scores are weight/rank (no smoothing constant).
    // -----------------------------------------------------------------------
    #[test]
    fn test_k_zero_scores_are_weight_over_rank() {
        // k=0: score = weight / (0 + rank) = weight / rank
        // list: [A, B, C], weight=1.0
        // A: 1/1 = 1.0, B: 1/2 = 0.5, C: 1/3 ≈ 0.333
        let list = vec![uuid_a(), uuid_b(), uuid_c()];
        let result = reciprocal_rank_fusion(&[list], &[1.0], 0.0);

        assert_eq!(result.len(), 3);

        let scores: HashMap<Uuid, f32> = result.iter().cloned().collect();
        assert!(approx_eq(scores[&uuid_a()], 1.0), "A: 1/1 = 1.0");
        assert!(approx_eq(scores[&uuid_b()], 0.5), "B: 1/2 = 0.5");
        assert!(approx_eq(scores[&uuid_c()], 1.0 / 3.0), "C: 1/3 ≈ 0.333");
    }

    // -----------------------------------------------------------------------
    // Test 9: Large K → scores converge (flatter distribution).
    // -----------------------------------------------------------------------
    #[test]
    fn test_large_k_scores_converge() {
        // With k=10000, scores ≈ weight/k for all ranks (very flat).
        // list: [A, B, C], weight=1.0, k=10000.0
        // A: 1/10001 ≈ 0.0000999..., B: 1/10002, C: 1/10003
        // Score difference between consecutive ranks is tiny.
        let list = vec![uuid_a(), uuid_b(), uuid_c()];
        let result = reciprocal_rank_fusion(&[list], &[1.0], 10_000.0);

        assert_eq!(result.len(), 3);

        let scores: HashMap<Uuid, f32> = result.iter().cloned().collect();
        let score_a = 1.0_f32 / 10_001.0;
        let score_b = 1.0_f32 / 10_002.0;
        let score_c = 1.0_f32 / 10_003.0;

        assert!(approx_eq(scores[&uuid_a()], score_a), "A score wrong");
        assert!(approx_eq(scores[&uuid_b()], score_b), "B score wrong");
        assert!(approx_eq(scores[&uuid_c()], score_c), "C score wrong");

        // Scores still in correct relative order.
        assert!(scores[&uuid_a()] > scores[&uuid_b()]);
        assert!(scores[&uuid_b()] > scores[&uuid_c()]);

        // The spread is very small (convergence).
        let spread = scores[&uuid_a()] - scores[&uuid_c()];
        assert!(
            spread < 1e-6,
            "scores should converge with large k, spread={spread}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 10: All weights zero → all scores zero.
    // -----------------------------------------------------------------------
    #[test]
    fn test_all_weights_zero_gives_zero_scores() {
        let list1 = vec![uuid_a(), uuid_b()];
        let list2 = vec![uuid_c(), uuid_d()];
        let result = reciprocal_rank_fusion(&[list1, list2], &[0.0, 0.0], 60.0);

        assert_eq!(result.len(), 4);
        for (_, score) in &result {
            assert!(
                approx_eq(*score, 0.0),
                "all scores must be zero when weights are zero"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 11: Empty individual lists are skipped (mixed empty/non-empty).
    // -----------------------------------------------------------------------
    #[test]
    fn test_empty_sublists_are_skipped() {
        // list1 is empty, list2 has [A, B]
        let list1: Vec<Uuid> = vec![];
        let list2 = vec![uuid_a(), uuid_b()];
        let result = reciprocal_rank_fusion(&[list1, list2], &[1.0, 1.0], 60.0);

        assert_eq!(result.len(), 2, "empty sublist contributes no items");

        let scores: HashMap<Uuid, f32> = result.iter().cloned().collect();
        // list2 only: A at rank 1, B at rank 2 with weight=1.0
        assert!(
            approx_eq(scores[&uuid_a()], 1.0_f32 / 61.0),
            "A score wrong"
        );
        assert!(
            approx_eq(scores[&uuid_b()], 1.0_f32 / 62.0),
            "B score wrong"
        );
    }

    // -----------------------------------------------------------------------
    // Test 12: Stable sort — items with identical scores preserve relative order.
    // -----------------------------------------------------------------------
    #[test]
    fn test_stable_sort_preserves_first_appearance_order_on_tie() {
        // Two symmetric lists that produce identical scores for all items.
        // list1: [A], list2: [B] both at rank 1 with equal weight → tied.
        // With stable sort, A (first encountered) should come before B.
        let list1 = vec![uuid_a()];
        let list2 = vec![uuid_b()];
        let result = reciprocal_rank_fusion(&[list1, list2], &[1.0, 1.0], 60.0);

        assert_eq!(result.len(), 2);

        let score_a = result
            .iter()
            .find(|(id, _)| *id == uuid_a())
            .map(|(_, s)| *s)
            .unwrap();
        let score_b = result
            .iter()
            .find(|(id, _)| *id == uuid_b())
            .map(|(_, s)| *s)
            .unwrap();

        // Both have exactly 1.0/61.0.
        assert!(approx_eq(score_a, 1.0_f32 / 61.0));
        assert!(approx_eq(score_b, 1.0_f32 / 61.0));

        // Stable sort: equal scores retain insertion order → A before B.
        assert_eq!(
            result[0].0,
            uuid_a(),
            "stable sort should keep A before B on tie"
        );
        assert_eq!(result[1].0, uuid_b());
    }

    // -----------------------------------------------------------------------
    // Test 13: DEFAULT_K constant equals 60.0.
    // -----------------------------------------------------------------------
    #[test]
    fn test_default_k_value() {
        assert!(approx_eq(DEFAULT_K, 60.0), "DEFAULT_K must be 60.0");
    }

    // -----------------------------------------------------------------------
    // Test 14: Six items across two overlapping lists with DEFAULT_K.
    // -----------------------------------------------------------------------
    #[test]
    fn test_six_items_two_overlapping_lists_with_default_k() {
        // BM25 list: [A, B, C, D], Vector list: [B, C, E, F]
        // Overlap: B (bm25 rank2, vec rank1), C (bm25 rank3, vec rank2)
        // k = DEFAULT_K = 60.0, weights=[1.0, 1.0]
        let bm25_list = vec![uuid_a(), uuid_b(), uuid_c(), uuid_d()];
        let vec_list = vec![uuid_b(), uuid_c(), uuid_e(), uuid_f()];
        let result = reciprocal_rank_fusion(&[bm25_list, vec_list], &[1.0, 1.0], DEFAULT_K);

        // A, B, C, D, E, F = 6 distinct items.
        assert_eq!(result.len(), 6);

        let scores: HashMap<Uuid, f32> = result.iter().cloned().collect();

        // B and C appear in both lists → must outscore A (single list, rank 1).
        let score_b = 1.0_f32 / 62.0 + 1.0_f32 / 61.0; // bm25 rank2 + vec rank1
        let score_c = 1.0_f32 / 63.0 + 1.0_f32 / 62.0; // bm25 rank3 + vec rank2
        let score_a = 1.0_f32 / 61.0; // bm25 rank1 only

        assert!(approx_eq(scores[&uuid_b()], score_b), "B score wrong");
        assert!(approx_eq(scores[&uuid_c()], score_c), "C score wrong");
        assert!(approx_eq(scores[&uuid_a()], score_a), "A score wrong");

        assert!(
            scores[&uuid_b()] > scores[&uuid_a()],
            "B (appears in both lists) must outscore A (only in BM25 at rank 1)"
        );
        assert!(
            scores[&uuid_c()] > scores[&uuid_a()],
            "C (appears in both lists) must outscore A"
        );

        // Top 2 results must be B and C (overlap items).
        let top2: Vec<Uuid> = result[..2].iter().map(|(id, _)| *id).collect();
        assert!(top2.contains(&uuid_b()), "B must be in top 2");
        assert!(top2.contains(&uuid_c()), "C must be in top 2");
    }
}
