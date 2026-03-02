//! Vector similarity functions.

use ndarray::ArrayView1;

/// Compute the cosine similarity between two f32 slices.
///
/// Returns `0.0` for empty slices, mismatched lengths, or zero vectors.
/// Returns a value in `[-1.0, 1.0]` for valid non-zero vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }

    let a = ArrayView1::from(a);
    let b = ArrayView1::from(b);

    let dot = a.dot(&b);
    let norm_a = a.dot(&a).sqrt();
    let norm_b = b.dot(&b).sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// L2-normalize a vector, returning a new `Vec<f32>`.
///
/// Returns a zero vector of the same length if the input is a zero vector.
/// Returns an empty `Vec` for empty input.
pub fn normalize_l2(v: &[f32]) -> Vec<f32> {
    if v.is_empty() {
        return Vec::new();
    }

    let arr = ArrayView1::from(v);
    let norm = arr.dot(&arr).sqrt();

    if norm == 0.0 {
        return vec![0.0; v.len()];
    }

    v.iter().map(|x| x / norm).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-6;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn test_cosine_identical_vectors() {
        let v = [1.0_f32, 2.0, 3.0];
        assert!(approx_eq(cosine_similarity(&v, &v), 1.0));
    }

    #[test]
    fn test_cosine_orthogonal_vectors() {
        let a = [1.0_f32, 0.0];
        let b = [0.0_f32, 1.0];
        assert!(approx_eq(cosine_similarity(&a, &b), 0.0));
    }

    #[test]
    fn test_cosine_opposite_vectors() {
        let a = [1.0_f32, 0.0, 0.0];
        let b = [-1.0_f32, 0.0, 0.0];
        assert!(approx_eq(cosine_similarity(&a, &b), -1.0));
    }

    #[test]
    fn test_cosine_known_vectors() {
        // a = [3, 4], b = [4, 3]
        // dot = 12 + 12 = 24, |a| = 5, |b| = 5 -> 24/25 = 0.96
        let a = [3.0_f32, 4.0];
        let b = [4.0_f32, 3.0];
        assert!(approx_eq(cosine_similarity(&a, &b), 0.96));
    }

    #[test]
    fn test_cosine_empty_vectors() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn test_cosine_mismatched_lengths() {
        let a = [1.0_f32, 2.0];
        let b = [1.0_f32, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = [0.0_f32, 0.0, 0.0];
        let b = [1.0_f32, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
        assert_eq!(cosine_similarity(&a, &a), 0.0);
    }

    #[test]
    fn test_normalize_l2_unit_magnitude() {
        let v = [3.0_f32, 4.0];
        let n = normalize_l2(&v);
        let mag: f32 = n.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(approx_eq(mag, 1.0));
        assert!(approx_eq(n[0], 0.6));
        assert!(approx_eq(n[1], 0.8));
    }

    #[test]
    fn test_normalize_l2_zero_vector() {
        let v = [0.0_f32, 0.0, 0.0];
        let n = normalize_l2(&v);
        assert_eq!(n, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_normalize_l2_empty() {
        let n = normalize_l2(&[]);
        assert!(n.is_empty());
    }

    #[test]
    fn test_normalize_l2_already_normalized() {
        let v = [1.0_f32, 0.0, 0.0];
        let n = normalize_l2(&v);
        assert!(approx_eq(n[0], 1.0));
        assert!(approx_eq(n[1], 0.0));
        assert!(approx_eq(n[2], 0.0));
    }
}
