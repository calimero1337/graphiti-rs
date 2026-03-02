use criterion::{criterion_group, criterion_main, Criterion};

// TODO: Add benchmarks for:
// - BM25 search over large node sets
// - Vector cosine similarity search
// - Hybrid search pipeline end-to-end
// - RRF reranking performance

fn search_benchmarks(_c: &mut Criterion) {
    // Placeholder — implement after search subsystem is complete
}

criterion_group!(benches, search_benchmarks);
criterion_main!(benches);
