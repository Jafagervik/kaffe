use kaffe::{Matrix, MatrixLinAlg};

use criterion::{black_box, criterion_group, criterion_main, Criterion};

/// Benchmark for matrix multiplication
fn matmul_bench(c: &mut Criterion) {
    let A = black_box(Matrix::randomize((4, 100)));
    let B = black_box(Matrix::randomize((100, 148)));

    c.bench_function("matmul transpose", |b| b.iter(|| A.matmul(&B)));
}

criterion_group!(benches, matmul_bench);
criterion_main!(benches);
