use kaffe::{Matrix, MatrixLinAlg};

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn add_bench(c: &mut Criterion) {
    let y = black_box(Matrix::randomize((500, 100)));
    let x = black_box(Matrix::randomize((500, 100)));

    c.bench_function("add", |b| b.iter(|| x.add(&y)));
}

fn sub_bench(c: &mut Criterion) {
    let y = black_box(Matrix::randomize((500, 100)));
    let x = black_box(Matrix::randomize((500, 100)));

    c.bench_function("sub", |b| b.iter(|| x.sub(&y)));
}

fn mul_bench(c: &mut Criterion) {
    let y = black_box(Matrix::randomize((500, 100)));
    let x = black_box(Matrix::randomize((500, 100)));

    c.bench_function("mul", |b| b.iter(|| x.mul(&y)));
}

criterion_group!(benches, add_bench, sub_bench, mul_bench);
criterion_main!(benches);
