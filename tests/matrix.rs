// TODO: Add more tests
use latte::matrix::{Matrix, MatrixLinAlg, MatrixPredicates};

#[test]
fn creation() {
    let a = Matrix::eye(2);

    assert!(a.data == vec![1.0, 0.0, 0.0, 1.0]);
    assert!(a.size() == 4);
    assert!(a.shape == (2, 2));
}

#[test]
fn matmul() {
    let a = Matrix::init(2f32, (2, 100));
    let b = Matrix::init(3f32, (100, 2));

    let c = a.matmul(&b);

    assert!(c.shape == (2, 2));
    assert!(c.size() == 4);
}
