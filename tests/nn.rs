// TODO: Add more tests
use latte::matrix::{Matrix, MatrixLinAlg, MatrixPredicates};
use latte::nn::activations::ReLU;

#[test]
fn relu() {
    assert!(ReLU(3f32) == 3f32);
    assert!(ReLU(-3f32) == 0f32);
}
