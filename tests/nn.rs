use kaffe::matrix::Matrix;
use kaffe::nn::activation::ReLU;

#[test]
fn relu() {
    let mat = Matrix::new(vec![-2.0, -4.0, 6.0], (3, 1));
    assert_eq!(ReLU(&mat).data, vec![0.0, 0.0, 6.0]);
}
