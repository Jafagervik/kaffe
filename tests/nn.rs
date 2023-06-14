use kaffe::nn::activation::ReLU;
use kaffe::Tensor;

#[test]
fn relu() {
    let mat = Tensor::new(vec![-2.0, -4.0, 6.0], vec![3, 1]).unwrap();
    assert_eq!(ReLU(&mat).data, vec![0.0, 0.0, 6.0]);
}
