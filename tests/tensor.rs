//! Integration tests
use kaffe::Tensor;

#[test]
fn creation() {
    let a: Tensor<f32> = Tensor::eye(2);

    assert!(a.data == vec![1.0, 0.0, 0.0, 1.0]);
    assert!(a.size() == 4);
    assert!(a.shape == vec![2, 2]);
}

#[test]
fn matmul() {
    let a = Tensor::init(2f32, vec![2, 100]);
    let b = Tensor::init(3f32, vec![100, 2]);

    let c = a.matmul(&b).unwrap_or_else(|_| Tensor::default());

    assert!(c.shape == vec![3, 3]);
    assert!(c.size() == 9);
}

#[test]
fn concat() {
    let tensor = Tensor::init(10.5, vec![7, 200, 3]);
    let tensor2 = Tensor::init(10.5, vec![150, 3]);

    let result = tensor.concat(&tensor2).unwrap();

    assert_eq!(result.shape, vec![7, 350, 3]);

    let mut tensor = Tensor::init(42.5, vec![7, 200, 3]);
    let tensor2 = Tensor::init(10.5, vec![77, 3]);

    tensor.extend(&tensor2);

    assert_eq!(tensor.shape, vec![7, 277, 3]);
}
