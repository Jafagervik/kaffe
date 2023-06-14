use kaffe::Tensor;

fn main() {
    let a: Tensor<f32> = Tensor::randomize(vec![2, 4]);
    let b: Tensor<f32> = Tensor::randomize(vec![4, 3]);

    if let Ok(tensor) = a.matmul(&b) {
        println!("{:?}", tensor.data);
    }
}
