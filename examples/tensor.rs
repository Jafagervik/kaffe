use kaffe::Tensor;

fn main() {
    let a: Tensor<f64> = Tensor::randomize(vec![2, 3]);
    let b: Tensor<f64> = Tensor::randomize(vec![2, 3]);

    let c = a.add(&b);

    if let Ok(tensor) = c {
        println!("{:?}", tensor.data);
    }
}
