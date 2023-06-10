use matrix::Matrix;
use matrix::MatrixOps;

fn main() {
    let mat = Matrix::default();
    mat.print();

    let vec = vec![1f32; 20];
    let v = Matrix::from_vec(vec, (5, 4)).unwrap();

    let eye = Matrix::eye(3);

    println!("{:?}", eye);

    let rand = Matrix::randomize_range(1f32, 10f32, (5, 4));
    rand.print();

    let a = Matrix::from_shape(4.45, (3, 5));
    let b = Matrix::from_shape(5.55, (3, 5));

    let mut c = a.add(&b);

    c = c.mul(&a);
    c.print();

    let m1 = Matrix::randomize_range(1f32, 8f32, (2, 42));
    let m2 = Matrix::randomize_range(1f32, 8f32, (42, 3));

    let m3 = m1.matmul(&m2);

    m3.print();
}
