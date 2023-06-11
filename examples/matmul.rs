use kaffe::{Matrix, MatrixLinAlg};

fn main() {
    let a = Matrix::randomize((2, 4));
    let b = Matrix::randomize((4, 3));

    let c = a.matmul(&b);

    c.print();
}
