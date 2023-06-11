use kaffe::{Matrix, MatrixLinAlg};

fn main() {
    let a = Matrix::randomize((3, 3));

    let det = a.determinant();

    a.print();

    println!("Determinant is {:.2}", det);
}
