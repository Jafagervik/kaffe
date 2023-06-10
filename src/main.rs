mod matrix;
mod nn;

use matrix::Matrix;
use matrix::MatrixOps;
use matrix::MatrixPredicates;

use nn::*;

fn main() {
    let u = Matrix::init(2f32, (2, 2));
    let v = Matrix::init(3f32, (2, 2));
    let w = u.matmul(&v);
    w.print();

    let mat = Matrix::default();
    mat.print();

    test();

    let a = Matrix::eye(3);
    let iff = a.any(|&e| e == 1.0);
    println!("\nAnswer is {iff}\n");

    let pos = a.find(|&e| e == 1.0).unwrap();
    println!("Pos {:?}", pos);
    // println!("\nPos is {:.2}\n", pos.iter().count());
    //pos.iter().for_each(|e| println!("{:?}", e));

    let mrand1 = Matrix::randomize((3, 9872));
    let mrand2 = Matrix::randomize((9872, 3));

    let mch = mrand1.matmul(&mrand2);
    mch.print();

    println!("\nMin is {:.2}\n", mch.min());
    println!("\nMax is {:.2}\n", mch.max());
}
