mod activations;
mod losses;
mod optimizers;
// mod pooling;

use crate::matrix::{Matrix, MatrixOps, Shape};
use activations::*;

pub struct Layer {}

/// Convolve a matrix with a certain kernel
pub fn convolution(matrix: Matrix, kernel: Matrix, stride: usize, padding: usize) -> Matrix {
    todo!()
}

pub struct Net {
    layers: Vec<Layer>,
}

pub fn test() {
    let a = Matrix::eye(4);
    a.print();
}
