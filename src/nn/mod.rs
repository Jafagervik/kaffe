#![warn(missing_docs)]
//! Here we go
pub mod activations;
pub mod losses;
pub mod optimizers;
// mod pooling;

use crate::matrix::{Matrix, MatrixLinAlg, Shape};
use activations::*;

/// struct Layer represents a layer in our net
pub struct Layer {}

/// Convolve a matrix with a certain kernel
pub fn convolution(matrix: Matrix, kernel: Matrix, stride: usize, padding: usize) -> Matrix {
    todo!()
}

/// Net is a net builder people can use to create epic neural networks
pub struct Net {
    layers: Vec<Layer>,
}

#[test]
fn name() {
    let a = Matrix::eye(4);
    a.print();
    assert!(1 == 1);
}
