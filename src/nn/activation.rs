//! This module contains the most used non-linear activation functions
//!
//! Sigmoid and variants of ReLU are for now the ones implemented
#![warn(missing_docs)]

/// Yes
use crate::constants::{E, PI};
use crate::{Matrix, MatrixLinAlg};

/// ReLU is the most used activation funcion besides Sigmoid
///
/// # Examples
///
/// ```
/// use kaffe::{Matrix, MatrixLinAlg};
/// use kaffe::nn::activation::ReLU;
///
/// let matrix = Matrix::new(vec![-1.0, -2.0, 1.0, 2.0], (2,2));
///
/// assert_eq!(ReLU(&matrix).data, vec![0.0, 0.0, 1.0, 2.0]);
/// ```
pub fn ReLU(x: &Matrix) -> Matrix {
    let data: Vec<f32> = x
        .data
        .iter()
        .map(|&e| if e >= 0.0 { e } else { 0.0 })
        .collect();

    Matrix::new(data, x.shape)
}

/// PReLU is a slight modification to ReLU
///
/// # Examples
///
/// ```
/// use kaffe::{Matrix, MatrixLinAlg};
/// use kaffe::nn::activation::PReLU;
///
/// let matrix = Matrix::new(vec![-1.0, -2.0, 1.0, 2.0], (2,2));
///
/// assert_eq!(PReLU(&matrix, -1.0).data, vec![1.0, 2.0, 1.0, 2.0]);
/// ```
pub fn PReLU(x: &Matrix, alpha: f32) -> Matrix {
    let data: Vec<f32> = x
        .data
        .iter()
        .map(|&e| if e >= 0.0 { e } else { e * alpha })
        .collect();

    Matrix::new(data, x.shape)
}

/// Sigmoid function
///
/// # Examples
///
/// ```
/// use kaffe::{Matrix, MatrixLinAlg};
/// use kaffe::nn::activation::Sigmoid;
///
/// let matrix = Matrix::new(vec![1.0, 2.0, 1.0, 2.0], (2,2));
///
/// // assert_eq!(Sigmoid(&matrix).data, vec![1.0, 2.0, 1.0, 2.0]);
/// ```
pub fn Sigmoid(x: &Matrix) -> Matrix {
    let data: Vec<f32> = x
        .data
        .iter()
        .map(|&e| E.powf(e) / (E.powf(e) + 1f32))
        .collect();

    Matrix::new(data, x.shape)
}

/// GeLU activation function
///
/// 0.5x (1 + tanh[sqrt(2/pi) * (x + 0.044715x^3)])
///
/// # Examples
///
/// ```
/// use kaffe::{Matrix, MatrixLinAlg};
/// use kaffe::nn::activation::GeLU;
///
/// let matrix = Matrix::new(vec![1.0, 2.0, 1.0, 2.0], (2,2));
///
/// // assert_eq!(GeLU(&matrix).data, vec![1.0, 2.0, 1.0, 2.0]);
/// ```
pub fn GeLU(x: &Matrix) -> Matrix {
    let lhs = x.mul_val(0.5);

    let inner = x
        .pow(3)
        .mul_val(0.044715)
        .add(x)
        .mul_val((2.0 / PI).sqrt() as f32);

    let result = lhs.mul(&inner.tanh().add_val(1.0));

    return result;
}
