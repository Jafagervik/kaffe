//! Common loss functions
#![warn(missing_docs)]

use crate::{Matrix, MatrixLinAlg, MatrixPredicates};

/// Binary Cross entropy loss
///
/// # Examples
///
/// ```
/// use kaffe::{Matrix, MatrixLinAlg};
/// use kaffe::nn::losses::CEntroypyLoss;
///
/// let m1 = Matrix::init(2.0, (2,2));
/// let m2 = Matrix::init(4.0, (2,2));
///
/// ```
pub fn BCEntroypyLoss(y: &Matrix, y_hat: &Matrix) -> f32 {
    todo!()
}

/// Cross entropy loss
///
/// # Examples
///
/// ```
/// use kaffe::{Matrix, MatrixLinAlg};
/// use kaffe::nn::losses::CEntroypyLoss;
///
/// let m1 = Matrix::init(2.0, (2,2));
/// let m2 = Matrix::init(4.0, (2,2));
///
/// ```
pub fn CEntroypyLoss(y: &Matrix, y_hat: &Matrix) -> f32 {
    todo!()
}

/// Also known as MAE loss
///
/// MAE = |x - y|
///
/// # Examples
///
/// ```
/// use kaffe::{Matrix, MatrixLinAlg};
/// use kaffe::nn::losses::L1Loss;
///
/// let m1 = Matrix::init(2.0, (2,2));
/// let m2 = Matrix::init(4.0, (2,2));
///
/// assert_eq!(L1Loss(&m1, &m2), 8.0);
/// ```
pub fn L1Loss(y: &Matrix, y_hat: &Matrix) -> f32 {
    y.sub_abs(y_hat).cumsum()
}

/// Also known as MSE loss
///
/// MSE = (x - y)^2
///
/// # Examples
///
/// ```
/// use kaffe::{Matrix, MatrixLinAlg};
/// use kaffe::nn::losses::L2Loss;
///
/// let m1 = Matrix::init(3.0, (2,2));
/// let m2 = Matrix::init(2.0, (2,2));
///
/// assert_eq!(L2Loss(&m1, &m2), 4.0);
/// ```
pub fn L2Loss(y: &Matrix, y_hat: &Matrix) -> f32 {
    y.sub(y_hat).pow(2).cumsum()
}
