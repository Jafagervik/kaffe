//! Common loss functions
#![warn(missing_docs)]

use crate::{Tensor, TensorLinAlg, TensorOps};

/// Represents all methods necessary to create a loss function
pub trait Loss {
    fn backward();
}

/// Binary Cross entropy loss
///
/// # Examples
///
/// ```
/// use kaffe::{Tensor, TensorLinAlg};
/// use kaffe::nn::loss::CEntroypyLoss;
///
/// let m1 = Tensor::init(2.0, (2,2));
/// let m2 = Tensor::init(4.0, (2,2));
/// ```
pub fn BCEntroypyLoss(y: &Tensor, y_hat: &Tensor) -> f32 {
    todo!()
}

/// Cross entropy loss
///
/// # Examples
///
/// ```
/// use kaffe::{Tensor, TensorLinAlg};
/// use kaffe::nn::loss::CEntroypyLoss;
///
/// let m1 = Tensor::init(2.0, (2,2));
/// let m2 = Tensor::init(4.0, (2,2));
/// ```
pub fn CEntroypyLoss(y: &Tensor, y_hat: &Tensor) -> f32 {
    todo!()
}

/// Also known as MAE loss
///
/// MAE = |x - y|
///
/// # Examples
///
/// ```
/// use kaffe::{Tensor, TensorLinAlg};
/// use kaffe::nn::loss::L1Loss;
///
/// let m1 = Tensor::init(2.0, (2,2));
/// let m2 = Tensor::init(4.0, (2,2));
///
/// assert_eq!(L1Loss(&m1, &m2), 8.0);
/// ```
pub fn L1Loss(y: &Tensor, y_hat: &Tensor) -> f32 {
    y.sub_abs(y_hat).cumsum()
}

/// Also known as MSE loss
///
/// MSE = (x - y)^2
///
/// # Examples
///
/// ```
/// use kaffe::{Tensor, TensorLinAlg};
/// use kaffe::nn::loss::L2Loss;
///
/// let m1 = Tensor::init(3.0, (2,2));
/// let m2 = Tensor::init(2.0, (2,2));
///
/// assert_eq!(L2Loss(&m1, &m2), 4.0);
/// ```
pub fn L2Loss(y: &Tensor, y_hat: &Tensor) -> f32 {
    y.sub(y_hat).pow(2).cumsum()
}
