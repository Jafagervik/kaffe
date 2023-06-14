//! Common loss functions
#![warn(missing_docs)]

use std::{error::Error, str::FromStr};

use crate::{Tensor, TensorElement};

/// Represents all methods necessary to create a loss function
pub trait Loss<'a, T>
where
    T: TensorElement,
    <T as FromStr>::Err: Error,
{
    fn loss(y: &Tensor<'a, T>, y_hat: &Tensor<'a, T>) -> T;
    fn backward();
}

// TODO: Implement the different losses as structures

/// Binary Cross entropy loss
///
/// # Examples
///
/// ```
/// use kaffe::Tensor;
/// use kaffe::nn::loss::CEntroypyLoss;
///
/// let m1 = Tensor::init(2.0, vec![2,2]);
/// let m2 = Tensor::init(4.0, vec![2,2]);
/// ```
pub fn BCEntroypyLoss<'a, T>(y: &Tensor<'a, T>, y_hat: &Tensor<'a, T>) -> T
where
    T: TensorElement,
    <T as FromStr>::Err: Error,
{
    unimplemented!()
}

/// Cross entropy loss
///
/// # Examples
///
/// ```
/// use kaffe::Tensor;
/// use kaffe::nn::loss::CEntroypyLoss;
///
/// let m1 = Tensor::init(2.0, vec![2,2]);
/// let m2 = Tensor::init(4.0, vec![2,2]);
/// ```
pub fn CEntroypyLoss<'a, T>(y: &Tensor<'a, T>, y_hat: &Tensor<'a, T>) -> T
where
    T: TensorElement,
    <T as FromStr>::Err: Error,
{
    unimplemented!()
}

/// Also known as MAE loss
///
/// MAE = |x - y|
///
/// # Examples
///
/// ```
/// use kaffe::Tensor;
/// use kaffe::nn::loss::L1Loss;
///
/// let m1 = Tensor::init(2.0, vec![2,2]);
/// let m2 = Tensor::init(4.0, vec![2,2]);
///
/// assert_eq!(L1Loss(&m1, &m2), 8.0);
/// ```
pub fn L1Loss<'a, T>(y: &Tensor<'a, T>, y_hat: &Tensor<'a, T>) -> T
where
    T: TensorElement,
    <T as FromStr>::Err: Error,
{
    y.sub_abs(y_hat).unwrap().cumsum()
}

/// Also known as MSE loss
///
/// MSE = (x - y)^2
///
/// # Examples
///
/// ```
/// use kaffe::Tensor;
/// use kaffe::nn::loss::L2Loss;
///
/// let m1 = Tensor::init(3.0, vec![2,2]);
/// let m2 = Tensor::init(2.0, vec![2,2]);
///
/// assert_eq!(L2Loss(&m1, &m2), 4.0);
/// ```
pub fn L2Loss<'a, T>(y: &Tensor<'a, T>, y_hat: &Tensor<'a, T>) -> T
where
    T: TensorElement,
    <T as FromStr>::Err: Error,
{
    y.sub(y_hat).unwrap().pow(2).cumsum()
}
