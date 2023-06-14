//! This module contains the most used non-linear activation functions
//!
//! Sigmoid and variants of ReLU are for now the ones implemented
#![warn(missing_docs)]

use std::error::Error;
use std::str::FromStr;

use crate::constants::{E, PI};
use crate::{Tensor, TensorElement};
use rayon::prelude::*;

/// ReLU is the most used activation funcion besides Sigmoid
///
/// # Examples
///
/// ```
/// use kaffe::Tensor;
/// use kaffe::nn::activation::ReLU;
///
/// let matrix = Tensor::new(vec![-1.0, -2.0, 1.0, 2.0], vec![2,2]).unwrap();
///
/// assert_eq!(ReLU(&matrix).data, vec![0.0, 0.0, 1.0, 2.0]);
/// ```
pub fn ReLU<'a, 'b, T>(x: &Tensor<'a, T>) -> Tensor<'b, T>
where
    T: TensorElement,
    <T as FromStr>::Err: Error,
    Vec<T>: FromParallelIterator<T>,
{
    let data: Vec<T> = x
        .data
        .par_iter()
        .map(|&e| if e >= T::zero() { e } else { T::zero() })
        .collect();

    Tensor::new(data, x.shape.clone()).unwrap()
}

/// PReLU is a slight modification to ReLU
///
/// # Examples
///
/// ```
/// use kaffe::Tensor;
/// use kaffe::nn::activation::PReLU;
///
/// let matrix = Tensor::new(vec![-1.0, -2.0, 1.0, 2.0], vec![2,2]).unwrap();
///
/// assert_eq!(PReLU(&matrix, -1.0).data, vec![1.0, 2.0, 1.0, 2.0]);
/// ```
pub fn PReLU<'a, 'b, T>(x: &Tensor<'a, T>, alpha: T) -> Tensor<'b, T>
where
    T: TensorElement,
    <T as FromStr>::Err: Error,
    Vec<T>: FromParallelIterator<T>,
{
    let data: Vec<T> = x
        .data
        .par_iter()
        .map(|&e| if e >= T::zero() { e } else { e * alpha })
        .collect();

    Tensor::new(data, x.shape.clone()).unwrap()
}

/// Sigmoid function
///
/// # Examples
///
/// ```
/// use kaffe::Tensor;
/// use kaffe::nn::activation::Sigmoid;
///
/// let matrix = Tensor::new(vec![1.0, 2.0, 1.0, 2.0], vec![2,2]).unwrap();
///
/// // assert_eq!(Sigmoid(&matrix).data, vec![1.0, 2.0, 1.0, 2.0]);
/// ```
pub fn Sigmoid<'a, T>(x: &Tensor<'a, T>) -> Tensor<'a, T>
where
    T: TensorElement,
    <T as FromStr>::Err: Error,
    Vec<T>: FromParallelIterator<T>,
{
    let data: Vec<T> = x
        .data
        .par_iter()
        .map(|&e| T::from(E).unwrap().powf(e) / (T::from(E).unwrap().powf(e) + T::one()))
        .collect();

    Tensor::new(data, x.shape.clone()).unwrap()
}

/// GeLU activation function
///
/// 0.5x (1 + tanh[sqrt(2/pi) * (x + 0.044715x^3)])
///
/// # Examples
///
/// ```
/// use kaffe::Tensor;
/// use kaffe::nn::activation::GeLU;
///
/// let matrix = Tensor::new(vec![1.0, 2.0, 1.0, 2.0], vec![2,2]).unwrap();
///
/// // assert_eq!(GeLU(&matrix).data, vec![1.0, 2.0, 1.0, 2.0]);
/// ```
pub fn GeLU<'a, T>(x: &Tensor<'a, T>) -> Tensor<'a, T>
where
    T: TensorElement,
    <T as FromStr>::Err: Error,
{
    let lhs = x.mul_val(T::from(0.5).unwrap());

    let inner = x
        .pow(3)
        .mul_val(T::from(0.044715).unwrap())
        .add(x)
        .unwrap()
        .mul_val((T::from(2).unwrap() / T::from(PI).unwrap()).sqrt());

    let result = lhs.mul(&inner.tanh().add_val(T::from(1).unwrap())).unwrap();

    return result;
}
