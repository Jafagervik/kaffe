//!  Pooling
#![warn(missing_docs)]
use std::{error::Error, str::FromStr};

use crate::{Tensor, TensorElement};
use rayon::prelude::*;

/// Max Pooling
///
/// # Examples
///
/// ```
/// use kaffe::Tensor;
/// use kaffe::nn::pooling::MaxPool;
///
/// let matrix = Tensor::init(2.0, vec![4,4]);
///
/// let res = MaxPool(&matrix, 2, 0);
///
/// assert_eq!(res.shape, vec![2,2]);
///
/// ```
pub fn MaxPool<'a, T>(x: &Tensor<'a, T>, stride: usize, padding: usize) -> Tensor<'a, T>
where
    T: TensorElement,
    <T as FromStr>::Err: Error,
{
    let pred = |slice: &[T]| {
        slice
            .par_iter()
            .cloned()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
    };

    pool(x, stride, padding, pred)
}

/// Min Pooling
///
/// # Examples
///
/// ```
/// use kaffe::Tensor;
/// use kaffe::nn::pooling::MinPool;
///
/// let mut matrix = Tensor::init(2.0, vec![4,4]);
///
/// matrix.set(vec![0,0], -2.0);
/// matrix.set(vec![0,3], -2.0);
/// matrix.set(vec![3,0], -2.0);
/// matrix.set(vec![3,3], -2.0);
///
/// let res = MinPool(&matrix, 2, 0);
///
/// assert_eq!(res.shape, vec![2,2]);
/// assert_eq!(res.get(vec![0,0]).unwrap(), -2.0);
/// assert_eq!(res.get(vec![0,1]).unwrap(), -2.0);
/// assert_eq!(res.get(vec![1,0]).unwrap(), -2.0);
/// assert_eq!(res.get(vec![1,1]).unwrap(), -2.0);
///
/// ```
pub fn MinPool<'a, T>(x: &Tensor<'a, T>, stride: usize, padding: usize) -> Tensor<'a, T>
where
    T: TensorElement,
    <T as FromStr>::Err: Error,
{
    let pred = |slice: &[T]| {
        slice
            .par_iter()
            .cloned()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
    };

    pool(x, stride, padding, pred)
}

/// Avg Pooling
///
/// # Examples
///
/// ```
/// use kaffe::Tensor;
/// use kaffe::nn::pooling::AvgPool;
///
/// let matrix = Tensor::init(2.0, vec![4,4]);
///
/// let res = AvgPool(&matrix, 2, 0);
///
/// assert_eq!(res.shape, vec![2,2]);
/// //assert_eq!(res.get(vec![0,0]), 2.0);
///
/// ```
pub fn AvgPool<'a, T>(x: &Tensor<'a, T>, stride: usize, padding: usize) -> Tensor<'a, T>
where
    T: TensorElement,
    <T as FromStr>::Err: Error,
{
    let average =
        |slice: &[T]| Some(slice.par_iter().cloned().sum::<T>() / T::from(slice.len()).unwrap());

    pool(x, stride, padding, average)
}

/// Internal Helper function that actually performs the
/// predicates given, so pooling becomes easier
fn pool<'a, T, F>(x: &Tensor<'a, T>, stride: usize, padding: usize, mut pred: F) -> Tensor<'a, T>
where
    F: FnMut(&[T]) -> Option<T> + Send + Sync + 'static,
    T: TensorElement,
    <T as FromStr>::Err: Error,
{
    // Calculate the dimensions of the resulting matrix
    let out_rows = ((x.shape.iter().nth_back(1).unwrap() + 2 * padding - stride) / stride) + 1;
    let out_cols = ((x.shape.iter().nth_back(0).unwrap() + 2 * padding - stride) / stride) + 1;

    let mut pooled = Tensor::zeros(vec![out_rows, out_cols]);

    // Perform the pooling operation
    for i in 0..out_rows {
        for j in 0..out_cols {
            // Calculate the start indices of the pooling window
            let start_row = i * stride;
            let start_col = j * stride;

            // Extract the slice from the original matrix
            let slice = x.get_vec_slice(vec![start_row, start_col], stride, stride);

            // Apply the predicate to the slice and get the result
            if let Some(value) = pred(&slice) {
                pooled.set(vec![i, j], value);
            }
        }
    }

    pooled
}
