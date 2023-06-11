//!  Pooling
#![warn(missing_docs)]
use crate::{Matrix, MatrixLinAlg, MatrixPredicates};
use rayon::prelude::*;

/// Max Pooling
///
/// # Examples
///
/// ```
/// use kaffe::{Matrix, MatrixLinAlg};
/// use kaffe::nn::pooling::MaxPool;
///
/// let matrix = Matrix::init(2.0, (4,4));
///
/// let res = MaxPool(&matrix, 2, 0);
///
/// assert_eq!(res.shape, (2,2));
///
/// ```
pub fn MaxPool(x: &Matrix, stride: usize, padding: usize) -> Matrix {
    let pred = |slice: &[f32]| {
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
/// use kaffe::{Matrix, MatrixLinAlg};
/// use kaffe::nn::pooling::MinPool;
///
/// let matrix = Matrix::init(2.0, (4,4));
///
/// let res = MinPool(&matrix, 2, 0);
///
/// assert_eq!(res.shape, (2,2));
/// assert_eq!(res.get(1,1), 2.0);
///
/// ```
pub fn MinPool(x: &Matrix, stride: usize, padding: usize) -> Matrix {
    let pred = |slice: &[f32]| {
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
/// use kaffe::{Matrix, MatrixLinAlg};
/// use kaffe::nn::pooling::AvgPool;
///
/// let matrix = Matrix::init(2.0, (4,4));
///
/// let res = AvgPool(&matrix, 2, 0);
///
/// assert_eq!(res.shape, (2,2));
/// //assert_eq!(res.get(0,0), 2.0);
///
/// ```
pub fn AvgPool(x: &Matrix, stride: usize, padding: usize) -> Matrix {
    let pred = |slice: &[f32]| Some(slice.par_iter().cloned().sum::<f32>() / slice.len() as f32);

    pool(x, stride, padding, pred)
}

/// Internal Helper function that actually performs the
/// predicates given, so pooling becomes easier
fn pool<F>(x: &Matrix, stride: usize, padding: usize, mut pred: F) -> Matrix
where
    F: FnMut(&[f32]) -> Option<f32> + Send + Sync + 'static,
{
    // Calculate the dimensions of the resulting matrix
    let out_rows = ((x.rows() + 2 * padding - stride) / stride) + 1;
    let out_cols = ((x.cols() + 2 * padding - stride) / stride) + 1;

    let mut pooled = Matrix::zeros((out_rows, out_cols));

    // Perform the pooling operation
    for i in 0..out_rows {
        for j in 0..out_cols {
            // Calculate the start indices of the pooling window
            let start_row = i * stride;
            let start_col = j * stride;

            // Extract the slice from the original matrix
            let slice = x.get_vec_slice(start_row, start_col, stride, stride);

            // Apply the predicate to the slice and get the result
            if let Some(value) = pred(&slice) {
                pooled.set(i, j, value);
            }
        }
    }

    pooled
}
