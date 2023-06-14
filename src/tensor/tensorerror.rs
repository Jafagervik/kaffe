//! Errors on tensors
#![warn(missing_docs)]

use std::fmt::{Display, Formatter, Result};

#[derive(Debug, PartialEq)]
/// Common Tensor errors that can occur
pub enum TensorError {
    /// Upon creation of a tensor, this could occur
    TensorCreationError,
    /// Index out of bound error
    TensorIndexOutOfBoundsError,
    /// This can only happen on matmul, where if the 2 matrices are not in the form of
    /// (M x N) @ (N x P) then this error will occur.
    MatrixMultiplicationDimensionMismatchError,
    /// Occurs on matrix operations where there is a dimension mismatch between
    /// the two matrices.
    TensorDimensionMismatchError,
    /// If reading tensor from file and an error occurs,
    /// this will be thrown
    TensorParseError,
    /// Divide by zero
    TensorDivideByZeroError,
    /// File read error
    TensorFileReadError(&'static str),
}

impl Display for TensorError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            TensorError::TensorCreationError => {
                write!(f, "There was an error creating the tensor.")
            }
            TensorError::TensorIndexOutOfBoundsError => {
                write!(f, "The indexes are out of bounds for the matrix")
            }
            TensorError::MatrixMultiplicationDimensionMismatchError => {
                write!(
                    f,
                    "The two matrices supplied are not on the form M x N @ N x P"
                )
            }
            TensorError::TensorDimensionMismatchError => {
                write!(f, "The tensors provided are both not on the form M x N")
            }
            TensorError::TensorParseError => write!(f, "Failed to parse tensor from file"),
            TensorError::TensorDivideByZeroError => write!(f, "Tried to divide by zero"),
            TensorError::TensorFileReadError(path) => {
                write!(f, "Could not read file from path: {}", path)
            }
        }
    }
}
