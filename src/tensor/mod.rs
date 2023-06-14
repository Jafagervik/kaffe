//! A tensor represents multidimensional data
//!
//! Matrix is only good enough up until a certain point

mod matrixmultiply;

use serde::{Deserialize, Serialize};
use std::{
    error::Error,
    fmt,
    fmt::{Debug, Display},
    fs,
    marker::PhantomData,
    ops::Div,
    str::FromStr,
};

use anyhow::Result;
use itertools::iproduct;
use itertools::Itertools;
use num_traits::{
    pow,
    sign::{abs, Signed},
    Float, Num, NumAssign, NumAssignOps, NumAssignRef, NumOps, One, Zero,
};
use rand::{distributions::uniform::SampleUniform, Rng};
use rayon::prelude::*;
use std::iter::{Product, Sum};

/// Shape represents the dimension size
/// of the tensor as a tuple of usize
pub type Shape = Vec<usize>;

/// Helper method to swap to usizes
fn swap(lhs: &mut usize, rhs: &mut usize) {
    let temp = *lhs;
    *lhs = *rhs;
    *rhs = temp;
}

// ======================================
//     Macros
// ======================================

/// Calculate 1D index from a 2D tensor
macro_rules! at {
    ($i:expr, $j:expr, $ncols:expr) => {
        $i * $ncols + $j
    };
}

/// Calculates 1D index from list of indexes
macro_rules! index {
    ($indexes:expr, $dimensions:expr) => {{
        if $indexes.len() != $dimensions.len() {
            32
        } else {
            let mut stride = 1;
            let mut result = 0;

            for (&index, &dimension) in $indexes.iter().rev().zip($dimensions.iter().rev()) {
                result += index * stride;
                stride *= dimension;
            }

            result
        }
    }};
}

/// Calculates a list of indexes from a single index and a vector of Tensor dim sizes
macro_rules! index_list {
    ($single_index:expr, $dimensions:expr) => {{
        let mut indexes = Vec::with_capacity($dimensions.len());
        let mut remaining_index = $single_index;

        for &dimension in $dimensions.iter().rev() {
            indexes.push(remaining_index % dimension);
            remaining_index /= dimension;
        }

        indexes.reverse();

        indexes
    }};
}

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

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

impl<'a, T> std::error::Error for Tensor<'a, T>
where
    T: TensorElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
}

#[derive(Clone, PartialEq, PartialOrd, Debug, Serialize, Deserialize)]
pub struct Tensor<'a, T>
where
    T: TensorElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    /// Vector containing all data
    pub data: Vec<T>,
    /// Shape of the tensor
    pub shape: Shape,
    /// Number of dimensions
    pub ndims: usize,
    _lifetime: PhantomData<&'a T>,
}

/// TensorElement is a trait explaining
/// all traits an item has to have.
/// FOr now, these restrictions make it so
/// only f32s and f64s are supported,
/// and for AI, this is mostly ok
pub trait TensorElement:
    Copy
    + Clone
    + PartialOrd
    + Signed
    + Float
    + Sum
    + Product
    + Display
    + Debug
    + FromStr
    + Default
    + One
    + PartialEq
    + Zero
    + Send
    + Sync
    + Sized
    + Num
    + NumOps
    + NumAssignOps
    + NumAssignRef
    + NumAssign
    + SampleUniform
{
}

unsafe impl<'a, T> Send for Tensor<'a, T>
where
    T: TensorElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
}

unsafe impl<'a, T> Sync for Tensor<'a, T>
where
    T: TensorElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
}

// Currently the only two supported datatypes
impl TensorElement for f32 {}
impl TensorElement for f64 {}

/// FromStr parses a Matrix from a file into a
impl<'a, T> FromStr for Tensor<'a, T>
where
    T: TensorElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    type Err = TensorError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // TODO: Add parse error
        let v: Vec<T> = s
            .trim()
            .lines()
            .map(|l| {
                l.split_whitespace()
                    .map(|num| num.parse::<T>().unwrap())
                    .collect::<Vec<T>>()
            })
            .collect::<Vec<Vec<T>>>()
            .into_iter()
            .flatten()
            .collect();

        let rows = s.trim().lines().count();
        let cols = s.trim().lines().nth(0).unwrap().split_whitespace().count();

        if let Ok(tensor) = Self::new(v, vec![rows, cols]) {
            return Ok(tensor);
        }

        Err(TensorError::TensorCreationError.into())
    }
}

impl<'a, T> Display for Tensor<'a, T>
where
    T: TensorElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}

impl<'a, T> Default for Tensor<'a, T>
where
    T: TensorElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    /// Represents a default 3x3 2D identity tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor: Tensor<f32> = Tensor::default();
    ///
    /// assert_eq!(tensor.size(), 9);
    /// ```
    fn default() -> Self {
        Self::eye(3)
    }
}

/// Implementations of all creatins of matrices
impl<'a, T> Tensor<'a, T>
where
    T: TensorElement,
    <T as FromStr>::Err: Error,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    /// Creates a new tensor from a vector and the shape you want.
    /// Will default init if it does not work
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor = Tensor::new(vec![1.0,2.0,3.0,4.0], vec![2,2]).unwrap();
    ///
    /// assert_eq!(tensor.size(), 4);
    /// assert_eq!(tensor.shape, vec![2,2]);
    /// ```
    pub fn new(data: Vec<T>, shape: Shape) -> Result<Self, TensorError> {
        if shape.iter().product::<usize>() != data.len() {
            return Err(TensorError::TensorCreationError.into());
        }

        Ok(Self {
            data,
            ndims: shape.len(),
            shape,
            _lifetime: PhantomData::default(),
        })
    }

    /// Initializes a tensor with the same value
    /// given from parameter 'value'
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor = Tensor::init(4f32, vec![1,1,1,4]);
    ///
    /// assert_eq!(tensor.data, vec![4f32; 4]);
    /// assert_eq!(tensor.shape, vec![1,1,1,4]);
    /// ```
    pub fn init(value: T, shape: Shape) -> Self {
        Self::from_shape(value, shape.clone())
    }

    /// Returns an eye tensor which for now is the same as the
    /// identity tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor: Tensor<f32> = Tensor::eye(2);
    ///
    /// assert_eq!(tensor.data, vec![1f32, 0f32, 0f32, 1f32]);
    /// assert_eq!(tensor.shape, vec![2,2]);
    /// ```
    pub fn eye(size: usize) -> Self {
        let mut data: Vec<T> = vec![T::zero(); size * size];

        (0..size).for_each(|i| data[i * size + i] = T::one());

        // This is safe to do since we decide the shape, not the user
        Self::new(data, vec![size, size]).unwrap()
    }

    /// Identity is same as eye, just for nerds
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor: Tensor<f64> = Tensor::identity(2);
    ///
    /// assert_eq!(tensor.data, vec![1f64, 0f64, 0f64, 1f64]);
    /// assert_eq!(tensor.shape, vec![2,2]);
    /// ```
    pub fn identity(size: usize) -> Self {
        Self::eye(size)
    }

    /// Tries to create a tensor from a slize and shape
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let s = vec![1f32, 2f32, 3f32, 4f32];
    /// let tensor = Tensor::from_slice(&s, vec![4,1]).unwrap();
    ///
    /// assert_eq!(tensor.shape, vec![4,1]);
    /// assert_eq!(tensor.get(vec![2, 0]).unwrap(), 3f32);
    /// ```
    pub fn from_slice(arr: &[T], shape: Shape) -> Result<Self, TensorError> {
        if shape.iter().product::<usize>() != arr.len() {
            return Err(TensorError::TensorCreationError.into());
        }

        Ok(Self::new(arr.to_owned(), shape).unwrap())
    }

    /// Creates a tensor where all values are 0.
    /// All sizes are based on a shape
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor: Tensor<f32> = Tensor::zeros(vec![4,1]);
    ///
    /// assert_eq!(tensor.shape, vec![4,1]);
    /// assert_eq!(tensor.data, vec![0f32; 4]);
    /// ```
    pub fn zeros(shape: Shape) -> Self {
        Self::from_shape(T::zero(), shape.clone())
    }

    /// Creates a tensor where all values are 1.
    /// All sizes are based on a shape
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor: Tensor<f64> = Tensor::ones(vec![4,1]);
    ///
    /// assert_eq!(tensor.shape, vec![4,1]);
    /// assert_eq!(tensor.data, vec![1f64; 4]);
    /// ```
    pub fn ones(shape: Shape) -> Self {
        Self::from_shape(T::one(), shape.clone())
    }

    /// Creates a tensor where all values are 0.
    /// All sizes are based on an already existent tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor1: Tensor<f32> = Tensor::default();
    /// let tensor2 = Tensor::zeros_like(&tensor1);
    ///
    /// assert_eq!(tensor2.shape, tensor1.shape);
    /// assert_eq!(tensor2.data[0], 0f32);
    /// ```
    pub fn zeros_like(other: &Self) -> Self {
        Self::from_shape(T::zero(), other.shape.clone())
    }

    /// Creates a tensor where all values are 1.
    /// All sizes are based on an already existent tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor1: Tensor<f64> = Tensor::default();
    /// let tensor2 = Tensor::ones_like(&tensor1);
    ///
    /// assert_eq!(tensor2.shape, tensor1.shape);
    /// assert_eq!(tensor2.data[0], 1f64);
    /// ```
    pub fn ones_like(other: &Self) -> Self {
        Self::from_shape(T::one(), other.shape.clone())
    }

    /// Creates a tensor where all values are random between 0 and 1.
    /// All sizes are based on an already existent tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor1: Tensor<f32> = Tensor::default();
    /// let tensor2 = Tensor::random_like(&tensor1);
    ///
    /// assert_eq!(tensor1.shape, tensor2.shape);
    /// ```
    pub fn random_like(tensor: &Self) -> Self {
        Self::randomize_range(T::zero(), T::one(), tensor.shape.clone())
    }

    /// Creates a tensor where all values are random between start..=end.
    /// Shape in new array is given through parameter 'shape'
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor = Tensor::randomize_range(1f32, 2f32, vec![2,3]);
    /// let elem = tensor.get(vec![1,1]);
    ///
    /// assert_eq!(tensor.shape, vec![2,3]);
    /// //assert!(elem >= 1f32 && 2f32 <= elem);
    /// ```
    pub fn randomize_range(start: T, end: T, shape: Shape) -> Self {
        let mut rng = rand::thread_rng();

        let len: usize = shape.iter().product();

        let data: Vec<T> = (0..len).map(|_| rng.gen_range(start..=end)).collect();

        Self::new(data, shape.clone()).unwrap()
    }

    /// Creates a tensor where all values are random between 0..=1.
    /// Shape in new array is given through parameter 'shape'
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor: Tensor<f64> = Tensor::randomize(vec![2,3]);
    ///
    /// assert_eq!(tensor.shape, vec![2,3]);
    /// ```
    pub fn randomize(shape: Shape) -> Self {
        Self::randomize_range(T::zero(), T::one(), shape.clone())
    }

    /// Parses from file, but will return a default tensor if nothing is given
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// // let m: Tensor<f32> = Tensor::from_file("../../test.txt")?;
    ///
    /// // m.print(5);
    /// ```
    pub fn from_file(path: &'static str) -> Result<Self, TensorError> {
        let data =
            fs::read_to_string(path).map_err(|_| TensorError::TensorFileReadError(path).into())?;

        data.parse::<Self>()
            .map_err(|_| TensorError::TensorParseError.into())
    }

    /// HELPER, name is too retarded for public usecases
    fn from_shape(value: T, shape: Shape) -> Self {
        let len: usize = shape.iter().product();

        let data = vec![value; len];

        Self::new(data, shape).unwrap()
    }
}

/// Enum for specifying which dimension / axis to work with
pub enum Dimension {
    /// Row is defined as 0
    Row = 0,
    /// Col is defined as 1
    Col = 1,
}

/// Regular tensor methods that are not operating math on them
impl<'a, T> Tensor<'a, T>
where
    T: TensorElement + Div<Output = T> + Sum<T>,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    /// Reshapes a tensor if possible.
    /// If the shapes don't match up, the old shape will be retained
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let mut tensor = Tensor::init(10.5, vec![2,3]);
    /// tensor.reshape(vec![3,2]);
    ///
    /// assert_eq!(tensor.shape, vec![3,2]);
    /// ```
    pub fn reshape(&mut self, new_shape: Shape) {
        if new_shape.iter().product::<usize>() != self.size() {
            println!("Can not reshape.. Keeping old dimensions for now");
            return;
        }

        self.shape = new_shape;
    }

    /// Remove any dimensions of size 1
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let mut tensor = Tensor::init(10.5, vec![1,1,1,2,3,1,2]);
    /// tensor.squeeze();
    ///
    /// assert_eq!(tensor.shape, vec![2,3,2]);
    /// ```
    pub fn squeeze(&mut self) {
        self.shape.retain(|&num| num > 1);

        self.ndims = self.shape.len()
    }

    /// Get the total size of the tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor = Tensor::init(10.5, vec![2,3]);
    ///
    /// assert_eq!(tensor.size(), 6);
    /// ```
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    ///  Gets element based on is and js
    ///
    ///  Returns None if index is out of bounds
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let mut tensor = Tensor::init(10.5, vec![7,200,3]);
    ///
    /// tensor.set(vec![1,2,3], 4.2);
    /// tensor.set(vec![4,100,1], 4.9);
    ///
    /// assert_eq!(tensor.get(vec![1,2,3]).unwrap(), 4.2);
    /// assert_eq!(tensor.get(vec![4,100,1]).unwrap(), 4.9);
    /// ```
    pub fn get(&self, idx: Shape) -> Option<T> {
        let i: usize = index!(idx, self.shape);
        if i >= self.size() {
            return None;
        }

        Some(self.data[i])
    }

    ///  Gets a piece of the tensor out as a vector
    ///
    ///
    ///  If some indexes are out of bounds, these will not be part of the slice
    ///  If all are outside, nothing gets returned
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor = Tensor::init(10.5, vec![4,4]);
    ///
    /// let slice = tensor.get_vec_slice(vec![1,1], 2, 2);
    ///
    /// assert_eq!(slice, vec![10.5; 4]);
    /// ```
    pub fn get_vec_slice(&self, start_idx: Shape, dy: usize, dx: usize) -> Vec<T> {
        let start_row = start_idx.iter().nth_back(1).unwrap().clone();
        let start_col = start_idx.iter().nth_back(0).unwrap().clone();

        let y_range = start_row..start_row + dy;
        let x_range = start_col..start_col + dx;

        iproduct!(y_range, x_range)
            .filter_map(|(i, j)| self.get(vec![i, j]))
            .collect()
    }

    ///  Gets a piece of the tensor out as a tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor = Tensor::init(10.5, vec![4,4]);
    /// ```
    fn get_sub_tensor(&self, start_idx: Shape, size: Shape) -> Vec<T> {
        unimplemented!()
    }

    ///  Sets element based on vector of indexes.
    ///  If index is out of bounds, the old value is kept
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let mut tensor = Tensor::init(10.5, vec![2,3]);
    /// tensor.set(vec![1,2], 11.5);
    ///
    /// assert_eq!(tensor.get(vec![1,2]).unwrap(), 11.5);
    /// ```
    pub fn set(&mut self, idx: Shape, value: T) {
        if idx.iter().product::<usize>() >= self.size() {
            eprintln!("Error: Index out of bounds. Keeping old value");
            return;
        }

        let i: usize = index!(idx, self.shape);

        self.data[i] = value;
    }

    ///  Sets multiple elements based on vector of indexes.
    ///  If index is out of bounds, the old value is kept
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let mut tensor = Tensor::init(10.5, vec![2,3]);
    /// tensor.set_many(vec![vec![1,2], vec![0,1]], 11.5);
    ///
    /// assert_eq!(tensor.get(vec![1,2]).unwrap(), 11.5);
    /// assert_eq!(tensor.get(vec![0,1]).unwrap(), 11.5);
    ///
    /// ```
    pub fn set_many(&mut self, indexes: Vec<Shape>, value: T) {
        indexes.iter().for_each(|idx| self.set(idx.clone(), value));
    }

    /// Finds maximum element in the tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor = Tensor::init(10.5, vec![2,3]);
    ///
    /// assert_eq!(tensor.max(), 10.5);
    /// ```
    pub fn max(&self) -> T {
        // Tensor must have at least one element, thus we can unwrap
        *self
            .data
            .par_iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    /// Finds minimum element in the tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let mut tensor = Tensor::init(10.5, vec![2,3]);
    /// tensor.data[2] = 1.0;
    ///
    /// assert_eq!(tensor.min(), 1.0);
    /// ```
    pub fn min(&self) -> T {
        // Tensor must have at least one element, thus we can unwrap
        *self
            .data
            .par_iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    /// Finds total sum of tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor = Tensor::init(10f32, vec![2,2]);
    ///
    /// assert_eq!(tensor.cumsum(), 40.0);
    /// ```
    pub fn cumsum(&self) -> T {
        self.data.par_iter().copied().sum()
    }

    /// Multiplies  all elements in tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor = Tensor::init(10f32, vec![2,2]);
    ///
    /// assert_eq!(tensor.cumprod(), 10000.0);
    /// ```
    pub fn cumprod(&self) -> T {
        self.data.par_iter().copied().product()
    }

    /// Gets the average of the tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor = Tensor::init(10f32, vec![2,2]);
    ///
    /// assert_eq!(tensor.avg(), 10.0);
    /// ```
    pub fn avg(&self) -> T {
        let mut size: T = T::zero();

        self.data.iter().for_each(|_| size += T::one());

        let tot: T = self.data.par_iter().copied().sum::<T>();

        tot / size
    }

    /// Gets the mean of the tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor = Tensor::init(10f32, vec![2,2]);
    ///
    /// assert_eq!(tensor.mean(), 10.0);
    /// ```
    pub fn mean(&self) -> T {
        self.avg()
    }

    /// Gets the median of the tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor = Tensor::new(vec![1.0, 4.0, 6.0, 5.0], vec![2,2]).unwrap();
    ///
    /// assert!(tensor.median() >= 4.45 && tensor.median() <= 4.55);
    /// ```
    pub fn median(&self) -> T {
        match self.data.len() % 2 {
            0 => {
                let half: usize = self.data.len() / 2;

                self.data
                    .iter()
                    .sorted_by(|a, b| a.partial_cmp(&b).unwrap())
                    .skip(half - 1)
                    .take(2)
                    .copied()
                    .sum::<T>()
                    / (T::from(2).unwrap())
            }
            1 => {
                let half: usize = self.data.len() / 2;

                self.data
                    .iter()
                    .sorted_by(|a, b| a.partial_cmp(&b).unwrap())
                    .nth(half)
                    .copied()
                    .unwrap()
            }
            _ => unreachable!(),
        }
    }

    /// Sums up elements over given dimension and axis
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    /// use kaffe::Dimension;
    ///
    /// let tensor = Tensor::init(10f32, vec![2,2]);
    /// ```
    fn sum(&self, rowcol: usize, dimension: Dimension) -> T {
        unimplemented!()
    }

    /// Prods up elements over given rowcol and dimension
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    /// use kaffe::Dimension;
    ///
    /// let tensor = Tensor::init(10f32, vec![2,2]);
    ///
    /// ```
    fn prod(&self, rowcol: usize, dimension: Dimension) -> T {
        unimplemented!()
    }
}

/// trait TensorLinAlg contains all common Linear Algebra functions to be
/// performed on matrices
impl<'a, T> Tensor<'a, T>
where
    T: TensorElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    /// Adds one tensor to another
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor1 = Tensor::init(10.0, vec![2,2]);
    /// let tensor2 = Tensor::init(10.0, vec![2,2]);
    ///
    /// assert_eq!(tensor1.add(&tensor2).unwrap().data[0], 20.0);
    /// ```
    pub fn add(&self, other: &Self) -> Result<Self, TensorError> {
        if self.shape != other.shape {
            return Err(TensorError::TensorDimensionMismatchError.into());
        }

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&x, &y)| x + y)
            .collect_vec();

        Ok(Self::new(data, self.shape.clone()).unwrap())
    }

    /// Subtracts one array from another
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor1 = Tensor::init(20.0, vec![2,2]);
    /// let tensor2 = Tensor::init(10.0, vec![2,2]);
    ///
    /// assert_eq!(tensor1.sub(&tensor2).unwrap().data[0], 10.0);
    /// ```
    pub fn sub(&self, other: &Self) -> Result<Self, TensorError> {
        if self.shape != other.shape {
            return Err(TensorError::TensorDimensionMismatchError.into());
        }

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&x, &y)| x - y)
            .collect_vec();

        Ok(Self::new(data, self.shape.clone()).unwrap())
    }

    /// Subtracts one array from another and returns the absolute value
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor1 = Tensor::init(10.0f32, vec![2,2]);
    /// let tensor2 = Tensor::init(15.0f32, vec![2,2]);
    ///
    /// assert_eq!(tensor1.sub_abs(&tensor2).unwrap().data[0], 5.0);
    /// ```
    pub fn sub_abs(&self, other: &Self) -> Result<Self, TensorError> {
        if self.shape != other.shape {
            return Err(TensorError::TensorDimensionMismatchError.into());
        }

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&x, &y)| if x > y { x - y } else { y - x })
            .collect_vec();

        Ok(Self::new(data, self.shape.clone()).unwrap())
    }

    /// Dot product of two matrices
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor1 = Tensor::init(20.0, vec![2,2]);
    /// let tensor2 = Tensor::init(10.0, vec![2,2]);
    ///
    /// assert_eq!(tensor1.mul(&tensor2).unwrap().data[0], 200.0);
    /// ```
    pub fn mul(&self, other: &Self) -> Result<Self, TensorError> {
        if self.shape != other.shape {
            return Err(TensorError::TensorDimensionMismatchError.into());
        }

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&x, &y)| x * y)
            .collect_vec();

        Ok(Self::new(data, self.shape.clone()).unwrap())
    }

    /// Bad handling of zero div
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor1 = Tensor::init(20.0, vec![2,2]);
    /// let tensor2 = Tensor::init(10.0, vec![2,2]);
    ///
    /// assert_eq!(tensor1.div(&tensor2).unwrap().data[0], 2.0);
    /// ```
    pub fn div(&self, other: &Self) -> Result<Self, TensorError> {
        if other.any(|e| e == &T::zero()) {
            return Err(TensorError::TensorDivideByZeroError.into());
        }

        if self.shape != other.shape {
            return Err(TensorError::TensorDimensionMismatchError.into());
        }

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&x, &y)| x / y)
            .collect_vec();

        Ok(Self::new(data, self.shape.clone()).unwrap())
    }

    /// Adds a value to a tensor and returns a new tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor = Tensor::init(20.0, vec![2,2]);
    /// let value: f32 = 2.0;
    /// assert_eq!(tensor.add_val(value).data[0], 22.0);
    /// ```
    pub fn add_val(&self, val: T) -> Self {
        let data: Vec<T> = self.data.par_iter().map(|&e| e + val).collect();

        Self::new(data, self.shape.clone()).unwrap()
    }

    /// Substracts a value to a tensor and returns a new tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor = Tensor::init(20.0, vec![2,2]);
    /// let value: f32 = 2.0;
    /// assert_eq!(tensor.sub_val(value).data[0], 18.0);
    /// ```
    pub fn sub_val(&self, val: T) -> Self {
        let data: Vec<T> = self.data.par_iter().map(|&e| e - val).collect();

        Self::new(data, self.shape.clone()).unwrap()
    }

    /// Multiplies a value to a tensor and returns a new tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor = Tensor::init(20.0, vec![2,2]);
    /// let value: f32 = 2.0;
    /// assert_eq!(tensor.mul_val(value).data[0], 40.0);
    /// ```
    pub fn mul_val(&self, val: T) -> Self {
        let data: Vec<T> = self.data.par_iter().map(|&e| e * val).collect();

        Self::new(data, self.shape.clone()).unwrap()
    }

    /// Divides a value to a tensor and returns a new tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor = Tensor::init(20.0, vec![2,2]);
    /// let value: f32 = 2.0;
    ///
    /// let result_mat = tensor.div_val(value);
    ///
    /// assert_eq!(result_mat.data, vec![10.0; 4]);
    /// ```
    ///
    /// # Panics
    ///
    /// When val is 0.
    pub fn div_val(&self, val: T) -> Self {
        let data: Vec<T> = self.data.par_iter().map(|&e| e / val).collect();

        Self::new(data, self.shape.clone()).unwrap()
    }

    /// Takes the logarithm of each element
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor = Tensor::init(10.0, vec![2,2]);
    ///
    /// let result = tensor.log(10.0);
    ///
    /// assert_eq!(result.shape, vec![2,2]);
    /// assert_eq!(result.data[0], 1.0);
    /// ```
    pub fn log(&self, base: T) -> Self {
        let data: Vec<T> = self.data.iter().map(|&e| e.log(base)).collect();

        Self::new(data, self.shape.clone()).unwrap()
    }

    /// Takes the natural logarithm of each element in a tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    /// use kaffe::constants::E64;
    ///
    /// let tensor = Tensor::init(E64, vec![2,2]);
    /// let result = tensor.ln();
    ///
    /// assert_eq!(result.shape, vec![2,2]);
    /// ```
    pub fn ln(&self) -> Self {
        let data: Vec<T> = self.data.iter().map(|&e| e.ln()).collect();

        Self::new(data, self.shape.clone()).unwrap()
    }

    /// Gets tanh of every value
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    /// use kaffe::constants::E;
    ///
    /// let tensor = Tensor::init(E, vec![2,2]);
    ///
    /// ```
    pub fn tanh(&self) -> Self {
        let data: Vec<T> = self.data.iter().map(|&e| e.tanh()).collect();

        Self::new(data, self.shape.clone()).unwrap()
    }

    /// Gets sinh of every value
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    /// use kaffe::constants::E;
    ///
    /// let tensor = Tensor::init(E, vec![2,2]);
    ///
    /// ```
    pub fn sinh(&self) -> Self {
        let data: Vec<T> = self.data.iter().map(|&e| e.tanh()).collect();

        Self::new(data, self.shape.clone()).unwrap()
    }

    /// Gets cosh of every value
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    /// use kaffe::constants::E;
    ///
    /// let tensor = Tensor::init(E, vec![2,2]);
    ///
    /// ```
    pub fn cosh(&self) -> Self {
        let data: Vec<T> = self.data.iter().map(|&e| e.cosh()).collect();

        Self::new(data, self.shape.clone()).unwrap()
    }

    /// Pows each value in a tensor by val times
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor = Tensor::init(2.0, vec![2,2]);
    ///
    /// let result_mat = tensor.pow(2);
    ///
    /// assert_eq!(result_mat.data, vec![4.0, 4.0, 4.0, 4.0]);
    /// ```
    pub fn pow(&self, val: usize) -> Self {
        let data: Vec<T> = self.data.par_iter().map(|&e| pow(e, val)).collect();

        Self::new(data, self.shape.clone()).unwrap()
    }

    /// Takes the absolute values of the tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor = Tensor::init(20.0, vec![2,2]);
    ///
    /// let res = tensor.abs();
    ///
    /// // assert_eq!(tensor1.data[0], 22.0);
    /// ```
    pub fn abs(&self) -> Self {
        let data: Vec<T> = self.data.par_iter().map(|&e| abs(e)).collect();

        Self::new(data, self.shape.clone()).unwrap()
    }

    /// Adds a tensor in-place to a tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let mut tensor1 = Tensor::init(20.0, vec![2,2]);
    /// let tensor2 = Tensor::init(2.0, vec![2,2]);
    ///
    /// tensor1.add_self(&tensor2);
    ///
    /// assert_eq!(tensor1.data[0], 22.0);
    /// ```
    pub fn add_self(&mut self, other: &Self) {
        self.data
            .par_iter_mut()
            .zip(&other.data)
            .for_each(|(a, b)| *a += *b);
    }

    /// Subtracts a tensor in-place to a tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let mut tensor1 = Tensor::init(20.0, vec![2,2]);
    /// let tensor2 = Tensor::init(2.0, vec![2,2]);
    ///
    /// tensor1.sub_self(&tensor2);
    ///
    /// assert_eq!(tensor1.data[0], 18.0);
    /// ```
    pub fn sub_self(&mut self, other: &Self) {
        self.data
            .par_iter_mut()
            .zip(&other.data)
            .for_each(|(a, b)| *a -= *b);
    }

    /// Multiplies a tensor in-place to a tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let mut tensor1 = Tensor::init(20.0, vec![2,2]);
    /// let tensor2 = Tensor::init(2.0, vec![2,2]);
    ///
    /// tensor1.mul_self(&tensor2);
    ///
    /// assert_eq!(tensor1.data[0], 40.0);
    /// ```
    pub fn mul_self(&mut self, other: &Self) {
        self.data
            .par_iter_mut()
            .zip(&other.data)
            .for_each(|(a, b)| *a *= *b);
    }

    /// Divides a tensor in-place to a tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let mut tensor1 = Tensor::init(20.0, vec![2,2]);
    /// let tensor2 = Tensor::init(2.0, vec![2,2]);
    ///
    /// tensor1.div_self(&tensor2);
    ///
    /// assert_eq!(tensor1.data[0], 10.0);
    /// ```
    pub fn div_self(&mut self, other: &Self) {
        self.data
            .par_iter_mut()
            .zip(&other.data)
            .for_each(|(a, b)| *a /= *b);
    }

    /// Abs tensor in-place to a tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let mut tensor = Tensor::init(20.0, vec![2,2]);
    ///
    /// tensor.abs_self()
    ///
    /// // assert_eq!(tensor1.data[0], 22.0);
    /// ```
    pub fn abs_self(&mut self) {
        self.data.par_iter_mut().for_each(|e| *e = abs(*e))
    }

    /// Adds a value in-place to a tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let mut tensor = Tensor::init(20.0, vec![2,2]);
    /// let value: f32 = 2.0;
    ///
    /// tensor.add_val_self(value);
    ///
    /// assert_eq!(tensor.data[0], 22.0);
    /// ```
    pub fn add_val_self(&mut self, val: T) {
        self.data.par_iter_mut().for_each(|e| *e += val);
    }

    /// Subtracts a value in-place to a tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let mut tensor = Tensor::init(20.0, vec![2,2]);
    /// let value: f32 = 2.0;
    ///
    /// tensor.sub_val_self(value);
    ///
    /// assert_eq!(tensor.data[0], 18.0);
    /// ```
    pub fn sub_val_self(&mut self, val: T) {
        self.data.par_iter_mut().for_each(|e| *e -= val);
    }

    /// Mults a value in-place to a tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let mut tensor = Tensor::init(20.0, vec![2,2]);
    /// let value: f32 = 2.0;
    ///
    /// tensor.mul_val_self(value);
    ///
    /// assert_eq!(tensor.data[0], 40.0);
    /// ```
    pub fn mul_val_self(&mut self, val: T) {
        self.data.par_iter_mut().for_each(|e| *e *= val);
    }

    /// Divs a value in-place to a tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let mut tensor = Tensor::init(20.0, vec![2,2]);
    /// let value: f32 = 2.0;
    ///
    /// tensor.div_val_self(value);
    ///
    /// assert_eq!(tensor.data[0], 10.0);
    /// ```
    pub fn div_val_self(&mut self, val: T) {
        self.data.par_iter_mut().for_each(|e| *e /= val);
    }

    /// Transposed tensor multiplications
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor1 = Tensor::init(2.0, vec![2,4]);
    /// let tensor2 = Tensor::init(2.0, vec![4,2]);
    ///
    /// let result = tensor1.matmul(&tensor2).unwrap();
    ///
    /// ```
    pub fn matmul(&self, other: &Self) -> Result<Self, TensorError> {
        if self.ndims == 1 && other.ndims == 1 {
            return Ok(self.mul(&other).unwrap());
        }

        if self.ndims == 1 && other.ndims == 0 {
            return Ok(self.mul_val(other.data[0]));
        }

        if self.ndims == 0 && other.ndims == 1 {
            return Ok(self.mul_val(other.data[0]));
        }

        if self.shape[1] != other.shape[0] {
            return Err(TensorError::MatrixMultiplicationDimensionMismatchError.into());
        }

        Ok(Self::default())
    }

    /// Performs matrix multiply on MN x NP tensors
    /// tensor.mm works for MxN @ NxP tensors only,
    /// so use tensor.matmul for tensors of higher dimensions
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor1 = Tensor::init(2.0, vec![2,4]);
    /// let tensor2 = Tensor::init(2.0, vec![4,2]);
    ///
    /// let result = tensor1.mm(&tensor2).unwrap();
    ///
    /// assert_eq!(result.shape, vec![2,2]);
    /// assert_eq!(result.data, vec![16.0; 4]);
    ///
    /// ```
    pub fn mm(&self, other: &Self) -> Result<Self, TensorError> {
        if self.ndims != 2 || other.ndims != 2 {
            return self.matmul(&other);
        }

        let r1 = self.shape[0];
        let c1 = self.shape[1];
        let c2 = other.shape[1];

        let mut data = vec![T::zero(); c2 * r1];

        let t_other = other.transpose_copy();

        for i in 0..r1 {
            for j in 0..c2 {
                data[at!(i, j, c2)] = (0..c1)
                    .into_par_iter()
                    .map(|k| self.data[at!(i, k, c1)] * t_other.data[at!(j, k, t_other.shape[1])])
                    .sum();
            }
        }

        Self::new(data, vec![c2, r1])
    }

    /// Transpose a tensor in-place
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let mut tensor = Tensor::init(2.0, vec![2,100]);
    /// tensor.transpose();
    ///
    /// assert_eq!(tensor.shape, vec![100,2]);
    /// ```
    pub fn transpose(&mut self) {
        if self.ndims < 2 {
            eprintln!("Error: You need at least a 2D tensor (matrix) to transpose.");
            return;
        }

        let ncols = self.shape.iter().nth_back(1).unwrap().clone();
        let nrows = self.shape.iter().nth_back(0).unwrap().clone();

        for i in 0..nrows {
            for j in (i + 1)..ncols {
                let lhs = index!(vec![i, j], self.shape);
                let rhs = index!(vec![j, i], self.shape);
                self.data.swap(lhs, rhs);
            }
        }

        let ncols_idx = self.shape.len() - 1;
        let nrows_idx = self.shape.len() - 2;

        self.shape.swap(ncols_idx, nrows_idx);
    }

    /// Shorthand call for transpose
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let mut tensor = Tensor::init(2.0, vec![2,100]);
    /// tensor.t();
    ///
    /// assert_eq!(tensor.shape, vec![100,2]);
    /// ```
    pub fn t(&mut self) {
        self.transpose()
    }

    /// Transpose a tensor and return a copy
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor = Tensor::init(2.0, vec![2,100]);
    /// let result = tensor.transpose_copy();
    ///
    /// assert_eq!(result.shape, vec![100,2]);
    /// ```
    pub fn transpose_copy(&self) -> Self {
        let mut res = self.clone();
        res.transpose();
        res
    }

    /// Find the eigenvale of a tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let mut tensor = Tensor::init(2.0, vec![2,100]);
    ///
    /// assert_eq!(42f32, 42f32);
    /// ```
    pub fn eigenvalue(&self) -> T {
        todo!()
    }
}

/// Implementations for predicates
impl<'a, T> Tensor<'a, T>
where
    T: TensorElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    /// Counts all occurances where predicate holds
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor = Tensor::init(2.0f64, vec![2,4]);
    ///
    /// assert_eq!(tensor.count_where(|&e| e == 2.0), 8);
    /// ```
    pub fn count_where<F>(&'a self, pred: F) -> usize
    where
        F: Fn(&T) -> bool + Sync,
    {
        self.data.par_iter().filter(|&e| pred(e)).count()
    }

    /// Sums all occurances where predicate holds
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor = Tensor::init(2.0, vec![2,4]);
    ///
    /// assert_eq!(tensor.sum_where(|&e| e == 2.0), 16.0);
    /// ```
    pub fn sum_where<F>(&self, pred: F) -> T
    where
        F: Fn(&T) -> bool + Sync,
    {
        self.data
            .par_iter()
            .filter(|&e| pred(e))
            .copied()
            .sum::<T>()
    }

    /// Sets all values based on a predicate
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let mut tensor = Tensor::init(2.0, vec![2, 4]);
    ///
    /// assert_eq!(tensor.data[0], 2.0);
    ///
    /// tensor.set_where(|e| {
    ///     if *e == 2.0 {
    ///         *e = 2.3;
    ///     }
    /// });
    ///
    /// assert_eq!(tensor.data[0], 2.3);
    /// ```
    pub fn set_where<P>(&mut self, mut pred: P)
    where
        P: FnMut(&mut T) + Sync + Send,
    {
        for element in self.data.iter_mut() {
            pred(element);
        }
        // self.data.par_iter_mut().for_each(|e| pred(e));
    }

    /// Return whether or not a predicate holds at least once
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor = Tensor::init(2.0, vec![2,4]);
    ///
    /// assert_eq!(tensor.any(|&e| e == 2.0), true);
    /// ```
    pub fn any<F>(&self, pred: F) -> bool
    where
        F: Fn(&T) -> bool + Sync + Send,
    {
        self.data.par_iter().any(pred)
    }

    /// Returns whether or not predicate holds for all values
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor = Tensor::randomize_range(1.0, 4.0, vec![2,4]);
    ///
    /// assert_eq!(tensor.all(|&e| e >= 1.0), true);
    /// ```
    pub fn all<F>(&self, pred: F) -> bool
    where
        F: Fn(&T) -> bool + Sync + Send,
    {
        self.data.par_iter().all(pred)
    }

    /// Finds first index where predicates holds if possible
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor = Tensor::init(2f32, vec![2,4]);
    ///
    /// assert_eq!(tensor.find(|&e| e >= 1f32), Some(vec![0,0]));
    /// ```
    pub fn find<F>(&self, pred: F) -> Option<Shape>
    where
        F: Fn(&T) -> bool + Sync,
    {
        if let Some((idx, _)) = self.data.iter().find_position(|&e| pred(e)) {
            return Some(index_list!(idx, self.shape));
        }

        None
    }

    /// Finds all indeces where predicates holds if possible
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor = Tensor::init(2.0, vec![2,4]);
    ///
    /// assert_eq!(tensor.find_all(|&e| e >= 3.0), None);
    /// ```
    pub fn find_all<F>(&self, pred: F) -> Option<Vec<Shape>>
    where
        F: Fn(&T) -> bool + Sync,
    {
        let data: Vec<Shape> = self
            .data
            .par_iter()
            .enumerate()
            .filter_map(|(idx, elem)| {
                if pred(elem) {
                    Some(index_list!(idx, self.shape))
                } else {
                    None
                }
            })
            .collect();

        if data.is_empty() {
            None
        } else {
            Some(data)
        }
    }
}
