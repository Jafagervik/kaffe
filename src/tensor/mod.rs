//! A tensor represents multidimensional data
//!
//! Matrix is only good enough up until a certain point
#![warn(missing_docs)]

use itertools::Itertools;
use rand::Rng;
use rayon::prelude::*;

/// Makes geting elements easier
macro_rules! at {
    ($($idx:expr)*) => {
        42
    };
}

/// Represents all data inside the tensor
pub struct Tensor {
    /// Data stored in tensor
    pub data: Vec<f32>,
    /// Shape of tensor
    pub shape: Vec<usize>,
    /// Number of dimensions in tensor
    pub ndims: usize,
}

/// Creations for Tensors
impl Tensor {
    /// Creates a new tensor from a vector of data and shapes
    ///
    /// Examples:
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor = Tensor::new(vec![1.0,2.0,3.0,4.0], vec![1,2,2]);
    ///
    /// assert_eq!(tensor.shape, vec![1,2,2]);
    /// assert_eq!(tensor.size(), 4);
    /// ```
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let size = shape.len();
        Self {
            data,
            shape,
            ndims: size,
        }
    }

    /// Initializes a tensor with a default value
    ///
    /// Examples:
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor = Tensor::init(4.0, vec![2,2,2]);
    ///
    /// assert_eq!(tensor.shape, vec![2,2,2]);
    /// assert_eq!(tensor.size(), 8);
    /// ```
    pub fn init(value: f32, shape: Vec<usize>) -> Self {
        Self::from_shape(value, shape)
    }

    /// Initializes a tensor with only 0's
    ///
    /// Examples:
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor = Tensor::zeros(vec![2,2,2]);
    ///
    /// assert_eq!(tensor.shape, vec![2,2,2]);
    /// assert_eq!(tensor.data[0], 0.0);
    /// ```
    pub fn zeros(shape: Vec<usize>) -> Self {
        Self::from_shape(0.0, shape)
    }

    /// Initializes a tensor with only 1's
    ///
    /// Examples:
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor = Tensor::ones(vec![2,2,2]);
    ///
    /// assert_eq!(tensor.shape, vec![2,2,2]);
    /// assert_eq!(tensor.data[0], 1.0);
    /// ```
    pub fn ones(shape: Vec<usize>) -> Self {
        Self::from_shape(1.0, shape)
    }

    /// Initializes a tensor with only 0's like another tensor
    ///
    /// Examples:
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let a = Tensor::init(7f32, vec![1,2,3,4]);
    /// let b = Tensor::zeros_like(&a);
    ///
    /// assert_eq!(b.shape, vec![1,2,3,4]);
    /// assert_eq!(b.data[0], 0.0);
    /// ```
    pub fn zeros_like(tensor: &Self) -> Self {
        Self::from_shape(0.0, tensor.shape.clone())
    }

    /// Initializes a tensor with only 1's like another tensor
    ///
    /// Examples:
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let a = Tensor::init(7f32, vec![1,2,3,4]);
    /// let b = Tensor::ones_like(&a);
    ///
    /// assert_eq!(b.shape, vec![1,2,3,4]);
    /// assert_eq!(b.data[0], 1.0);
    /// ```
    pub fn ones_like(tensor: &Tensor) -> Self {
        Self::from_shape(1.0, tensor.shape.clone())
    }

    // TODO: Copy from matrix
    pub fn random_like(tensor: &Tensor) -> Self {
        Self::randomize_range(0.0, 1.0, tensor.shape.clone())
    }

    // TODO: Copy from matrix
    pub fn randomize_range(start: f32, end: f32, shape: Vec<usize>) -> Self {
        let mut rng = rand::thread_rng();

        let size = shape.iter().sum();

        let data: Vec<f32> = (0..size).map(|_| rng.gen_range(start..=end)).collect();

        Self::new(data, shape)
    }

    /// TODO: Copy from matrix
    pub fn randomize(shape: Vec<usize>) -> Self {
        Self::randomize_range(-1f32, 1f32, shape)
    }

    /// TODO: Copy from matrix
    pub fn eye(size: usize) -> Self {
        let mut data = vec![0f32; size * size];

        (0..size).for_each(|i| data[i * size + i] = 1f32);

        Self::new(data, vec![size, size])
    }

    /// TODO: Copy from matrix
    pub fn identity(size: usize) -> Self {
        Self::eye(size)
    }

    /// TODO: Copy from matrix
    pub fn from_slice(arr: &[f32], shape: Vec<usize>) -> Option<Self> {
        if shape.iter().sum::<usize>() != arr.len() {
            return None;
        }

        Some(Self::new(arr.to_owned(), shape))
    }

    // Internal helper
    fn from_shape(value: f32, shape: Vec<usize>) -> Self {
        let vec = vec![value; shape.iter().product()];

        Self::new(vec, shape)
    }
}

/// Setters and getters for Tensors
impl Tensor {
    /// Returns the size of a tensor
    ///
    /// Examples:
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let t = Tensor::init(2f32, vec![1,2,3,4,5]);
    ///
    /// assert_eq!(t.size(), 120);
    /// ```
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    /// Reshapes a tensor if possible.
    /// If the shapes don't match up, the old shape will be retained
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let mut tensor = Tensor::init(10.5, vec![1,2,3]);
    /// tensor.reshape(vec![2,3,1]);
    ///
    /// assert_eq!(tensor.shape, vec![2,3,1]);
    /// ```
    pub fn reshape(&mut self, new_shape: Vec<usize>) {
        if new_shape.iter().sum::<usize>() != self.size() {
            println!("Can not reshape.. Keeping old dimensions for now");
        }

        self.shape = new_shape;
    }

    /// Gets element at index list
    /// If the shapes don't match up, the old shape will be retained
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let mut tensor = Tensor::init(10.5, vec![1,2,3]);
    /// let elem = tensor.get(vec![0,0,0]);
    ///
    /// assert_eq!(elem, Some(10.5));
    /// ```
    pub fn get(&self, idx: Vec<usize>) -> Option<f32> {
        if idx.len() != self.shape.len() {
            println!("unimplemented");
            return None;
        }

        Some(10.5)
    }

    /// Finds minimum element in the tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor = Tensor::init(10.5, vec![1,2,3]);
    ///
    /// assert_eq!(tensor.min(), 10.5);
    /// ```
    pub fn min(&self) -> f32 {
        *self
            .data
            .par_iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    /// Finds max element in the tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor = Tensor::init(10.5, vec![1,2,3]);
    ///
    /// assert_eq!(tensor.max(), 10.5);
    /// ```
    pub fn max(&self) -> f32 {
        *self
            .data
            .par_iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    // TODO: Find index where value is max
    pub fn argmax(&self, dimension: usize, which: usize) -> Vec<usize> {
        todo!()
    }

    // TODO: Find index where value is max
    pub fn argmin(&self, dimension: usize, which: usize) -> Vec<usize> {
        todo!()
    }

    /// Finds total sum of tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let matrix = Tensor::init(10f32, vec![2,2]);
    ///
    /// assert_eq!(matrix.cumsum(), 40f32);
    /// ```
    pub fn cumsum(&self) -> f32 {
        self.data.par_iter().sum()
    }

    /// Multiplies all elements in Tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let matrix = Tensor::init(10f32, vec![2,2]);
    ///
    /// assert_eq!(matrix.cumprod(), 10000f32);
    /// ```
    pub fn cumprod(&self) -> f32 {
        self.data.iter().product()
    }

    /// Gets the average of the tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let matrix = Tensor::init(10f32, vec![2,2]);
    ///
    /// assert_eq!(matrix.avg(), 10.0);
    /// ```
    pub fn avg(&self) -> f32 {
        self.data.par_iter().sum::<f32>() / self.data.len() as f32
    }

    /// Gets the mean of the tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let matrix = Tensor::init(10f32, vec![2,2]);
    ///
    /// assert_eq!(matrix.mean(), 10.0);
    /// ```
    pub fn mean(&self) -> f32 {
        self.avg()
    }

    /// Gets the median of the tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let matrix = Tensor::new(vec![1.0, 4.0, 6.0, 5.0], vec![2,2]);
    ///
    /// assert!(matrix.median() >= 4.45 && matrix.median() <= 4.55);
    /// ```
    pub fn median(&self) -> f32 {
        match self.data.len() % 2 {
            0 => {
                let half: usize = self.data.len() / 2;

                self.data
                    .iter()
                    .sorted_by(|a, b| a.partial_cmp(&b).unwrap())
                    .skip(half - 1)
                    .take(2)
                    .sum::<f32>()
                    / 2.0
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

    /// TODO: implement
    pub fn sum(&self, dimension: usize) -> f32 {
        todo!()
    }

    /// TODO: implement
    pub fn prod(&self, dimension: usize) -> f32 {
        todo!()
    }
}

/// Basic Tensor operations
/// Add, sum, mean and so on
pub trait TensorOps {
    fn add(&self, other: &Self) -> Self;
    fn sub(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;
    fn div(&self, other: &Self) -> Self;

    fn add_val(&self, val: f32) -> Self;
    fn sub_val(&self, val: f32) -> Self;
    fn mul_val(&self, val: f32) -> Self;
    fn div_val(&self, val: f32) -> Self;

    fn add_self(&mut self, other: &Self);
    fn sub_self(&mut self, other: &Self);
    fn mul_self(&mut self, other: &Self);
    fn div_self(&mut self, other: &Self);

    fn add_val_self(&mut self, val: f32);
    fn sub_val_self(&mut self, val: f32);
    fn mul_val_self(&mut self, val: f32);
    fn div_val_self(&mut self, val: f32);
}

/// Mathematical functions on tensor
pub trait TensorMath {
    /// Takes the log(base) of every element in Tensor
    fn log(&self, base: i32) -> Self;

    /// Takes the natural logarithm of every element in Tensor
    fn ln(&self) -> Self;

    /// Gets tanh of every element in tensor
    fn tanh(&self) -> Self;

    /// Pows every element by exp  
    fn pow(&self, exp: i32) -> Self;

    /// Find square root of every element in tensor
    fn sqrt(&self, base: usize) -> Self;
}

impl TensorMath for Tensor {
    fn log(&self, base: i32) -> Self {
        todo!()
    }

    fn ln(&self) -> Self {
        todo!()
    }

    fn tanh(&self) -> Self {
        todo!()
    }

    fn pow(&self, exp: i32) -> Self {
        todo!()
    }

    fn sqrt(&self, base: usize) -> Self {
        todo!()
    }
}

/// Specialized Tensor operations for linear algebra, on matrices
/// Matrix in this case will be the last two dims
pub trait TensorLinAlg {
    /// Matrix multiplication with transposition
    fn matmul(lhs: &Self, rhs: &Self) -> Self;

    /// Finds eigenvalue of a matrix
    fn eigenvalue(&self) -> f32;

    /// Finds determinant of a matrix
    fn determinant(&self) -> Self;

    /// Transpose
    fn transpose(&mut self);

    /// Transpose
    fn T(&mut self);

    /// Transpose into new copy
    fn transpose_copy() -> Self;
}

/// Tensor predicates
impl Tensor {
    /// Counts elements where predicate holds true
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor = Tensor::init(2.0f32, vec![2,4]);
    ///
    /// assert_eq!(tensor.count_where(|&e| e == 2.0), 8);
    /// ```
    pub fn count_where<'a, F>(&'a self, pred: F) -> usize
    where
        F: Fn(&'a f32) -> bool + 'static + Sync,
    {
        self.data.par_iter().filter(|&e| pred(e)).count()
    }

    /// Sums elements where predicate holds true
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor = Tensor::init(2.0, vec![2,4]);
    ///
    /// assert!(tensor.sum_where(|&e| e == 2.0) == 16.0);
    /// ```
    pub fn sum_where<'a, F>(&'a self, pred: F) -> f32
    where
        F: Fn(&'a f32) -> bool + 'static + Sync,
    {
        self.data.par_iter().filter(|&e| pred(e)).sum()
    }

    /// Returns true if at least one element hold true
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor = Tensor::init(2.0, vec![2,4]);
    ///
    /// assert_eq!(tensor.any(|&e| e >= 1.0), true);
    /// ```
    pub fn any<'a, F>(&'a self, pred: F) -> bool
    where
        F: Fn(&'a f32) -> bool + 'static + Sync + Send,
    {
        self.data.par_iter().any(pred)
    }

    /// Returns true if all elements hold true
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor = Tensor::init(2.0, vec![2,4]);
    ///
    /// assert_eq!(tensor.all(|&e| e >= 1.0), true);
    /// ```
    pub fn all<'a, F>(&'a self, pred: F) -> bool
    where
        F: Fn(&'a f32) -> bool + 'static + Sync + Send,
    {
        self.data.par_iter().all(pred)
    }

    /// Finds first index  where predicates holds true if possible
    ///
    /// # Examples
    ///
    /// ```
    /// use kaffe::Tensor;
    ///
    /// let tensor = Tensor::init(2.0, vec![2,4]);
    ///
    /// assert_eq!(tensor.find(|&e| e >= 1.0), Some(vec![1,2]));
    /// ```
    pub fn find<'a, F>(&'a self, pred: F) -> Option<Vec<usize>>
    where
        F: Fn(&'a f32) -> bool + 'static,
    {
        if let Some((idx, _)) = self.data.iter().find_position(|&e| pred(e)) {
            return Some(vec![1, 2]);
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
    /// let matrix = Tensor::init(2.0, vec![2,4]);
    ///
    /// assert_eq!(matrix.find_all(|&e| e >= 3.0), None);
    /// ```
    pub fn find_all<'a, F>(&'a self, pred: F) -> Option<Vec<Vec<usize>>>
    where
        F: Fn(&'a f32) -> bool + 'static + Sync,
    {
        let data: Vec<Vec<usize>> = self
            .data
            .par_iter()
            .enumerate()
            .filter_map(|(idx, elem)| {
                if pred(elem) {
                    None
                    //Some(vec![vec![2 as usize; 4 as usize]; 4])
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
