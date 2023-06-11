//! First time writing docs for my code, please be nice
#![warn(missing_docs)]

mod determinant_helpers;

use determinant_helpers::*;
use itertools::Itertools;
use rand::Rng;
use rayon::prelude::*;
use std::{
    fmt::{Debug, Error},
    str::FromStr,
};

/// Type definitions
pub type Shape = (usize, usize);

/// calculates index for matrix
macro_rules! at {
    ($i:ident, $j:ident, $cols:expr) => {
        $i * $cols + $j
    };
}

/// Represents all data we want to get
#[derive(Clone)]
pub struct Matrix {
    /// Data contained inside the matrix with f32 values
    pub data: Vec<f32>,
    /// Shape of matrix, (rows, cols)
    pub shape: Shape,
}

impl Debug for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[");

        for i in 0..self.rows() {
            for j in 0..self.cols() {
                if i == 0 {
                    write!(f, "{} ", self.data[at!(i, j, self.cols())]);
                } else {
                    write!(f, " {}", self.data[at!(i, j, self.cols())]);
                }
            }
            // Print ] on same line if youre at the end
            if i == self.shape.0 - 1 {
                break;
            }
            write!(f, "\n");
        }
        write!(f, "], dtype=f32")
    }
}

impl FromStr for Matrix {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Parse the input string and construct the matrix dynamically
        let v: Vec<f32> = s
            .trim()
            .lines()
            .map(|l| {
                l.split_whitespace()
                    .map(|num| num.parse::<f32>().unwrap())
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<Vec<f32>>>()
            .into_iter()
            .flatten()
            .collect();

        let rows = s.trim().lines().count();
        let cols = s.trim().lines().nth(0).unwrap().split_whitespace().count();

        Ok(Self::new(v, (rows, cols)))
    }
}

/// Printer functions for the matrix
impl Matrix {
    /// Prints out the matrix based on shape. Also outputs datatype
    pub fn print(&self) {
        print!("[");

        // Large matrices
        if self.rows() > 10 || self.cols() > 10 {
            print!("...");
        }

        for i in 0..self.rows() {
            for j in 0..self.cols() {
                if i == 0 {
                    print!("{:.2} ", self.data[at!(i, j, self.cols())]);
                } else {
                    print!(" {:.2}", self.data[at!(i, j, self.cols())]);
                }
            }
            // Print ] on same line if youre at the end
            if i == self.shape.0 - 1 {
                break;
            }
            print!("\n");
        }
        println!("], dtype=f32")
    }
}

/// Implementations of all creatins of matrices
impl Matrix {
    /// Creates a new matrix from a vector and the shape you want.
    /// Will default init if it does not work
    pub fn new(data: Vec<f32>, shape: Shape) -> Self {
        if shape.0 * shape.1 != data.len() {
            return Self::default();
        }

        Self { data, shape }
    }

    /// Represents a default identity matrix
    pub fn default() -> Self {
        Self {
            data: vec![0f32; 9],
            shape: (3, 3),
        }
    }

    /// Init works like zeros or ones
    pub fn init(value: f32, shape: Shape) -> Self {
        Self::from_shape(value, shape)
    }

    /// Returns an eye matrix
    pub fn eye(size: usize) -> Self {
        let mut data = vec![0f32; size * size];

        (0..size).for_each(|i| data[i * size + i] = 1f32);

        Self::new(data, (size, size))
    }

    /// Identity is same as eye, just for nerds
    pub fn identity(size: usize) -> Self {
        Self::eye(size)
    }

    /// Some sort of error handling
    /// Really trivial function
    pub fn from_slice(arr: &[f32], shape: Shape) -> Option<Self> {
        if shape.0 * shape.1 != arr.len() {
            return None;
        }

        Some(Self::new(arr.to_owned(), shape))
    }

    /// All zeroes baby
    pub fn zeros(shape: Shape) -> Self {
        Self::from_shape(0f32, shape)
    }

    /// All zeroes baby
    pub fn ones(shape: Shape) -> Self {
        Self::from_shape(1f32, shape)
    }

    /// Gives you a zero like matrix based on another matrix
    pub fn zeros_like(other: &Self) -> Self {
        Self::from_shape(0f32, other.shape)
    }

    /// Gives you a one like matrix based on another matrix
    pub fn ones_like(other: &Self) -> Self {
        Self::from_shape(1f32, other.shape)
    }

    /// Gives you a random 0-1 like matrix based on another matrix
    /// Range is 0 to 1
    pub fn random_like(matrix: &Self) -> Self {
        Self::randomize_range(0f32, 1f32, matrix.shape)
    }

    /// Random initialize values
    pub fn randomize_range(start: f32, end: f32, shape: Shape) -> Self {
        let mut rng = rand::thread_rng();

        let (rows, cols) = shape;

        let len: usize = rows * cols;

        let data: Vec<f32> = (0..len).map(|_| rng.gen_range(start..=end)).collect();

        Self::new(data, shape)
    }

    /// Range here will be -1 to 1
    pub fn randomize(shape: Shape) -> Self {
        Self::randomize_range(-1f32, 1f32, shape)
    }

    /// Parses from file, but will return a default matrix if nothing is given
    pub fn from_file(path: &'static str) -> Self {
        path.parse::<Self>().unwrap_or_else(|_| Self::default())
    }

    /// HELPER, name is too retarded for public usecases
    fn from_shape(value: f32, shape: Shape) -> Self {
        let (rows, cols) = shape;

        let len: usize = rows * cols;

        let data = vec![value; len];

        Self::new(data, shape)
    }
}

/// Enum for specifying which dimension / axis to work with
pub enum Dimension {
    /// Row is defined as 0
    Row = 0,
    /// Col is defined as 1
    Col = 1,
}

/// Regular matrix methods that are not operating math on them
impl Matrix {
    /// Converts matrix to a tensor
    pub fn to_tensor(&self) {
        todo!()
    }

    /// Reshapes a matrix if possible.
    /// If the shapes don't match up, the old shape will be retained
    pub fn reshape(&mut self, new_shape: Shape) {
        if new_shape.0 * new_shape.1 != self.size() {
            println!("Can not reshape.. Keeping old dimensions for now");
        }

        self.shape = new_shape;
    }

    /// Get number of columns in the matrix
    pub fn cols(&self) -> usize {
        self.shape.1
    }

    /// Get number of rows in the matrix
    pub fn rows(&self) -> usize {
        self.shape.0
    }

    /// Total size of matrix
    pub fn size(&self) -> usize {
        self.shape.0 * self.shape.1
    }

    /// Could get an out of bounds
    pub fn get(&self, i: usize, j: usize) -> f32 {
        self.data[at!(i, j, self.cols())]
    }

    /// Calculates the (row, col) for a matrix by a single index
    fn inverse_at(&self, idx: usize) -> Shape {
        let mut idx = idx;

        // Get amount of rows
        let s0 = idx % self.cols();

        // The rest is now cols
        idx %= self.cols();

        (s0, idx)
    }

    /// Finds maximum element in the matrix
    pub fn max(&self) -> f32 {
        // Matrix must have at least one element, thus we can unwrap
        *self
            .data
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    /// Finds maximum element in the matrix
    pub fn min(&self) -> f32 {
        // Matrix must have at least one element, thus we can unwrap
        *self
            .data
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    /// Finds maximum element in the matrix based on dimension
    ///
    /// dimension: whether to look in rows or cols
    /// rowcol: which of these to look into
    ///
    /// Example:
    ///     let a = Matrix::eye(3);
    ///     a.argmax(2, Dimension::Col)
    pub fn argmax(&self, rowcol: usize, dimension: Dimension) -> Option<f32> {
        match dimension {
            Dimension::Row => {
                if rowcol >= self.rows() - 1 {
                    return None;
                }

                self.data
                    .iter()
                    .skip(rowcol)
                    .step_by(self.cols())
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .copied()
            }

            Dimension::Col => {
                if rowcol >= self.cols() - 1 {
                    return None;
                }

                // TODO: Fix this skipping
                self.data
                    .iter()
                    .skip(rowcol)
                    .step_by(self.rows())
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .copied()
            }
        }
    }

    /// Finds minimum element in the matrix based on dimension
    ///
    /// dimension: whether to look in rows or cols
    /// rowcol: which of these to look into
    ///
    /// Example:
    ///     let a = Matrix::eye(3);
    ///     a.argmin(2, Dimension::Row)
    pub fn argmin(&self, rowcol: usize, dimension: Dimension) -> Option<f32> {
        match dimension {
            Dimension::Row => {
                if rowcol >= self.rows() - 1 {
                    return None;
                }

                self.data
                    .iter()
                    .skip(rowcol)
                    .step_by(self.cols())
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .copied()
            }

            Dimension::Col => {
                if rowcol >= self.cols() - 1 {
                    return None;
                }

                // TODO: Fix this skipping
                self.data
                    .iter()
                    .skip(rowcol)
                    .step_by(self.rows())
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .copied()
            }
        }
    }

    /// Sums up all elements in matrix
    pub fn cumsum(&self) -> f32 {
        self.data.par_iter().sum()
    }

    /// Sums up all elements in matrix
    pub fn cumprod(&self) -> f32 {
        self.data.par_iter().product()
    }

    /// Sums up elements over given dimension and axis
    pub fn sum(&self, rowcol: usize, dimension: Dimension) -> f32 {
        match dimension {
            Dimension::Row => self
                .data
                .par_iter()
                .skip(rowcol * self.cols())
                .step_by(rowcol)
                .sum(),
            Dimension::Col => self.data.par_iter().skip(rowcol).step_by(self.cols()).sum(),
        }
    }

    /// Prods up elements over given rowcol and dimension
    pub fn prod(&self, rowcol: usize, dimension: Dimension) -> f32 {
        match dimension {
            Dimension::Row => self
                .data
                .par_iter()
                .skip(rowcol * self.cols())
                .step_by(rowcol)
                .product(),
            Dimension::Col => self
                .data
                .par_iter()
                .skip(rowcol)
                .step_by(self.cols())
                .product(),
        }
    }
}

/// trait MatrixLinAlg contains all common Linear Algebra functions to be
/// performed on matrices
pub trait MatrixLinAlg {
    /// Adds one matrix to another
    fn add(&self, other: &Self) -> Self;

    /// Subtracts one array from another
    fn sub(&self, other: &Self) -> Self;

    /// Dot product of two matrices
    fn mul(&self, other: &Self) -> Self;

    /// Bad handling of zero div
    fn div(&self, other: &Self) -> Self;

    /// Adds a value to a matrix and returns a new matrix
    fn add_val(&self, val: f32) -> Self;

    /// Substracts a value to a matrix and returns a new matrix
    fn sub_val(&self, val: f32) -> Self;

    /// Multiplies a value to a matrix and returns a new matrix
    fn mul_val(&self, val: f32) -> Self;

    /// Divides a value to a matrix and returns a new matrix
    fn div_val(&self, val: f32) -> Self;

    /// Adds a matrix in-place to a matrix
    fn add_self(&mut self, other: &Self);

    /// Subtracts a matrix in-place to a matrix
    fn sub_self(&mut self, other: &Self);

    /// Multiplies a matrix in-place to a matrix
    fn mul_self(&mut self, other: &Self);

    /// Divides a matrix in-place to a matrix
    fn div_self(&mut self, other: &Self);

    /// Adds a value in-place to a matrix
    fn add_val_self(&mut self, val: f32);

    /// Subtracts a value in-place to a matrix
    fn sub_val_self(&mut self, val: f32);

    /// Mults a value in-place to a matrix
    fn mul_val_self(&mut self, val: f32);

    /// Divs a value in-place to a matrix
    fn div_val_self(&mut self, val: f32);

    /// Transposed matrix multiplications
    fn matmul(&self, other: &Self) -> Self;

    /// Find the determinant of a matrix
    fn determinant(&self) -> f32;

    /// Transpose a matrix in-place
    fn transpose(&mut self);

    /// Shorthand call for transpose
    fn t(&mut self);

    /// Transpose a matrix and return a copy
    fn transpose_copy(&self) -> Self;

    /// Find the eigenvale of a matrix
    fn eigenvalue(&self) -> f32;
}

impl MatrixLinAlg for Matrix {
    fn add(&self, other: &Self) -> Self {
        if self.rows() != other.rows() || self.cols() != other.cols() {
            panic!("NOOO!");
        }

        let mut mat = Self::zeros_like(self);

        for i in 0..self.rows() {
            for j in 0..self.cols() {
                let a = self.get(i, j);
                let b = other.get(i, j);
                mat.data[at!(i, j, self.cols())] = a + b;
            }
        }
        mat
    }

    fn sub(&self, other: &Self) -> Self {
        if self.rows() != other.rows() || self.cols() != other.cols() {
            panic!("NOOO!");
        }

        let mut mat = Self::zeros_like(self);

        for i in 0..self.rows() {
            for j in 0..self.cols() {
                let a = self.get(i, j);
                let b = other.get(i, j);
                mat.data[at!(i, j, self.cols())] = a - b;
            }
        }
        mat
    }

    fn mul(&self, other: &Self) -> Self {
        if self.rows() != other.rows() || self.cols() != other.cols() {
            panic!("NOOO!");
        }

        let mut mat = Self::zeros_like(self);

        for i in 0..self.rows() {
            for j in 0..self.cols() {
                let a = self.get(i, j);
                let b = other.get(i, j);
                mat.data[at!(i, j, self.cols())] = a * b;
            }
        }
        mat
    }

    fn div(&self, other: &Self) -> Self {
        if self.rows() != other.rows() || self.cols() != other.cols() {
            panic!("NOOO!");
        }

        // div by 0 :/
        if other.any(|e| e == &0f32) {
            panic!("NOOOOO")
        }

        let mut mat = Self::zeros_like(self);

        for i in 0..self.rows() {
            for j in 0..self.cols() {
                let a = self.get(i, j);
                let b = other.get(i, j);
                mat.data[at!(i, j, self.cols())] = a / b;
            }
        }
        mat
    }

    fn add_val(&self, val: f32) -> Self {
        let data: Vec<f32> = self.data.par_iter().map(|&e| e + val).collect();

        Self::new(data, self.shape)
    }

    fn sub_val(&self, val: f32) -> Self {
        let data: Vec<f32> = self.data.par_iter().map(|&e| e - val).collect();

        Self::new(data, self.shape)
    }

    fn mul_val(&self, val: f32) -> Self {
        let data: Vec<f32> = self.data.par_iter().map(|&e| e * val).collect();

        Self::new(data, self.shape)
    }

    fn div_val(&self, val: f32) -> Self {
        let data: Vec<f32> = self.data.par_iter().map(|&e| e / val).collect();

        Self::new(data, self.shape)
    }

    fn add_self(&mut self, other: &Self) {
        self.data
            .par_iter_mut()
            .zip(&other.data)
            .for_each(|(a, b)| *a += b);
    }

    fn sub_self(&mut self, other: &Self) {
        self.data
            .par_iter_mut()
            .zip(&other.data)
            .for_each(|(a, b)| *a -= b);
    }

    fn mul_self(&mut self, other: &Self) {
        self.data
            .par_iter_mut()
            .zip(&other.data)
            .for_each(|(a, b)| *a *= b);
    }

    fn div_self(&mut self, other: &Self) {
        self.data
            .par_iter_mut()
            .zip(&other.data)
            .for_each(|(a, b)| *a /= b);
    }

    fn add_val_self(&mut self, val: f32) {
        self.data.par_iter_mut().for_each(|e| *e += val);
    }

    fn sub_val_self(&mut self, val: f32) {
        self.data.par_iter_mut().for_each(|e| *e -= val);
    }

    fn mul_val_self(&mut self, val: f32) {
        self.data.par_iter_mut().for_each(|e| *e *= val);
    }

    fn div_val_self(&mut self, val: f32) {
        self.data.par_iter_mut().for_each(|e| *e /= val);
    }

    fn matmul(&self, other: &Self) -> Self {
        // assert M N x N P
        if self.cols() != other.rows() {
            panic!("Oops, dimensions do not match");
        }

        let r1 = self.rows();
        let c1 = self.cols();
        let c2 = other.cols();

        let mut data = vec![0f32; c2 * r1];

        let t_other = other.transpose_copy();

        for i in 0..r1 {
            for j in 0..c2 {
                let dot_product = (0..c1)
                    .into_par_iter()
                    .map(|k| self.data[at!(i, k, c1)] * t_other.data[at!(j, k, t_other.cols())])
                    .sum();

                data[at!(i, j, c2)] = dot_product;
            }
        }

        Self::new(data, (c2, r1))
    }

    fn determinant(&self) -> f32 {
        match self.size() {
            1 => self.data[0],
            4 => determinant_2x2(self),
            9 => determinant_3x3(self),
            _ => {
                todo!()
                // let mut det = 0.0;
                // let mut sign = 1.0;
                //
                // for i in 0..3 {
                //     let minor = get_minor(self, 3, i);
                //     det += sign * self.data[i] * determinant(&minor);
                //     sign *= -1.0;
                // }
                //
                // det
            }
        }
    }

    fn transpose(&mut self) {
        for i in 0..self.rows() {
            for j in (i + 1)..self.cols() {
                let lhs = at!(i, j, self.cols());
                let rhs = at!(j, i, self.rows());
                self.data.swap(lhs, rhs);
            }
        }

        swap(&mut self.shape.0, &mut self.shape.1);
    }

    fn t(&mut self) {
        self.transpose()
    }

    fn transpose_copy(&self) -> Matrix {
        let mut res = self.clone();
        res.transpose();
        res
    }

    fn eigenvalue(&self) -> f32 {
        todo!()
    }
}

/// Matrix functions with predicates
pub trait MatrixPredicates {
    /// Counts all occurances where predicate holds
    fn count_where<'a, F>(&'a self, pred: F) -> usize
    where
        F: Fn(&'a f32) -> bool + 'static;

    /// Sums all occurances where predicate holds
    fn sum_where<'a, F>(&'a self, pred: F) -> f32
    where
        F: Fn(&'a f32) -> bool + 'static;

    /// Return whether or not a predicate holds at least once
    fn any<'a, F>(&'a self, pred: F) -> bool
    where
        F: Fn(&'a f32) -> bool + 'static;

    /// Returns whether or not predicate holds for all values
    fn all<'a, F>(&'a self, pred: F) -> bool
    where
        F: Fn(&'a f32) -> bool + 'static;

    /// Finds first index where predicates holds if possible
    fn find<'a, F>(&'a self, pred: F) -> Option<Shape>
    where
        F: Fn(&'a f32) -> bool + 'static;

    /// Finds all indeces where predicates holds if possible
    fn find_all<'a, F>(&'a self, pred: F) -> Option<Vec<Shape>>
    where
        F: Fn(&'a f32) -> bool + 'static;
}

impl MatrixPredicates for Matrix {
    fn count_where<'a, F>(&'a self, pred: F) -> usize
    where
        F: Fn(&'a f32) -> bool + 'static,
    {
        self.data.iter().filter(|&e| pred(e)).count()
    }

    fn sum_where<'a, F>(&'a self, pred: F) -> f32
    where
        F: Fn(&'a f32) -> bool + 'static,
    {
        self.data.iter().filter(|&e| pred(e)).sum()
    }

    fn any<'a, F>(&'a self, pred: F) -> bool
    where
        F: Fn(&'a f32) -> bool + 'static,
    {
        self.data.iter().any(pred)
    }

    fn all<'a, F>(&'a self, pred: F) -> bool
    where
        F: Fn(&'a f32) -> bool + 'static,
    {
        self.data.iter().all(pred)
    }

    fn find<'a, F>(&'a self, pred: F) -> Option<Shape>
    where
        F: Fn(&'a f32) -> bool + 'static,
    {
        if let Some((idx, _)) = self.data.iter().find_position(|&e| pred(e)) {
            return Some(self.inverse_at(idx));
        }

        None
    }

    fn find_all<'a, F>(&'a self, pred: F) -> Option<Vec<Shape>>
    where
        F: Fn(&'a f32) -> bool + 'static,
    {
        let data: Vec<Shape> = self
            .data
            .iter()
            .enumerate()
            .filter_map(|(idx, elem)| {
                if pred(elem) {
                    Some(self.inverse_at(idx))
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

/// Helper method to swap to usizes
fn swap(lhs: &mut usize, rhs: &mut usize) {
    let temp = *lhs;
    *lhs = *rhs;
    *rhs = temp;
}
