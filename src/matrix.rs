use rand::Rng;
use rayon::prelude::*;
use std::{
    fmt::{Debug, Error},
    str::FromStr,
};

/// calculates index for matrix
macro_rules! at {
    ($i:ident, $j:ident, $cols:expr) => {
        $i * $cols + $j
    };
}

/// Represents all data we want to get
#[derive(Clone)]
pub struct Matrix {
    data: Vec<f32>,
    len: usize,
    shape: (usize, usize),
}

impl Debug for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[");

        // Large matrices
        if self.rows() > 10 || self.cols() > 10 {
            write!(f, "....");
        }

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
                    .map(|num| num.parse::<f32>().ok().unwrap())
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

/// Pretty printing
impl Matrix {
    pub fn print(&self) {
        print!("[");

        // Large matrices
        if self.rows() > 10 || self.cols() > 10 {
            print!("....");
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
    pub fn new(data: Vec<f32>, shape: (usize, usize)) -> Self {
        Self {
            data,
            len: shape.0,
            shape,
        }
    }

    /// represents a default identity matrix
    pub fn default() -> Self {
        Self {
            data: vec![0f32; 9],
            len: 3,
            shape: (3, 3),
        }
    }

    /// Returns an eye matrix
    pub fn eye(size: usize) -> Self {
        let mut data = vec![0f32; size * size];

        (0..size).for_each(|i| data[i * size + i] = 1f32);

        Self::new(data, (size, size))
    }

    /// Some sort of error handling
    /// Really trivial function
    pub fn from_vec(vec: Vec<f32>, shape: (usize, usize)) -> Option<Self> {
        if shape.0 * shape.1 != vec.len() {
            return None;
        }

        Some(Self::new(vec, shape))
    }

    /// Gives you a zero like matrix based on another matrix
    pub fn zeros_like(other: &Self) -> Self {
        Self::from_shape(0f32, other.shape)
    }

    /// Gives you a one like matrix based on another matrix
    pub fn ones_like(other: &Self) -> Self {
        Self::from_shape(1f32, other.shape)
    }

    /// Random initialize values
    pub fn randomize_range(start: f32, end: f32, shape: (usize, usize)) -> Self {
        let mut rng = rand::thread_rng();

        let (rows, cols) = shape;

        let len: usize = rows * cols;

        let data: Vec<f32> = (0..len).map(|_| rng.gen_range(start..=end)).collect();

        Self::new(data, shape)
    }

    /// Range here will be -1 to 1
    pub fn randomize(shape: (usize, usize)) -> Self {
        Self::randomize_range(-1f32, 1f32, shape)
    }

    pub fn from_shape(value: f32, shape: (usize, usize)) -> Self {
        let (rows, cols) = shape;

        let len: usize = rows * cols;

        let data = vec![value; len];

        Self::new(data, shape)
    }
}

/// Getters
impl Matrix {
    pub fn to_tensor(&self) {
        todo!()
    }

    pub fn cols(&self) -> usize {
        self.shape.1
    }

    pub fn rows(&self) -> usize {
        self.shape.0
    }

    /// Could get an out of bounds
    pub fn get(&self, i: usize, j: usize) -> f32 {
        self.data[at!(i, j, self.cols())]
    }
}

/// Other identities on matrices

pub trait MatrixOps {
    fn add(&self, other: &Self) -> Self;
    fn sub(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;
    fn div(&self, other: &Self) -> Self;
    fn matmul(&self, other: &Self) -> Self;
    fn determinant(&self) -> f32;
    fn transpose(&mut self);
    fn transpose_new(&self) -> Self;
}

/// Represents the different operations that can be performed
enum Operation {
    PLUS(f32, f32),
    MINUS(f32, f32),
    MULTIPLY(f32, f32),
    DIVIDE(f32, f32),
}

fn inner_op(op: Operation) -> f32 {
    match op {
        Operation::PLUS(lhs, rhs) => lhs + rhs,
        Operation::MINUS(lhs, rhs) => lhs - rhs,
        Operation::MULTIPLY(lhs, rhs) => lhs * rhs,
        Operation::DIVIDE(lhs, rhs) => lhs / rhs,
    }
}

impl MatrixOps for Matrix {
    fn add(&self, other: &Self) -> Self {
        if self.rows() != other.rows() || self.cols() != other.cols() {
            panic!("NOOO!");
        }

        let mut mat = Self::zeros_like(self);

        for i in 0..self.rows() {
            for j in 0..self.cols() {
                let a = self.get(i, j);
                let b = other.get(i, j);
                mat.data[at!(i, j, self.cols())] = inner_op(Operation::PLUS(a, b));
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
                mat.data[at!(i, j, self.cols())] = inner_op(Operation::MINUS(a, b));
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
                mat.data[at!(i, j, self.cols())] = inner_op(Operation::MULTIPLY(a, b));
            }
        }
        mat
    }

    /// Bad handling of zero div
    fn div(&self, other: &Self) -> Self {
        if self.rows() != other.rows() || self.cols() != other.cols() {
            panic!("NOOO!");
        }

        // div by 0 :/
        if other.data.iter().any(|e| e == &0f32) {
            panic!("NOOOOO")
        }

        let mut mat = Self::zeros_like(self);

        for i in 0..self.rows() {
            for j in 0..self.cols() {
                let a = self.get(i, j);
                let b = other.get(i, j);
                mat.data[at!(i, j, self.cols())] = inner_op(Operation::DIVIDE(a, b));
            }
        }
        mat
    }

    /// Transposed Matrix multiplication
    fn matmul(&self, other: &Self) -> Self {
        // assert M N x N P
        if self.cols() != other.rows() {
            panic!("Oops, dimensions do not match");
        }

        let r1 = self.rows();
        let c1 = self.cols();
        let c2 = other.cols();

        // let mut res = Self::from_shape(0f32, (c2, r1));
        let mut data = vec![0f32; c2 * r1];

        let t_other = other.transpose_new();

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

    /// Jacobian determinant
    fn determinant(&self) -> f32 {
        todo!()
    }

    fn transpose(&mut self) {
        for i in 0..self.rows() {
            for j in (i + 1)..self.cols() {
                let lhs = at!(i, j, self.cols());
                let rhs = at!(j, i, self.rows());
                self.data.swap(lhs, rhs);
            }
        }

        self.shape = (self.shape.1, self.shape.0);
    }

    /// Transpose into new matrix
    fn transpose_new(&self) -> Matrix {
        let mut res = self.clone();
        res.transpose();
        res
    }
}

#[test]
fn lezgoo() {
    unimplemented!();
}
