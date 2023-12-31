//! Common optimizers for neural networks
#![warn(missing_docs)]

use std::{error::Error, marker::PhantomData, str::FromStr};

use crate::{Tensor, TensorElement};
use rayon::prelude::*;

/// Trait contains all the functions needed to run an optimizer
pub trait Optimizer<'a, T>
where
    T: TensorElement,
    <T as FromStr>::Err: Error,
{
    /// Initializes a new optimizer
    fn init(learning_rate: T, momentum: f32) -> Self;

    /// Function that minimizes based on cost function
    fn minimize<F>(&mut self, cost: F, vars: &mut Vec<T>)
    where
        F: Fn(&Tensor<'a, T>, &Tensor<'a, T>) -> T;
}

/// Adam Optimizer
///
/// # Examples
///
/// ```
/// use kaffe::Tensor;
/// use kaffe::nn::optimizer::Adam;
/// use kaffe::nn::optimizer::Optimizer;
///
/// let optim = Adam::init(1e-6, 0.8);
///
/// ```
pub struct Adam<T> {
    /// Learning rate
    lr: T,
    /// Momentum
    momentum: f32,
    /// Decay rate
    decay_rate: f32,
    /// Beta 1
    beta1: f32,
    /// Beta 2:
    beta2: f32,
    /// Epsilon
    epsilon: f32,
    m_dw: f32,
    v_dw: f32,
    m_db: f32,
    v_db: f32,
}

impl<'a, T> Optimizer<'a, T> for Adam<T>
where
    T: TensorElement,
    <T as FromStr>::Err: Error,
{
    fn init(learning_rate: T, momentum: f32) -> Self {
        Self {
            lr: learning_rate,
            momentum,
            decay_rate: 0.2,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            m_dw: 0.0,
            m_db: 0.0,
            v_dw: 0.0,
            v_db: 0.0,
        }
    }

    fn minimize<F>(&mut self, cost: F, vars: &mut Vec<T>)
    where
        F: Fn(&Tensor<'a, T>, &Tensor<'a, T>) -> T,
    {
        unimplemented!()
    }
}

/// SGD
///
/// # Examples
///
/// ```
/// use kaffe::Tensor;
/// use kaffe::nn::optimizer::SGD;
/// use crate::kaffe::nn::optimizer::Optimizer;
///
/// let optim = SGD::init(1e-6, 0.8);
///
/// ```
pub struct SGD<T> {
    /// Learning rate
    lr: T,
    /// Momentum
    momentum: f32,
    /// Decay rate
    decay_rate: f32,
}

impl<'a, T> Optimizer<'a, T> for SGD<T>
where
    T: TensorElement,
    <T as FromStr>::Err: Error,
{
    fn init(learning_rate: T, momentum: f32) -> Self {
        Self {
            lr: learning_rate,
            momentum,
            decay_rate: 0.2,
        }
    }

    fn minimize<F>(&mut self, cost: F, vars: &mut Vec<T>)
    where
        F: Fn(&Tensor<'a, T>, &Tensor<'a, T>) -> T,
    {
        todo!()
    }
}
