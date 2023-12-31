#![warn(missing_docs)]
//! Neural Network Base module
//!
//! Should be kept as simple as possible to avoid confusion,
//! and if people want to make their own versions
//! of optimizers, losses, and so on; that should be supported
pub mod activation;
pub mod dataloader;
pub mod dataset;
pub mod loss;
pub mod optimizer;
pub mod pooling;
pub mod transform;

use std::{error::Error, str::FromStr};

use crate::{tensor::Tensor, TensorElement};

/// struct Layer represents a layer in our net
pub struct Layer {}

/// Convolve a matrix with a certain kernel
pub fn convolution<'a, T>(
    tensor: Tensor<'a, T>,
    kernel: Tensor<'a, T>,
    stride: usize,
    padding: usize,
) -> Tensor<'a, T>
where
    T: TensorElement,
    <T as FromStr>::Err: Error,
{
    todo!()
}

/// Net is a net builder people can use to create epic neural networks
pub struct Net {
    /// Represents all layers in the network
    layers: Vec<Layer>,
}

/// Module is simular to how nn.Module from pytorch works
pub trait Module {
    /// Initializes a neural network
    fn init() -> Net;

    /// Default forward pass must be implemented
    fn forward(&mut self);
}
