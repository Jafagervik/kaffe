#![warn(missing_docs)]
//! hello

pub mod constants;
pub mod matrix;
pub mod nn;

pub use constants::*;
pub use matrix::*;

/// I am
fn test() -> f32 {
    1f32 + 2f32
}
