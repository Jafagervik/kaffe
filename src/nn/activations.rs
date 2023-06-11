//!This module contains the most used non-linear activation functions
#[warn(non_snake_case)]
#[warn(missing_docs)]

/// Yes
const E: f32 = std::f32::consts::E;

/// ReLU is the most used activation funcion besides Sigmoid
pub fn ReLU(x: f32) -> f32 {
    if x >= 0f32 {
        x
    } else {
        0f32
    }
}

/// PReLU is a slight modification to ReLU
pub fn PReLU(alpha: f32, x: f32) -> f32 {
    if x >= 0f32 {
        x
    } else {
        x * alpha
    }
}

/// Sigmoid function
pub fn Sigmoid(x: f32) -> f32 {
    E.powf(x) / (E.powf(x) + 1f32)
}

// pub fn GeLU(x: f32) -> f32 {
//     0.5 * x * (1f32 + tanh(math.sqrt(2, std::f32::consts::PI) * (x + 0.044715 * x.powi(3))))
// }
