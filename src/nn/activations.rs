#[warn(non_snake_case)]
const E: f32 = std::f32::consts::E;

pub fn ReLU(x: f32) -> f32 {
    if x >= 0f32 {
        x
    } else {
        0f32
    }
}

pub fn PReLU(alpha: f32, x: f32) -> f32 {
    if x >= 0f32 {
        x
    } else {
        x * alpha
    }
}

pub fn Sigmoid(x: f32) -> f32 {
    E.powf(x) / (E.powf(x) + 1f32)
}

// pub fn GeLU(x: f32) -> f32 {
//     0.5 * x * (1f32 + tanh(math.sqrt(2, std::f32::consts::PI) * (x + 0.044715 * x.powi(3))))
// }
