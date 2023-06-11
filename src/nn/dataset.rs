//! Module for defining and implementing datasets

/// Represents what a single entity of our dataset will be
type Data = f32;

/// Dataloader represents A iterator basically
pub struct DataSet {
    data: Vec<f32>,
}
