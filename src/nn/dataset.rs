//! Module for defining and implementing datasets
#![warn(missing_docs)]
use crate::nn::transform::Transform;
use crate::Tensor;
use std::os;

/// Dataset represents a set of many matrices
pub trait DataSet {
    /// Initializes a new dataset
    fn new(
        img_labels: Vec<String>,
        img_dir: &'static str,
        transform: Option<Transform>,
        target_transform: Option<Transform>,
    ) -> Self;

    /// Gives the size of the dataset
    fn size(&self) -> usize;

    /// Gets an element at a certain index
    fn get(&self, idx: usize) -> (Tensor, String);
}

/// Represents a custom dataset
/// All common datasets will be added here by default
pub struct CustomImageDataset {
    img_labels: Vec<String>,
    img_dir: &'static str,
    transform: Option<Transform>,
    target_transform: Option<Transform>,
}

impl DataSet for CustomImageDataset {
    fn new(
        img_labels: Vec<String>,
        img_dir: &'static str,
        transform: Option<Transform>,
        target_transform: Option<Transform>,
    ) -> Self {
        Self {
            img_labels,
            img_dir,
            transform,
            target_transform,
        }
    }

    fn size(&self) -> usize {
        self.img_labels.len()
    }

    fn get(&self, idx: usize) -> (Tensor, String) {
        todo!()
        // img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        // image = read_image(img_path)
        // label = self.img_labels.iloc[idx, 1]
        // if self.transform:
        //     image = self.transform(image)
        // if self.target_transform:
        //     label = self.target_transform(label)
        // return image, label
    }
}
