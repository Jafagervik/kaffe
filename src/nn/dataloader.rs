//! Module for defining and implementing dataloaders
use crate::nn::dataset::DataSet;

/// Dataloader represents A iterator basically
#[derive(Clone)]
pub struct DataLoader<D: DataSet + Clone> {
    /// Dataset
    data: D,
    /// Batch size
    batch_size: usize,
    /// Whether or not to shuffle data
    shuffle: bool,
}

impl<D: DataSet + Clone> Iterator for DataLoader<D> {
    /// Item is what we'll get out when we iterate
    type Item = D;

    /// Yields the next element
    ///
    /// Examples:
    ///
    /// ```
    ///
    /// ```
    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

impl<D: DataSet + Clone> DataLoader<D> {
    /// Creates a new dataloader for us
    ///
    /// Examples:
    ///
    /// ```
    ///
    /// ```
    fn new(data: D, batch_size: usize, shuffle: bool) -> DataLoader<D> {
        Self {
            data,
            batch_size,
            shuffle,
        }
    }

    /// Creates an iterator from a dataset
    ///
    /// Examples:
    ///
    /// ```
    ///
    /// ```
    fn iter(&mut self) -> DataLoader<D> {
        // Return a new instance of your custom struct,
        // which will serve as the iterator
        self.clone()
    }
}
