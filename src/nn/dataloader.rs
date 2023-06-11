//! Module for defining and implementing dataloaders
use crate::nn::dataset::DataSet;

type Data = f32;

/// Dataloader represents A iterator basically
pub struct DataLoader {
    data: DataSet,
}

impl Iterator for DataLoader {
    type Item = Data;

    /// Yields the next element
    ///
    /// Examples:
    ///
    /// ```
    ///
    /// ```
    fn next(&mut self) -> Option<Self::Item> {
        // Implement the logic for returning the next item
        // from your data vector or None if there are no more items
        todo!()
    }
}

impl DataLoader {
    /// Creates an iterator for us
    ///
    /// Examples:
    ///
    /// ```
    ///
    /// ```
    fn iter(&mut self) -> DataLoader {
        // Return a new instance of your custom struct,
        // which will serve as the iterator
        todo!()
    }
}
