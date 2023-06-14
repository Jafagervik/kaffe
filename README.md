# Kaffe - A Pytorch inspired library written in rust 

[![Crates.io](https://img.shields.io/crates/v/kaffe.svg)](https://crates.io/crates/kaffe)
[![Documentation](https://docs.rs/kaffe/badge.svg)](https://docs.rs/kaffe/)
[![Coverage Status](https://codecov.io/gh/Jafagervik/kaffe/branch/master/graph/badge.svg)](https://codecov.io/gh/Jafagervik/kaffe)
![Maintenance](https://img.shields.io/badge/maintenance-experimental-blue.svg)

Kaffe is as per the title, a way to create Neural Networks in Rust.

The goal is to create a simple way to write your own models, and test them.
Syntax should be familiar to pytorch, but some features might take 
names from numpy or even tensorflow.

In the future, the matrix library might be completely moved to its
own project, but for now they're all in the same crate;

Why? Because sometimes you wanna make cool and fast stuff in rust :)


## Examples

### Matrix basic example 

```rust 
use kaffe::Matrix;

fn main() {
    let a = Matrix::init(2f32, (2,3));
    let b = Matrix::init(4f32, (2,3));

    let c = a.add(&b);

    // To print this beautiful matrix:
    c.print();
}
```

### Neural net basic example - To Be Implemented
```rust
use kaffe::Matrix;
use kaffe::{Net, Layer, optimizer::*, loss::*};

// Here lies our model 
struct MyNet {
    layers: Vec<Layer>
}

// Implement default functions
impl Net for MyNet {
    /// Set's up parametes for the struct 
    fn init() -> Self {
        let mut layers: Vec<Layers> = Vec::new();
        self.layers.push(nn.Conv2d(1,32,3,1));
        self.layers.push(nn.Conv2d(32,64,3,1));
        self.layers.push(nn.Dropout(0.25));
        self.layers.push(nn.Dropout(0.5));
        self.layers.push(nn.FCL(9216, 128));
        self.layers.push(nn.FCL(128,10));

        Self { layers }
    }

    /// Define a forward pass 
    fn forward(x: &Matrix) {
        x = layers[0](x)
        x = ReLU(x);
        x = layers[1](x)
        x = ReLU(x);
        x = layers[2](x)
        x = ReLU(x);
        let output = log_softmax(x);
        return output;
    }
}

fn train(model: &Model, 
        train_dataloader: &DataLoader, 
        optimizer: &Optimizer, 
        epoch: usize) {
    model.train();

    for (batch_idx, (data, target)) in train_dataloader.iter().enumerate() {
        optimizer.zero_grad();
        let output = model(data);
        let loss = BCELoss(output, target);
        loss.backward();
        optimizer.step();
    }
}

fn test(model: &Model, 
        test_dataloader: &DataLoader, 
        optimizer: &Optimizer, 
        epoch: usize) {
    model.eval();

    let mut test_loss = 0.0;
    let mut correct = 0.0;

    optimizer.no_grad();

    for (batch_idx, (data, target)) in train_dataloader.iter().enumerate() {
        let output = model(data);
        test_loss += BCELoss(output, target);

        let pred = output.argmax(Dimension::Row);
        correct += pred.eq(target.view_as(pred)).sum();
    }
    test_loss /= test_dataloader.count();
}

// TODO: Add Clap

fn main() {
    let d1 = download_dataset(url, "../data", true, true, transform);
    let d2 = download_dataset(url, "../data", false, false, transform);

    let train_dl = DataLoader::new(&d1);
    let test_dl = DataLoader::new(&d2);

    let model = Net::init();
    let optimizer = SGD::init(0.001, 0.8);

    for epoch in 1..EPOCHS+1 {
        train(&model, &train_dl, &optimizer, epoch);
        test(&model, &test_dl, &optimizer, epoch);
    }

    if args.SAVE_MODEL {
        model.save_model("mnist_test.kaffe_pt");        
    }
}
```

## GPU Support 

As per right now, support for training on GPU is not happening anytime soon.
Although.. transpilation IS a thing you know.

For more examples, please see [examples](./examples/)

## Documentation
Full API documentation can be found [here](https://docs.rs/kaffe/latest/kaffe/).

## Features 
- [X] Blazingly fast
- [X] Common tensor operations exists under tensor  module
- [X] Optimizers 
- [X] Support for both f32 and f64 
- [X] ReLU, GeLU, PReLU, Sigmoid 
- [ ] Basic neural net features

