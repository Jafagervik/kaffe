# Kaffe - A Pytorch inspired library written in rust 

## Why Kaffe?

Because sometimes you wanna make cool and fast stuff in rust :)

## Documentation
Full API documentation can be found [here](https://docs.rs/kaffe/latest/kaffe/).

## Example Usage

### Basic example 

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

For more examples, please see [examples](./examples/)

## Features 
- [X] Blazingly fast
- [X] Common matrix operations exists under matrix module
- [ ] Basic neural net features

### Version Control 

- Breaking changes: X.0.0
- New features: X.X+1.0
- Adjustments/Fixes: X.X.X+1
