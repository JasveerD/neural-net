# neural-net

A neural network library built from scratch in C++20. No PyTorch, no Eigen, no ML frameworks.
Hand-derived backprop, He initialization, and a clean matrix library with full Rule of 5 compliance.
Trains to **97.91% accuracy** on MNIST in 20 epochs.

## Highlights

- `Matrix<T>` — templated matrix class with heap-allocated row-major storage, full Rule of 5, operator overloading, and elementwise/scalar arithmetic
- Backpropagation implemented from scratch — gradients derived analytically, not via autograd
- He initialization — proper weight scaling for ReLU networks, eliminates vanishing gradients
- Modular architecture — layers, activations, loss, optimizer, and dataloader are all independent
- SGD with momentum optimizer
- MNIST IDX binary format parser with big-endian byte swapping and one-hot encoding
- Binary serialization — save and resume training from checkpoints
- Apple Silicon ready — Accelerate framework linked, cblas_sgemm swap in progress

## Results

| Metric | Value |
|--------|-------|
| Test accuracy | 97.91% |
| Epochs | 20 |
| Batch size | 32 |
| Learning rate | 0.01 |
| Momentum | 0.9 |
| Architecture | 784 → 128 → 64 → 64 → 10 |

Loss curve:

| Epoch | Loss |
|-------|------|
| 1 | 0.2757 |
| 5 | 0.0475 |
| 10 | 0.0209 |
| 15 | 0.0170 |
| 20 | 0.0079 |

## Project Structure

```
include/
  matrix.hpp         — Matrix<T>: storage, arithmetic, transpose, apply, factories
  activations.hpp    — ReLU, Sigmoid, Tanh, Softmax (elementwise + matrix-level)
  loss.hpp           — Cross-entropy loss and derivative
  layer.hpp          — DenseLayer: forward pass, backprop, gradient storage
  network.hpp        — Network: owns layers, chains forward/backward
  optimizer.hpp      — SGD with momentum
  dataloader.hpp     — MNIST IDX binary parser, mini-batch iterator
  serializer.hpp     — Binary save/load for network weights
src/                 — empty (template implementations live in headers)
tools/
  train.cpp          — Training script: build network, train, evaluate, save
benchmarks/
  bench_matmul.cpp   — Naive matmul vs Accelerate timing harness
tests/
  test_matrix.cpp    — Catch2 unit tests for Matrix<T>
  test_gradients.cpp — Numerical gradient checking for backprop correctness
  test_mnist.cpp     — End-to-end training test
metal/
  matmul.metal       — Metal compute kernel (stretch goal)
data/                — MNIST files (download separately)
```

## Building

Requires macOS with Xcode Command Line Tools (Accelerate framework included automatically).

```zsh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)
```

## Training

Download MNIST first:

```zsh
cd data
curl -L -O https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
curl -L -O https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
curl -L -O https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
curl -L -O https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
cd ..
```

Then train:

```zsh
./build/train
```

Output:

```
epoch 1/20  loss: 0.275706
epoch 2/20  loss: 0.121338
...
epoch 20/20  loss: 0.00785543
test accuracy: 97.91%
model saved to model.bin
```

## Testing

```zsh
cd build && ctest --output-on-failure
```

## Resuming Training

The network is saved to `model.bin` after training. To resume, load weights into a network with the same architecture:

```cpp
load(net, "model.bin");
```

## Roadmap

- [ ] cblas_sgemm via Accelerate — hardware-accelerated matmul on Apple AMX coprocessor
- [ ] Benchmark naive matmul vs Accelerate across matrix sizes and batch sizes
- [ ] Numerical gradient checking to verify backprop correctness
- [ ] Adam optimizer
- [ ] Dropout regularization
- [ ] Metal compute kernel for GPU matmul
- [ ] Convolutional layer → CNN → 99%+ accuracy
