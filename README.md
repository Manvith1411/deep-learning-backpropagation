# Deep Learning Backpropagation (Python)

This repository contains simple, beginner-friendly Python examples that demonstrate **backpropagation** concepts in different neural network settings:

- **ANN backpropagation** (single-neuron example with sigmoid + MSE)
- **CNN backpropagation** (toy convolution + kernel update)
- **RNN backpropagation** (toy recurrent forward pass + weight update)

## Repository Structure

- `backpropagation-codes/`
  - `ann_backpropagation.py` — basic ANN-style backpropagation demo
  - `cnn_backpropagation.py` — basic CNN convolution + kernel update demo
  - `rnn_backpropagation.py` — basic RNN forward pass + weight update demo
- `Difference between Backpropagation.pptx` — presentation material
- `ChatGPT Image Mar 17, 2026, ... .png` — images/figures related to the topic

## Requirements

- Python 3.x
- NumPy

Install NumPy:

```bash
pip install numpy
```

## How to Run

Run any script from the repository root:

### ANN Backpropagation
```bash
python backpropagation-codes/ann_backpropagation.py
```

### CNN Backpropagation
```bash
python backpropagation-codes/cnn_backpropagation.py
```

### RNN Backpropagation
```bash
python backpropagation-codes/rnn_backpropagation.py
```

## What each script demonstrates (high level)

### 1) ANN (`ann_backpropagation.py`)
- Forward pass: `z = X·W + b`, `y_pred = sigmoid(z)`
- Loss: Mean Squared Error (MSE)
- Backprop: computes gradient and updates the weight

### 2) CNN (`cnn_backpropagation.py`)
- Computes a small **feature map** by convolving a 3×3 “image” with a 2×2 kernel
- Uses a **simulated gradient** to show a basic kernel update step

### 3) RNN (`rnn_backpropagation.py`)
- Runs a forward pass across a short input sequence using `tanh`
- Uses a **simulated gradient** to show how weights can be updated

## Notes

These scripts are intended for learning/demonstration purposes (toy examples), not for training real models or matching full deep-learning framework behavior.

## License

Add a license if you plan to share/redistribute widely (e.g., MIT).
