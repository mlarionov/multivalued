# Learning Multi-valued Functions using RNNs

This project implements a sequential solution to the **multi-valued function problem** (also known as the inverse problem) as discussed in Christopher Bishop's *Pattern Recognition and Machine Learning* (PRML), Chapter 5.6.

## The Problem

In many real-world "inverse" problems, a single input $x$ can map to multiple valid output values $y$. Standard neural networks, which minimize Mean Squared Error (MSE), tend to predict the **average** of all valid $y$ values, resulting in poor performance (the "mean line" artifact).

While Bishop proposes Mixture Density Networks (MDNs) as a solution, this project explores an alternative approach: **Recurrent Neural Networks (RNNs)**.

## The Solution: Sequential Generation

Instead of predicting all values at once, we treat the set of output values as a **sequence** sorted in ascending order. For a given input $x$, the RNN:
1.  Is initialized with the context of $x$.
2.  Generates the first value $y_1$ and a probability that the sequence has ended.
3.  If the sequence continues, it uses $y_1$ and the context of $x$ to generate $y_2$, and so on.

This approach naturally handles varying numbers of branches (e.g., 1 branch vs. 3 branches) for different inputs.

## Architecture

- **Contextual Initialization**: The GRU's initial hidden state is generated via a non-linear projection of the scalar input $x$.
- **GRU Core**: A Gated Recurrent Unit processes the sequence.
- **Deep Heads**: Separate deep MLP heads predict the scalar value $y_t$ and the "stop" logit $p_{stop,t}$.
- **Hardware Acceleration**: Automatic support for CUDA (NVIDIA) and MPS (Apple Silicon).
- **Vectorized Inference**: Parallel sampling of multiple $x$ values simultaneously using active masking.

## Project Structure

- `model.py`: The `MultivaluedRNN` architecture.
- `data_utils.py`: Synthetic data generation for $x = y + 0.3 \sin(2\pi y)$.
- `train.py`: Vectorized training script with input jittering for robustness.
- `predict.py`: Vectorized inference and visualization.
- `config.py`: Centralized hyperparameters and device selection.

## Getting Started

### Prerequisites

This project uses `uv` for lightning-fast dependency management.

```bash
# Install dependencies
uv sync
```

### Usage

1.  **Train the model**:
    ```bash
    uv run train.py
    ```
2.  **Generate predictions and plot**:
    ```bash
    uv run predict.py
    ```

The results will be saved to `prediction_results.png`, showing the predicted branches (in red) against the ground truth manifold (in gray).

## Results

The model successfully identifies:
- Single-valued regions (generating 1 value).
- Multi-valued regions (generating 3 values in ascending order).
- Smooth transitions near vertical tangents.
