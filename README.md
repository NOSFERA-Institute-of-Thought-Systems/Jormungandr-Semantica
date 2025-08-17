# Jörmungandr-Semantica

[![Build and Test CI](https://github.com/your-username/Jormungandr-Semantica/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/Jormungandr-Semantica/actions/workflows/ci.yml)

A framework for geometric deep learning in semantic analysis, built on first principles.

---

### Guiding Principle

> Every empirical result must be supported by a theoretical claim, and every theoretical claim must be tested by a rigorous empirical or synthetic experiment. No compromise on reproducibility, rigor, or ambition.

This repository contains the full implementation for the Jörmungandr-Semantica project, a research initiative aimed at developing novel methods for document clustering and semantic representation by leveraging the geometry of data manifolds.

### Core Features

- **High-Performance C++ Backend:** Core computations, like k-NN graph construction, are implemented in C++ using **Faiss** and exposed to Python via Pybind11 for maximum speed.
- **Graph Wavelet Analysis:** Utilizes the robust **PyGSP** library to perform graph signal processing, analyzing data through the lens of heat wavelets.
- **End-to-End Reproducibility:** Every experiment is tracked using **Weights & Biases**, capturing code versions, hyperparameters, and results automatically.
- **Continuous Integration:** A GitHub Actions pipeline automatically builds the C++ backend and runs a suite of tests on every push, guaranteeing code stability.

### Installation

This project is managed with Conda. Ensure you have a Conda distribution (like Miniforge or Anaconda) installed.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/Jormungandr-Semantica.git
    cd Jormungandr-Semantica
    ```

2.  **Create and activate the Conda environment:**

    ```bash
    conda env create -f environment.yml
    conda activate jormungandr
    ```

3.  **Perform an editable installation:** This command builds the C++ backend and installs the Python package. The `-e` flag allows you to edit the Python source code and have the changes immediately reflected without reinstalling.
    ```bash
    pip install -e .
    ```

### Quick Start

Once installed, you can use the core components of the library.

```python
import numpy as np
import aglt
import pygsp.graphs as graphs

# 1. Create sample data
data = np.random.rand(100, 16).astype('float32')

# 2. Build a k-NN graph with the high-performance C++ backend
neighbors, distances = aglt.build_faiss_knn_graph(data, k=10)
print(f"k-NN graph built. Neighbors shape: {neighbors.shape}")

# 3. Use the graph with PyGSP and compute wavelets
G = graphs.Graph(W=distances) # A simplified example
G.compute_fourier_basis()
signal = data[:, 0] # Use the first feature as a signal
coeffs = aglt.compute_heat_wavelets(G, signal)
print(f"Wavelet coefficients computed. Shape: {coeffs.shape}")
```
