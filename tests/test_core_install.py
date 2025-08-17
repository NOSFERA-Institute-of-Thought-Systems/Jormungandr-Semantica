# File: tests/test_core_install.py

import numpy as np
import aglt


def test_faiss_knn_graph_backend():
    """
    Verifies that the core C++ backend (build_faiss_knn_graph) is installed
    correctly and produces outputs with the expected shape and type.
    This is the primary installation verification test.
    """
    # 1. Define test parameters
    n_points, n_dims, k = 100, 16, 10

    # 2. Create synthetic data
    data = np.random.rand(n_points, n_dims).astype("float32")

    # 3. Call the function under test
    neighbors, distances = aglt.build_faiss_knn_graph(data, k)

    # 4. Assert correctness
    assert neighbors.shape == (n_points, k), "Neighbors matrix has incorrect shape"
    assert distances.shape == (n_points, k), "Distances matrix has incorrect shape"
    assert neighbors.dtype == np.int64, "Neighbors should be integer indices"
    assert distances.dtype == np.float32, "Distances should be float32"
