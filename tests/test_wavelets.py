# File: tests/test_wavelets.py

import numpy as np
import pygsp.graphs as graphs
import pytest
import aglt  # Use the new library name


def test_compute_heat_wavelets_shape_and_type():
    """
    Tests that the heat wavelet function returns coefficients with the
    expected shape, dtype, and that it contains no NaN/inf values.
    """
    # 1. Define graph and signal parameters for the test
    n_nodes, n_features = 50, 4
    scales_to_test = [10, 20, 30, 40, 50]
    n_scales = len(scales_to_test)

    # 2. Create a toy graph and signal
    G = graphs.Ring(N=n_nodes)
    G.compute_fourier_basis()
    signal = np.random.rand(n_nodes, n_features)

    # 3. Call the function under test
    coeffs = aglt.compute_heat_wavelets(G, signal, scales=scales_to_test)

    # 4. Assert correctness (the core of the test)
    expected_shape = (n_nodes, n_features, n_scales)

    # Check if the output shape is exactly what we expect
    assert coeffs.shape == expected_shape, "Output shape is incorrect"

    # Check if the data type is a numpy float
    assert (
        coeffs.dtype == np.float64 or coeffs.dtype == np.float32
    ), "Output dtype is not float"

    # Check for numerical stability (no non-finite values)
    assert np.all(np.isfinite(coeffs)), "Output contains non-finite values (NaN or inf)"


def test_compute_heat_wavelets_with_single_feature_signal():
    """
    Tests that the function handles a 1D (single-feature) signal correctly.
    """
    n_nodes = 30
    scales_to_test = [10, 20]
    n_scales = len(scales_to_test)

    G = graphs.Path(N=n_nodes)
    G.compute_fourier_basis()

    # Create a 1D signal
    signal_1d = np.random.rand(n_nodes)

    coeffs = aglt.compute_heat_wavelets(G, signal_1d, scales=scales_to_test)

    # The function should internally promote the 1D signal to (N, 1, S)
    expected_shape = (n_nodes, 1, n_scales)
    assert coeffs.shape == expected_shape


# This allows running the test file directly, though `pytest` is the standard way
if __name__ == "__main__":
    pytest.main()
