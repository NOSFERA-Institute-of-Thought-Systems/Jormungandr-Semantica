# File: tests/test_dagn_operator.py

import numpy as np
import networkx as nx
import pytest
import torch
from scipy.sparse import csgraph

# We need to import the internal operator class for direct testing
from aglt.pipeline.steps import DifferentiableChebyshevOperator


def test_chebyshev_approximation_accuracy():
    """
    Tests if the Chebyshev approximation of the heat kernel is reasonably
    close to the exact solution computed via dense eigendecomposition.
    """
    # 1. Create toy data
    N, D, t_scale = 50, 4, 5.0
    nx_g = nx.path_graph(N)
    L_np = csgraph.laplacian(nx.to_scipy_sparse_array(nx_g), normed=True)
    L_dense = torch.from_numpy(L_np.toarray()).float()
    X_torch = torch.randn(N, D)

    # 2. Compute Ground Truth (Exact)
    eigenvalues, eigenvectors = torch.linalg.eigh(L_dense)
    exp_lambda_t = torch.exp(-t_scale * eigenvalues)
    ground_truth = eigenvectors @ torch.diag(exp_lambda_t) @ eigenvectors.T @ X_torch

    # 3. Compute Approximation
    # We need a sparse Laplacian for the operator
    coo = L_np.tocoo()
    indices = torch.from_numpy(np.vstack((coo.row, coo.col))).long()
    values = torch.from_numpy(coo.data).float()
    L_sparse = torch.sparse_coo_tensor(indices, values, L_dense.shape)

    # We need a dummy DifferentiableChebyshevOperator from the pipeline
    # The initial_center isn't used for the forward pass test, so we can use a dummy value
    model = DifferentiableChebyshevOperator(initial_center=0.0, chebyshev_order=40)

    # Extract the actual operator logic for testing
    # In a future refactor, this might be a standalone function
    lambda_max = 2.0
    I_indices = torch.arange(N).view(1, -1).repeat(2, 1)
    identity_matrix = torch.sparse_coo_tensor(I_indices, torch.ones(N), L_sparse.shape)
    L_rescaled = (2.0 / lambda_max) * L_sparse - identity_matrix
    coeffs = model.compute_chebyshev_coeffs(t_scale, lambda_max, device="cpu")

    approximation = model.chebyshev_op(L_rescaled, X_torch, coeffs)

    # 4. Assert correctness
    relative_error = torch.norm(ground_truth - approximation) / torch.norm(ground_truth)
    assert relative_error < 0.1, "Approximation error is too high"


def test_chebyshev_operator_differentiability():
    """
    Tests that gradients can flow through the Chebyshev operator.
    """
    # 1. Create toy data
    N, D = 30, 3
    L_np = csgraph.laplacian(
        nx.to_scipy_sparse_array(nx.grid_graph([5, 6])), normed=True
    )
    coo = L_np.tocoo()
    indices = torch.from_numpy(np.vstack((coo.row, coo.col))).long()
    values = torch.from_numpy(coo.data).float()
    L_sparse = torch.sparse_coo_tensor(indices, values, (N, N))

    # Input tensor requires gradients
    X_torch = torch.randn(N, D, requires_grad=True)

    # 2. Run forward and backward pass
    model = DifferentiableChebyshevOperator(initial_center=0.0)

    # We don't need the full DAGN builder, just the operator part
    lambda_max = 2.0
    I_indices = torch.arange(N).view(1, -1).repeat(2, 1)
    identity_matrix = torch.sparse_coo_tensor(I_indices, torch.ones(N), L_sparse.shape)
    L_rescaled = (2.0 / lambda_max) * L_sparse - identity_matrix
    coeffs = model.compute_chebyshev_coeffs(5.0, lambda_max, device="cpu")

    approximation = model.chebyshev_op(L_rescaled, X_torch, coeffs)

    try:
        approximation.sum().backward()
    except Exception as e:
        pytest.fail(f"Backward pass failed with an exception: {e}")

    # 3. Assert that gradients were computed
    assert X_torch.grad is not None, "Gradient was not computed for the input tensor"
    assert torch.all(
        torch.isfinite(X_torch.grad)
    ), "Gradients contain non-finite values"
