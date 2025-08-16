import torch
import numpy as np
import networkx as nx
from torch_geometric.utils.convert import from_scipy_sparse_matrix

print("--- Verification Script for Scalable Differentiable Operator ---")

# --- 1. Define the Operator ---
class DifferentiableChebyshevOperator(torch.nn.Module):
    def __init__(self, chebyshev_order=30):
        super().__init__()
        self.chebyshev_order = chebyshev_order

    def forward(self, L_sparse, X, t_scale=5.0):
        """
        Differentiably approximates e^(-tL)X using Chebyshev polynomials.
        This version uses a robust numerical method to compute coefficients.
        """
        N = L_sparse.shape[0]
        # For normalized Laplacian, lambda_max is approximately 2.
        lambda_max = 2.0
        
        # Rescale Laplacian to have spectrum in [-1, 1]
        L_rescaled = (2.0 / lambda_max) * L_sparse - torch.sparse.spdiags(
            torch.ones(N), torch.tensor([0]), (N, N)
        )

        # --- THE CORRECT & ROBUST COEFFICIENT CALCULATION ---
        # Compute coefficients c_k for our target function on the rescaled interval
        M = self.chebyshev_order
        # Sample points in the cosine domain
        j = np.arange(M, dtype=np.float64)
        x = np.cos(np.pi * (j + 0.5) / M) # Chebyshev nodes
        
        # Evaluate the target function f(lambda) at these nodes
        # lambda = (lambda_max / 2) * (x + 1)
        lambdas = (lambda_max / 2.0) * (x + 1.0)
        f_vals = np.exp(-t_scale * lambdas)
        
        # Compute coefficients via Discrete Cosine Transform
        c = np.fft.fft(f_vals) / M
        c = np.real(c) # Should be real anyway
        coeffs = torch.from_numpy(c).float()

        # --- Three-term recurrence for T_k(L_rescaled)X ---
        T0_X = X
        T1_X = torch.sparse.mm(L_rescaled, X)
        
        # The sum starts with the first two terms. Need to handle c0 carefully.
        # DCT gives coefficients for a slightly different basis.
        # Let's use a simpler, direct summation for clarity.
        
        # Let's re-implement the coefficient calculation exactly as per textbooks
        coeffs = torch.zeros(M)
        for k in range(M):
            T_k_x = np.cos(k * np.arccos(x))
            coeffs[k] = (2.0 / M) * np.sum(f_vals * T_k_x)
        coeffs[0] /= 2.0
        
        # Let's run the recurrence
        current_sum = (coeffs[0] * T0_X) + (coeffs[1] * T1_X)
        
        for k in range(2, self.chebyshev_order):
            Tk_X = 2 * torch.sparse.mm(L_rescaled, T1_X) - T0_X
            current_sum = current_sum + (coeffs[k] * Tk_X)
            T0_X, T1_X = T1_X, Tk_X
            
        return current_sum

# --- 2. Create Toy Data ---
print("\nStep 1: Creating toy graph and data...")
N = 50
D = 4
t_scale = 5.0
nx_g = nx.path_graph(N)
L_np = nx.normalized_laplacian_matrix(nx_g)

# Convert to PyTorch sparse tensor
edge_index, edge_weight = from_scipy_sparse_matrix(L_np)
L_torch_sparse = torch.sparse_coo_tensor(edge_index, edge_weight.float(), (N, N))
X_torch = torch.randn(N, D, requires_grad=True)

# --- 3. Compute Ground Truth (Exact) ---
print("Step 2: Computing ground truth via exact eigendecomposition...")
L_dense = L_torch_sparse.to_dense()
eigenvalues, eigenvectors = torch.linalg.eigh(L_dense)
exp_lambda_t = torch.exp(-t_scale * eigenvalues)
ground_truth = eigenvectors @ torch.diag(exp_lambda_t) @ eigenvectors.T @ X_torch

# --- 4. Compute Approximation ---
print("Step 3: Computing approximation via differentiable operator...")
model = DifferentiableChebyshevOperator(chebyshev_order=40) # Increase order for more accuracy
approximation = model(L_torch_sparse, X_torch, t_scale=t_scale)

# --- 5. Verify and Test ---
print("\n--- VERIFICATION ---")

error = torch.norm(ground_truth - approximation) / torch.norm(ground_truth)
print(f"Approximation Error (Relative L2 Norm): {error.item():.6f}")
assert error < 0.1, "Approximation error is too high!"

print("Testing backward pass (differentiability)...")
try:
    approximation.sum().backward()
    assert X_torch.grad is not None, "Gradient was not computed!"
    print("Backward pass successful. Gradients were computed.")
except Exception as e:
    print(f"Backward pass FAILED: {e}")
    assert False, "Differentiability test failed."

print("\n********************************************************")
print("**** SUCCESS: Scalable Operator is Correct & Diff. ****")
print("********************************************************")