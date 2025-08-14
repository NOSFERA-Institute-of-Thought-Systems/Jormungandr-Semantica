import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import diags
from pygsp.graphs import Graph
from pygsp.filters import Heat
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import pygsp.graphs as graphs


def safe_compute_normalized_laplacian(W: sp.spmatrix) -> sp.spmatrix:
    """
    Computes a normalized Laplacian, robust to isolated nodes.
    L = I - D^(-1/2) * W * D^(-1/2)
    """
    # --- DEBUG PROBE ---
    print("    [DEBUG] Entering safe_compute_normalized_laplacian...")
    
    # Calculate degree matrix D
    d = np.array(W.sum(axis=1)).flatten()
    
    # --- DEBUG PROBE ---
    print(f"    [DEBUG] Degree vector 'd' min: {np.min(d)}, max: {np.max(d)}, mean: {np.mean(d)}")
    if np.any(d < 0):
        print("    [CRITICAL DEBUG] NEGATIVE DEGREES FOUND IN 'd'. This should be impossible.")
    
    # Check for zero-degree nodes, which cause the division error
    zero_degree_nodes = np.where(d == 0)[0]
    if len(zero_degree_nodes) > 0:
        print(f"    [DEBUG] Found {len(zero_degree_nodes)} isolated nodes (degree == 0).")

    # Create D^(-1/2), carefully handling zeros
    d_inv_sqrt = np.zeros_like(d, dtype=np.float64)
    non_zero_mask = d > 1e-12 # Use a tolerance for floating point comparison
    
    # --- DEBUG PROBE ---
    print(f"    [DEBUG] {np.sum(non_zero_mask)} nodes have non-zero degree.")

    d_inv_sqrt[non_zero_mask] = np.power(d[non_zero_mask], -0.5)
    
    # --- DEBUG PROBE ---
    if np.any(np.isinf(d_inv_sqrt)) or np.any(np.isnan(d_inv_sqrt)):
        print("    [CRITICAL DEBUG] 'inf' or 'NaN' found in d_inv_sqrt. This indicates a problem.")

    D_inv_sqrt = diags(d_inv_sqrt)
    
    # Compute the normalized Laplacian
    I = sp.identity(W.shape[0], dtype=np.float64)
    L_norm = I - D_inv_sqrt @ W @ D_inv_sqrt
    
    print("    [DEBUG] Safely computed normalized Laplacian.")
    return L_norm

def compute_heat_wavelets(graph: Graph, signal: np.ndarray, scales: list[float] | None = None, n_eigenvectors: int | None = None) -> np.ndarray:
    if signal.shape[0] != graph.N:
        raise ValueError("Signal and graph must have the same number of nodes.")
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    if scales is None:
        scales = [5, 10, 25, 50]
    
    if not hasattr(graph, '_e') or getattr(graph, '_e', None) is None:
        print("  -> Manually computing a safe normalized Laplacian...")
        L_safe = safe_compute_normalized_laplacian(graph.W)
        graph.L = L_safe
        print(f"  -> Laplacian computed. Dtype: {graph.L.dtype}, Shape: {graph.L.shape}")

        if n_eigenvectors is not None and n_eigenvectors < graph.N:
            print(f"  -> Configuring graph to compute first {n_eigenvectors} eigenvectors (using sparse 'eigs' solver)...")
            graph.eigensolver = 'eigs'
            graph.eigensolver_params = {'k': n_eigenvectors}
        else:
            print("  -> Configuring graph to compute full Fourier basis (using dense 'eigh' solver)...")
            graph.eigensolver = 'eigh'
            graph.eigensolver_params = {}

        print("  -> Computing Fourier basis...")
        graph.compute_fourier_basis()

    heat_filter = Heat(graph, tau=scales)

    all_coeffs = []
    for i in range(signal.shape[1]):
        coeffs_one_feature = heat_filter.filter(signal[:, i], method='exact')
        all_coeffs.append(coeffs_one_feature)

    return np.stack(all_coeffs, axis=1)

def compute_acmw_wavelets(graph: Graph, signal: np.ndarray, scales: list[float] | None = None, alpha: float = 4.0, beta: float = 10.0, n_eigenvectors: int | None = None) -> np.ndarray:
    print("Step (ACMW): Converting to NetworkX and computing Ricci curvature...")
    W_f64 = graph.W.astype(np.float64)
    nx_graph = nx.from_scipy_sparse_array(W_f64)
    orc = OllivierRicci(nx_graph, alpha=0.0, verbose="INFO")
    orc.compute_ricci_curvature()
    
    print("Step (ACMW): Creating curvature-modulated anisotropic weight matrix...")
    W_prime = W_f64.copy().tolil()
    
    # --- THE ULTIMATE FIX ---
    # The original h(kappa) function could produce negative values, leading to
    # negative edge weights and negative degrees, which is mathematically invalid.
    # We now clamp the modulation_factor at a small positive epsilon to ensure
    # all edge weights remain non-negative.
    h = lambda kappa: alpha * (1 / (1 + np.exp(-beta * kappa)) - 0.5) + 1.0
    epsilon = 1e-9 # A small positive floor
    
    for u, v, data in orc.G.edges(data=True):
        kappa = data.get('ricciCurvature', 0.0)
        modulation_factor = h(kappa)
        
        # Clamp the factor to be non-negative.
        safe_modulation_factor = max(modulation_factor, epsilon)
        
        W_prime[u, v] *= safe_modulation_factor
        W_prime[v, u] *= safe_modulation_factor
    
    anisotropic_graph = graphs.Graph(W=W_prime.tocsr())
    print(f"Anisotropic graph constructed. Adjacency matrix (W) dtype: {anisotropic_graph.W.dtype}")

    # This will now be called with a graph that is guaranteed to have non-negative weights.
    return compute_heat_wavelets(anisotropic_graph, signal, scales=scales, n_eigenvectors=n_eigenvectors)