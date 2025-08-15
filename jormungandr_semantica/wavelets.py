import numpy as np
import networkx as nx
from pygsp.graphs import Graph
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import scipy.sparse as sp

# --- NEW HELPER FUNCTION ---
def safe_compute_normalized_laplacian(W: sp.spmatrix) -> sp.spmatrix:
    """
    Computes a numerically stable normalized Laplacian.
    Handles nodes with zero or very small degrees to prevent division by zero.
    """
    print("    [DEBUG] Entering safe_compute_normalized_laplacian...")
    # Calculate degree, ensuring it's a column vector
    d = np.array(W.sum(axis=1)).flatten()
    print(f"    [DEBUG] Degree vector 'd' min: {d.min()}, max: {d.max()}, mean: {d.mean()}")
    
    # Identify nodes with non-zero degree to avoid division by zero
    non_zero_degree_indices = d > 1e-12
    print(f"    [DEBUG] {np.sum(non_zero_degree_indices)} nodes have non-zero degree.")
    
    # Create the inverse square root of the degree matrix, but only for non-zero degrees
    d_inv_sqrt = np.zeros_like(d)
    d_inv_sqrt[non_zero_degree_indices] = 1.0 / np.sqrt(d[non_zero_degree_indices])
    
    # Create the sparse diagonal matrix D^-1/2
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    
    # Compute I - D^-1/2 * W * D^-1/2
    I = sp.identity(W.shape[0], dtype=W.dtype)
    L_norm = I - D_inv_sqrt @ W @ D_inv_sqrt
    print("    [DEBUG] Safely computed normalized Laplacian.")
    return L_norm


def compute_heat_wavelets(graph: Graph, signal: np.ndarray, scales: list[float] | None = None, n_eigenvectors: int | None = None) -> np.ndarray:
    # ... This function is now correct and does not need changes ...
    if signal.shape[0] != graph.N:
        raise ValueError("Signal and graph must have the same number of nodes.")
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    if scales is None:
        scales = [5, 10, 25, 50]
    
    if not hasattr(graph, '_e') or getattr(graph, '_e', None) is None:
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

    # PyGSP's filter requires the Laplacian to be computed.
    if not hasattr(graph, 'L'):
        graph.compute_laplacian('normalized')

    from pygsp.filters import Heat # Local import to avoid circular dependency issues
    heat_filter = Heat(graph, tau=scales)

    all_coeffs = []
    for i in range(signal.shape[1]):
        coeffs_one_feature = heat_filter.filter(signal[:, i], method='exact')
        all_coeffs.append(coeffs_one_feature)

    return np.stack(all_coeffs, axis=1)


def compute_acmw_wavelets(
    graph: Graph, 
    signal: np.ndarray, 
    scales: list[float] | None = None, 
    n_eigenvectors: int | None = None,
    # --- NEW: Expose the key hyperparameters ---
    alpha: float = 4.0, 
    beta: float = 1.0, 
    epsilon: float = 0.01
) -> np.ndarray:
    
    print("Step (CPAL): Computing Ricci curvature for anisotropic modulation...")
    nx_graph = nx.from_scipy_sparse_array(graph.W)
    orc = OllivierRicci(nx_graph, alpha=0.0, verbose="INFO")
    orc.compute_ricci_curvature()
    
    # Get all curvature values to find the median for centering
    all_curvatures = [data['ricciCurvature'] for _, _, data in orc.G.edges(data=True)]
    kappa_0 = np.median(all_curvatures)
    print(f"Step (CPAL): Centering curvature around median value kappa_0 = {kappa_0:.4f}")

    print("Step (CPAL): Creating connectivity-preserving anisotropic weight matrix...")
    W_prime = graph.W.copy().tolil()
    
    # --- THE CPAL IMPLEMENTATION ---
    # This is the "soft" modulation function from your research plan.
    sigmoid = lambda z: 1 / (1 + np.exp(-z))
    h = lambda kappa: epsilon + (1 - epsilon) * sigmoid(alpha * (kappa - kappa_0))
    
    for u, v, data in orc.G.edges(data=True):
        modulation_factor = h(data['ricciCurvature'])
        # No need for max(0, ...) as h(kappa) is guaranteed to be > epsilon > 0
        W_prime[u, v] *= modulation_factor
        W_prime[v, u] *= modulation_factor
        
    W_prime = W_prime.tocsr()

    print("Step (CPAL): Initializing new anisotropic PyGSP graph...")
    anisotropic_graph = Graph(W=W_prime)
    
    print("  -> Manually computing a safe normalized Laplacian...")
    L_safe = safe_compute_normalized_laplacian(anisotropic_graph.W)
    anisotropic_graph.L = L_safe
    
    return compute_heat_wavelets(anisotropic_graph, signal, scales=scales, n_eigenvectors=n_eigenvectors)
