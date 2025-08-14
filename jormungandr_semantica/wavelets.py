import numpy as np
import networkx as nx
from pygsp.graphs import Graph
from pygsp.filters import Heat
from GraphRicciCurvature.OllivierRicci import OllivierRicci

def compute_heat_wavelets(
    graph: Graph,
    signal: np.ndarray,
    scales: list[float] | None = None,
) -> np.ndarray:
    """
    Computes the heat wavelet transform of a signal on a graph.
    This is the robust version that handles multi-feature signals correctly.
    """
    if signal.shape[0] != graph.N:
        raise ValueError("Signal and graph must have the same number of nodes.")
    if signal.ndim == 1:  # Ensure signal is always 2D for consistency
        signal = signal.reshape(-1, 1)

    if scales is None:
        scales = [5, 10, 25, 50]

    # Ensure Fourier basis is computed for efficiency and to avoid warnings
    if not hasattr(graph, 'e'):
        print("Computing Fourier basis for wavelet transform...")
        graph.compute_fourier_basis()

    heat_filter = Heat(graph, tau=scales)
    
    # --- ROBUST FILTERING ---
    # Filter each feature column individually and stack the results.
    all_coeffs = []
    for i in range(signal.shape[1]):
        feature_signal = signal[:, i]
        # Filtering a 1D signal reliably returns a (n_nodes, n_scales) array
        coeffs_per_feature = heat_filter.filter(feature_signal)
        all_coeffs.append(coeffs_per_feature)

    # Stack along a new axis to get the final (n_nodes, n_features, n_scales) shape
    return np.stack(all_coeffs, axis=1)


def compute_acmw_wavelets(
    graph: Graph,
    signal: np.ndarray,
    scales: list[float] | None = None,
    alpha: float = 4.0,
    beta: float = 10.0
) -> np.ndarray:
    """
    Computes the Anisotropic Curvature-Modulated Wavelet (ACMW) transform.
    """
    if signal.shape[0] != graph.N:
        raise ValueError("Signal and graph must have the same number of nodes.")
    
    if scales is None:
        scales = [5, 10, 25, 50]

    print("Step (ACMW): Converting to NetworkX and computing Ricci curvature...")
    nx_graph = nx.from_scipy_sparse_array(graph.W)
    orc = OllivierRicci(nx_graph, alpha=0.0, verbose="INFO")
    orc.compute_ricci_curvature()
    
    print("Step (ACMW): Creating curvature-modulated anisotropic weight matrix...")
    W_prime = graph.W.copy().tolil()

    def h(kappa):
        return alpha * (1 / (1 + np.exp(-beta * kappa)) - 0.5) + 1.0

    for u, v, data in orc.G.edges(data=True):
        kappa = data['ricciCurvature']
        modulation_factor = h(kappa)
        W_prime[u, v] *= modulation_factor
        W_prime[v, u] *= modulation_factor

    print("Step (ACMW): Initializing new anisotropic PyGSP graph...")
    anisotropic_graph = Graph(W=W_prime.tocsr())

    print("Step (ACMW): Computing heat wavelets on the anisotropic graph...")
    # This now calls the robust, corrected version of compute_heat_wavelets
    acmw_coefficients = compute_heat_wavelets(
        anisotropic_graph, 
        signal, 
        scales=scales
    )

    return acmw_coefficients