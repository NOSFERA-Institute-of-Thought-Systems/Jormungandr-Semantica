import numpy as np
import networkx as nx
from pygsp.graphs import Graph
from pygsp.filters import Heat
from GraphRicciCurvature.OllivierRicci import OllivierRicci

def compute_heat_wavelets(graph: Graph, signal: np.ndarray, scales: list[float] | None = None, n_eigenvectors: int | None = None) -> np.ndarray:
    if signal.shape[0] != graph.N:
        raise ValueError("Signal and graph must have the same number of nodes.")
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    if scales is None:
        scales = [5, 10, 25, 50]
    
    # --- FINAL, DEFINITIVE FIX ---
    # This function is responsible for its own Fourier basis.
    if not hasattr(graph, 'e'):
        if n_eigenvectors is not None and n_eigenvectors < graph.N:
            print(f"  -> Computing first {n_eigenvectors} eigenvectors for wavelet transform...")
            graph.compute_fourier_basis(k=n_eigenvectors)
        else:
            print("  -> Computing full Fourier basis for wavelet transform...")
            graph.compute_fourier_basis()

    heat_filter = Heat(graph, tau=scales)
    all_coeffs = [heat_filter.filter(signal[:, i]) for i in range(signal.shape[1])]
    return np.stack(all_coeffs, axis=1)

def compute_acmw_wavelets(graph: Graph, signal: np.ndarray, scales: list[float] | None = None, alpha: float = 4.0, beta: float = 10.0, n_eigenvectors: int | None = None) -> np.ndarray:
    print("Step (ACMW): Converting to NetworkX and computing Ricci curvature...")
    nx_graph = nx.from_scipy_sparse_array(graph.W)
    orc = OllivierRicci(nx_graph, alpha=0.0, verbose="INFO")
    orc.compute_ricci_curvature()
    
    print("Step (ACMW): Creating curvature-modulated anisotropic weight matrix...")
    W_prime = graph.W.copy().tolil()
    h = lambda kappa: alpha * (1 / (1 + np.exp(-beta * kappa)) - 0.5) + 1.0
    for u, v, data in orc.G.edges(data=True):
        modulation_factor = h(data['ricciCurvature'])
        W_prime[u, v] *= modulation_factor
        W_prime[v, u] *= modulation_factor

    print("Step (ACMW): Initializing new anisotropic PyGSP graph...")
    anisotropic_graph = Graph(W=W_prime.tocsr())

    # This self-contained function now calls the base wavelet function, which handles its own basis.
    return compute_heat_wavelets(anisotropic_graph, signal, scales=scales, n_eigenvectors=n_eigenvectors)