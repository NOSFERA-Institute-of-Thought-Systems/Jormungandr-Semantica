# jormungandr_semantica/wavelets.py

import numpy as np
import networkx as nx
from pygsp.graphs import Graph
import scipy.sparse as sp
from pygsp.filters import Heat
from GraphRicciCurvature.OllivierRicci import OllivierRicci

def compute_heat_wavelets(graph: Graph, signal: np.ndarray, scales: list[float] | None = None, n_eigenvectors: int | None = None) -> np.ndarray:
    """
    Computes heat wavelet coefficients for a signal on a graph.
    This function is self-contained and manages the Fourier basis computation.
    """
    if signal.shape[0] != graph.N:
        raise ValueError("Signal and graph must have the same number of nodes.")
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    if scales is None:
        scales = [5, 10, 25, 50]
    
    # --- THE FINAL, ABSOLUTELY CORRECT FIX ---
    # We must check the internal private attribute _e to see if the basis has been
    # computed, to avoid triggering the @property getter which automatically
    # runs the default (dense) computation.
    if not hasattr(graph, '_e') or getattr(graph, '_e', None) is None:
        if n_eigenvectors is not None and n_eigenvectors < graph.N:
            print(f"  -> Configuring graph to compute first {n_eigenvectors} eigenvectors (using sparse 'eigs' solver)...")
            graph.eigensolver = 'eigs'
            graph.eigensolver_params = {'k': n_eigenvectors}
        else:
            print("  -> Configuring graph to compute full Fourier basis (using dense 'eigh' solver)...")
            graph.eigensolver = 'eigh'
            graph.eigensolver_params = {}

        # This call will now use the configuration we just set, and will only
        # run once.
        print("  -> Computing Fourier basis...")
        graph.compute_fourier_basis()

    heat_filter = Heat(graph, tau=scales)

    all_coeffs = []
    for i in range(signal.shape[1]):
        # This will now correctly use the pre-computed partial or full basis.
        coeffs_one_feature = heat_filter.filter(signal[:, i], method='exact')
        all_coeffs.append(coeffs_one_feature)

    return np.stack(all_coeffs, axis=1)


def compute_acmw_wavelets(graph: Graph, signal: np.ndarray, scales: list[float] | None = None, alpha: float = 4.0, beta: float = 10.0, n_eigenvectors: int | None = None) -> np.ndarray:
    """
    Computes Anisotropic Curvature-Modulated Wavelets.
    This function now correctly leverages the fixed compute_heat_wavelets.
    """
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