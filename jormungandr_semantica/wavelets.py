import numpy as np
from pygsp.graphs import Graph
from pygsp.filters import Heat
# We no longer need to import gfilter

def compute_heat_wavelets(
    graph: Graph,
    signal: np.ndarray,
    scales: list[float] | None = None,
) -> np.ndarray:
    """
    Computes the heat wavelet transform of a signal on a graph.

    This function leverages PyGSP to define a heat kernel filter and applies
    it to the input signal at various scales.

    Parameters
    ----------
    graph : pygsp.graphs.Graph
        The PyGSP graph object. It must be instantiated before calling this
        function, as it contains the graph Laplacian required for the transform.
    signal : numpy.ndarray
        The signal(s) on the graph nodes. Must be of shape (n_nodes, n_features).
    scales : list[float], optional
        A list of scales (tau values) for the heat kernel. These control the
        "width" of the wavelets. If None, a default set of scales is used.

    Returns
    -------
    numpy.ndarray
        The wavelet coefficients. An array of shape (n_nodes, n_features, n_scales).
    """
    if signal.shape[0] != graph.N:
        raise ValueError(
            f"The signal must have the same number of nodes as the graph. "
            f"Got signal with {signal.shape[0]} nodes and graph with {graph.N} nodes."
        )

    if scales is None:
        scales = [5, 10, 25, 50]

    # 1. Instantiate the Heat filter bank with the provided graph and scales.
    heat_filter = Heat(graph, tau=scales)

    # 2. Apply the filter to the signal to get the wavelet coefficients.
    # --- THIS IS THE CORRECTED LINE ---
    # The modern PyGSP API calls the .filter() method on the filter object itself.
    wavelet_coefficients = heat_filter.filter(signal)

    return wavelet_coefficients