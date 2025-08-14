"""
Jörmungandr-Semantica

A framework for geometric deep learning in semantic analysis.
"""
# Import from the C++ backend
from ._core import build_faiss_knn_graph
# Import from the wavelets module
from .wavelets import compute_heat_wavelets

# Update the public API
__all__ = [
    "build_faiss_knn_graph",
    "compute_heat_wavelets",
]