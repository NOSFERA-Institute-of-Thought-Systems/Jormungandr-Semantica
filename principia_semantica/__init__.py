"""
Principia Semantica

A framework for geometric deep learning in semantic analysis.
"""

# Import our new C++ function
from principia_semantica._core import build_faiss_knn_graph

# Update the public API
__all__ = [
    "build_faiss_knn_graph",
]