from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize
import networkx as nx
import jormungandr_semantica as js
from umap import UMAP
from sklearn.cluster import KMeans
from bertopic import BERTopic
from hdbscan import HDBSCAN
import pygsp.graphs as graphs
from scipy.sparse.csgraph import connected_components

# ==============================================================================
# 1. Core Data Structure
# ==============================================================================

class PipelineData:
    """A data structure to hold and pass data through the pipeline steps."""
    def __init__(self, docs, embeddings, labels_true):
        self.docs: list[str] = docs
        self.embeddings: np.ndarray = embeddings
        self.labels_true: np.ndarray = labels_true
        
        # These attributes are populated by the pipeline steps
        self.graph: graphs.Graph | None = None
        self.representation: np.ndarray | None = None
        self.reduced_representation: np.ndarray | None = None
        self.labels_pred: np.ndarray | None = None

# ==============================================================================
# 2. Abstract Base Classes for Pipeline Steps
# ==============================================================================

class PipelineStep(ABC):
    """Abstract base class for a step in the processing pipeline."""
    def __init__(self, config: dict):
        self.config = config
    
    @abstractmethod
    def run(self, data: PipelineData) -> PipelineData:
        """Executes the pipeline step."""
        pass

class GraphConstructor(PipelineStep): pass
class RepresentationBuilder(PipelineStep): pass
class Reducer(PipelineStep): pass
class Clusterer(PipelineStep): pass

# ==============================================================================
# 3. Concrete Implementations of Pipeline Steps
# ==============================================================================

class FaissGraphConstructor(GraphConstructor):
    """Constructs a k-NN graph using the C++/Faiss backend."""
    def run(self, data: PipelineData) -> PipelineData:
        print("Step: Building k-NN graph with Faiss (sparse)...")
        # Faiss C++ backend expects float32 for performance.
        embeddings_f32 = data.embeddings.astype(np.float32)
        n_points, k = embeddings_f32.shape[0], self.config['k']
        
        neighbors, distances = js.build_faiss_knn_graph(embeddings_f32, k=k)
        
        rows = np.repeat(np.arange(n_points), k)
        cols = neighbors.flatten()
        
        # --- THE FINAL, CORRECT FIX ---
        # 1. Cast distances to float64 *before* computing weights. This ensures
        #    the resulting weights array is float64.
        # 2. Construct the sparse matrix with this float64 weights array.
        # 3. The PyGSP Graph object will then inherit this double-precision dtype,
        #    ensuring numerical stability for the sparse eigensolver.
        sigma = np.mean(distances[:, -1])
        weights_f64 = np.exp(-distances.flatten().astype(np.float64) / (sigma + 1e-8))
        
        adj = sp.coo_matrix(
            (weights_f64, (rows, cols)), 
            shape=(n_points, n_points)
        ).tocsr()
        
        adj = adj.maximum(adj.T) # Symmetrize
        
        # Clean up small weights
        epsilon = 1e-9
        adj.data[adj.data < epsilon] = 0
        adj.eliminate_zeros()
        
        # Handle disconnected components by taking the largest one
        n_components, labels = connected_components(csgraph=adj, directed=False, return_labels=True)
        
        if n_components > 1:
            print(f"[WARNING] Graph not connected. Found {n_components}. Using largest component.")
            unique, counts = np.unique(labels, return_counts=True)
            largest_component_label = unique[np.argmax(counts)]
            largest_idx = np.where(labels == largest_component_label)[0]
            
            # Filter all data to match the subgraph
            data.docs = [data.docs[i] for i in largest_idx]
            data.embeddings = data.embeddings[largest_idx]
            data.labels_true = data.labels_true[largest_idx]
            adj = adj[largest_idx, :][:, largest_idx]

        adj.setdiag(0) # Remove self-loops
        
        # PyGSP Graph will inherit the float64 dtype from the sparse matrix `adj`.
        data.graph = graphs.Graph(W=adj)
        
        print(f"Graph constructed. Adjacency matrix (W) dtype: {data.graph.W.dtype}")
        
        return data

class DirectRepresentationBuilder(RepresentationBuilder):
    """Uses the initial embeddings as the representation."""
    def run(self, data: PipelineData) -> PipelineData:
        print("Step: Using direct embeddings as representation.")
        data.representation = data.embeddings
        return data


class WaveletRepresentationBuilder(RepresentationBuilder):
    """Builds a multi-scale representation using the SGWT."""
    def run(self, data: PipelineData) -> PipelineData:
        print("Step: Building representation with SGWT...")
        coeffs = js.compute_heat_wavelets(
            data.graph, 
            data.embeddings, 
            self.config.get('wavelet_scales'), 
            self.config.get('n_eigenvectors')
        )
        
        n_nodes, n_features, n_scales = coeffs.shape
        representation = coeffs.reshape((n_nodes, n_features * n_scales))
        
        # --- FINAL POLISH ---
        # Sanitize the representation to remove any non-finite values (NaN, inf)
        # that can be produced by the wavelet transform in edge cases.
        # This prevents numerical warnings in downstream UMAP/KMeans.
        if not np.all(np.isfinite(representation)):
            print("[WARNING] Non-finite values found in wavelet representation. Sanitizing...")
            representation = np.nan_to_num(representation, nan=0.0, posinf=0.0, neginf=0.0)
            
        data.representation = representation
        return data

class ACMWRepresentationBuilder(RepresentationBuilder):
    """Builds a representation using Anisotropic Curvature-Modulated Wavelets."""
    def run(self, data: PipelineData) -> PipelineData:
        print("Step: Building representation with ACMW...")
        coeffs = js.compute_acmw_wavelets(
            data.graph, 
            data.embeddings, 
            self.config.get('wavelet_scales'),
            n_eigenvectors=self.config.get('n_eigenvectors')
        )
        
        n_nodes, n_features, n_scales = coeffs.shape
        representation = coeffs.reshape((n_nodes, n_features * n_scales))

        # --- FINAL POLISH (also applied here for consistency) ---
        if not np.all(np.isfinite(representation)):
            print("[WARNING] Non-finite values found in ACMW representation. Sanitizing...")
            representation = np.nan_to_num(representation, nan=0.0, posinf=0.0, neginf=0.0)

        data.representation = representation
        return data
class UMAPReducer(Reducer):
    """Reduces dimensionality using UMAP."""
    def run(self, data: PipelineData) -> PipelineData:
        print("Step: Reducing dimensionality with UMAP...")
        reducer = UMAP(
            n_components=self.config['umap_dims'], 
            random_state=self.config['seed'],
            n_jobs=1 # Enforce single-threaded for stability
        )
        # Normalize the final embedding for stable clustering
        data.reduced_representation = normalize(reducer.fit_transform(data.representation))
        return data

class KMeansClusterer(Clusterer):
    """Performs clustering using KMeans."""
    def run(self, data: PipelineData) -> PipelineData:
        print("Step: Clustering with KMeans...")
        num_clusters = len(np.unique(data.labels_true))
        kmeans = KMeans(
            n_clusters=num_clusters, 
            random_state=self.config['seed'], 
            n_init='auto'
        )
        data.labels_pred = kmeans.fit_predict(data.reduced_representation)
        return data