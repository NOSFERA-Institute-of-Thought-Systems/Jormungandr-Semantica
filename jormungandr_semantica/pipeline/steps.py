# jormungandr_semantica/pipeline/steps.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

# A simple data structure to pass data between steps
class PipelineData:
    def __init__(self, docs: list[str], embeddings: np.ndarray, labels_true: np.ndarray):
        self.docs = docs
        self.embeddings = embeddings
        self.labels_true = labels_true
        self.graph = None
        self.representation = None
        self.reduced_representation = None
        self.labels_pred = None

# Abstract Base Class for any pipeline step
class PipelineStep(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def run(self, data: PipelineData) -> PipelineData:
        """Executes the step and modifies the PipelineData object."""
        pass

# --- Define Interfaces for Each Stage ---

class GraphConstructor(PipelineStep):
    @abstractmethod
    def run(self, data: PipelineData) -> PipelineData:
        pass

class RepresentationBuilder(PipelineStep):
    @abstractmethod
    def run(self, data: PipelineData) -> PipelineData:
        pass

class Reducer(PipelineStep):
    @abstractmethod
    def run(self, data: PipelineData) -> PipelineData:
        pass

class Clusterer(PipelineStep):
    @abstractmethod
    def run(self, data: PipelineData) -> PipelineData:
        pass

# jormungandr_semantica/pipeline/steps.py (continued)
import jormungandr_semantica as js
from umap import UMAP
from sklearn.cluster import KMeans
from bertopic import BERTopic
from hdbscan import HDBSCAN
import pygsp.graphs as graphs

# --- Concrete Implementations for Jörmungandr ---

class FaissGraphConstructor(GraphConstructor):
    def run(self, data: PipelineData) -> PipelineData:
        print("Step: Building k-NN graph with Faiss (sparse)...")
        n_points = data.embeddings.shape[0]
        k = self.config['k']
        
        neighbors, distances = js.build_faiss_knn_graph(
            data.embeddings.astype('float32'), 
            k=k
        )
        
        # --- Build a SciPy sparse matrix instead of a dense one ---
        # We create the matrix in Coordinate (COO) format, which is efficient for construction.
        print("Constructing sparse adjacency matrix...")
        rows = np.repeat(np.arange(n_points), k)
        cols = neighbors.flatten()
        # Use a Gaussian kernel on the squared L2 distances from Faiss
        # Adding a small epsilon to avoid division by zero if distance is 0
        sigma = np.mean(distances[:, -1]) # Use distance to k-th neighbor as adaptive sigma
        weights = np.exp(-distances.flatten() / (sigma + 1e-8))
        
        adjacency_sparse = sp.coo_matrix((weights, (rows, cols)), shape=(n_points, n_points))

        # Symmetrize the matrix to make it undirected
        adjacency_sparse = adjacency_sparse.maximum(adjacency_sparse.T)
        
        print("Initializing PyGSP graph...")
        data.graph = graphs.Graph(W=adjacency_sparse)
        return data

class DirectRepresentationBuilder(RepresentationBuilder):
    """A 'dummy' step that just passes the original embeddings through."""
    def run(self, data: PipelineData) -> PipelineData:
        print("Step: Using direct embeddings as representation...")
        data.representation = data.embeddings
        return data

class UMAPReducer(Reducer):
    def run(self, data: PipelineData) -> PipelineData:
        print("Step: Reducing dimensions with UMAP...")
        reducer = UMAP(
            n_components=self.config['umap_dims'],
            random_state=self.config['seed']
        )
        reduced_embeddings = reducer.fit_transform(data.representation)
        
        # --- ADD THIS LINE ---
        # Normalize the embeddings to prevent numerical issues in KMeans
        data.reduced_representation = normalize(reduced_embeddings)
        
        return data

class KMeansClusterer(Clusterer):
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