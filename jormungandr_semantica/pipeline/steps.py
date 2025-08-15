from abc import ABC, abstractmethod
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
import jormungandr_semantica as js
from umap import UMAP
from sklearn.cluster import KMeans
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
    """Constructs a k-NN graph, ensuring the final graph is fully connected and valid for UMAP."""
    def run(self, data: PipelineData) -> PipelineData:
        print("Step: Building k-NN graph with Faiss (sparse)...")
        k = self.config.get('k', 15)
        
        # Initial graph construction
        embeddings_f32 = data.embeddings.astype(np.float32)
        n_points = embeddings_f32.shape[0]
        neighbors, distances = js.build_faiss_knn_graph(embeddings_f32, k=k)
        
        rows = np.repeat(np.arange(n_points), k)
        cols = neighbors.flatten()
        sigma = np.mean(distances[:, -1])
        weights_f64 = np.exp(-distances.flatten().astype(np.float64) / (sigma + 1e-8))
        
        adj = sp.coo_matrix((weights_f64, (rows, cols)), shape=(n_points, n_points)).tocsr()
        adj = adj.maximum(adj.T)
        
        # Check for connected components
        n_components, labels = connected_components(csgraph=adj, directed=False, return_labels=True)
        
        if n_components > 1:
            print(f"[WARNING] Graph not connected. Found {n_components}. Taking largest component.")
            unique, counts = np.unique(labels, return_counts=True)
            largest_component_label = unique[np.argmax(counts)]
            largest_idx = np.where(labels == largest_component_label)[0]
            
            # Filter the original data to keep only the largest component
            print(f"  -> Filtering data from {len(data.embeddings)} to {len(largest_idx)} points.")
            data.docs = [data.docs[i] for i in largest_idx]
            data.embeddings = data.embeddings[largest_idx]
            data.labels_true = data.labels_true[largest_idx]
            
            # --- THE GUARANTEED FIX ---
            # Re-run the k-NN graph construction on the *filtered* data.
            # This ensures the final graph has no nodes with < k neighbors.
            print("  -> Re-building graph on the largest component for UMAP compatibility.")
            embeddings_f32_filtered = data.embeddings.astype(np.float32)
            n_points_filtered = embeddings_f32_filtered.shape[0]
            
            neighbors, distances = js.build_faiss_knn_graph(embeddings_f32_filtered, k=k)
            rows = np.repeat(np.arange(n_points_filtered), k)
            cols = neighbors.flatten()
            sigma = np.mean(distances[:, -1])
            weights_f64 = np.exp(-distances.flatten().astype(np.float64) / (sigma + 1e-8))
            
            adj = sp.coo_matrix(
                (weights_f64, (rows, cols)), 
                shape=(n_points_filtered, n_points_filtered)
            ).tocsr()
            adj = adj.maximum(adj.T)

        adj.setdiag(0)
        adj.eliminate_zeros()
        
        data.graph = graphs.Graph(W=adj)
        print(f"Graph constructed. Adjacency (W) dtype: {data.graph.W.dtype}")
        
        return data


class DirectRepresentationBuilder(RepresentationBuilder):
    """Uses the initial embeddings as the representation."""
    def run(self, data: PipelineData) -> PipelineData:
        print("Step: Using direct embeddings as representation.")
        # For the graph path, this representation isn't used by the reducer, but we set it for consistency.
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

        if not np.all(np.isfinite(representation)):
            print("[WARNING] Non-finite values found in ACMW representation. Sanitizing...")
            representation = np.nan_to_num(representation, nan=0.0, posinf=0.0, neginf=0.0)

        data.representation = representation
        return data

# --- NO DUPLICATE CLASSES ---
class GraphUMAPReducer(Reducer):
    """Reduces dimensionality using UMAP on the pre-computed graph."""
    def run(self, data: PipelineData) -> PipelineData:
        print("Step: Reducing dimensionality with Graph-UMAP...")
        print("  -> Using metric='precomputed' on the graph adjacency matrix.")
        
        k = self.config.get('k', 15)
        print(f"  -> Setting UMAP n_neighbors to match graph k={k}.")
        
        # --- DEFENSIVE PROGRAMMING: UMAP SAFETY CHECK ---
        graph_matrix = data.graph.W.tocsr()
        
        # Check the number of non-zero entries per row (node degree)
        degrees = np.ediff1d(graph_matrix.indptr)
        
        if np.min(degrees) < k:
            print(f"[WARNING] Graph has nodes with degree < k ({np.min(degrees)} < {k}). This is likely due to numerical underflow and zero-elimination.")
            print("          Running UMAP with a corrected n_neighbors value.")
            # We must use a value for n_neighbors that is <= the minimum degree
            safe_k = int(np.min(degrees))
            if safe_k < 2:
                # This is an unrecoverable graph. UMAP will fail.
                raise RuntimeError(f"Graph is too sparse for UMAP. Minimum degree is {safe_k}.")
            k = safe_k
            print(f"  -> Using safe_k = {k} for UMAP.")

        reducer = UMAP(
            n_components=self.config['umap_dims'], 
            random_state=self.config['seed'],
            n_jobs=1,
            metric='precomputed',
            n_neighbors=k
        )
        
        data.reduced_representation = normalize(reducer.fit_transform(graph_matrix))
        return data

class FeatureUMAPReducer(Reducer):
    """Reduces dimensionality using UMAP on a feature representation."""
    def run(self, data: PipelineData) -> PipelineData:
        print("Step: Reducing dimensionality with Feature-UMAP...")
        print(f"  -> Using standard UMAP on representation of shape {data.representation.shape}")
        
        reducer = UMAP(
            n_components=self.config['umap_dims'], 
            random_state=self.config['seed'],
            n_jobs=1
        )
        
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