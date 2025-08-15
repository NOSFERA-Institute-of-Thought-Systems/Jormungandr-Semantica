from abc import ABC, abstractmethod
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
import jormungandr_semantica as js
from umap import UMAP
from sklearn.cluster import KMeans
import pygsp.graphs as graphs
from scipy.sparse.csgraph import connected_components
from hdbscan import HDBSCAN
import networkx as nx
from jormungandr_semantica.wavelets import safe_compute_normalized_laplacian
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
    """Builds a representation using the CPAL operator."""
    def run(self, data: PipelineData) -> PipelineData:
        print("Step: Building representation with CPAL...")
        coeffs = js.compute_acmw_wavelets(
            data.graph, 
            data.embeddings, 
            self.config.get('wavelet_scales'),
            n_eigenvectors=self.config.get('n_eigenvectors'),
            alpha=self.config.get('cpal_alpha'),
            beta=self.config.get('cpal_beta'),
            epsilon=self.config.get('cpal_epsilon')
        )
        
        n_nodes, n_features, n_scales = coeffs.shape
        representation = coeffs.reshape((n_nodes, n_features * n_scales))

        if not np.all(np.isfinite(representation)):
            print("[WARNING] Non-finite values found in representation. Sanitizing...")
            representation = np.nan_to_num(representation, nan=0.0, posinf=0.0, neginf=0.0)

        data.representation = representation
        
        # --- THE FIX ---
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
        
        # --- TUNING EXPERIMENT ---
        # Let's try more robust UMAP parameters.
        # n_neighbors=15 (default) is good.
        # min_dist=0.0 encourages tighter packing of clusters.
        # metric='cosine' is often better for high-dimensional text embeddings.
        print("  -> Using tuned UMAP parameters (min_dist=0.0, metric='cosine').")
        
        reducer = UMAP(
            n_components=self.config['umap_dims'], 
            random_state=self.config['seed'],
            n_jobs=1,
            n_neighbors=self.config.get('k', 15),
            min_dist=0.0,
            metric='cosine'
        )
        
        data.reduced_representation = normalize(reducer.fit_transform(data.representation))
        return data

class CommunitySGWTRepresentationBuilder(RepresentationBuilder):
    """
    Builds a representation using a community-enhanced anisotropic graph.
    This method uses a fast community detection algorithm to identify likely
    clusters, then re-weights the graph to enhance intra-community connections
    and dampen inter-community connections before running the wavelet transform.
    """
    def run(self, data: PipelineData) -> PipelineData:
        print("Step: Building representation with Community-Enhanced SGWT...")
        
        # --- 1. Detect Communities ---
        print("  -> Detecting communities with the Louvain method...")
        # Convert to NetworkX graph to use its community detection algorithms
        nx_graph = nx.from_scipy_sparse_array(data.graph.W)

        
        # The Louvain method is fast and effective. It returns a list of sets,
        # where each set contains the nodes of a community.
        communities = nx.community.louvain_communities(nx_graph, weight='weight', seed=self.config['seed'])
        print(f"  -> Found {len(communities)} communities.")
        
        # Create a mapping from each node to its community ID for fast lookup
        node_to_community_id = {node: i for i, community in enumerate(communities) for node in community}

        # --- 2. Construct Community-Modulated Anisotropic Graph ---
        print("  -> Creating community-modulated anisotropic weight matrix...")
        
        # Get the hyperparameters from the config
        epsilon = self.config.get('community_epsilon', 0.1) # Dampening factor for inter-community edges
        
        W_prime = data.graph.W.copy().tolil()
        
        # Iterate through all edges and re-weight them
        for i, j in zip(*W_prime.nonzero()):
            # Check if nodes i and j are in the same community
            if node_to_community_id.get(i) == node_to_community_id.get(j):
                # This is an intra-community edge, keep its weight (or slightly boost it)
                pass # W_prime[i, j] remains the same
            else:
                # This is an inter-community edge (a "bridge"), dampen its weight
                W_prime[i, j] *= epsilon
        
        W_prime = W_prime.tocsr()
        
        # --- 3. Compute Wavelets on the New Anisotropic Graph ---
        print("  -> Initializing new anisotropic graph and computing wavelets...")
        anisotropic_graph = graphs.Graph(W=W_prime)
        
        # We need to compute a stable Laplacian for this new graph.
        # Since we only dampened weights, the graph remains valid, but let's be safe.
        # L_safe = graphs.compute_laplacian(anisotropic_graph.W, lap_type='normalized')
        L_safe = safe_compute_normalized_laplacian(anisotropic_graph.W)
        anisotropic_graph.L = L_safe
        
        # Now, call the standard wavelet function with our new, community-enhanced graph
        coeffs = js.compute_heat_wavelets(
            anisotropic_graph, 
            data.embeddings, 
            self.config.get('wavelet_scales'),
            n_eigenvectors=self.config.get('n_eigenvectors')
        )
        
        # --- 4. Finalize the Representation ---
        n_nodes, n_features, n_scales = coeffs.shape
        representation = coeffs.reshape((n_nodes, n_features * n_scales))
        
        if not np.all(np.isfinite(representation)):
            print("[WARNING] Non-finite values found in representation. Sanitizing...")
            representation = np.nan_to_num(representation, nan=0.0, posinf=0.0, neginf=0.0)

        data.representation = representation
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
    
class HDBSCANClusterer(Clusterer):
    """Performs clustering using the HDBSCAN algorithm."""
    def run(self, data: PipelineData) -> PipelineData:
        print("Step: Clustering with HDBSCAN...")
        # min_cluster_size is the most important hyperparameter for HDBSCAN
        min_cluster_size = self.config.get('min_cluster_size', 15)
        
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean',
            core_dist_n_jobs=1 # Ensure single-threaded for stability
        )
        
        data.labels_pred = clusterer.fit_predict(data.reduced_representation)
        # HDBSCAN labels noise points as -1. The benchmark script already handles this.
        return data
    

class RankSGWTRepresentationBuilder(RepresentationBuilder):
    """
    Builds a representation using a rank-based anisotropic graph.
    This method enhances/dampens a small percentage of edges based on their
    curvature rank, providing a gentle nudge to the geometry.
    """
    def run(self, data: PipelineData) -> PipelineData:
        print("Step: Building representation with Rank-Enhanced SGWT...")
        
        # 1. Compute Curvature and Get Edges
        print("  -> Computing Forman-Ricci curvature...")
        nx_graph = nx.from_scipy_sparse_array(data.graph.W)
        frc = FormanRicci(nx_graph, verbose="INFO")
        frc.compute_ricci_curvature()
        
        # Get a list of all edges with their curvatures
        edges_with_curvature = [
            (u, v, attr['formanCurvature']) 
            for u, v, attr in frc.G.edges(data=True)
        ]
        
        # Sort edges by curvature
        edges_with_curvature.sort(key=lambda x: x[2])
        
        # 2. Identify Top and Bottom Edges by Rank
        print("  -> Identifying top/bottom edges by curvature rank...")
        n_edges = len(edges_with_curvature)
        quantile = self.config.get('rank_quantile', 0.1) # e.g., top/bottom 10%
        n_modulate = int(n_edges * quantile)
        
        bottom_edges = edges_with_curvature[:n_modulate]
        top_edges = edges_with_curvature[-n_modulate:]
        
        # 3. Construct Rank-Modulated Anisotropic Graph
        print(f"  -> Modulating {n_modulate*2} edges...")
        enhancement_factor = self.config.get('rank_enhancement', 1.5) # e.g., +50%
        dampening_factor = self.config.get('rank_dampening', 0.5)   # e.g., -50%
        
        W_prime = data.graph.W.copy().tolil()
        
        # Dampen "bridges" (most negative curvature)
        for u, v, _ in bottom_edges:
            W_prime[u, v] *= dampening_factor
            W_prime[v, u] *= dampening_factor
            
        # Enhance "threads" (most positive curvature)
        for u, v, _ in top_edges:
            W_prime[u, v] *= enhancement_factor
            W_prime[v, u] *= enhancement_factor

        W_prime = W_prime.tocsr()
        
        # 4. Compute Wavelets on the New Anisotropic Graph
        anisotropic_graph = graphs.Graph(W=W_prime)
        L_safe = safe_compute_normalized_laplacian(anisotropic_graph.W)
        anisotropic_graph.L = L_safe
        
        coeffs = js.compute_heat_wavelets(
            anisotropic_graph, 
            data.embeddings, 
            self.config.get('wavelet_scales'),
            n_eigenvectors=self.config.get('n_eigenvectors')
        )
        
        # 5. Finalize Representation
        n_nodes, n_features, n_scales = coeffs.shape
        representation = coeffs.reshape((n_nodes, n_features * n_scales))
        
        if not np.all(np.isfinite(representation)):
            print("[WARNING] Non-finite values found. Sanitizing...")
            representation = np.nan_to_num(representation, nan=0.0, posinf=0.0, neginf=0.0)

        data.representation = representation
        return data