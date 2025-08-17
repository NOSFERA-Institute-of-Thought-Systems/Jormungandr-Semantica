from abc import ABC, abstractmethod
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
import aglt as js
from umap import UMAP
import pygsp.graphs as graphs
from scipy.sparse.csgraph import connected_components
from hdbscan import HDBSCAN
import networkx as nx
from tqdm import tqdm

# chunked_chebyshev.py
import torch
import torch.nn as nn

# from .chunked_chebyshev import ChunkedChebyshev
# --- Critical Imports for Advanced Builders ---
from GraphRicciCurvature.FormanRicci import FormanRicci

# from chunked_chebyshev import ChunkedChebyshev

# Import the helper function from the wavelets module
from aglt.wavelets import safe_compute_normalized_laplacian

from pathlib import Path
import pickle


# ==============================================================================
# 1. Core Data Structure & Abstract Classes
# ==============================================================================
class PipelineData:
    """A data structure to hold and pass data through the pipeline steps.

    This class acts as a central data bus for the entire Jörmungandr pipeline.
    An instance of this class is created at the beginning of an experiment
    and is progressively populated and modified by each `PipelineStep`.

    Attributes
    ----------
    docs : list[str]
        The raw text documents of the dataset.
    embeddings : np.ndarray
        The high-dimensional sentence embeddings of the documents.
    labels_true : np.ndarray
        The ground-truth integer labels for each document.
    graph : pygsp.graphs.Graph or None
        The PyGSP graph object representing the data's connectivity,
        populated by a `GraphConstructor`.
    representation : np.ndarray or None
        The feature representation used for dimensionality reduction,
        populated by a `RepresentationBuilder`.
    reduced_representation : np.ndarray or None
        The final low-dimensional representation, populated by a `Reducer`.
    labels_pred : np.ndarray or None
        The predicted cluster labels, populated by a `Clusterer`.
    """

    def __init__(self, docs, embeddings, labels_true):
        self.docs: list[str] = docs
        self.embeddings: np.ndarray = embeddings
        self.labels_true: np.ndarray = labels_true
        self.graph: graphs.Graph | None = None
        self.representation: np.ndarray | None = None
        self.reduced_representation: np.ndarray | None = None
        self.labels_pred: np.ndarray | None = None


class PipelineStep(ABC):
    """Abstract base class for a step in the processing pipeline.

    Each concrete subclass of `PipelineStep` represents a distinct, modular
    operation in the overall experiment, such as graph construction,
    representation building, or clustering.
    """

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def run(self, data: PipelineData) -> PipelineData:
        """Executes the pipeline step.

        This method takes a `PipelineData` object, performs an operation
        (e.g., computes a graph, creates a representation), modifies the
        data object with the results, and returns it.

        Parameters
        ----------
        data : PipelineData
            The central data object for the pipeline run.

        Returns
        -------
        PipelineData
            The modified data object.
        """
        pass


class GraphConstructor(PipelineStep):
    """Abstract base class for steps that construct a graph.

    Subclasses of `GraphConstructor` are responsible for taking the initial
    embeddings from a `PipelineData` object and populating the `graph`
    attribute with a `pygsp.graphs.Graph`.
    """

    pass


class RepresentationBuilder(PipelineStep):
    """Abstract base class for steps that build a feature representation.

    Subclasses of `RepresentationBuilder` are responsible for populating the
    `representation` attribute of the `PipelineData` object. This can range
    from using the embeddings directly to computing complex, multi-scale
    wavelet coefficients.
    """

    pass


class Reducer(PipelineStep):
    """Abstract base class for steps that perform dimensionality reduction.

    Subclasses of `Reducer` take a high-dimensional representation and
    populate the `reduced_representation` attribute of the `PipelineData`
    object with a low-dimensional version.
    """

    pass


class Clusterer(PipelineStep):
    """Abstract base class for steps that perform clustering.

    Subclasses of `Clusterer` take a low-dimensional representation and
    populate the `labels_pred` attribute of the `PipelineData` object with
    the final predicted cluster labels.
    """

    pass


# ==============================================================================
# 2. Concrete Implementations of Pipeline Steps
# ==============================================================================


class FaissGraphConstructor(GraphConstructor):
    """Constructs a k-NN graph using the high-performance Faiss library.

    This class is responsible for the first major step of the pipeline:
    transforming a point cloud of high-dimensional embeddings into a sparse
    graph representation. It uses Faiss for efficient k-nearest neighbor
    search and includes critical logic to ensure the final graph is a single,
    connected component, which is a prerequisite for many downstream spectral
    methods and for UMAP's `metric='precomputed'` path.
    """

    def run(self, data: PipelineData) -> PipelineData:
        """Builds a k-NN graph and ensures it is a single connected component.

        The process involves:
        1. Building an initial k-NN graph on all data points.
        2. Identifying all connected components.
        3. If more than one component exists, it filters the data to keep only
           the largest component.
        4. It then re-builds the k-NN graph on this filtered subset to ensure
           the final graph is well-formed and connected.

        Parameters
        ----------
        data : PipelineData
            The pipeline data object containing the initial embeddings.

        Returns
        -------
        PipelineData
            The data object with the `graph` attribute populated and potentially
            filtered `docs`, `embeddings`, and `labels_true` attributes.
        """

        print("Step: Building k-NN graph with Faiss (sparse)...")
        k = self.config.get("k", 15)
        embeddings_f32 = data.embeddings.astype(np.float32)
        n_points = embeddings_f32.shape[0]
        neighbors, distances = js.build_faiss_knn_graph(embeddings_f32, k=k)
        rows, cols = np.repeat(np.arange(n_points), k), neighbors.flatten()
        sigma = np.mean(distances[:, -1])
        weights_f64 = np.exp(-distances.flatten().astype(np.float64) / (sigma + 1e-8))
        adj = sp.coo_matrix(
            (weights_f64, (rows, cols)), shape=(n_points, n_points)
        ).tocsr()
        adj = adj.maximum(adj.T)
        n_components, labels = connected_components(
            csgraph=adj, directed=False, return_labels=True
        )
        if n_components > 1:
            print(
                f"[WARNING] Graph not connected. Found {n_components}. Using largest component."
            )
            unique, counts = np.unique(labels, return_counts=True)
            largest_idx = np.where(labels == unique[np.argmax(counts)])[0]
            data.docs = [data.docs[i] for i in largest_idx]
            data.embeddings = data.embeddings[largest_idx]
            data.labels_true = data.labels_true[largest_idx]
            print(
                "  -> Re-building graph on the largest component for UMAP compatibility."
            )
            embeddings_f32_filtered = data.embeddings.astype(np.float32)
            n_points_filtered = embeddings_f32_filtered.shape[0]
            neighbors, distances = js.build_faiss_knn_graph(
                embeddings_f32_filtered, k=k
            )
            rows, cols = np.repeat(np.arange(n_points_filtered), k), neighbors.flatten()
            sigma = np.mean(distances[:, -1])
            weights_f64 = np.exp(
                -distances.flatten().astype(np.float64) / (sigma + 1e-8)
            )
            adj = sp.coo_matrix(
                (weights_f64, (rows, cols)),
                shape=(n_points_filtered, n_points_filtered),
            ).tocsr()
            adj = adj.maximum(adj.T)
        adj.setdiag(0)
        adj.eliminate_zeros()
        data.graph = graphs.Graph(W=adj)
        print(f"Graph constructed. Adjacency (W) dtype: {data.graph.W.dtype}")
        return data


class DirectRepresentationBuilder(RepresentationBuilder):
    """Uses the initial embeddings directly as the representation.

    This builder serves as a crucial baseline. It represents the simplest
    possible approach where no graph information or geometric priors are used
    to modify the initial feature space provided by the sentence-transformer.
    Its performance is a benchmark against which all more complex,
    geometry-aware builders are measured.
    """

    def run(self, data: PipelineData) -> PipelineData:
        """Assigns the initial `embeddings` to the `representation` attribute.

        Parameters
        ----------
        data : PipelineData
            The pipeline data object, containing the initial embeddings.

        Returns
        -------
        PipelineData
            The data object with the `representation` attribute populated.
        """
        print("Step: Using direct embeddings as representation.")
        data.representation = data.embeddings
        return data


class WaveletRepresentationBuilder(RepresentationBuilder):
    """Builds a multi-scale representation using the isotropic SGWT.

    This builder implements a standard, isotropic Signal Processing on Graphs
    (GSP) approach. It filters the initial embeddings through a bank of heat
    kernel filters (a low-pass filter) at multiple scales. This serves as the
    strong "geometry-blind" baseline, demonstrating the power of graph-based
    diffusion without any explicit geometric modifications.

    The resulting representation captures features at different levels of
    "smoothness" across the graph.
    """

    def run(self, data: PipelineData) -> PipelineData:
        """Computes heat wavelet coefficients and reshapes them into a representation.

        Parameters
        ----------
        data : PipelineData
            The pipeline data object, containing the graph and embeddings.

        Returns
        -------
        PipelineData
            The data object with the `representation` attribute populated by the
            flattened, multi-scale wavelet coefficients.
        """
        print("Step: Building representation with SGWT...")
        coeffs = js.compute_heat_wavelets(
            data.graph,
            data.embeddings,
            self.config.get("wavelet_scales"),
            self.config.get("n_eigenvectors"),
        )
        n_nodes, n_features, n_scales = coeffs.shape
        rep = coeffs.reshape((n_nodes, n_features * n_scales))
        data.representation = np.nan_to_num(rep)
        return data


class ACMWRepresentationBuilder(RepresentationBuilder):
    """Builds a representation using the Connectivity-Preserving Anisotropic Laplacian (CPAL).

    This builder was the first heuristic-based intervention, testing the
    hypothesis that a global, smooth re-weighting of the graph based on
    Ollivier-Ricci curvature could improve representations. It applies a
    sigmoid function to the curvature of every edge to create an anisotropic
    graph before computing wavelets.

    This experiment was a key part of the falsification process, as its
    poor performance demonstrated the flaws of a global, monotonic mapping
    on a skewed curvature distribution.
    """

    def run(self, data: PipelineData) -> PipelineData:
        """Creates an anisotropic graph via CPAL and computes wavelets on it.

        Parameters
        ----------
        data : PipelineData
            The pipeline data object, containing the graph and embeddings.

        Returns
        -------
        PipelineData
            The data object with the `representation` attribute populated by
            coefficients from the anisotropic graph.
        """
        print("Step: Building representation with CPAL...")
        coeffs = js.compute_acmw_wavelets(
            data.graph,
            data.embeddings,
            self.config.get("wavelet_scales"),
            n_eigenvectors=self.config.get("n_eigenvectors"),
            alpha=self.config.get("cpal_alpha"),
            epsilon=self.config.get("cpal_epsilon"),
        )
        n_nodes, n_features, n_scales = coeffs.shape
        rep = coeffs.reshape((n_nodes, n_features * n_scales))
        data.representation = np.nan_to_num(rep)
        return data


class CommunitySGWTRepresentationBuilder(RepresentationBuilder):
    """Builds a representation using a community-enhanced anisotropic graph.

    This builder tested the hypothesis that a coarse-grained, community-based
    geometric signal is superior to the noisy, local curvature signal. It uses
    the Louvain method to detect communities and then aggressively dampens the
    weights of inter-community edges.

    Its failure demonstrated that hard partitioning of the graph destroys
    critical information and is too blunt an instrument for effective
    anisotropic diffusion.
    """

    def run(self, data: PipelineData) -> PipelineData:
        """Detects communities, re-weights the graph, and computes wavelets.

        Parameters
        ----------
        data : PipelineData
            The pipeline data object.

        Returns
        -------
        PipelineData
            The data object with the `representation` attribute populated.
        """
        print("Step: Building representation with Community-Enhanced SGWT...")
        nx_graph = nx.from_scipy_sparse_array(data.graph.W)
        communities = nx.community.louvain_communities(
            nx_graph, weight="weight", seed=self.config["seed"]
        )
        print(f"  -> Found {len(communities)} communities.")
        node_to_community_id = {
            node: i for i, comm in enumerate(communities) for node in comm
        }
        epsilon = self.config.get("community_epsilon", 0.1)
        W_prime = data.graph.W.copy().tolil()
        for i, j in zip(*W_prime.nonzero()):
            if node_to_community_id.get(i) != node_to_community_id.get(j):
                W_prime[i, j] *= epsilon
        anisotropic_graph = graphs.Graph(W=W_prime.tocsr())
        anisotropic_graph.L = safe_compute_normalized_laplacian(anisotropic_graph.W)
        coeffs = js.compute_heat_wavelets(
            anisotropic_graph,
            data.embeddings,
            self.config.get("wavelet_scales"),
            n_eigenvectors=self.config.get("n_eigenvectors"),
        )
        n_nodes, n_features, n_scales = coeffs.shape
        rep = coeffs.reshape((n_nodes, n_features * n_scales))
        data.representation = np.nan_to_num(rep)
        return data


class RankSGWTRepresentationBuilder(RepresentationBuilder):
    """Builds a representation using a rank-based anisotropic graph.

    This was the final and most sophisticated heuristic-based intervention. It
    tests the hypothesis that a surgical modification of the graph's geometry
    is more effective than a global one. It identifies and modulates only the
    top and bottom quantiles of edges based on their Forman-Ricci curvature rank.

    While the best-performing heuristic, its failure to surpass the isotropic
    baseline was the final piece of evidence needed to abandon hand-crafted
    rules and pivot to a learning-based approach.
    """

    def run(self, data: PipelineData) -> PipelineData:
        """Sorts edges by curvature, modulates the extremes, and computes wavelets.

        Parameters
        ----------
        data : PipelineData
            The pipeline data object.

        Returns
        -------
        PipelineData
            The data object with the `representation` attribute populated.
        """
        print("Step: Building representation with Rank-Enhanced SGWT...")
        nx_graph = nx.from_scipy_sparse_array(data.graph.W)
        frc = FormanRicci(nx_graph)
        frc.compute_ricci_curvature()
        edges_with_curvature = sorted(
            [(u, v, attr["formanCurvature"]) for u, v, attr in frc.G.edges(data=True)],
            key=lambda x: x[2],
        )
        n_edges = len(edges_with_curvature)
        n_modulate = int(n_edges * self.config.get("rank_quantile", 0.1))
        bottom_edges, top_edges = (
            edges_with_curvature[:n_modulate],
            edges_with_curvature[-n_modulate:],
        )
        W_prime = data.graph.W.copy().tolil()
        dampening, enhancement = self.config.get("rank_dampening"), self.config.get(
            "rank_enhancement"
        )
        for u, v, _ in bottom_edges:
            W_prime[u, v] *= dampening
            W_prime[v, u] *= dampening
        for u, v, _ in top_edges:
            W_prime[u, v] *= enhancement
            W_prime[v, u] *= enhancement
        anisotropic_graph = graphs.Graph(W=W_prime.tocsr())
        anisotropic_graph.L = safe_compute_normalized_laplacian(anisotropic_graph.W)
        coeffs = js.compute_heat_wavelets(
            anisotropic_graph,
            data.embeddings,
            self.config.get("wavelet_scales"),
            n_eigenvectors=self.config.get("n_eigenvectors"),
        )
        n_nodes, n_features, n_scales = coeffs.shape
        rep = coeffs.reshape((n_nodes, n_features * n_scales))
        data.representation = np.nan_to_num(rep)
        return data


def sparse_mm(L, X):
    return torch.sparse.mm(L, X)


class ChunkedChebyshev(nn.Module):
    """Computes a linear combination of Chebyshev polynomials chunk-wise."""

    def __init__(self, m=30, chunk_size=64):
        super().__init__()
        self.m = m
        self.chunk_size = chunk_size

    def chebyshev_block(self, L_rescaled, X_block, coeffs):
        if self.m < 2:
            return coeffs[0] * X_block if self.m == 1 else X_block
        T0, T1 = X_block, sparse_mm(L_rescaled, X_block)
        current_sum = (coeffs[0] * T0) + (coeffs[1] * T1)
        for k in range(2, self.m):
            Tk = 2.0 * sparse_mm(L_rescaled, T1) - T0
            current_sum += coeffs[k] * Tk
            T0, T1 = T1, Tk
        return current_sum

    def forward(self, L_rescaled, X, coeffs):
        N, D = X.shape
        c = self.chunk_size
        out = X.new_zeros((N, D))
        for start in range(0, D, c):
            end = min(D, start + c)
            out_block = self.chebyshev_block(
                L_rescaled, X[:, start:end].contiguous(), coeffs
            )
            out[:, start:end] = out_block
        return out


class DifferentiableChebyshevOperator(torch.nn.Module):
    """Manages parameters and delegates computation to the ChunkedChebyshev module."""

    def __init__(self, initial_center, chebyshev_order=30, chunk_size=64, epsilon=0.01):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))
        self.center = torch.nn.Parameter(torch.tensor(float(initial_center)))
        self.epsilon = epsilon
        self.chebyshev_op = ChunkedChebyshev(m=chebyshev_order, chunk_size=chunk_size)

    def h_theta(self, kappa_values):
        sensitivity = torch.nn.functional.softplus(self.alpha)
        return self.epsilon + (1 - self.epsilon) * torch.sigmoid(
            sensitivity * (kappa_values - self.center)
        )

    def forward(self, W_sparse, kappa_sparse_values, X_signal, t_scale=5.0):
        conductance_values = self.h_theta(kappa_sparse_values)
        W_prime_sparse = torch.sparse_coo_tensor(
            W_sparse.indices(), W_sparse.values() * conductance_values, W_sparse.shape
        ).coalesce()
        D_prime_vec = torch.sparse.sum(W_prime_sparse, dim=1).to_dense()
        D_inv_sqrt_vec = 1.0 / (torch.sqrt(D_prime_vec) + 1e-8)
        rows, cols = W_prime_sparse.indices()
        W_norm_vals = (
            D_inv_sqrt_vec[rows] * W_prime_sparse.values() * D_inv_sqrt_vec[cols]
        )
        N = W_sparse.shape[0]
        I_indices = torch.arange(N, device=W_sparse.device).view(1, -1).repeat(2, 1)
        identity_matrix = torch.sparse_coo_tensor(
            I_indices, torch.ones(N, device=W_sparse.device), W_sparse.shape
        ).coalesce()
        L_aniso = (
            identity_matrix
            - torch.sparse_coo_tensor(
                W_prime_sparse.indices(), W_norm_vals, W_prime_sparse.shape
            )
        ).coalesce()
        # Also update the rescaled Laplacian calculation a few lines down
        lambda_max = 2.0

        L_rescaled = (2.0 / lambda_max) * L_aniso - identity_matrix

        coeffs = self.compute_chebyshev_coeffs(
            t_scale, lambda_max, device=W_sparse.device
        )
        return self.chebyshev_op(L_rescaled, X_signal, coeffs)

    def compute_chebyshev_coeffs(self, t_scale, lambda_max, device):
        M = self.chebyshev_op.m
        x = np.cos(np.pi * (np.arange(M, dtype=np.float64) + 0.5) / M)
        lambdas = (lambda_max / 2.0) * (x + 1.0)
        f_vals = np.exp(-t_scale * lambdas)
        coeffs = np.zeros(M)
        for k in range(M):
            coeffs[k] = (2.0 / M) * np.sum(f_vals * np.cos(k * np.arccos(x)))
        coeffs[0] /= 2.0
        return torch.from_numpy(coeffs).float().to(device)


class LearnableSGWTRepresentationBuilder(RepresentationBuilder):
    """Builds a representation by learning the optimal anisotropic operator."""

    def get_triplet(self, y):
        anchor_idx, p_idx, n_idx = 0, 0, 0
        while p_idx == anchor_idx or n_idx == anchor_idx or p_idx == n_idx:
            anchor_idx = np.random.randint(len(y))
            anchor_label = y[anchor_idx]
            positive_indices = np.where(y == anchor_label)[0]
            if len(positive_indices) < 2:
                continue
            positive_idx = np.random.choice(positive_indices)
            negative_indices = np.where(y != anchor_label)[0]
            if len(negative_indices) < 1:
                continue
            negative_idx = np.random.choice(negative_indices)
        return anchor_idx, positive_idx, negative_idx

    def run(self, data: PipelineData) -> PipelineData:
        print("Step: Building representation with SCALABLE Learnable SGWT...")
        CACHE_DIR = Path("cache")
        k = self.config.get("k", 15)
        dataset_name = self.config.get("dataset")
        cache_path = CACHE_DIR / f"{dataset_name}_k{k}_forman_curvature.pkl"
        if cache_path.exists():
            print("  -> Loading pre-computed Forman-Ricci curvature from cache...")
            with open(cache_path, "rb") as f:
                curvature_dict = pickle.load(f)
        else:
            print(
                "  -> No cache found. Pre-computing Forman-Ricci curvature (this will be slow)..."
            )
            nx_graph = nx.from_scipy_sparse_array(data.graph.W)
            frc = FormanRicci(nx_graph)
            frc.compute_ricci_curvature()
            curvature_dict = {
                (u, v): attr["formanCurvature"] for u, v, attr in frc.G.edges(data=True)
            }
            print("  -> Saving curvature to cache for future runs...")
            with open(cache_path, "wb") as f:
                pickle.dump(curvature_dict, f)

        print("  -> [CONTROL EXPERIMENT] Overriding curvature with all zeros.")
        curvature_dict = {edge: 0.0 for edge in curvature_dict}

        W_coo = data.graph.W.tocoo()
        W_indices = torch.from_numpy(np.vstack((W_coo.row, W_coo.col))).long()
        W_values = torch.from_numpy(W_coo.data).float()
        W_torch_sparse = torch.sparse_coo_tensor(
            W_indices, W_values, torch.Size(W_coo.shape)
        ).coalesce()
        kappa_values = torch.zeros_like(W_torch_sparse.values())
        for i, (r, c) in enumerate(W_torch_sparse.indices().T):
            u, v = r.item(), c.item()
            kappa_values[i] = curvature_dict.get((u, v), curvature_dict.get((v, u), 0))
        initial_center = torch.median(kappa_values[kappa_values.nonzero()]).squeeze()

        device = torch.device("cpu")
        print("  -> [NOTE] Using CPU for learnable operator.")
        print(f"  -> Beginning training (device: {device})...")
        W_torch_sparse, kappa_values = W_torch_sparse.to(device), kappa_values.to(
            device
        )
        X_signal_torch = torch.from_numpy(data.embeddings).float().to(device)
        model = DifferentiableChebyshevOperator(initial_center=initial_center).to(
            device
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        loss_fn = torch.nn.TripletMarginLoss(margin=1.0)
        num_epochs = self.config.get("learnable_epochs", 50)
        for epoch in range(num_epochs):
            model.train()
            print(
                f"    Epoch {epoch:03d}/{num_epochs}: Forward pass...",
                end="",
                flush=True,
            )
            X_prime = model(W_torch_sparse, kappa_values, X_signal_torch)
            print("done. Loss & Backward...", end="", flush=True)
            batch_loss = 0
            num_triplets = 200
            for _ in range(num_triplets):
                a_idx, p_idx, n_idx = self.get_triplet(data.labels_true)
                anchor, positive, negative = (
                    X_prime[a_idx],
                    X_prime[p_idx],
                    X_prime[n_idx],
                )
                batch_loss += loss_fn(anchor, positive, negative)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            print(f"done. --> Loss = {batch_loss.item()/num_triplets:.4f}")
        print("  -> Training complete.")
        print("  -> Computing final representation with learned operator...")
        model.eval()
        with torch.no_grad():
            final_rep = model(
                W_torch_sparse, kappa_values, X_signal_torch, t_scale=50.0
            )
        data.representation = final_rep.cpu().numpy()
        return data


class GraphUMAPReducer(Reducer):
    """Reduces dimensionality using UMAP directly on the graph structure.

    This reducer is intended for pipelines where the graph itself is considered
    the primary representation. It uses UMAP with `metric='precomputed'`,
    operating on the graph's adjacency matrix. It includes defensive checks
    to ensure that the `n_neighbors` parameter for UMAP is compatible with the
    graph's actual minimum degree, preventing common runtime errors.
    """

    def run(self, data: PipelineData) -> PipelineData:
        """Applies UMAP to the graph's adjacency matrix.

        Parameters
        ----------
        data : PipelineData
            The pipeline data object with a populated `graph` attribute.

        Returns
        -------
        PipelineData
            The data object with the `reduced_representation` attribute populated.
        """
        print("Step: Reducing dimensionality with Graph-UMAP...")
        k = self.config.get("k", 15)
        graph_matrix = data.graph.W.tocsr()
        degrees = np.ediff1d(graph_matrix.indptr)
        if np.min(degrees) < k:
            safe_k = int(np.min(degrees))
            if safe_k < 2:
                raise RuntimeError(
                    f"Graph is too sparse for UMAP. Minimum degree is {safe_k}."
                )
            k = safe_k
            print(
                f"  -> [WARNING] Graph has nodes with degree < k. Using safe_k = {k} for UMAP."
            )
        reducer = UMAP(
            n_components=self.config["umap_dims"],
            random_state=self.config["seed"],
            n_jobs=1,
            metric="precomputed",
            n_neighbors=k,
        )
        data.reduced_representation = normalize(reducer.fit_transform(graph_matrix))
        return data


class HDBSCANClusterer(Clusterer):
    """Performs clustering on the reduced representation using HDBSCAN.

    This class implements a more advanced, density-based clustering algorithm.
    HDBSCAN has the advantage of not requiring the number of clusters to be
    specified beforehand and can identify noise points (labeling them as -1).
    The `min_cluster_size` is its most important hyperparameter.
    """

    def run(self, data: PipelineData) -> PipelineData:
        """Fits HDBSCAN and predicts cluster labels.

        Parameters
        ----------
        data : PipelineData
            The pipeline data object with a populated `reduced_representation`.

        Returns
        -------
        PipelineData
            The data object with the `labels_pred` attribute populated.
        """
        print("Step: Clustering with HDBSCAN...")
        min_cluster_size = self.config.get("min_cluster_size", 15)
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size, metric="euclidean", core_dist_n_jobs=1
        )
        data.labels_pred = clusterer.fit_predict(data.reduced_representation)
        return data


class LearnableRepresentationBuilder(RepresentationBuilder):
    """
    Builds a representation by learning an optimal anisotropic graph aggregation.
    This is a fast, GCN-style operator where the edge weights (attention)
    are a learned function of the graph's Ricci curvature.
    """

    def get_triplet(self, y):
        """
        Correctly samples a triplet (anchor, positive, negative) ensuring all indices are valid.
        """
        while True:
            # 1. Select a random anchor
            anchor_idx = np.random.randint(len(y))
            anchor_label = y[anchor_idx]

            # 2. Find all possible positive samples for this anchor's class
            positive_indices = np.where(y == anchor_label)[0]

            # 3. Ensure there is at least one OTHER positive sample to choose from
            if len(positive_indices) < 2:
                continue  # This class is too small, try another anchor

            # 4. Select a positive sample that is GUARANTEED to not be the anchor
            positive_choices = positive_indices[positive_indices != anchor_idx]
            p_idx = np.random.choice(positive_choices)

            # 5. Find all possible negative samples
            negative_indices = np.where(y != anchor_label)[0]

            # This should not happen in a multi-class dataset, but is safe to check
            if len(negative_indices) < 1:
                continue

            # 6. Select a random negative sample
            n_idx = np.random.choice(negative_indices)

            # We have successfully found a valid triplet by construction, so we can exit the loop
            return anchor_idx, p_idx, n_idx

    def run(self, data: PipelineData) -> PipelineData:
        print("Step: Building representation with Learnable Anisotropic Aggregation...")

        CACHE_DIR = Path("cache")
        # Make sure cache directory exists
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        k = self.config.get("k", 15)
        dataset_name = self.config.get("dataset")
        cache_path = CACHE_DIR / f"{dataset_name}_k{k}_forman_curvature.pkl"
        if cache_path.exists():
            print("  -> Loading pre-computed Forman-Ricci curvature from cache...")
            with open(cache_path, "rb") as f:
                curvature_dict = pickle.load(f)
        else:
            print("  -> No cache found. Pre-computing (this will be slow)...")
            nx_graph = nx.from_scipy_sparse_array(data.graph.W)
            frc = FormanRicci(nx_graph)
            frc.compute_ricci_curvature()
            curvature_dict = {
                (u, v): attr["formanCurvature"] for u, v, attr in frc.G.edges(data=True)
            }
            print("  -> Saving curvature to cache for future runs...")
            with open(cache_path, "wb") as f:
                pickle.dump(curvature_dict, f)

        W_coo = data.graph.W.tocoo()
        W_indices = torch.from_numpy(np.vstack((W_coo.row, W_coo.col))).long()
        W_values = torch.from_numpy(W_coo.data).float()
        W_torch_sparse = torch.sparse_coo_tensor(
            W_indices, W_values, torch.Size(W_coo.shape)
        ).coalesce()

        kappa_values = torch.zeros_like(W_torch_sparse.values())
        for i, (r, c) in enumerate(W_torch_sparse.indices().T):
            u, v = r.item(), c.item()
            kappa_values[i] = curvature_dict.get((u, v), curvature_dict.get((v, u), 0))

        initial_center = torch.median(kappa_values[kappa_values.nonzero()]).squeeze()

        class AnisotropicAttention(torch.nn.Module):
            def __init__(self, initial_center, epsilon=0.01):
                super().__init__()
                self.alpha = torch.nn.Parameter(torch.tensor(1.0))
                self.center = torch.nn.Parameter(torch.tensor(float(initial_center)))
                self.epsilon = epsilon

            def h_theta(self, kappa):
                sensitivity = torch.nn.functional.softplus(self.alpha)
                return self.epsilon + (1 - self.epsilon) * torch.sigmoid(
                    sensitivity * (kappa - self.center)
                )

            def forward(self, W_sparse, kappa_values, X_signal):
                conductance_values = self.h_theta(kappa_values)
                W_prime = torch.sparse_coo_tensor(
                    W_sparse.indices(),
                    W_sparse.values() * conductance_values,
                    W_sparse.shape,
                ).coalesce()

                # --- EFFICIENT SPARSE ROW-NORMALIZATION ---
                D_prime_vec = torch.sparse.sum(W_prime, dim=1).to_dense()
                D_prime_inv_vec = 1.0 / (D_prime_vec + 1e-8)
                D_prime_inv_vec[torch.isinf(D_prime_inv_vec)] = 0
                rows, _ = W_prime.indices()
                D_inv_for_values = D_prime_inv_vec[rows]
                A_prime_values = W_prime.values() * D_inv_for_values
                A_prime = torch.sparse_coo_tensor(
                    W_prime.indices(), A_prime_values, W_prime.shape
                )

                return torch.sparse.mm(A_prime, X_signal)

        device = torch.device("cpu")
        print(f"  -> Beginning training (device: {device})...")
        W_torch_sparse, kappa_values = W_torch_sparse.to(device), kappa_values.to(
            device
        )
        X_signal_torch = torch.from_numpy(data.embeddings).float().to(device)

        model = AnisotropicAttention(initial_center=initial_center).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        loss_fn = torch.nn.TripletMarginLoss(margin=0.5)

        num_epochs = self.config.get("learnable_epochs", 100)

        epoch_iterator = tqdm(range(num_epochs), desc="Training Progress")
        for epoch in epoch_iterator:
            model.train()
            X_prime = model(W_torch_sparse, kappa_values, X_signal_torch)

            batch_loss = 0
            num_triplets = 500
            for _ in range(num_triplets):
                a_idx, p_idx, n_idx = self.get_triplet(data.labels_true)
                anchor, positive, negative = (
                    X_prime[a_idx],
                    X_prime[p_idx],
                    X_prime[n_idx],
                )
                batch_loss += loss_fn(anchor, positive, negative)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            current_loss = batch_loss.item() / num_triplets
            epoch_iterator.set_description(
                f"Training Progress (Loss: {current_loss:.4f})"
            )

        print("\n  -> Training complete.")
        print("  -> Computing final representation...")
        model.eval()
        with torch.no_grad():
            final_rep = model(W_torch_sparse, kappa_values, X_signal_torch)

        data.representation = final_rep.cpu().numpy()
        return data


class DAGNRepresentationBuilder(RepresentationBuilder):
    """Builds a representation by learning an optimal anisotropic graph operator.

    This is the core learnable model of the project, which will be evolved into
    the full Deep Anisotropic Graph Network (DAGN). It abandons hand-crafted
    heuristics and instead learns a direct mapping from Forman-Ricci curvature
    to the graph's diffusion properties.

    The model is trained with a triplet margin loss to produce an embedding space
    where documents from the same class are closer together and documents from
    different classes are farther apart.
    """

    def get_triplet(self, y: np.ndarray) -> tuple[int, int, int]:
        """Correctly samples a triplet (anchor, positive, negative).

        This helper function robustly samples a valid triplet of indices,
        ensuring that the anchor, positive, and negative samples are all
        distinct and that the positive sample belongs to the same class as
        the anchor while the negative sample does not.

        Parameters
        ----------
        y : np.ndarray
            The array of ground-truth labels.

        Returns
        -------
        tuple[int, int, int]
            A tuple containing the anchor, positive, and negative indices.
        """
        pass

    def run(self, data: PipelineData) -> PipelineData:
        """Trains the learnable operator and computes the final representation.

        This method orchestrates the entire learning pipeline:
        1. Loads or computes the Forman-Ricci curvature cache.
        2. Converts the graph and signal to PyTorch tensors.
        3. Initializes and trains the AnisotropicAttention model.
        4. Performs inference with the trained model to get the final representation.

        Parameters
        ----------
        data : PipelineData
            The pipeline data object.


        -------
        PipelineData
            The data object with the `representation` attribute populated by the
            learned, geometry-aware embeddings.
        """
        pass
