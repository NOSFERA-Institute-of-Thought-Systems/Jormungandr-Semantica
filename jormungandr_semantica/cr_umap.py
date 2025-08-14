# jormungandr_semantica/cr_umap.py
import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix
from sklearn.preprocessing import normalize

def curvature_regularized_umap(
    high_dim_data: np.ndarray,
    graph: nx.Graph,
    n_components: int = 2,
    n_epochs: int = 200,
    learning_rate: float = 1.0,
    gamma: float = 1.0,
    random_state: int = 42
) -> np.ndarray:
    """
    A proof-of-concept implementation of Curvature-Regularized UMAP (CR-UMAP).

    This function performs dimensionality reduction, guided by both the topological
    structure of the data (like standard UMAP) and an explicit geometric prior
    derived from the graph's Ricci curvature.

    Parameters
    ----------
    high_dim_data : np.ndarray
        The data to be embedded, shape (n_samples, n_features).
    graph : networkx.Graph
        The graph representation of the data, with 'ricciCurvature' as an edge attribute.
        It MUST have the same number of nodes as n_samples.
    n_components : int
        The dimension of the space to embed into.
    n_epochs : int
        The number of optimization epochs to run.
    learning_rate : float
        The initial learning rate for the optimization.
    gamma : float
        The weight of the curvature regularization term.
    random_state : int
        Seed for the random initialization of the low-dimensional embedding.

    Returns
    -------
    np.ndarray
        The low-dimensional embedding of the data, shape (n_samples, n_components).
    """
    print("--- Running Curvature-Regularized UMAP (Proof-of-Concept) ---")
    
    # --- 1. Simplified UMAP High-Dimensional Representation ---
    # In real UMAP, this is a complex fuzzy simplicial set. Here, we use
    # the given graph's adjacency matrix as a simpler proxy for p_ij.
    n_samples = high_dim_data.shape[0]
    if n_samples != graph.number_of_nodes():
        raise ValueError("Data and graph must have the same number of nodes.")
        
    p_ij = nx.to_scipy_sparse_array(graph, format='coo')
    p_ij.data = np.ones_like(p_ij.data) # For simplicity, treat all connections equally

    # --- 2. Create the Curvature Regularization Weights ---
    print("Step (CR-UMAP): Calculating curvature regularization weights...")
    avg_curvature = {node: 0.0 for node in graph.nodes()}
    node_degree = {node: 0 for node in graph.nodes()}

    for u, v, data in graph.edges(data=True):
        kappa = data.get('ricciCurvature', 0.0)
        avg_curvature[u] += kappa
        avg_curvature[v] += kappa
        node_degree[u] += 1
        node_degree[v] += 1

    for node in graph.nodes():
        if node_degree[node] > 0:
            avg_curvature[node] /= node_degree[node]

    # The weight w_ij' is the sum of the positive parts of the average curvatures
    reg_rows, reg_cols, reg_weights = [], [], []
    for u, v in graph.edges():
        w_prime = max(0, avg_curvature[u]) + max(0, avg_curvature[v])
        if w_prime > 0:
            reg_rows.append(u)
            reg_cols.append(v)
            reg_weights.append(w_prime)
    
    w_prime_matrix = coo_matrix(
        (reg_weights, (reg_rows, reg_cols)), 
        shape=(n_samples, n_samples)
    ).tocsr()
    w_prime_matrix = w_prime_matrix.maximum(w_prime_matrix.T) # Symmetrize

    # --- 3. Initialize Low-Dimensional Embedding ---
    print("Step (CR-UMAP): Initializing low-dimensional embedding...")
    rng = np.random.RandomState(random_state)
    y = rng.uniform(-10, 10, (n_samples, n_components)).astype(np.float32)

    # --- 4. Optimization Loop (Simplified Stochastic Gradient Descent) ---
    print(f"Step (CR-UMAP): Running optimization for {n_epochs} epochs...")
    edges = np.array(graph.edges())
    # We will use float64 for the embedding for better numerical stability during optimization
    y = y.astype(np.float64)

    for epoch in range(n_epochs):
        # Adaptive learning rate that decays over time
        current_lr = learning_rate * (1.0 - (epoch / n_epochs))

        # --- Attractive force (UMAP + Curvature) ---
        edge_batch = edges[rng.choice(edges.shape[0], size=1000, replace=True)]
        
        for i, j in edge_batch:
            y_i, y_j = y[i], y[j]
            delta = y_i - y_j
            dist_sq = np.sum(np.square(delta))
            
            grad_coeff_umap = 1.0 / (1.0 + dist_sq)
            grad_coeff_cr = gamma * w_prime_matrix[i, j]

            grad_coeff = grad_coeff_umap + grad_coeff_cr
            grad = grad_coeff * delta

            # Gradient Clipping
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 4.0:
                grad = (4.0 / grad_norm) * grad
            
            y[i] -= current_lr * grad
            y[j] += current_lr * grad

        # --- Repulsive force (negative sampling) ---
        neg_i = rng.choice(n_samples, size=1000)
        neg_j = rng.choice(n_samples, size=1000)

        for i, j in zip(neg_i, neg_j):
            if i == j: continue
            y_i, y_j = y[i], y[j]
            delta = y_i - y_j
            dist_sq = np.sum(np.square(delta))
            
            # Add a small epsilon for stability
            grad_coeff = -1.0 / ((0.001 + dist_sq) * (1.0 + dist_sq))
            grad = grad_coeff * delta

            # Gradient Clipping
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 4.0:
                grad = (4.0 / grad_norm) * grad
            
            # Only update one point for negative samples
            y[i] -= current_lr * grad
            
        if epoch % (n_epochs // 10) == 0:
            print(f"  Epoch {epoch}/{n_epochs}")
    
    # Clip any potential runaway values before normalizing
    y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)

    print("Step (CR-UMAP): Optimization complete.")
    return normalize(y.astype(np.float32)) # Convert back to float32 at the end
