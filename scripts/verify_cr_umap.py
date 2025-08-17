# scripts/verify_cr_umap.py
import numpy as np
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import matplotlib.pyplot as plt
import aglt as js


def main():
    """
    Runs a verification test of the Curvature-Regularized UMAP.
    """
    print("--- Jörmungandr-Semantica CR-UMAP Module Verification ---")

    # 1. Create a synthetic "barbell" dataset and graph
    print("Creating synthetic barbell dataset...")
    n_cluster = 50
    # Two distinct clusters
    cluster1 = np.random.randn(n_cluster, 2) + np.array([5, 0])
    cluster2 = np.random.randn(n_cluster, 2) + np.array([-5, 0])
    # A single "bridge" point
    bridge = np.array([[0, 0]])

    data = np.vstack([cluster1, cluster2, bridge])
    labels = np.array([0] * n_cluster + [1] * n_cluster + [2])

    # Build a graph where the bridge connects the two clusters
    G = nx.Graph()
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            dist = np.linalg.norm(data[i] - data[j])
            if (
                (labels[i] == labels[j] and dist < 1.5)
                or (labels[i] != labels[j] and labels[j] == 2 and dist < 3)
                or (labels[j] != labels[i] and labels[i] == 2 and dist < 3)
            ):
                G.add_edge(i, j)

    # 2. Compute Ricci Curvature
    print("Computing Ricci curvature...")
    orc = OllivierRicci(G, alpha=0.0, verbose="INFO")
    orc.compute_ricci_curvature()

    # 3. Run CR-UMAP
    embedding = js.curvature_regularized_umap(data, orc.G, n_epochs=100, gamma=5.0)

    # 4. Visualize the result
    print("Visualizing the embedding...")
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="viridis", s=50)
    plt.title("CR-UMAP Embedding of Barbell Graph", fontsize=16)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)

    output_path = "outputs/cr_umap_verification.png"
    plt.savefig(output_path, dpi=150)
    print(f"SUCCESS: Verification plot saved to {output_path}")


if __name__ == "__main__":
    # Run in single-threaded mode to prevent any conflicts
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    main()
