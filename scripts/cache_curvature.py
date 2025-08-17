import numpy as np
import pandas as pd
from pathlib import Path
import pickle

# Import the necessary components from your pipeline
from aglt.pipeline.steps import FaissGraphConstructor, PipelineData
from GraphRicciCurvature.FormanRicci import FormanRicci
import networkx as nx


def main():
    """
    Pre-computes and caches the Forman-Ricci curvature for the 20 Newsgroups graph.

    This script is a crucial optimization step, as curvature computation is
    computationally expensive. By caching the results, subsequent benchmark runs
    (especially those involving the learnable operator) can proceed much faster.
    """

    print("--- Curvature Pre-computation and Caching Script ---")
    DATASET = "20newsgroups"

    # --- 1. Load data and build the graph (same as in the benchmark) ---
    print(f"Step 1: Loading data for {DATASET} and constructing graph...")
    config = {"k": 15}
    DATA_DIR = Path("data")
    embeddings = np.load(DATA_DIR / f"{DATASET}_embeddings.npy")
    df = pd.read_csv(DATA_DIR / f"{DATASET}_labels.csv")
    data = PipelineData(
        docs=df["text"].tolist(),
        embeddings=embeddings,
        labels_true=df["label"].to_numpy(),
    )

    graph_constructor = FaissGraphConstructor(config)
    data_with_graph = graph_constructor.run(data)

    # --- 2. Compute Forman-Ricci Curvature ---
    print("\nStep 2: Computing Forman-Ricci Curvature (this will be slow)...")
    nx_graph = nx.from_scipy_sparse_array(data_with_graph.graph.W)
    frc = FormanRicci(nx_graph)
    frc.compute_ricci_curvature()

    # --- 3. Save the Curvature Data ---
    print("\nStep 3: Saving curvature to cache file...")
    # We will save a simple dictionary mapping edges to curvature
    curvature_dict = {
        (u, v): attr["formanCurvature"] for u, v, attr in frc.G.edges(data=True)
    }

    CACHE_DIR = Path("cache")
    CACHE_DIR.mkdir(exist_ok=True)
    output_path = CACHE_DIR / f"{DATASET}_forman_curvature.pkl"

    with open(output_path, "wb") as f:
        pickle.dump(curvature_dict, f)

    print(
        f"\nSUCCESS: Curvature for {len(curvature_dict)} edges saved to {output_path}"
    )


if __name__ == "__main__":
    main()
