# File: scripts/generate_failure_viz.py

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP

# Import the necessary pipeline steps from our aglt library
from aglt.pipeline.steps import (
    PipelineData,
    FaissGraphConstructor,
    RankSGWTRepresentationBuilder,
)


def main():
    """
    Runs a single pipeline for the RankSGWTRepresentationBuilder to generate
    a 2D UMAP visualization that demonstrates its failure mode.
    """
    print("--- Generating Diagnostic Visualization for Failed Heuristic (RankSGWT) ---")

    # --- 1. Configuration ---
    # We only need one seed. We force UMAP to output 2 dimensions for plotting.
    config = {
        "k": 15,
        "seed": 42,
        "umap_dims": 2,  # CRITICAL: We need a 2D output for the scatter plot
        "rank_quantile": 0.1,
        "rank_enhancement": 1.5,
        "rank_dampening": 0.5,
        "wavelet_scales": [5, 15, 50, 100],
        "n_eigenvectors": 200,
    }

    # --- 2. Load Data ---
    print("Step 1: Loading 20 Newsgroups data...")
    DATA_DIR = Path("data")
    embeddings = np.load(DATA_DIR / "20newsgroups_embeddings.npy")
    df_labels = pd.read_csv(DATA_DIR / "20newsgroups_labels.csv")
    data = PipelineData(
        docs=df_labels["text"].tolist(),
        embeddings=embeddings,
        labels_true=df_labels["label"].to_numpy(),
    )

    # --- 3. Run the Core Pipeline ---
    # We don't need the full benchmark harness, just the necessary steps.
    print("Step 2: Running the pipeline (Graph -> RankSGWT Representation)...")
    graph_builder = FaissGraphConstructor(config)
    data = graph_builder.run(data)

    rep_builder = RankSGWTRepresentationBuilder(config)
    data = rep_builder.run(data)

    # --- 4. Perform Final 2D UMAP Reduction ---
    # The pipeline produces the high-dim representation. Now we reduce it to 2D for plotting.
    print("Step 3: Reducing representation to 2D with UMAP for visualization...")
    reducer = UMAP(
        n_components=2,  # Ensure 2D output
        random_state=config["seed"],
        n_jobs=1,
        min_dist=0.0,
        metric="cosine",
    )
    embedding_2d = reducer.fit_transform(data.representation)

    # --- 5. Create and Save the Plot ---
    print("Step 4: Generating and saving the publication-quality plot...")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 8))

    # We use small, semi-transparent points for clarity with many overlapping classes
    plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=data.labels_true,
        cmap="turbo",  # 'turbo' is a good colormap for many categories
        s=5,
        alpha=0.7,
    )

    plt.title("UMAP of Rank-Modulated Representation (ARI ≈ 0.17)", fontsize=16)
    plt.xlabel("UMAP Component 1", fontsize=12)
    plt.ylabel("UMAP Component 2", fontsize=12)
    plt.grid(False)  # A clean background is better for this type of plot
    plt.xticks([])
    plt.yticks([])

    # Ensure the output directory exists
    output_dir = Path("paper/figures")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "failed_heuristic_visualization.png"

    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    print("\n--- SUCCESS ---")
    print(f"Diagnostic plot saved to: {output_path}")
    print("You can now compile your LaTeX document.")


if __name__ == "__main__":
    main()
