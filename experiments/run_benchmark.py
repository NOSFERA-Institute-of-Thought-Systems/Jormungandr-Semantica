import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import git
import jormungandr_semantica as js
import numpy as np
import torch
import wandb
from bertopic import BERTopic
from hdbscan import HDBSCAN


def set_seed(seed: int):
    """Sets the random seed for all relevant libraries to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Global random seed set to {seed}")


def get_git_hash() -> str | None:
    """Gets the current git hash of the repository."""
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha
    except git.InvalidGitRepositoryError:
        return None


def run_experiment(args):
    """
    Main function to run a single, fully-reproducible experiment.
    """
    set_seed(args.seed)

    config = vars(args)
    config["git_hash"] = get_git_hash()
    config["run_timestamp_utc"] = datetime.utcnow().isoformat()

    run = wandb.init(
        project="Jormungandr-Semantica",
        job_type="benchmark",
        config=config,
    )
    print(f"W&B Run URL: {run.url}")

    # --- Placeholder for Data Loading ---
    # In a real run, this would load text data and pre-computed embeddings
    print(f"Simulating data loading for: {args.dataset}")
    mock_embeddings = np.random.rand(500, 384) # e.g., from sentence-transformers
    mock_docs = ["doc_" + str(i) for i in range(500)]
    # ------------------------------------

    print(f"\nRunning experiment with method: {args.method}")
    start_time = time.time()

    # --- Method Dispatcher ---
    # This block selects which algorithm to run based on the --method argument.
    if args.method == "jormungandr":
        # This is where our full pipeline will go.
        # For now, we simulate it.
        _ = js.build_faiss_knn_graph(mock_embeddings.astype('float32'), k=args.k)
        # In the future: build graph -> compute wavelets -> UMAP -> cluster
        mock_ari_score = 0.75 + (np.random.rand() * 0.1) # Our method's mock score
        print("Jörmungandr pipeline finished.")

    elif args.method == "bertopic":
        # BERTopic needs a clustering model. We use HDBSCAN for a fair comparison.
        hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', 
                                prediction_data=True)
        topic_model = BERTopic(hdbscan_model=hdbscan_model, verbose=False)
        # BERTopic is deterministic if the underlying models are.
        _, _ = topic_model.fit_transform(mock_docs, mock_embeddings)
        mock_ari_score = 0.70 + (np.random.rand() * 0.1) # BERTopic's mock score
        print("BERTopic pipeline finished.")

    elif args.method == "hdbscan":
        # Just run HDBSCAN on the embeddings directly.
        clusterer = HDBSCAN(min_cluster_size=15)
        _ = clusterer.fit_predict(mock_embeddings)
        mock_ari_score = 0.65 + (np.random.rand() * 0.1) # HDBSCAN's mock score
        print("HDBSCAN pipeline finished.")
        
    else:
        raise ValueError(f"Unknown method: {args.method}")
    # -------------------------

    end_time = time.time()
    duration_seconds = end_time - start_time
    print(f"Method finished in {duration_seconds:.2f} seconds.")
    print(f"Mock ARI Score: {mock_ari_score:.4f}")

    wandb.log({
        "ARI": mock_ari_score,
        "runtime_seconds": duration_seconds,
    })

    output_dir = Path(f"outputs/{run.name}-{run.id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Run manifest saved to: {manifest_path}")

    run.finish()


def main():
    parser = argparse.ArgumentParser(
        description="Run a deterministic benchmark experiment for Jörmungandr-Semantica."
    )
    # New argument to select the method
    parser.add_argument("--method", type=str, required=True, 
                        choices=["jormungandr", "bertopic", "hdbscan"],
                        help="The clustering method to run.")
                        
    parser.add_argument("--seed", type=int, required=True, help="Random seed.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name.")
    parser.add_argument("--k", type=int, default=15, help="k-NN graph neighbors (for Jörmungandr).")
    
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()