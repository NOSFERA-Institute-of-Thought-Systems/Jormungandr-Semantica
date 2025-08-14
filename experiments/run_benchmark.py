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


def set_seed(seed: int):
    """Sets the random seed for all relevant libraries to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # If using CUDA, this is also needed
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
    # 1. Set seed for determinism
    set_seed(args.seed)

    # 2. Setup run configuration for logging
    config = vars(args)  # Convert argparse namespace to a dictionary
    config["git_hash"] = get_git_hash()
    config["run_timestamp_utc"] = datetime.utcnow().isoformat()

    # 3. Initialize MLOps tracking
    run = wandb.init(
        project="Jormungandr-Semantica",
        job_type="benchmark",
        config=config,
    )
    print(f"W&B Run URL: {run.url}")

    # --- Placeholder for Real Experiment Logic ---
    # In the future, this section will load data, run our pipeline, etc.
    # For now, we simulate an experiment to test the harness.
    print(f"\nSimulating experiment for dataset: {args.dataset} with k={args.k}")
    start_time = time.time()
    # Simulate building the graph
    mock_data = np.random.rand(500, 128).astype('float32')
    # We pass the seed to UMAP/KMeans etc. via their 'random_state' params
    # For our C++ backend, its randomness is controlled by np.random.seed()
    neighbors, distances = js.build_faiss_knn_graph(mock_data, k=args.k)
    # Simulate a result
    mock_ari_score = 0.75 + (np.random.rand() * 0.1) # A mock result
    end_time = time.time()
    duration_seconds = end_time - start_time
    print(f"Simulation finished in {duration_seconds:.2f} seconds.")
    print(f"Mock ARI Score: {mock_ari_score:.4f}")
    # ---------------------------------------------

    # 4. Log metrics to W&B
    wandb.log({
        "ARI": mock_ari_score,
        "runtime_seconds": duration_seconds,
    })

    # 5. Save local manifest for this run
    # Create a unique output directory for this specific run
    output_dir = Path(f"outputs/{run.name}-{run.id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Run manifest saved to: {manifest_path}")

    # 6. Finish MLOps run
    run.finish()


def main():
    """Parses command-line arguments and launches the experiment."""
    parser = argparse.ArgumentParser(
        description="Run a deterministic benchmark experiment for Jörmungandr-Semantica."
    )
    # Core arguments for reproducibility and identification
    parser.add_argument("--seed", type=int, required=True, help="Random seed for all operations.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to use (e.g., '20newsgroups').")
    
    # Example hyperparameter for our pipeline
    parser.add_argument("--k", type=int, default=15, help="Number of nearest neighbors for the graph.")
    
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()