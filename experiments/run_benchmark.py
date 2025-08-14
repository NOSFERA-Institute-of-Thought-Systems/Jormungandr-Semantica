import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import git
import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import adjusted_rand_score

# Import our new pipeline tools
from jormungandr_semantica.pipeline.steps import (
    PipelineData,
    FaissGraphConstructor,
    DirectRepresentationBuilder,
    UMAPReducer,
    KMeansClusterer
)
# We will import baselines directly here for simplicity for now
from bertopic import BERTopic
from hdbscan import HDBSCAN


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Global random seed set to {seed}")


def get_git_hash() -> str | None:
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha
    except git.InvalidGitRepositoryError:
        return None


def run_experiment(args):
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

    print(f"Loading pre-computed data for: {args.dataset}")
    DATA_DIR = Path("data")
    embeddings = np.load(DATA_DIR / f"{args.dataset}_embeddings.npy")
    df = pd.read_csv(DATA_DIR / f"{args.dataset}_labels.csv")
    
    data = PipelineData(
        docs=df["text"].tolist(),
        embeddings=embeddings,
        labels_true=df["label"].to_numpy()
    )
    num_clusters = len(np.unique(data.labels_true))
    print(f"Data loaded: {len(data.docs)} documents, {data.embeddings.shape[1]} dims, {num_clusters} classes.")
    
    print(f"\nRunning experiment with method: {args.method}")
    start_time = time.time()
    
    if args.method == "jormungandr":
        # --- Build and Run the Jörmungandr Pipeline ---
        pipeline_steps = [
            FaissGraphConstructor(config),
            DirectRepresentationBuilder(config), # This will be replaced by a Wavelet step later
            UMAPReducer(config),
            KMeansClusterer(config)
        ]
        
        # Sequentially run each step
        for step in pipeline_steps:
            data = step.run(data)
        
        labels_pred = data.labels_pred

    elif args.method == "bertopic":
        hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean')
        topic_model = BERTopic(hdbscan_model=hdbscan_model, verbose=False, nr_topics=num_clusters)
        _, _ = topic_model.fit_transform(data.docs, data.embeddings)
        labels_pred = topic_model.topics_

    elif args.method == "hdbscan":
        clusterer = HDBSCAN(min_cluster_size=15)
        labels_pred = clusterer.fit_predict(data.embeddings)
        
    else:
        raise ValueError(f"Unknown method: {args.method}")

    end_time = time.time()
    duration_seconds = end_time - start_time

    valid_indices = labels_pred != -1
    ari_score = adjusted_rand_score(data.labels_true[valid_indices], labels_pred[valid_indices])
    
    print(f"\nMethod finished in {duration_seconds:.2f} seconds.")
    print(f"Final Adjusted Rand Index (ARI): {ari_score:.4f}")

    wandb.log({"ARI": ari_score, "runtime_seconds": duration_seconds})

    output_dir = Path(f"outputs/{run.name}-{run.id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_data = {**config, "results": {"ARI": ari_score}}
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(result_data, f, indent=4)
    print(f"Run manifest saved to: {output_dir / 'manifest.json'}")

    run.finish()


def main():
    parser = argparse.ArgumentParser(description="Run a deterministic benchmark experiment.")
    parser.add_argument("--method", type=str, required=True, choices=["jormungandr", "bertopic", "hdbscan"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["20newsgroups", "agnews"])
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--umap_dims", type=int, default=5)
    
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()