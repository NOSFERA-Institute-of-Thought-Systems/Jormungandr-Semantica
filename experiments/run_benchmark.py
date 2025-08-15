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
from bertopic import BERTopic
from hdbscan import HDBSCAN

# Import all the final, correct pipeline steps
from jormungandr_semantica.pipeline.steps import (
    PipelineData,
    FaissGraphConstructor,
    DirectRepresentationBuilder,
    WaveletRepresentationBuilder,
    ACMWRepresentationBuilder,
    GraphUMAPReducer,
    FeatureUMAPReducer,
    KMeansClusterer,
    HDBSCANClusterer,
    CommunitySGWTRepresentationBuilder,
    RankSGWTRepresentationBuilder
)

def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Global random seed set to {seed}")

def get_git_hash() -> str | None:
    """Gets the current git commit hash."""
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha
    except git.InvalidGitRepositoryError:
        return None

def run_experiment(args):
    """Main function to run a single experiment configuration."""
    set_seed(args.seed)
    config = vars(args)
    config["git_hash"] = get_git_hash()
    config["run_timestamp_utc"] = datetime.utcnow().isoformat()
    
    if isinstance(config.get('wavelet_scales'), str):
        config['wavelet_scales'] = [int(s) for s in config['wavelet_scales'].split(',')]

    job_type = "ablation" if args.method == "jormungandr" else "baseline"
    
    run = wandb.init(
        project="Jormungandr-Semantica",
        job_type=job_type,
        config=config
    )
    print(f"W&B Run URL: {run.url}")

    DATA_DIR = Path("data")
    embeddings = np.load(DATA_DIR / f"{args.dataset}_embeddings.npy")
    df = pd.read_csv(DATA_DIR / f"{args.dataset}_labels.csv")
    data = PipelineData(docs=df["text"].tolist(), embeddings=embeddings, labels_true=df["label"].to_numpy())
    num_clusters = len(np.unique(data.labels_true))
    
    print(f"Data loaded: {len(data.docs)} documents, {data.embeddings.shape[1]} dims, {num_clusters} classes.")
    print(f"\nRunning experiment with method: {args.method}")
    start_time = time.time()
    
    if args.method == "jormungandr":
        pipeline_steps = [FaissGraphConstructor(config)]
        
        if args.representation == "direct":
            pipeline_steps.append(DirectRepresentationBuilder(config))
            pipeline_steps.append(GraphUMAPReducer(config))
        elif args.representation == "wavelet":
            pipeline_steps.append(WaveletRepresentationBuilder(config))
            pipeline_steps.append(FeatureUMAPReducer(config))
        elif args.representation == "cpal":
            pipeline_steps.append(ACMWRepresentationBuilder(config))
            pipeline_steps.append(FeatureUMAPReducer(config))      
        elif args.representation == "community": # The new option
            pipeline_steps.append(CommunitySGWTRepresentationBuilder(config))
            pipeline_steps.append(FeatureUMAPReducer(config))   
        elif args.representation == "rank":
            pipeline_steps.append(RankSGWTRepresentationBuilder(config))
            pipeline_steps.append(FeatureUMAPReducer(config))
                   
        # --- THE CORRECTED LOGIC ---
        # Select one and only one clusterer based on the argument.
        if args.clusterer == "kmeans":
            pipeline_steps.append(KMeansClusterer(config))
        elif args.clusterer == "hdbscan":
            pipeline_steps.append(HDBSCANClusterer(config))
        
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
    result_data = {**config, "results": {"ARI": ari_score, "runtime_seconds": duration_seconds}}
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(result_data, f, indent=4)
    print(f"Run manifest saved to: {output_dir / 'manifest.json'}")
    run.finish()

def main():
    """Parses arguments and runs the experiment."""
    parser = argparse.ArgumentParser(description="Run a modular benchmark/ablation experiment.")
    parser.add_argument("--method", type=str, required=True, choices=["jormungandr", "bertopic", "hdbscan"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["20newsgroups", "agnews"])
    
    parser.add_argument("--representation", type=str, default="direct", 
                        choices=["direct", "wavelet", "cpal", "community", "rank"],
                        help="Representation builder for the Jörmungandr pipeline.")

    parser.add_argument("--rank_quantile", type=float, default=0.1, help="Quantile of edges to modulate based on curvature rank.")
    parser.add_argument("--rank_enhancement", type=float, default=1.5, help="Factor to boost high-curvature edges.")
    parser.add_argument("--rank_dampening", type=float, default=0.5, help="Factor to dampen low-curvature edges.")

    parser.add_argument("--cpal_alpha", type=float, default=4.0)
    parser.add_argument("--cpal_beta", type=float, default=1.0) # beta is not used in new func, but good practice
    parser.add_argument("--cpal_epsilon", type=float, default=0.01)    
    parser.add_argument("--community_epsilon", type=float, default=0.1,
                        help="Dampening factor for inter-community edges.")
    # --- ADD THE NEW ARGUMENT PARSING ---
    parser.add_argument("--clusterer", type=str, default="kmeans", choices=["kmeans", "hdbscan"],
                        help="Clustering algorithm to use at the end of the pipeline.")
    
    parser.add_argument("--k", type=int, default=15, help="Number of nearest neighbors for the graph.")
    parser.add_argument("--umap_dims", type=int, default=5, help="Target dimensionality for UMAP.")
    parser.add_argument("--wavelet_scales", type=str, default="5,15,50,100", 
                        help="Comma-separated list of wavelet scales.")
    parser.add_argument("--n_eigenvectors", type=int, default=200,
                        help="Number of eigenvectors to compute for spectral methods.")
    parser.add_argument("--min_cluster_size", type=int, default=15, help="min_cluster_size for HDBSCAN.")
    
    args = parser.parse_args()
    
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    
    run_experiment(args)

if __name__ == "__main__":
    main()