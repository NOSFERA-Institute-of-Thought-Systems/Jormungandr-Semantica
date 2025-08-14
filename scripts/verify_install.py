# scripts/verify_install.py
import numpy as np
import jormungandr_semantica
import wandb # Import the W&B library

def main():
    """
    Runs a quick verification test of the C++ backend and logs it to W&B.
    """
    
    # --- 1. Initialize a new W&B Run ---
    # All W&B logging for a single experiment happens within a run.
    # `project` is the name of the project on your W&B dashboard.
    # `job_type` helps you group runs (e.g., "verification", "benchmark").
    run = wandb.init(
        project="Jormungandr-Semantica", 
        job_type="verification"
    )

    print("--- Jörmungandr-Semantica Installation Verification ---")
    
    # --- 2. Define and Log Hyperparameters ---
    # We use a dictionary for all our settings. W&B will log this automatically.
    config = {
        "n_points": 100,
        "n_dims": 16,
        "k": 10,
        "data_source": "random_uniform"
    }
    wandb.config.update(config)

    print("Creating simple sample data...")
    data = np.random.rand(config["n_points"], config["n_dims"]).astype('float32')
    
    print(f"Building k-NN graph with k={config['k']}...")
    try:
        neighbors, distances = jormungandr_semantica.build_faiss_knn_graph(data, config["k"])

        print("\n--- Verification ---")
        print(f"Neighbors shape: {neighbors.shape}")
        print(f"Distances shape: {distances.shape}")

        assert neighbors.shape == (config["n_points"], config["k"])
        assert distances.shape == (config["n_points"], config["k"])
        
        # --- 3. Log Results and Metrics ---
        # Log summary metrics about the run. These will be plotted on your dashboard.
        # Here, we'll log the average distance to the nearest neighbor.
        avg_dist_k1 = np.mean(distances[:, 0])
        wandb.log({"avg_dist_k1": avg_dist_k1})

        # Mark the run as successful
        run.finish(exit_code=0)

        print("\n\n******************************************************")
        print("****      SUCCESS: FOUNDATION IS COMPLETE       ****")
        print("******************************************************")
        print(f"**** Find this run at: {run.url} ****")

    except Exception as e:
        # Mark the run as failed
        run.finish(exit_code=1)
        print(f"\nAn error occurred during execution: {e}")
        print("****      VERIFICATION FAILED!                ****")

if __name__ == "__main__":
    main()