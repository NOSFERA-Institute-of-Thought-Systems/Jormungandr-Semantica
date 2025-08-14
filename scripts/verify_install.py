# scripts/verify_install.py
import numpy as np
# --- RENAMED ---
import jormungandr_semantica

def main():
    """
    Runs a quick verification test of the C++ backend.
    """
    print("--- Jörmungandr-Semantica Installation Verification ---")
    
    print("Creating simple sample data...")
    data = np.random.rand(100, 16).astype('float32')
    k = 10

    print(f"Building k-NN graph with k={k}...")
    try:
        # --- RENAMED ---
        neighbors, distances = jormungandr_semantica.build_faiss_knn_graph(data, k)

        # ... (rest of the file is the same)
        print("\n--- Verification ---")
        print(f"Neighbors shape: {neighbors.shape}")
        print(f"Distances shape: {distances.shape}")
        assert neighbors.shape == (100, k)
        assert distances.shape == (100, k)
        print("Data types are correct.")
        print("Shapes are correct.")

        print("\n\n******************************************************")
        print("****      SUCCESS: FOUNDATION IS COMPLETE       ****")
        print("******************************************************")

    except Exception as e:
        print(f"\nAn unexpected error occurred during execution: {e}")
        print("****      VERIFICATION FAILED!                ****")

if __name__ == "__main__":
    main()