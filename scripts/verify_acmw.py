# scripts/verify_acmw.py
import numpy as np
import pygsp.graphs as graphs
from jormungandr_semantica import compute_acmw_wavelets

def main():
    """
    Runs a quick verification test of the Anisotropic Curvature-Modulated Wavelet.
    """
    print("--- Jörmungandr-Semantica ACMW Module Verification ---")

    # 1. Define graph and signal parameters
    n_nodes = 30 # Use a smaller graph as curvature calculation can be slow
    n_features = 2
    scales_to_test = [10, 20]
    n_scales = len(scales_to_test)

    # 2. Create a toy graph (a path graph, which has interesting curvature)
    print(f"Creating a toy graph with {n_nodes} nodes...")
    G = graphs.Path(N=n_nodes)

    # 3. Create a random signal on the graph nodes
    print(f"Creating a random signal with shape ({n_nodes}, {n_features})...")
    signal = np.random.rand(n_nodes, n_features)

    # 4. Call our new ACMW function
    print(f"Computing ACMW at {n_scales} scales...")
    try:
        coeffs = compute_acmw_wavelets(G, signal, scales=scales_to_test)

        # 5. Verify the results
        print("\n--- Verification ---")
        expected_shape = (n_nodes, n_features, n_scales)
        print(f"Input signal shape:   {signal.shape}")
        print(f"Expected coeffs shape: {expected_shape}")
        print(f"Actual coeffs shape:   {coeffs.shape}")
        
        assert coeffs.shape == expected_shape
        
        print("\n\n********************************************************")
        print("**** SUCCESS: ACMW MODULE IS WORKING CORRECTLY ****")
        print("********************************************************")

    except Exception as e:
        print(f"\nAn error occurred during ACMW computation: {e}")
        print("****      VERIFICATION FAILED!                  ****")
        # Print full traceback for debugging
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()