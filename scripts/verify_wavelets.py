# scripts/verify_wavelets.py
import numpy as np
import pygsp.graphs as graphs
# --- RENAMED ---
from jormungandr_semantica import compute_heat_wavelets

def main():
    """
    Runs a quick verification test of the PyGSP wavelet wrapper.
    """
    print("--- Jörmungandr-Semantica Wavelet Module Verification ---")
    # ... (rest of the file is the same)
    n_nodes = 50
    n_features = 4
    scales_to_test = [10, 20, 30, 40, 50]
    n_scales = len(scales_to_test)

    print(f"Creating a toy graph with {n_nodes} nodes...")
    G = graphs.Ring(N=n_nodes)
    G.compute_fourier_basis()

    print(f"Creating a random signal with shape ({n_nodes}, {n_features})...")
    signal = np.random.rand(n_nodes, n_features)

    print(f"Computing wavelets at {n_scales} scales...")
    try:
        coeffs = compute_heat_wavelets(G, signal, scales=scales_to_test)

        print("\n--- Verification ---")
        expected_shape = (n_nodes, n_features, n_scales)
        print(f"Input signal shape:   {signal.shape}")
        print(f"Expected coeffs shape: {expected_shape}")
        print(f"Actual coeffs shape:   {coeffs.shape}")
        
        assert coeffs.shape == expected_shape
        
        print("\n\n********************************************************")
        print("**** SUCCESS: WAVELET MODULE IS WORKING CORRECTLY ****")
        print("********************************************************")

    except Exception as e:
        print(f"\nAn error occurred during wavelet computation: {e}")
        print("****      VERIFICATION FAILED!                  ****")

if __name__ == "__main__":
    main()