# experiments/run_theorem_4_3_validation.py
import networkx as nx
import numpy as np
import pandas as pd
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from pathlib import Path

def create_barbell_graph(m: int) -> nx.Graph:
    """Creates a barbell graph of two m-cliques connected by a single edge."""
    G = nx.barbell_graph(m1=m, m2=0) # m1 is the clique size, m2 is the path length (0)
    return G

def compute_spectral_gap(G: nx.Graph) -> float:
    """Computes the spectral gap (lambda_2) of the graph's normalized Laplacian."""
    L = nx.normalized_laplacian_matrix(G)
    eigenvalues = np.linalg.eigvalsh(L.toarray())
    # The eigenvalues are sorted, the second one is the gap
    return eigenvalues[1]

def compute_bridge_curvature(G: nx.Graph, bridge_edge: tuple) -> float:
    """Computes the Ollivier-Ricci curvature of the specified bridge edge."""
    # The OllivierRicci library expects an alpha parameter to balance original vs. neighbor distribution
    # alpha=0 gives the original definition.
    orc = OllivierRicci(G, alpha=0, verbose="INFO")
    orc.compute_ricci_curvature()
    
    # The library stores curvature on edges in a new graph attribute
    for u, v, data in orc.G.edges(data=True):
        if (u, v) == bridge_edge or (v, u) == bridge_edge:
            return data['ricciCurvature']
    return float('nan') # Should not happen

def main():
    """
    Runs the synthetic experiment to validate Theorem 4.3.
    """
    print("--- Validation Harness for Theorem 4.3: Curvature vs. Spectral Gap ---")
    
    # We will test for a range of clique sizes
    clique_sizes = np.arange(5, 51, 2) # From 5x5 to 50x50 cliques
    results = []

    for m in clique_sizes:
        print(f"\nProcessing barbell graph with clique size m={m}...")
        
        G = create_barbell_graph(m)
        
        # The bridge edge connects node (m-1) and node (m) in NetworkX's construction
        bridge_edge = (m - 1, m)
        
        spectral_gap = compute_spectral_gap(G)
        curvature = compute_bridge_curvature(G, bridge_edge)
        
        results.append({
            "m": m,
            "lambda_2": spectral_gap,
            "kappa_bridge": curvature
        })
        
        print(f"  -> Spectral Gap (lambda_2): {spectral_gap:.4f}")
        print(f"  -> Bridge Curvature (kappa): {curvature:.4f}")

    # Save results to a CSV file for plotting
    results_df = pd.DataFrame(results)
    output_path = Path("outputs/theorem_4_3_validation_data.csv")
    output_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    print(f"\n--- Experiment Complete ---")
    print(f"Validation data saved to {output_path}")

if __name__ == "__main__":
    main()