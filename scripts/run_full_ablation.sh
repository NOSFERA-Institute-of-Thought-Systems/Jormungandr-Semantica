#!/bin/bash

# ==============================================================================
# Jörmungandr-Semantica: Full Ablation Suite (Phase 2)
# ==============================================================================
# This script systematically runs the Jörmungandr pipeline with different
# representation builders to quantify the performance contribution of each
# geometric innovation (Wavelets, ACMW).
#
# It executes on the 20 Newsgroups dataset, as the exact eigendecomposition
# required for these methods is not scalable to the larger AG News dataset.
# This scalability limitation is the primary motivation for Phase 3.
# ==============================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Starting Full Ablation Suite (Phase 2) ---"

# --- Configuration ---
DATASET="20newsgroups"
# Run across 10 distinct random seeds for statistical significance
SEEDS=($(seq 42 51)) 
# The different representation methods to ablate
# REPRESENTATIONS=("direct" "wavelet" "acmw") 
REPRESENTATIONS=("acmw") 
# For a graph of ~11k nodes, 200 eigenvectors is a reasonable compromise
# between computational speed and spectral accuracy.
N_EIGENVECTORS=200

# --- Execution Loop ---
for representation in "${REPRESENTATIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo ""
    echo "================================================================="
    echo "RUNNING: Dataset=${DATASET}, Representation=${representation}, Seed=${seed}"
    echo "================================================================="
    
    # Run the benchmark script with the specified configuration.
    # We enforce single-threaded execution to prevent library conflicts.
    OMP_NUM_THREADS=1 python experiments/run_benchmark.py \
      --method "jormungandr" \
      --dataset "$DATASET" \
      --representation "$representation" \
      --seed "$seed" \
      --n_eigenvectors "$N_EIGENVECTORS"
      
    echo "-----------------------------------------------------------------"
    echo "COMPLETED: Representation=${representation}, Seed=${seed}"
    echo "-----------------------------------------------------------------"
    # Add a small delay to prevent overwhelming the wandb sync process
    sleep 2
  done
done

echo ""
echo "--- Full Ablation Suite Finished Successfully ---"
echo "All runs have been logged to Weights & Biases."
echo "You can now run 'notebooks/phase2_ablation_analysis.ipynb' to generate the summary table."