#!/bin/bash
# File: scripts/run_full_ablation.sh

# ==============================================================================
# AGLT: Heuristics Falsification Suite
# ==============================================================================
# This script systematically runs the AGLT pipeline with different
# hand-crafted heuristic representation builders to quantify their
# performance against the isotropic baseline.
# ==============================================================================

set -e

echo "--- Starting Heuristics Falsification Suite ---"

# --- Configuration ---
DATASET="20newsgroups"
SEEDS=$(seq 42 51) # Run across 10 distinct seeds
# The different representation methods to ablate, including the baseline
REPRESENTATIONS=("wavelet" "acmw" "community" "rank")
N_EIGENVECTORS=200

# --- Execution Loop ---
for representation in "${REPRESENTATIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo ""
    echo "================================================================="
    echo "RUNNING: Dataset=${DATASET}, Representation=${representation}, Seed=${seed}"
    echo "================================================================="

    # Run the consolidated benchmark script
    # Use python -u for unbuffered output to see logs in real-time
    python -u experiments/run_benchmark.py \
      --method "jormungandr" \
      --dataset "$DATASET" \
      --representation "$representation" \
      --clusterer "hdbscan" \
      --seed "$seed" \
      --n_eigenvectors "$N_EIGENVECTORS"

    echo "-----------------------------------------------------------------"
    echo "COMPLETED: Representation=${representation}, Seed=${seed}"
    echo "-----------------------------------------------------------------"
    sleep 2 # Small delay for W&B syncing
  done
done

echo ""
echo "--- Heuristics Falsification Suite Finished Successfully ---"
echo "All runs have been logged to Weights & Biases."
