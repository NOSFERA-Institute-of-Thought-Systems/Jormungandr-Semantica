#!/bin/bash

# This script runs the full benchmark suite required for Phase 1 analysis.
# It iterates through datasets, methods, and random seeds.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Starting Full Benchmark Suite ---"

# --- CONFIGURATION ---
DATASETS=("20newsgroups" "agnews")
METHODS=("jormungandr" "bertopic" "hdbscan")
# Define a range of 10 seeds for statistical robustness
SEEDS=($(seq 42 51))
# --------------------

# Get the directory of this script to ensure we can call python correctly
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT="$SCRIPT_DIR/.."

# Loop over each dataset
for dataset in "${DATASETS[@]}"; do
  # Loop over each method
  for method in "${METHODS[@]}"; do
    # Loop over each seed
    for seed in "${SEEDS[@]}"; do
      echo ""
      echo "================================================================="
      echo "RUNNING: Dataset=${dataset}, Method=${method}, Seed=${seed}"
      echo "================================================================="
      
      # Execute the experiment harness with the current parameters
      # We run from the project root to ensure paths are correct
      python "$PROJECT_ROOT/experiments/run_benchmark.py" \
        --dataset "$dataset" \
        --method "$method" \
        --seed "$seed"
        
      echo "-----------------------------------------------------------------"
      echo "COMPLETED: Dataset=${dataset}, Method=${method}, Seed=${seed}"
      echo "-----------------------------------------------------------------"
      # Add a small delay to avoid overwhelming the W&B sync process if needed
      sleep 2
    done
  done
done

echo ""
echo "--- Full Benchmark Suite Finished Successfully ---"