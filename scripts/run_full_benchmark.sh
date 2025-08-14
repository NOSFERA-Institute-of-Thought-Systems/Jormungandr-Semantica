#!/bin/bash
set -e

echo "--- Starting Full Benchmark Suite ---"

DATASETS=("20newsgroups" "agnews")
METHODS=("jormungandr" "bertopic" "hdbscan")
SEEDS=($(seq 42 51))

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT="$SCRIPT_DIR/.."

# --- CRITICAL FIX: Set OpenMP to single-threaded to prevent crashes ---
export OMP_NUM_THREADS=1

for dataset in "${DATASETS[@]}"; do
  for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo ""
      echo "================================================================="
      echo "RUNNING: Dataset=${dataset}, Method=${method}, Seed=${seed}"
      echo "================================================================="
      
      python "$PROJECT_ROOT/experiments/run_benchmark.py" \
        --dataset "$dataset" \
        --method "$method" \
        --seed "$seed"
        
      echo "-----------------------------------------------------------------"
      echo "COMPLETED: Dataset=${dataset}, Method=${method}, Seed=${seed}"
      echo "-----------------------------------------------------------------"
      sleep 2
    done
  done
done

echo ""
echo "--- Full Benchmark Suite Finished Successfully ---"