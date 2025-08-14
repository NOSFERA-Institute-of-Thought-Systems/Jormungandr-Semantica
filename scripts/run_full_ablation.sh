#!/bin/bash
set -e

echo "--- Starting Full Ablation Suite (Phase 2) ---"

# --- MODIFICATION: Run only on 20newsgroups due to scalability limits of eigendecomposition ---
DATASET="20newsgroups"
SEEDS=($(seq 42 51))
REPRESENTATIONS=("direct" "wavelet" "acmw")

# For a graph of ~11k nodes, 200 eigenvectors is a reasonable number
N_EIGENVECTORS=200

for representation in "${REPRESENTATIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo ""
    echo "================================================================="
    echo "RUNNING: Dataset=${DATASET}, Repr=${representation}, Seed=${seed}"
    echo "================================================================="
    
    python experiments/run_benchmark.py \
      --dataset "$DATASET" \
      --method "jormungandr" \
      --representation "$representation" \
      --seed "$seed" \
      --n_eigenvectors "$N_EIGENVECTORS" # Pass the new parameter
      
    echo "-----------------------------------------------------------------"
    echo "COMPLETED: Repr=${representation}, Seed=${seed}"
    echo "-----------------------------------------------------------------"
    sleep 2
  done
done

echo ""
echo "--- Full Ablation Suite Finished Successfully ---"
echo "NOTE: AG News was skipped due to the computational cost of eigendecomposition."
echo "This bottleneck will be solved in Phase 3 with Chebyshev approximations."