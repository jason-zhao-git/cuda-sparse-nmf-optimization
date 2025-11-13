#!/bin/bash
#SBATCH --account=cse587f25s001_class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --mem=16GB
#SBATCH --job-name=nmf_benchmark
#SBATCH --output=results/benchmark_%j.log

# ============================================================================
# NMF Sparse vs Dense Benchmarking Script
# ============================================================================

echo "=========================================="
echo "NMF Benchmark Suite"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

# Load required modules
module load cuda/12.2.0
module load gcc/11.2.0
module load python/3.10

echo "Loaded modules:"
module list
echo ""

# Check GPU
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
echo ""

# ============================================================================
# Configuration
# ============================================================================

SIZES="1000 5000 10000"
SPARSITIES="50 70 80 85 90 95 99"
K=20
ITERS=100

OUTPUT_FILE="results/results.csv"

# Create output directory
mkdir -p results

# Write CSV header
echo "size,sparsity,method,time_ms,memory_mb,final_error,nnz" > $OUTPUT_FILE

echo "Configuration:"
echo "  Sizes: $SIZES"
echo "  Sparsities: $SPARSITIES"
echo "  Rank (k): $K"
echo "  Iterations: $ITERS"
echo ""

# ============================================================================
# Main Benchmark Loop
# ============================================================================

TOTAL_TESTS=$(($(echo $SIZES | wc -w) * $(echo $SPARSITIES | wc -w) * 2))
CURRENT_TEST=0

for size in $SIZES; do
    for sp in $SPARSITIES; do
        echo "=========================================="
        echo "Testing: ${size}×${size}, ${sp}% sparse"
        echo "=========================================="

        # ----------------------------------------------------------------------
        # Dense GPU
        # ----------------------------------------------------------------------
        CURRENT_TEST=$((CURRENT_TEST + 1))
        echo "[$CURRENT_TEST/$TOTAL_TESTS] Dense GPU..."

        # For dense, we just use a random matrix
        ./nmf_dense_gpu $size $K $ITERS 2>&1 | tee -a results/dense_${size}_sp${sp}.log

        # Extract timing (you'll need to parse your output)
        # For now, we'll record manually - you can enhance this

        # ----------------------------------------------------------------------
        # Sparse GPU
        # ----------------------------------------------------------------------
        CURRENT_TEST=$((CURRENT_TEST + 1))
        echo "[$CURRENT_TEST/$TOTAL_TESTS] Sparse GPU..."

        # Convert sparsity percentage to fraction
        SPARSITY=$(echo "scale=2; $sp / 100" | bc)

        ./nmf_sparse_gpu $size $K $SPARSITY $ITERS 2>&1 | tee -a results/sparse_${size}_sp${sp}.log

        echo ""
    done
done

# ============================================================================
# Summary
# ============================================================================

echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="
echo "End time: $(date)"
echo "Results saved to: $OUTPUT_FILE"
echo ""

# Display first few lines of results
if [ -f "$OUTPUT_FILE" ]; then
    echo "Preview of results:"
    head -20 $OUTPUT_FILE
fi

echo ""
echo "Next steps:"
echo "  1. Download results: scp greatlakes:~/nmf-sparse-dense/results/results.csv ."
echo "  2. Visualize: python scripts/plot_results.py"
echo "  3. Analyze: check results/profiles/ for detailed profiling"
