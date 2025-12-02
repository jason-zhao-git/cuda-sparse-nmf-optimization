#!/bin/bash
#
# NMF Benchmark Suite
# Run all benchmarks with convergence logging for report
#
# Usage:
#   ./scripts/run_benchmarks.sh              # Run all benchmarks
#   ./scripts/run_benchmarks.sh --generate   # Only generate matrices
#   ./scripts/run_benchmarks.sh --single-gpu # Only single-GPU tests
#   ./scripts/run_benchmarks.sh --multi-gpu  # Only multi-GPU tests (cluster)
#

set -e  # Exit on error

# Configuration
SIZES="500 1000 2000 4000"
RANK=20
MU_ITERS=100
HALS_ITERS=50
LOG_INTERVAL=10

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "   NMF Benchmark Suite"
echo "=============================================="
echo ""

# Parse arguments
GENERATE_ONLY=false
SINGLE_GPU_ONLY=false
MULTI_GPU_ONLY=false

for arg in "$@"; do
    case $arg in
        --generate)
            GENERATE_ONLY=true
            ;;
        --single-gpu)
            SINGLE_GPU_ONLY=true
            ;;
        --multi-gpu)
            MULTI_GPU_ONLY=true
            ;;
    esac
done

# Step 1: Generate low-rank test matrices
generate_matrices() {
    echo -e "${YELLOW}[1/4] Generating low-rank test matrices...${NC}"
    python3 scripts/generate_lowrank_matrices.py --sizes $SIZES --rank $RANK
    echo ""
}

# Step 2: Run single-GPU benchmarks
run_single_gpu() {
    echo -e "${YELLOW}[2/4] Running single-GPU benchmarks...${NC}"

    mkdir -p results

    for size in $SIZES; do
        echo ""
        echo -e "${GREEN}=== Matrix size: ${size}x${size} ===${NC}"

        # MU L2 (Memory-Optimized)
        echo "Running MU L2..."
        ./nmf_memory_opt data/lowrank_${size}.bin $RANK $MU_ITERS --log-convergence $LOG_INTERVAL
        mv results/convergence_mu_l2.csv results/convergence_mu_l2_${size}.csv 2>/dev/null || true

        # MU L3 (Compute-Optimized)
        echo "Running MU L3..."
        ./nmf_compute_opt data/lowrank_${size}.bin $RANK $MU_ITERS 128 --log-convergence $LOG_INTERVAL
        mv results/convergence_mu_l3.csv results/convergence_mu_l3_${size}.csv 2>/dev/null || true

        # HALS Block-Parallel
        echo "Running HALS Block..."
        ./nmf_hals_gpu_block data/lowrank_${size}.bin $RANK $HALS_ITERS 5 --log-convergence 5
        mv results/convergence_hals_block.csv results/convergence_hals_block_${size}.csv 2>/dev/null || true
    done

    echo ""
    echo -e "${GREEN}Single-GPU benchmarks complete!${NC}"
}

# Step 3: Run multi-GPU benchmarks (cluster only)
run_multi_gpu() {
    echo -e "${YELLOW}[3/4] Running multi-GPU benchmarks...${NC}"

    # Check number of GPUs
    NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -1 || echo "1")

    if [ "$NUM_GPUS" -lt 2 ]; then
        echo -e "${RED}Warning: Only $NUM_GPUS GPU(s) detected. Skipping multi-GPU tests.${NC}"
        echo "Run this script on a cluster with multiple GPUs."
        return
    fi

    echo "Detected $NUM_GPUS GPUs"

    mkdir -p results

    # Test with 2 GPUs
    for size in 1000 2000; do
        echo ""
        echo -e "${GREEN}=== Multi-GPU: ${size}x${size}, 2 GPUs ===${NC}"
        ./nmf_multigpu data/lowrank_${size}.bin $RANK $MU_ITERS 2 --log-convergence $LOG_INTERVAL
        mv results/convergence_mu_l4.csv results/convergence_mu_l4_${size}_2gpu.csv 2>/dev/null || true
    done

    # Test with 4 GPUs if available
    if [ "$NUM_GPUS" -ge 4 ]; then
        for size in 2000 4000; do
            echo ""
            echo -e "${GREEN}=== Multi-GPU: ${size}x${size}, 4 GPUs ===${NC}"
            ./nmf_multigpu data/lowrank_${size}.bin $RANK $MU_ITERS 4 --log-convergence $LOG_INTERVAL
            mv results/convergence_mu_l4.csv results/convergence_mu_l4_${size}_4gpu.csv 2>/dev/null || true
        done
    fi

    echo ""
    echo -e "${GREEN}Multi-GPU benchmarks complete!${NC}"
}

# Step 4: Generate summary
generate_summary() {
    echo -e "${YELLOW}[4/4] Generating summary...${NC}"

    echo ""
    echo "=============================================="
    echo "   Benchmark Results Summary"
    echo "=============================================="
    echo ""
    echo "Output files in results/:"
    ls -la results/convergence_*.csv 2>/dev/null || echo "No convergence files found"
    echo ""
    echo "Metrics files:"
    ls -la results/*_metrics.txt 2>/dev/null || echo "No metrics files found"
    echo ""
    echo -e "${GREEN}Next step: Open notebooks/nmf_report_analysis.ipynb to generate plots${NC}"
}

# Main execution
if [ "$GENERATE_ONLY" = true ]; then
    generate_matrices
elif [ "$SINGLE_GPU_ONLY" = true ]; then
    run_single_gpu
    generate_summary
elif [ "$MULTI_GPU_ONLY" = true ]; then
    run_multi_gpu
    generate_summary
else
    generate_matrices
    run_single_gpu
    run_multi_gpu
    generate_summary
fi

echo ""
echo "=============================================="
echo "   Benchmark Complete!"
echo "=============================================="
