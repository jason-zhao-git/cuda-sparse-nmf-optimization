#!/bin/bash
#
# Comprehensive NMF Benchmark Suite
# Runs all 7 implementations across 5 matrix sizes
#
# Usage:
#   ./scripts/run_full_benchmarks.sh              # Full benchmark
#   ./scripts/run_full_benchmarks.sh --generate   # Only generate matrices
#   ./scripts/run_full_benchmarks.sh --quick      # Quick test (500, 1000 only)
#

set -e

# Configuration
SIZES="500 1000 2000 4000 8000"
RANK=20
MU_ITERS=100
HALS_ITERS=50

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
GENERATE_ONLY=false
QUICK_MODE=false

for arg in "$@"; do
    case $arg in
        --generate)
            GENERATE_ONLY=true
            ;;
        --quick)
            QUICK_MODE=true
            SIZES="500 1000"
            ;;
    esac
done

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         NMF Comprehensive Benchmark Suite                      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Matrix sizes: $SIZES"
echo "Rank: $RANK"
echo "MU iterations: $MU_ITERS"
echo "HALS iterations: $HALS_ITERS"
echo ""

# Create results directory
mkdir -p results
TIMING_FILE="results/benchmark_timing.csv"

# Initialize timing CSV
echo "size,method,time_ms,iterations,time_per_iter_ms,final_error" > $TIMING_FILE

# =============================================================================
# Step 1: Generate Test Matrices
# =============================================================================

echo -e "${YELLOW}[1/3] Generating low-rank test matrices...${NC}"
python3 scripts/generate_lowrank_matrices.py --sizes $SIZES --rank $RANK
echo ""

if [ "$GENERATE_ONLY" = true ]; then
    echo -e "${GREEN}Matrix generation complete!${NC}"
    exit 0
fi

# =============================================================================
# Step 2: Run All Benchmarks
# =============================================================================

echo -e "${YELLOW}[2/3] Running benchmarks...${NC}"
echo ""

# Function to extract metrics from program output
extract_metrics() {
    local output="$1"
    local time=$(echo "$output" | grep -oP 'Time: \K[\d.]+' | head -1)
    local error=$(echo "$output" | grep -oP 'Final_Error: \K[\d.e+-]+' | head -1)
    echo "$time,$error"
}

for size in $SIZES; do
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Matrix Size: ${size}×${size}${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo ""

    DATA_FILE="data/lowrank_${size}.bin"

    if [ ! -f "$DATA_FILE" ]; then
        echo -e "${RED}Error: $DATA_FILE not found${NC}"
        continue
    fi

    # -------------------------------------------------------------------------
    # MU Implementations
    # -------------------------------------------------------------------------

    # MU Naive (v1)
    echo -e "${GREEN}Running MU Naive...${NC}"
    OUTPUT=$(./nmf_naive $DATA_FILE $RANK $MU_ITERS 2>&1) || true
    METRICS=$(extract_metrics "$OUTPUT")
    TIME=$(echo $METRICS | cut -d',' -f1)
    ERROR=$(echo $METRICS | cut -d',' -f2)
    if [ -n "$TIME" ]; then
        TIME_PER_ITER=$(echo "scale=4; $TIME / $MU_ITERS" | bc)
        echo "$size,mu_naive,$TIME,$MU_ITERS,$TIME_PER_ITER,$ERROR" >> $TIMING_FILE
        echo "  Time: ${TIME} ms, Error: ${ERROR}"
    else
        echo "  Failed to extract metrics"
    fi

    # MU L2 Memory-Opt
    echo -e "${GREEN}Running MU L2 (Memory-Opt)...${NC}"
    OUTPUT=$(./nmf_memory_opt $DATA_FILE $RANK $MU_ITERS 2>&1) || true
    METRICS=$(extract_metrics "$OUTPUT")
    TIME=$(echo $METRICS | cut -d',' -f1)
    ERROR=$(echo $METRICS | cut -d',' -f2)
    if [ -n "$TIME" ]; then
        TIME_PER_ITER=$(echo "scale=4; $TIME / $MU_ITERS" | bc)
        echo "$size,mu_l2_memory,$TIME,$MU_ITERS,$TIME_PER_ITER,$ERROR" >> $TIMING_FILE
        echo "  Time: ${TIME} ms, Error: ${ERROR}"
    else
        echo "  Failed to extract metrics"
    fi

    # MU L3 Compute-Opt
    echo -e "${GREEN}Running MU L3 (Compute-Opt)...${NC}"
    OUTPUT=$(./nmf_compute_opt $DATA_FILE $RANK $MU_ITERS 128 2>&1) || true
    METRICS=$(extract_metrics "$OUTPUT")
    TIME=$(echo $METRICS | cut -d',' -f1)
    ERROR=$(echo $METRICS | cut -d',' -f2)
    if [ -n "$TIME" ]; then
        TIME_PER_ITER=$(echo "scale=4; $TIME / $MU_ITERS" | bc)
        echo "$size,mu_l3_compute,$TIME,$MU_ITERS,$TIME_PER_ITER,$ERROR" >> $TIMING_FILE
        echo "  Time: ${TIME} ms, Error: ${ERROR}"
    else
        echo "  Failed to extract metrics"
    fi

    # -------------------------------------------------------------------------
    # HALS Implementations
    # -------------------------------------------------------------------------

    # HALS CPU
    echo -e "${GREEN}Running HALS CPU...${NC}"
    OUTPUT=$(./nmf_hals_cpu $DATA_FILE $RANK $HALS_ITERS 2>&1) || true
    METRICS=$(extract_metrics "$OUTPUT")
    TIME=$(echo $METRICS | cut -d',' -f1)
    ERROR=$(echo $METRICS | cut -d',' -f2)
    if [ -n "$TIME" ]; then
        TIME_PER_ITER=$(echo "scale=4; $TIME / $HALS_ITERS" | bc)
        echo "$size,hals_cpu,$TIME,$HALS_ITERS,$TIME_PER_ITER,$ERROR" >> $TIMING_FILE
        echo "  Time: ${TIME} ms, Error: ${ERROR}"
    else
        echo "  Failed to extract metrics"
    fi

    # HALS GPU Strict
    echo -e "${GREEN}Running HALS GPU Strict...${NC}"
    OUTPUT=$(./nmf_hals_gpu_strict $DATA_FILE $RANK $HALS_ITERS 2>&1) || true
    METRICS=$(extract_metrics "$OUTPUT")
    TIME=$(echo $METRICS | cut -d',' -f1)
    ERROR=$(echo $METRICS | cut -d',' -f2)
    if [ -n "$TIME" ]; then
        TIME_PER_ITER=$(echo "scale=4; $TIME / $HALS_ITERS" | bc)
        echo "$size,hals_gpu_strict,$TIME,$HALS_ITERS,$TIME_PER_ITER,$ERROR" >> $TIMING_FILE
        echo "  Time: ${TIME} ms, Error: ${ERROR}"
    else
        echo "  Failed to extract metrics"
    fi

    # HALS GPU Block
    echo -e "${GREEN}Running HALS GPU Block...${NC}"
    OUTPUT=$(./nmf_hals_gpu_block $DATA_FILE $RANK $HALS_ITERS 5 2>&1) || true
    METRICS=$(extract_metrics "$OUTPUT")
    TIME=$(echo $METRICS | cut -d',' -f1)
    ERROR=$(echo $METRICS | cut -d',' -f2)
    if [ -n "$TIME" ]; then
        TIME_PER_ITER=$(echo "scale=4; $TIME / $HALS_ITERS" | bc)
        echo "$size,hals_gpu_block,$TIME,$HALS_ITERS,$TIME_PER_ITER,$ERROR" >> $TIMING_FILE
        echo "  Time: ${TIME} ms, Error: ${ERROR}"
    else
        echo "  Failed to extract metrics"
    fi

    echo ""
done

# =============================================================================
# Step 3: Summary
# =============================================================================

echo -e "${YELLOW}[3/3] Summary${NC}"
echo ""
echo "Results saved to: $TIMING_FILE"
echo ""
echo "Contents:"
cat $TIMING_FILE | column -t -s','
echo ""

# Check for multi-GPU
NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -1 || echo "1")
if [ "$NUM_GPUS" -ge 2 ]; then
    echo -e "${YELLOW}Multi-GPU detected ($NUM_GPUS GPUs). Run separately:${NC}"
    echo "  ./nmf_multigpu data/lowrank_2000.bin 20 100 2"
else
    echo -e "${YELLOW}Single GPU detected. Multi-GPU tests skipped.${NC}"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         Benchmark Complete!                                    ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Next: Run the Jupyter notebook for analysis and plots"
echo "  jupyter notebook notebooks/nmf_report_analysis.ipynb"
echo ""
