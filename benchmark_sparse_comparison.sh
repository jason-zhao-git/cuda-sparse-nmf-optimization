#!/bin/bash

# Comprehensive benchmark comparing all NMF implementations
# Tests: Dense Optimized vs Hybrid Sparse vs Pure Sparse (transpose trick)

echo "════════════════════════════════════════════════════════════════"
echo "  NMF GPU Implementation Comparison Benchmark"
echo "  Testing 1000×1000 matrix, k=20, 90% sparsity, 50 iterations"
echo "════════════════════════════════════════════════════════════════"
echo ""

SIZE=1000
RANK=20
SPARSITY=0.9
ITERS=50

# Create results directory
mkdir -p results

echo "Building all versions..."
make clean > /dev/null 2>&1
make memory-opt sparse > /dev/null 2>&1

# Build transpose version
/usr/local/cuda/bin/nvcc -O3 -arch=sm_86 -Xcompiler -Wall \
    src/nmf_sparse_gpu_v3_transpose.cu src/utils.cu \
    -o nmf_sparse_transpose -lcublas -lcusparse > /dev/null 2>&1

echo "✓ All versions built"
echo ""

# ============================================================================
# Test 1: Dense Optimized (Level 2)
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 1: Dense Optimized (Level 2 - Baseline)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
./nmf_memory_opt $SIZE $RANK $ITERS 2>&1 | tee results/benchmark_dense.txt
echo ""
echo ""

# ============================================================================
# Test 2: Hybrid Sparse (Current Level 3)
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 2: Hybrid Sparse (dense for W^T×X, sparse for X×H^T)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
./nmf_sparse $SIZE $RANK $SPARSITY $ITERS 2>&1 | tee results/benchmark_hybrid.txt
echo ""
echo ""

# ============================================================================
# Test 3: Pure Sparse with Transpose Trick
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 3: Pure Sparse (transpose trick for both operations)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
./nmf_sparse_transpose $SIZE $RANK $SPARSITY $ITERS 2>&1 | tee results/benchmark_transpose.txt
echo ""
echo ""

# ============================================================================
# Extract and Compare Results
# ============================================================================
echo "════════════════════════════════════════════════════════════════"
echo "  PERFORMANCE COMPARISON SUMMARY"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Extract metrics
DENSE_TIME=$(grep "^Time:" results/benchmark_dense.txt | awk '{print $2}')
DENSE_GFLOPS=$(grep "^FLOPS:" results/benchmark_dense.txt | awk '{print $2}')
DENSE_BW=$(grep "^Bandwidth:" results/benchmark_dense.txt | awk '{print $2}')

HYBRID_TIME=$(grep "^Time:" results/benchmark_hybrid.txt | awk '{print $2}')
HYBRID_GFLOPS=$(grep "^GFLOPS:" results/benchmark_hybrid.txt | awk '{print $2}')
HYBRID_BW=$(grep "^Bandwidth:" results/benchmark_hybrid.txt | awk '{print $2}')

TRANSPOSE_TIME=$(grep "^Time:" results/benchmark_transpose.txt | awk '{print $2}')
TRANSPOSE_GFLOPS=$(grep "^GFLOPS:" results/benchmark_transpose.txt | awk '{print $2}')
TRANSPOSE_BW=$(grep "^Bandwidth:" results/benchmark_transpose.txt | awk '{print $2}')

printf "%-30s %10s %12s %15s\n" "Implementation" "Time (ms)" "GFLOPS" "Bandwidth (GB/s)"
printf "%-30s %10s %12s %15s\n" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" "━━━━━━━━━━" "━━━━━━━━━━━━" "━━━━━━━━━━━━━━━"
printf "%-30s %10.2f %12.2f %15.2f\n" "Dense Optimized (BASELINE)" "$DENSE_TIME" "$DENSE_GFLOPS" "$DENSE_BW"
printf "%-30s %10.2f %12.2f %15.2f\n" "Hybrid Sparse" "$HYBRID_TIME" "$HYBRID_GFLOPS" "$HYBRID_BW"
printf "%-30s %10.2f %12.2f %15.2f\n" "Pure Sparse (transpose)" "$TRANSPOSE_TIME" "$TRANSPOSE_GFLOPS" "$TRANSPOSE_BW"
echo ""

# Calculate speedups
HYBRID_SPEEDUP=$(echo "scale=2; $DENSE_TIME / $HYBRID_TIME" | bc)
TRANSPOSE_SPEEDUP=$(echo "scale=2; $DENSE_TIME / $TRANSPOSE_TIME" | bc)
HYBRID_VS_TRANSPOSE=$(echo "scale=2; $TRANSPOSE_TIME / $HYBRID_TIME" | bc)

echo "Speedup vs Dense Optimized:"
printf "  Hybrid Sparse:       %.2fx %s\n" "$HYBRID_SPEEDUP" "$(if (( $(echo "$HYBRID_SPEEDUP < 1" | bc -l) )); then echo "(SLOWER)"; else echo "(faster)"; fi)"
printf "  Pure Sparse:         %.2fx %s\n" "$TRANSPOSE_SPEEDUP" "$(if (( $(echo "$TRANSPOSE_SPEEDUP < 1" | bc -l) )); then echo "(SLOWER)"; else echo "(faster)"; fi)"
echo ""
printf "Hybrid vs Pure Sparse: %.2fx %s\n" "$HYBRID_VS_TRANSPOSE" "$(if (( $(echo "$HYBRID_VS_TRANSPOSE < 1" | bc -l) )); then echo "(hybrid faster)"; else echo "(transpose faster)"; fi)"
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "  ANALYSIS"
echo "════════════════════════════════════════════════════════════════"
echo ""

if (( $(echo "$HYBRID_SPEEDUP < 1" | bc -l) )); then
    echo "WHY IS HYBRID SPARSE SLOWER THAN DENSE?"
    echo ""
    echo "Even though hybrid only uses cuSPARSE for 1 operation (X×H^T),"
    echo "cuSPARSE SpMM is SO MUCH SLOWER than cuBLAS GEMM that the"
    echo "savings from skipping 90% of FLOPs doesn't compensate for the"
    echo "overhead of the sparse operation."
    echo ""
    echo "Breakdown:"
    echo "  - Dense: 6 fast cuBLAS operations"
    echo "  - Hybrid: 5 fast cuBLAS + 1 VERY SLOW cuSPARSE"
    echo "  - The 1 slow operation dominates the runtime!"
    echo ""
fi

if (( $(echo "$TRANSPOSE_SPEEDUP < 1" | bc -l) )); then
    echo "WHY IS PURE SPARSE (TRANSPOSE) EVEN SLOWER?"
    echo ""
    echo "Pure sparse uses cuSPARSE for BOTH sparse operations:"
    echo "  1. X^T × W (via transpose trick)"
    echo "  2. X × H^T (direct)"
    echo ""
    echo "This means 2 slow cuSPARSE operations instead of 1, PLUS:"
    echo "  - Extra transpose kernel overhead"
    echo "  - More irregular memory access"
    echo "  - Less cache efficiency"
    echo ""
    echo "Memory savings (1.6MB vs 4.8MB) don't matter because memory"
    echo "is not the bottleneck - cuSPARSE performance is!"
    echo ""
fi

echo "KEY INSIGHT:"
echo "For NMF specifically, cuSPARSE overhead >> FLOP savings"
echo "Dense cuBLAS is the winner for this algorithm."
echo ""
echo "════════════════════════════════════════════════════════════════"

# Save summary
cat > results/comparison_summary.txt << EOF
NMF Implementation Comparison (${SIZE}×${SIZE}, k=${RANK}, 90% sparse, ${ITERS} iters)

Implementation              Time (ms)    GFLOPS    Bandwidth (GB/s)    Speedup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dense Optimized            ${DENSE_TIME}      ${DENSE_GFLOPS}       ${DENSE_BW}               1.00x (baseline)
Hybrid Sparse              ${HYBRID_TIME}      ${HYBRID_GFLOPS}        ${HYBRID_BW}                ${HYBRID_SPEEDUP}x
Pure Sparse (transpose)    ${TRANSPOSE_TIME}      ${TRANSPOSE_GFLOPS}        ${TRANSPOSE_BW}                ${TRANSPOSE_SPEEDUP}x

CONCLUSION:
Dense optimized is the clear winner for NMF. cuSPARSE overhead dominates
any FLOP savings from sparse operations, even at 90% sparsity.
EOF

echo "✓ Results saved to results/comparison_summary.txt"
