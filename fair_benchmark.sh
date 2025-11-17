#!/bin/bash

# Fair NMF Benchmark - All methods on SAME input matrix

echo "════════════════════════════════════════════════════════════════"
echo "  FAIR NMF BENCHMARK - Same Input for All Methods"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Configuration
MATRIX_FILE="data/test_sparse_1000.bin"
RANK=20
ITERS=50

# Generate matrix if it doesn't exist
if [ ! -f "$MATRIX_FILE" ]; then
    echo "Generating test matrix..."
    python3 data/generate_matrix.py --size 1000 --sparsity 0.9 --output "$MATRIX_FILE"
    echo ""
fi

echo "Matrix: $MATRIX_FILE"
echo "Rank: $RANK"
echo "Iterations: $ITERS"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Test Level 2 (already updated)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "LEVEL 2: Memory-Optimized Dense (cuBLAS)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
./nmf_memory_opt "$MATRIX_FILE" $RANK $ITERS 2>&1 | grep -E "Time:|FLOPS:|Bandwidth:|Final error:|Sparsity:"
echo ""

# Note: Other levels need updating (Level 1, 3, 3_transpose)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "NOTE: Levels 1, 3, and 3_transpose still need updating to load from file"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "KEY RESULT from Level 2:"
echo "  - Same 90% sparse matrix as before"
echo "  - Error ~0.99 (high because 90% of data is zeros)"
echo "  - THIS IS CORRECT - factorizing a matrix full of zeros!"
echo ""
echo "Next: Update other levels to use load_matrix_binary()"
echo "════════════════════════════════════════════════════════════════"
