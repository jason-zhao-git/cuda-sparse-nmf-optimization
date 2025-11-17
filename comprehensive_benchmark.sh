#!/bin/bash

echo "════════════════════════════════════════════════════════════════"
echo "  COMPREHENSIVE NMF BENCHMARK - FAIR COMPARISON"
echo "════════════════════════════════════════════════════════════════"
echo ""

RANK=20
ITERS=50

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 1: ALL METHODS ON DENSE MATRIX (0% sparsity)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "--- Level 2: Memory-Optimized Dense (cuBLAS) ---"
./nmf_memory_opt data/dense_1000.bin $RANK $ITERS 2>&1 | grep -E "Sparsity:|Time:|FLOPS:|Bandwidth:|Final error:"
echo ""

echo "--- Level 3: Sparse Hybrid (cuBLAS + cuSPARSE) ---"
./nmf_sparse data/dense_1000.bin $RANK $ITERS 2>&1 | grep -E "Sparsity:|Time:|GFLOPS:|Bandwidth:|Final error:"
echo ""

echo "--- Level 3: Sparse Transpose (Pure cuSPARSE) ---"
./nmf_sparse_transpose data/dense_1000.bin $RANK $ITERS 2>&1 | grep -E "Sparsity:|Time:|GFLOPS:|Bandwidth:|Final error:"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 2: ALL METHODS ON SPARSE MATRIX (90% sparsity)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "--- Level 2: Memory-Optimized Dense (cuBLAS) ---"
./nmf_memory_opt data/sparse_1000.bin $RANK $ITERS 2>&1 | grep -E "Sparsity:|Time:|FLOPS:|Bandwidth:|Final error:"
echo ""

echo "--- Level 3: Sparse Hybrid (cuBLAS + cuSPARSE) ---"
./nmf_sparse data/sparse_1000.bin $RANK $ITERS 2>&1 | grep -E "Sparsity:|Time:|GFLOPS:|Bandwidth:|Final error:"
echo ""

echo "--- Level 3: Sparse Transpose (Pure cuSPARSE) ---"
./nmf_sparse_transpose data/sparse_1000.bin $RANK $ITERS 2>&1 | grep -E "Sparsity:|Time:|GFLOPS:|Bandwidth:|Final error:"
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "  BENCHMARK COMPLETE"
echo "════════════════════════════════════════════════════════════════"
