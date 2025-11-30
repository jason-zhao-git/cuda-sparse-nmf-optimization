# HALS GPU Implementation: Strict Parallelization

## Overview

This document explains the HALS (Hierarchical Alternating Least Squares) algorithm for Non-negative Matrix Factorization (NMF) and how we parallelized it on GPU while maintaining mathematical correctness.

## What is HALS?

HALS is an efficient algorithm for NMF that solves:

```
minimize ||X - WH||²  subject to W ≥ 0, H ≥ 0
```

Where:
- **X** is the input data matrix (m × n)
- **W** is the basis/dictionary matrix (m × k)
- **H** is the coefficient/encoding matrix (k × n)
- **k** is the rank (number of features/components)

### Key Insight: Feature-wise Updates

Unlike Multiplicative Update (MU) which updates all elements simultaneously, HALS updates **one feature at a time** using a closed-form solution. This is more efficient because:

1. Each feature update has an analytical solution (no step-size tuning)
2. Faster convergence (~10-40 iterations vs ~200 for MU)
3. Better numerical stability

## HALS Algorithm

### Update Formulas

**For H (updating row f across all samples j):**
```
H[f,j] = max(ε, H[f,j] + (Numerator_H[f,j] - interaction_H) / Denom_H[f,f])

where:
  Numerator_H = W^T × X          (k × n matrix, precomputed)
  Denom_H = W^T × W              (k × k matrix, precomputed)
  interaction_H = Σ_l Denom_H[f,l] × H[l,j]   (dot product)
```

**For W (updating column f across all rows i):**
```
W[i,f] = max(ε, W[i,f] + (Numerator_W[i,f] - interaction_W) / Denom_W[f,f])

where:
  Numerator_W = X × H^T          (m × k matrix, precomputed)
  Denom_W = H × H^T              (k × k matrix, precomputed)
  interaction_W = Σ_l W[i,l] × Denom_W[l,f]   (dot product)

Then normalize: W[:,f] = W[:,f] / ||W[:,f]||₂
```

### Gauss-Seidel Property

The critical property of HALS is the **Gauss-Seidel update pattern**:

```
For f = 0, 1, 2, ..., k-1:
    Update feature f
    Feature f now sees the UPDATED values of features 0, 1, ..., f-1
```

This creates a **dependency chain**: feature f depends on the already-updated features 0 to f-1. This is what enables fast convergence but makes parallelization challenging.

```
Feature 0 ──────► Feature 1 ──────► Feature 2 ──────► ... ──────► Feature k-1
   │                  │                  │                            │
   │                  │                  │                            │
   ▼                  ▼                  ▼                            ▼
 Uses             Uses updated       Uses updated                Uses updated
 initial H        feature 0          features 0,1               features 0..k-2
```

## GPU Parallelization Strategy

### Level 1: Strict Single-Feature Parallelism

Our "strict" implementation maintains exact Gauss-Seidel semantics:

```
┌─────────────────────────────────────────────────────────────────┐
│                    SEQUENTIAL (features)                        │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐       ┌─────────┐     │
│  │Feature 0│ → │Feature 1│ → │Feature 2│ → ... │Feature k│     │
│  └────┬────┘   └────┬────┘   └────┬────┘       └────┬────┘     │
│       │             │             │                  │          │
│       ▼             ▼             ▼                  ▼          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              PARALLEL (samples/rows)                     │   │
│  │  Thread 0   Thread 1   Thread 2   ...   Thread N        │   │
│  │  4 rows     4 rows     4 rows           4 rows          │   │
│  └─────────────────────────────────────────────────────────┘   │
│       │             │             │                  │          │
│       ▼             ▼             ▼                  ▼          │
│   cudaDeviceSynchronize() after each feature                   │
└─────────────────────────────────────────────────────────────────┘
```

**What's Parallelized:**
- **H update**: For feature f, parallelize across n samples (each thread handles 4 samples)
- **W update**: For feature f, parallelize across m rows (each thread handles 4 rows)

**What's Sequential:**
- Features are processed one at a time (f = 0, 1, ..., k-1)
- `cudaDeviceSynchronize()` after each feature ensures dependencies are respected

### Memory Layout: Column-Major

We use column-major storage for cuBLAS compatibility:

```
Matrix W (m × k):        Matrix H (k × n):
┌─────────────┐          ┌─────────────────────┐
│ col 0 col 1 │          │ col 0  col 1  ...   │
│ ───── ───── │          │ ─────  ─────        │
│ W[0,0]W[0,1]│          │ H[0,0] H[0,1] ...   │
│ W[1,0]W[1,1]│          │ H[1,0] H[1,1] ...   │
│ W[2,0]W[2,1]│          │ ...                 │
│ ...   ...   │          │ H[k-1,0] ...        │
└─────────────┘          └─────────────────────┘

W[i,f] stored at: f*m + i
H[f,j] stored at: j*k + f
```

### Kernel Structure

```cuda
// For each iteration:
for (iter = 0; iter < max_iter; iter++) {

    // === Update H ===
    // 1. Precompute using cuBLAS (parallelized matrix multiply)
    Numerator_H = W^T × X    // cublasSgemm
    Denom_H = W^T × W        // cublasSgemm

    // 2. Update features sequentially
    for (f = 0; f < k; f++) {
        update_H_feature<<<grid, block>>>(H, Numerator_H, Denom_H, f);
        cudaDeviceSynchronize();  // CRITICAL: wait before next feature
    }

    // === Update W ===
    // 1. Precompute
    Numerator_W = X × H^T    // cublasSgemm
    Denom_W = H × H^T        // cublasSgemm

    // 2. Update and normalize features sequentially
    for (f = 0; f < k; f++) {
        update_W_feature<<<grid, block>>>(W, Numerator_W, Denom_W, f);
        cudaDeviceSynchronize();
        normalize_W_column<<<grid, block>>>(W, f);
        cudaDeviceSynchronize();
    }
}
```

### ILP Optimization: float4

Each thread processes 4 consecutive elements for Instruction-Level Parallelism:

```cuda
// Thread processes rows: row_start, row_start+1, row_start+2, row_start+3
int row_start = tid * 4;

float4 result;
for (int offset = 0; offset < 4; offset++) {
    int row = row_start + offset;
    // ... compute update for this row ...
    // Store to float4 component
}
// Write back as float4 (coalesced when possible)
```

## Bug Fix: Memory Layout Mismatch

### The Problem

The original code used `compute_relative_error_dense()` from `utils.cu` which assumes **row-major** storage:

```c
// utils.cu - WRONG for our matrices
sum += h_W[i * k + p] * h_H[p * n + j];  // Row-major indexing
```

But our GPU matrices are **column-major**:
```
W[i,f] at position f*m + i  (not i*k + f)
H[f,j] at position j*k + f  (not f*n + j)
```

This caused the error computation to read garbage values, reporting error > 1.0 (worse than no approximation!).

### The Fix

Added a column-major error computation function:

```c
float compute_error_column_major(const float* X, const float* W, const float* H,
                                  int m, int n, int k) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            float x_val = X[j * m + i];           // Column-major X
            float wh_val = 0.0f;
            for (int f = 0; f < k; f++) {
                wh_val += W[f * m + i] * H[j * k + f];  // Column-major W, H
            }
            // ... accumulate error ...
        }
    }
}
```

### Results After Fix

| Metric | CPU HALS | GPU HALS (Before) | GPU HALS (After) |
|--------|----------|-------------------|------------------|
| Time   | 106.32 ms | 93.33 ms | 95.78 ms |
| Error  | 0.4869 | **1.4249** (WRONG!) | **0.4869** ✓ |

## Performance Characteristics

### Current Performance (1000×1000, k=20, 15 iterations)

- **Time**: 95.78 ms (GPU) vs 106.32 ms (CPU) = **~11% speedup**
- **GFLOPS**: 13.03
- **Bandwidth**: 26.06 GB/s

### Why Only 11% Speedup?

1. **Synchronization overhead**: 40 `cudaDeviceSynchronize()` calls per iteration
2. **Sequential feature updates**: Cannot exploit full GPU parallelism
3. **Memory access patterns**: Non-coalesced reads for interaction computation
4. **Small problem size**: Kernel launch overhead significant for 1000×1000

### Potential Improvements (Level 2+)

1. **Batched Features**: Update T features in parallel (Jacobi-like), then synchronize
   - Trade-off: Slower convergence but more parallelism

2. **Shared Memory**: Cache Denom matrix in shared memory

3. **Larger Problems**: GPU advantage grows with matrix size

## File Structure

```
src/hals/
├── nmf_hals_cpu.cpp           # CPU baseline (column-major)
└── nmf_hals_gpu_v1_strict.cu  # GPU Level 1 (strict Gauss-Seidel)
```

## Usage

```bash
# Compile
make nmf_hals_cpu nmf_hals_gpu_strict

# Run CPU baseline
./nmf_hals_cpu data/dense_1000.bin 20 50

# Run GPU strict
./nmf_hals_gpu_strict data/dense_1000.bin 20 50

# Compare metrics
cat results/hals_cpu_metrics.txt
cat results/hals_gpu_strict_metrics.txt
```

## Level 2: Block-Parallel with Random Shuffling

### Overview

Level 2 achieves **1.97x speedup** over CPU by:
1. Partitioning k features into blocks (default: 5 features/block)
2. Updating blocks in parallel via CUDA streams
3. Maintaining sequential Gauss-Seidel within each block
4. **Critically:** Random shuffling features each iteration

### Why Random Shuffling?

**Problem with fixed groupings:**
```
Block 0 always = [0,1,2,3,4], Block 1 always = [5,6,7,8,9], ...
Feature 4 NEVER sees updated feature 5 → systematic bias → poor convergence
```

**Solution: Random shuffle each iteration:**
```
Iter 0: Block 0 = [3,17,8,1,12], Block 1 = [5,19,0,14,7], ...
Iter 1: Block 0 = [11,2,18,6,9], Block 1 = [4,15,1,13,8], ...
```
Over iterations, all feature pairs interact in both orders, eliminating bias.

### Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│ Iteration i (random shuffle: perm = [3,17,8,1,12,5,19,0,14,...])  │
│                                                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────┐ │
│  │  Block 0    │  │  Block 1    │  │  Block 2    │  │ Block 3  │ │
│  │  Stream 0   │  │  Stream 1   │  │  Stream 2   │  │ Stream 3 │ │
│  │  [3,17,8,   │  │  [5,19,0,   │  │  [11,6,16,  │  │ [9,4,13, │ │
│  │   1,12]     │  │   14,7]     │  │   2,18]     │  │  10,15]  │ │
│  └─────┬───────┘  └─────┬───────┘  └─────┬───────┘  └────┬─────┘ │
│        │                │                │               │        │
│        │  PARALLEL      │                │               │        │
│        └────────────────┴────────────────┴───────────────┘        │
│                                 │                                  │
│                     cudaDeviceSynchronize()                        │
└───────────────────────────────────────────────────────────────────┘
```

### Key Code Sections

**Fisher-Yates Shuffle:**
```cpp
void shuffle_features(int* perm, int k, unsigned int seed) {
    for (int i = 0; i < k; i++) perm[i] = i;
    std::mt19937 rng(seed);
    for (int i = k - 1; i > 0; i--) {
        std::uniform_int_distribution<int> dist(0, i);
        std::swap(perm[i], perm[dist(rng)]);
    }
}
```

**Block-Parallel Loop:**
```cpp
for (int iter = 0; iter < max_iter; iter++) {
    shuffle_features(perm, k, 42 + iter);  // NEW shuffle each iter!

    // H update - blocks in parallel
    for (int b = 0; b < num_blocks; b++) {
        for (int f_local = 0; f_local < block_size; f_local++) {
            int f = perm[b * block_size + f_local];
            update_H_column<<<grid, block, 0, streams[b]>>>(d_H, ..., f);
        }
    }
    for (int b = 0; b < num_blocks; b++)
        cudaStreamSynchronize(streams[b]);  // Single sync per phase

    // W update (similar)
}
```

### Performance Results (1000×1000, k=20, 15 iterations)

| Implementation | Time (ms) | Error | Speedup |
|---------------|-----------|-------|---------|
| CPU Baseline | 106.32 | 0.4869 | 1.0x |
| GPU Level 1 (Strict) | 95.78 | 0.4869 | 1.1x |
| **GPU Level 2 (Block)** | **53.95** | **0.4869** | **1.97x** |

### Sync Reduction

- Level 1: 2k syncs per iteration = 40 for k=20
- Level 2: 2 syncs per iteration (after H blocks, after W blocks)
- **Reduction: 20x fewer synchronization points!**

### Usage

```bash
# Build
make hals-gpu-block

# Run (default block size = 5)
./nmf_hals_gpu_block data/dense_1000.bin 20 15

# Custom block size
./nmf_hals_gpu_block data/dense_1000.bin 20 15 8
```

---

## Summary

The HALS algorithm provides fast convergence for NMF by using Gauss-Seidel feature-wise updates.

**GPU Level 1 (Strict)** maintains exact mathematical equivalence with CPU:
- Sequential feature processing (preserving all dependencies)
- Row-parallel within each feature update
- Limited speedup (1.1x) due to sync overhead

**GPU Level 2 (Block-Parallel)** achieves **1.97x speedup** by:
- Partitioning features into blocks processed in parallel via CUDA streams
- Sequential Gauss-Seidel within each block
- **Random shuffling** each iteration to ensure proper feature mixing
- 20x fewer synchronization points

The key insight is that **random shuffling is essential** - without it, fixed groupings create systematic bias where certain feature pairs never see each other's updates in the correct order.
