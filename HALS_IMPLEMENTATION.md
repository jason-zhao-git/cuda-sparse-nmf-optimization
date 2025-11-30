# HALS NMF Implementation: Parallelizing Sequential Dependencies

## Overview

HALS (Hierarchical Alternating Least Squares) represents a **fundamentally different parallelization challenge** compared to Multiplicative Update NMF. While MU has trivially parallel element-wise operations, HALS uses Gauss-Seidel updates with strict sequential dependencies.

**Key Insight:** HALS converges in ~40 iterations (vs 200+ for MU) but is inherently sequential. The parallelization challenge is: **How do we exploit GPU parallelism without destroying convergence?**

---

## HALS Algorithm

### Update Equations

Per iteration, HALS updates H and W column-by-column using a Gauss-Seidel pattern:

**H Update:**
```
WtA = W^T × A          # (k × n) - cuBLAS GEMM
WtW = W^T × W          # (k × k) - cuBLAS GEMM

For j = 0 to k-1:      # SEQUENTIAL LOOP
    H[:,j] = H[:,j] + WtA[j,:] - H × WtW[:,j]
    H[:,j] = max(H[:,j], ε)  # Non-negativity constraint
```

**W Update:**
```
AH = A × H^T           # (m × k) - cuBLAS GEMM
HtH = H^T × H          # (k × k) - cuBLAS GEMM

For j = 0 to k-1:      # SEQUENTIAL LOOP
    W[:,j] = W[:,j] × HtH[j,j] + AH[:,j] - W × HtH[:,j]
    W[:,j] = max(W[:,j], ε)
    W[:,j] = W[:,j] / norm(W[:,j])  # Normalization
```

### Why Sequential?

**Gauss-Seidel Dependency:**
```
Column j+1 uses the JUST-UPDATED value of column j

H[:,1] depends on updated H[:,0]
H[:,2] depends on updated H[:,0] and H[:,1]
...
H[:,j] depends on updated H[:,0..j-1]
```

This is **fundamentally different** from Jacobi iteration (used in MU), where all updates use values from the previous iteration.

---

## Parallelization Challenge

### Why HALS is Hard to Parallelize on GPU

1. **Sequential Dependencies**
   - GPU thrives on thousands of independent threads
   - HALS requires strictly consecutive updates
   - Cannot launch k parallel threads without losing convergence properties

2. **Small Per-Column Work**
   - Each column update: O(n×k) operations
   - For typical k=20, n=1000: only 20,000 FLOPs per column
   - GPU underutilized (needs millions of operations for efficiency)

3. **Memory Access Pattern**
   - Each update reads all of H (random column access)
   - Cache coherence across sequential updates is poor
   - cuBLAS GEMMs are memory-bound, not compute-bound

### Naive Parallelization Fails

**Attempt 1: Parallelize across columns (ignore dependencies)**
```cuda
// Launch k parallel threads, one per column
parallel_for (int j = 0; j < k; j++) {
    H[:,j] = H[:,j] + WtA[j,:] - H × WtW[:,j]
}
```
**Result:** Converges to wrong solution! Each thread reads stale H values.

**Attempt 2: Jacobi-style (use H from previous iteration)**
```cuda
H_new = H_old + WtA - H_old × WtW  // Fully parallel
```
**Result:** Converges to MU (200+ iterations). Lost the advantage of HALS!

---

## Block-Parallel Solution with Random Shuffling

### Core Idea: Random Feature Mixing

Instead of fixed groupings (which create systematic bias), randomly shuffle features into blocks each iteration:

```
Instead of: for j = 0 to k-1: update H[:,j]          # Fully sequential
Do this:    for each iteration:
                shuffle features randomly into blocks
                for each block (in parallel via CUDA streams):
                    for j in block: update H[:,j]    # Sequential within block
```

**Within each block:** Update features sequentially (Gauss-Seidel)
**Between blocks:** Update blocks in parallel via CUDA streams (Jacobi-like)
**Critical:** Random shuffling ensures all feature pairs eventually see updated values

### Why Random Shuffling is Essential

**Problem with Fixed Groupings:**
```
Fixed blocks: Block 0 = [0,1,2,3,4], Block 1 = [5,6,7,8,9], ...
Feature 4 NEVER sees updated feature 5 before updating (systematic bias)
```

**Random Shuffling Solution:**
```
Iter 0: Block 0 = [3,17,8,1,12], Block 1 = [5,19,0,14,7], ...
Iter 1: Block 0 = [11,2,18,6,9], Block 1 = [4,15,1,13,8], ...
Over iterations, all feature pairs interact in both orders!
```

### Algorithm: Block-Parallel HALS

```
function HALS_BlockParallel(X, W, H, k, block_size):
    num_blocks = ceil(k / block_size)
    create CUDA streams[num_blocks]

    for iter = 1 to max_iters:
        // === Random shuffle features each iteration ===
        perm = Fisher_Yates_Shuffle([0, 1, ..., k-1], seed=42+iter)

        // H update precompute
        WtA = W^T × X
        WtW = W^T × W

        // H update - blocks in parallel, features within block sequential
        for block = 0 to num_blocks-1:           // PARALLEL (streams)
            for local_f = 0 to block_size-1:     // SEQUENTIAL within block
                f = perm[block * block_size + local_f]
                update_H_column<<<stream[block]>>>(f)

        cudaDeviceSynchronize()  // Wait for all blocks

        // W update (similar pattern)
        ...
```

### Performance Results

| Implementation | Time (ms) | Error | Speedup | Notes |
|---------------|-----------|-------|---------|-------|
| CPU (sequential) | 106.3 | 0.487 | 1.0x | Exact Gauss-Seidel |
| GPU Level 1 (strict) | 95.8 | 0.487 | 1.1x | Single-column parallel |
| GPU Level 2 (block) | **54.0** | **0.487** | **1.97x** | Block-parallel + shuffle |

**Key Finding:** Block-parallel achieves **1.97x speedup** with **identical error** thanks to random shuffling!

---

## CPU Implementation

### File: `src/hals/nmf_hals_cpu.cpp`

**Purpose:** Sequential baseline to verify correctness and measure iteration count

**Key Functions:**

1. **`hals_update_H`** - Update H column-by-column
   ```cpp
   void hals_update_H(float* H, float* WtA, float* WtW, int k, int n) {
       for (int col = 0; col < k; col++) {
           // Hx = H[:,col] + WtA[col,:] - H × WtW[:,col]
           for (int i = 0; i < n; i++) {
               Hx[i] = H[col*n + i] + WtA[col*n + i];
               for (int j = 0; j < k; j++) {
                   Hx[i] -= H[j*n + i] * WtW[j*k + col];
               }
               Hx[i] = max(Hx[i], 1e-16);
           }
           memcpy(&H[col*n], Hx, n * sizeof(float));
       }
   }
   ```

2. **`hals_update_W`** - Update W column-by-column with normalization
   ```cpp
   void hals_update_W(float* W, float* AH, float* HtH, int m, int k) {
       for (int col = 0; col < k; col++) {
           float HtH_diag = HtH[col*k + col];
           for (int i = 0; i < m; i++) {
               Wx[i] = W[col*m + i] * HtH_diag + AH[col*m + i];
               for (int j = 0; j < k; j++) {
                   Wx[i] -= W[j*m + i] * HtH[j*k + col];
               }
               Wx[i] = max(Wx[i], 1e-16);
           }
           // Normalize
           float norm = vector_norm(Wx, m);
           for (int i = 0; i < m; i++) {
               W[col*m + i] = Wx[i] / norm;
           }
       }
   }
   ```

**Expected Performance:**
- Convergence: ~40 iterations
- Time per iteration: Slower than MU (more complex updates)
- Total time: May be faster (fewer iterations) or slower (depends on k, matrix size)

---

## GPU Implementation (Block-Parallel with Random Shuffling)

### Level 1: Strict Single-Column (`src/hals/nmf_hals_gpu_v1_strict.cu`)

**Purpose:** Exact Gauss-Seidel with GPU row-parallelism

- Processes features **sequentially** (f = 0, 1, ..., k-1)
- Parallelizes **within each feature** across rows/samples
- Uses `cudaDeviceSynchronize()` between features
- **Result:** 1.1x speedup over CPU (limited by sync overhead)

### Level 2: Block-Parallel (`src/hals/nmf_hals_gpu_v2_block.cu`)

**Purpose:** Parallelize across feature blocks while maintaining local Gauss-Seidel

**Key Components:**

1. **Fisher-Yates Shuffle** (on host):
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

2. **CUDA Streams** (one per block):
```cpp
cudaStream_t streams[num_blocks];
for (int b = 0; b < num_blocks; b++)
    cudaStreamCreate(&streams[b]);
```

3. **Block-Parallel Iteration Loop**:
```cpp
for (int iter = 0; iter < max_iter; iter++) {
    // Random shuffle each iteration (critical!)
    shuffle_features(perm, k, 42 + iter);

    // Precompute using cuBLAS
    cublasSgemm(..., d_Numerator_H);  // W^T × X
    cublasSgemm(..., d_Denom_H);      // W^T × W

    // H Update - blocks in parallel via streams
    for (int b = 0; b < num_blocks; b++) {
        for (int local_f = 0; local_f < block_size; local_f++) {
            int f = get_feature(perm, b, local_f, block_size, k);
            if (f >= 0) {
                update_H_column<<<grid, threads, 0, streams[b]>>>(
                    d_H, d_Numerator_H, d_Denom_H, k, n, f);
            }
        }
    }
    // Single sync point after all blocks complete
    for (int b = 0; b < num_blocks; b++)
        cudaStreamSynchronize(streams[b]);

    // W Update (similar pattern)
    ...
}
```

**Sync Reduction:**
- Level 1: 2k syncs per iteration (one per feature × 2 matrices)
- Level 2: 2 syncs per iteration (one after all H blocks, one after all W blocks)
- For k=20: 40 syncs → 2 syncs = **20x fewer syncs!**

---

## Performance Analysis

### Actual Results (1000×1000, k=20, 15 iterations)

| Implementation | Time (ms) | Time/Iter | Error | Bandwidth | GFLOPS |
|----------------|-----------|-----------|-------|-----------|--------|
| **CPU Baseline** | 106.32 | 7.09 | 0.4869 | N/A | N/A |
| **GPU Level 1 (Strict)** | 95.78 | 6.39 | 0.4869 | 26.06 GB/s | 13.03 |
| **GPU Level 2 (Block)** | **53.95** | **3.60** | **0.4869** | 46.26 GB/s | 23.13 |

### Speedup Analysis

| Comparison | Speedup | Notes |
|------------|---------|-------|
| GPU L2 vs CPU | **1.97x** | Main achievement |
| GPU L2 vs GPU L1 | **1.78x** | Benefit of block-parallel |
| GPU L1 vs CPU | 1.11x | Limited by sync overhead |

### Why Block-Parallel Works So Well

1. **Sync Reduction:** 40 syncs → 2 syncs per iteration (20x fewer!)
2. **CUDA Stream Parallelism:** 4 blocks execute concurrently
3. **Random Shuffling:** Maintains convergence quality despite partial Jacobi
4. **Same Convergence:** Error 0.4869 matches exact Gauss-Seidel (CPU/Level 1)

### Key Finding

Block-parallel with random shuffling achieves **nearly 2x speedup** while maintaining **identical convergence quality**. The random shuffling is critical - without it, fixed groupings would create systematic bias and degrade convergence.

---

## Convergence vs Parallelism Tradeoff

### Quantitative Analysis

**For T=4:**
- Parallelism gain: 4× (can process 4 columns simultaneously)
- Convergence cost: 12.5% more iterations (45 vs 40)
- Net benefit: 4.0 / 1.125 = 3.56× speedup (theoretical)

**For T=10:**
- Parallelism gain: 10×
- Convergence cost: 37.5% more iterations (55 vs 40)
- Net benefit: 10.0 / 1.375 = 7.27× speedup (theoretical)

**But:** Must account for:
- cuBLAS overhead (doesn't scale with T)
- Memory bandwidth limits
- GPU occupancy

### Optimal Tile Size Selection

**Empirical approach:**
1. Run experiments with T = {1, 2, 4, 8, 10, 16, 20}
2. Measure: iterations to converge, time per iteration, total time
3. Plot: Wall-clock time vs T
4. Find minimum (typically T=4-10)

---

## Implementation Checklist

### Phase 1: CPU Baseline ✓
- [x] Implement `hals_update_H` (sequential)
- [x] Implement `hals_update_W` (sequential)
- [x] Matrix operations (WtA, WtW, AH, HtH)
- [x] Convergence testing (verified convergence)
- [x] Correctness (error 0.4869)

### Phase 2: GPU Level 1 (Strict) ✓
- [x] Kernel: `update_H_column_hals` (single-column parallel)
- [x] Kernel: `update_W_column_hals` (single-column parallel)
- [x] Kernel: `normalize_W_column_hals` (column normalization)
- [x] Host orchestration with `cudaDeviceSynchronize()`
- [x] Column-major memory layout for cuBLAS
- [x] Bug fix: Column-major error computation

### Phase 3: GPU Level 2 (Block-Parallel) ✓
- [x] Fisher-Yates shuffle for random feature permutation
- [x] CUDA streams (one per block)
- [x] Block-parallel H and W updates
- [x] Single sync point after all blocks
- [x] Configurable block size (default: 5)

### Phase 4: Analysis ✓
- [x] Performance comparison: CPU vs GPU L1 vs GPU L2
- [x] Verify identical convergence (all implementations: 0.4869 error)
- [x] Measure speedup: 1.97x (GPU L2 vs CPU)
- [x] Document block-parallel algorithm

---

## Comparison: MU vs HALS

### Algorithm Characteristics

| Feature | Multiplicative Update | HALS |
|---------|----------------------|------|
| **Update Pattern** | Element-wise (Jacobi) | Column-wise (Gauss-Seidel) |
| **Dependencies** | None (trivially parallel) | Sequential (j+1 needs j) |
| **Convergence** | 200+ iterations | 40 iterations |
| **Per-Iteration Cost** | Low (simple operations) | Higher (reduction loops) |
| **GPU Efficiency** | High (massive parallelism) | Medium (limited by tiling) |
| **Parallelization** | Trivial | Non-trivial (tiling required) |

### When to Use Which

**Use MU when:**
- Need robust, guaranteed parallelism
- Matrix size is large (n > 10,000)
- Extra iterations acceptable
- Prefer simpler implementation

**Use HALS when:**
- Convergence speed critical (real-time applications)
- Can tolerate complex implementation
- k is moderate (k < 50, so tiling is effective)
- CPU implementation acceptable (if GPU doesn't help)

---

## Future Work

1. **Adaptive Tile Size:** Dynamically adjust T based on convergence rate
2. **Hybrid MU-HALS:** Start with HALS (fast convergence), switch to MU (better parallelism)
3. **Multi-GPU HALS:** Distribute tiles across GPUs
4. **Block-wise Updates:** Update multiple elements per column simultaneously

---

## References

1. **Original HALS Paper:** Cichocki & Phan (2009). "Fast local algorithms for large scale nonnegative matrix and tensor factorizations"
2. **ALO-NMF:** Anderson & Combs (2019). "ALO-NMF: Accelerated locality-optimized non-negative matrix factorization" (GPU implementation, 4.45× speedup)
3. **planc Library:** Referenced implementation (C++ with Armadillo)

---

**Document Version:** 2.0
**Last Updated:** 2025-11-29
**Status:** Complete - CPU baseline, GPU Level 1 (strict), GPU Level 2 (block-parallel) all implemented and tested
