# GPU-Accelerated NMF: Multiplicative Update vs HALS

A comprehensive study of GPU optimization and parallelization strategies for Non-negative Matrix Factorization (NMF), comparing **two complementary algorithms** with different parallelization challenges.

## Overview

This project implements and compares two NMF algorithms on GPU:
1. **Multiplicative Update (MU)** - Trivially parallel element-wise operations, Amdahl's Law limits
2. **HALS (Hierarchical Alternating Least Squares)** - Sequential Gauss-Seidel dependencies, tiled batching solution

### Key Questions
1. How do different optimization techniques affect performance?
2. When do sparse matrix formats provide benefits vs overhead?
3. What are the practical limits of single-GPU optimization?
4. How do we parallelize inherently sequential algorithms?
5. When does faster convergence translate to faster wall-clock time?

## Key Finding

**Dense matrix operations with kernel-level optimizations outperform sparse implementations even at 90% sparsity** due to cuBLAS efficiency and the algorithmic structure of NMF.

## What's Implemented

### Multiplicative Update (MU) - Progressive Optimization

| Level | Optimization | Speedup | Implementation | Parallelization |
|-------|-------------|---------|----------------|-----------------|
| **Level 1** | Naive GPU baseline | 1.0x | Separate multiply/divide kernels | Trivial (element-wise) |
| **Level 2** | Memory optimized | **2.0x** | Kernel fusion + 4-way ILP | Memory bandwidth focus |
| **Level 3** | Compute + Streams | 0.66x | 8-way ILP + CUDA streams | Stream overhead > benefit |
| **Level 4** | Multi-GPU | 1.7-1.9x* | OpenMP + AllReduce | Distributed computing |

*Theoretical speedup for 2 GPUs (91% efficiency)

### HALS - Tiled Batching Parallelization

| Level | Approach | Convergence | Implementation | Challenge |
|-------|----------|-------------|----------------|-----------|
| **Sequential** | CPU baseline | 40 iters | Gauss-Seidel updates | Inherently sequential |
| **Tiled (T=4-10)** | GPU parallel | 45-55 iters | Parallel tiles with stale dependencies | Convergence vs parallelism tradeoff |

### Sparse Implementations (Comparative Study)

- **Basic CSR** - Standard sparse format
- **Hybrid** - Mixed cuBLAS/cuSPARSE
- **Transpose** - Optimized memory layout

**Result:** All sparse variants are 1.5-2.5x **slower** than optimized dense due to algorithm structure

## Performance Summary

### 1000×1000 Matrix, Rank 20, 50 Iterations

| Implementation | Time (ms) | GFLOPS | Bandwidth (GB/s) | vs Dense L2 |
|----------------|-----------|--------|------------------|-------------|
| Dense L1 (Naive) | 43.67 | 95.45 | 9.89 | 0.49x |
| **Dense L2 (Optimized)** | **21.43** | **194.52** | **19.79** | **1.00x** ✓ |
| Sparse (90% sparse) | 52.55 | 18.60 | 1.93 | 0.41x |

**Level 2 is 2.5x faster than sparse despite 90% sparsity!**

## Why Sparse Doesn't Win

The NMF multiplicative update algorithm:
```
H = H .* (W^T × X) ./ (W^T × W × H + eps)
W = W .* (X × H^T) ./ (W × H × H^T + eps)
```

**Critical insight:** Only 2 out of 6 matrix operations can use sparsity:
- ✅ `W^T × X` (sparse × dense)
- ✅ `X × H^T` (sparse × dense)
- ❌ `W^T × W` (dense × dense)
- ❌ `W × H` (dense × dense)
- ❌ `H × H^T` (dense × dense)
- ❌ `W × (H × H^T)` (dense × dense)

Since W and H remain dense throughout, 67% of matrix operations get no benefit from sparse X, while CSR format adds overhead.

## MU vs HALS: Different Parallelization Challenges

### Multiplicative Update (MU)
**Algorithm:**
```
H = H .* (W^T × X) ./ (W^T × W × H + eps)
W = W .* (X × H^T) ./ (W × H × H^T + eps)
```

**Parallelization:**
- ✅ **Trivially parallel**: Element-wise operations can run independently
- ✅ **GPU-friendly**: Massive parallelism (millions of independent updates)
- ❌ **Slow convergence**: Requires 200+ iterations
- ❌ **Amdahl's Law**: cuBLAS dominates 85% of runtime (already optimal)

**Key Challenge:** Optimizing the 15% element-wise operations (achieved 2× via kernel fusion + ILP)

### HALS (Hierarchical Alternating Least Squares)
**Algorithm:**
```
For each column j = 0 to k-1:    H[:,j] = H[:,j] + WtA[j,:] - H × WtW[:,j]  (uses updated H[:,0..j-1])
```

**Parallelization:**
- ✅ **Fast convergence**: Requires only 40 iterations
- ❌ **Sequential dependencies**: Column j+1 depends on updated column j (Gauss-Seidel)
- ❌ **GPU-hostile**: Inherently sequential updates

**Key Challenge:** Breaking sequential dependencies via **tiled batching**

### Tiled Batching Solution
```
Instead of: for j = 0 to k-1: update H[:,j]  (sequential)
Do this:    for tile = 0 to k step T:
              parallel_for j in [tile, tile+T): update H[:,j]  (accept stale deps)
```

**Tradeoff:**
- T=1: Exact HALS (fully sequential, 40 iters)
- T=4-10: **Optimal balance** (4-10× parallelism, 45-55 iters)
- T=k: Maximum parallelism (converges to MU, 200+ iters)

## Repository Structure

```
MU_Parallel/
├── README.md                           # This file (MU + HALS overview)
├── MU_IMPLEMENTATION.md                # Multiplicative Update technical details
├── HALS_IMPLEMENTATION.md              # HALS + tiled batching details
├── COMPARATIVE_ANALYSIS.md             # MU vs HALS comparison
│
├── src/
│   ├── mu/                             # Multiplicative Update implementations
│   │   ├── nmf_dense_gpu_v1_naive.cu   # Level 1: Naive GPU baseline
│   │   ├── nmf_dense_gpu_v2_memory.cu  # Level 2: Kernel fusion + 4-way ILP
│   │   ├── nmf_dense_gpu_v3_compute.cu # Level 3: 8-way ILP + CUDA streams
│   │   └── nmf_dense_gpu_v4_multigpu.cu# Level 4: Multi-GPU with OpenMP
│   │
│   ├── hals/                           # HALS implementations
│   │   ├── nmf_hals_cpu.cpp            # Sequential CPU baseline
│   │   ├── nmf_hals_gpu_v1_naive.cu    # Basic GPU implementation
│   │   └── nmf_hals_gpu_v2_tiled.cu    # Tiled batching (T=4-10)
│   │
│   ├── nmf_cpu.py                      # NumPy baseline
│   ├── nmf_sparse_gpu_v3.cu            # Sparse: Hybrid cuBLAS/cuSPARSE
│   ├── nmf_sparse_gpu_v3_transpose.cu  # Sparse: Transpose optimization
│   └── utils.h / utils.cu              # Shared utilities
│
├── data/
│   ├── generate_matrix.py              # Matrix generation
│   └── README.md                       # Data module docs
│
└── results/
    ├── mu_metrics.txt                  # MU performance data
    ├── hals_metrics.txt                # HALS performance data
    └── comparative_analysis.txt        # MU vs HALS comparison
```

## Quick Start

### Prerequisites
- CUDA Toolkit 11.0+
- NVIDIA GPU with Compute Capability 6.0+
- gcc/g++ 7.0+
- Python 3.7+ with numpy, scipy, PIL, matplotlib (for image datasets)

### Build

```bash
# Build Multiplicative Update (MU) implementations
make naive         # MU Level 1: Naive GPU
make memory-opt    # MU Level 2: Kernel fusion + 4-way ILP
make compute-opt   # MU Level 3: 8-way ILP + CUDA streams
make multigpu      # MU Level 4: Multi-GPU with OpenMP

# Build HALS implementations
make hals-cpu      # HALS: Sequential CPU baseline
make hals-gpu      # HALS: GPU with tiled batching (coming soon)

# Build all
make all           # Builds MU + sparse implementations
```

### Option A: Test on Real Images (Recommended)

**Demonstrates NMF actually works and can reconstruct images:**

```bash
# 1. Prepare image datasets (downloads/creates 3 test images)
python3 data/prepare_image_datasets.py --sizes 128 256 512

# 2. Run NMF on images
./nmf_memory_opt data/images_256.bin 10 100

# 3. Visualize results (shows original vs reconstructed)
python3 scripts/visualize_nmf_results.py --size 256 --images 3 --rank 10
```

See **[IMAGE_WORKFLOW.md](IMAGE_WORKFLOW.md)** for complete guide.

### Option B: Test on Synthetic Matrices

**For performance benchmarking:**

```bash
# Generate random matrix
python3 data/generate_matrix.py --size 1000 --sparsity 0.0 --output data/dense_1000.bin

# Run benchmarks
./nmf_naive data/dense_1000.bin 20 50       # Level 1
./nmf_memory_opt data/dense_1000.bin 20 50  # Level 2
./nmf_compute_opt data/dense_1000.bin 20 50 # Level 3
```

## Key Optimizations (Level 1 → Level 2)

### 1. Kernel Fusion
**Before (Level 1):** Separate multiply and divide kernels
```cuda
elementwise_multiply<<<>>>(d_H, d_WtX, d_H, size);     // Read H, write H
elementwise_divide<<<>>>(d_H, d_temp_H, d_H, size);    // Read H again, write H
```

**After (Level 2):** Combined operation
```cuda
elementwise_multiply_divide_fused<<<>>>(d_H, d_WtX, d_temp_H, size);  // Read H once, write once
```

**Impact:** 50% fewer kernel launches, 20% less memory traffic

### 2. Instruction-Level Parallelism (ILP)
Each thread processes 4 elements instead of 1, hiding memory latency:
```cuda
// Process 4 elements per thread
float in0 = input[idx];
float in1 = input[idx + 1];
float in2 = input[idx + 2];
float in3 = input[idx + 3];

// Compute while waiting for memory
in0 = in0 * num0 / (den0 + eps);
in1 = in1 * num1 / (den1 + eps);
in2 = in2 * num2 / (den2 + eps);
in3 = in3 * num3 / (den3 + eps);
```

**Impact:** Better ALU utilization, 1.4x latency hiding

### Combined Result: 2x speedup

## Hardware Tested

- **NVIDIA GeForce RTX 3050 Ti Laptop** (192 GB/s bandwidth, 4GB memory)
- **NVIDIA GeForce RTX 4070** (504 GB/s bandwidth, 12GB memory)

Results consistent across both architectures.

## Documentation

### Technical Documentation
- **[IMPLEMENTATION.md](IMPLEMENTATION.md)** - Detailed technical implementation of all levels
- **[FINAL_COMPREHENSIVE_ANALYSIS.md](FINAL_COMPREHENSIVE_ANALYSIS.md)** - Complete performance analysis
- **[BENCHMARKS.md](BENCHMARKS.md)** - Fair benchmarking methodology

### Workflow Guides
- **[IMAGE_WORKFLOW.md](IMAGE_WORKFLOW.md)** - Complete guide for testing on real images
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Project status and cleanup summary
- **[data/README.md](data/README.md)** - Data generation module reference

## Conclusions

1. **Memory optimization matters:** Kernel fusion + ILP provides 2x speedup
2. **Algorithm structure determines format choice:** NMF's dense intermediate matrices make sparse formats inefficient
3. **cuBLAS is highly optimized:** Hard to beat even with specialized sparse operations
4. **For NMF specifically:** Use dense implementations regardless of input sparsity

## Future Work

- [ ] Complete Level 3 (compute optimizations)
- [ ] Multi-GPU implementation
- [ ] Mixed precision (FP16/FP32)
- [ ] Sparse NMF variants that maintain W, H sparsity

## License

MIT

## References

- Lee & Seung (1999). "Learning the parts of objects by non-negative matrix factorization"
- NVIDIA cuBLAS Documentation
- NVIDIA cuSPARSE Documentation
