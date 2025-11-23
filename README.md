# CUDA Sparse vs Dense NMF Optimization

A systematic study of GPU optimization techniques for Non-negative Matrix Factorization (NMF), exploring both dense and sparse matrix implementations.

## Overview

This project implements progressive optimization levels for GPU-accelerated NMF using CUDA, cuBLAS, and cuSPARSE. The goal is to understand:
1. How different optimization techniques affect performance
2. When sparse matrix formats provide benefits vs overhead
3. The practical limits of GPU optimization for this algorithm

## Key Finding

**Dense matrix operations with kernel-level optimizations outperform sparse implementations even at 90% sparsity** due to cuBLAS efficiency and the algorithmic structure of NMF.

## What's Implemented

### Dense Implementations

| Level | Optimization | Speedup | Implementation |
|-------|-------------|---------|----------------|
| **Level 1** | Naive GPU baseline | 1.0x | Separate multiply/divide kernels |
| **Level 2** | Memory optimized | **2.0x** | Kernel fusion + 4-way ILP |
| **Level 3** | Compute optimized | ~1.1x | 8-way ILP + tuning (in progress) |

### Sparse Implementations

- **Basic CSR** (`nmf_sparse_gpu.cu`) - Standard sparse format
- **Hybrid** (`nmf_sparse_gpu_v3.cu`) - Mixed cuBLAS/cuSPARSE
- **Transpose** (`nmf_sparse_gpu_v3_transpose.cu`) - Optimized memory layout

**Result:** All sparse variants are 1.5-2.5x **slower** than optimized dense (Level 2)

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

## Repository Structure

```
cuda-sparse-nmf-optimization/
├── README.md                           # This file
├── IMPLEMENTATION.md                   # Technical implementation details
├── FINAL_COMPREHENSIVE_ANALYSIS.md     # Complete performance analysis
├── BENCHMARKS.md                       # Benchmark methodology & results
│
├── src/
│   ├── nmf_cpu.py                      # NumPy baseline
│   ├── nmf_dense_gpu_v1_naive.cu       # Level 1: Naive GPU
│   ├── nmf_dense_gpu_v2_memory.cu      # Level 2: Memory optimized
│   ├── nmf_dense_gpu_v3_compute.cu     # Level 3: Compute optimized (WIP)
│   ├── nmf_sparse_gpu.cu               # Basic sparse CSR
│   ├── nmf_sparse_gpu_v3.cu            # Hybrid sparse/dense
│   ├── nmf_sparse_gpu_v3_transpose.cu  # Transpose optimization
│   ├── utils.h / utils.cu              # Shared utilities
│
├── data/
│   ├── generate_matrix.py              # Matrix generation
│   └── README.md                       # Data module docs
│
└── results/
    ├── naive_metrics.txt
    ├── memory_opt_metrics.txt
    └── sparse_metrics.txt
```

## Quick Start

### Prerequisites
- CUDA Toolkit 11.0+
- NVIDIA GPU with Compute Capability 6.0+
- gcc/g++ 7.0+

### Build

```bash
# Build all implementations
make all

# Or build specific levels
make level1    # Naive GPU
make level2    # Memory optimized
make level3    # Compute optimized
make sparse    # Sparse implementations
```

### Generate Test Data

```bash
# Dense matrix (1000×1000)
python3 data/generate_matrix.py --size 1000 --sparsity 0.0 --output data/dense_1000.bin

# Sparse matrix (1000×1000, 90% zeros)
python3 data/generate_matrix.py --size 1000 --sparsity 0.9 --output data/sparse_1000.bin
```

### Run Benchmarks

```bash
# Level 1 (Naive)
./nmf_naive data/dense_1000.bin 20 50

# Level 2 (Optimized)
./nmf_memory_opt data/dense_1000.bin 20 50

# Sparse (on 90% sparse data)
./nmf_sparse data/sparse_1000.bin 20 50
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

- **[IMPLEMENTATION.md](IMPLEMENTATION.md)** - Detailed technical implementation
- **[FINAL_COMPREHENSIVE_ANALYSIS.md](FINAL_COMPREHENSIVE_ANALYSIS.md)** - Complete performance analysis
- **[BENCHMARKS.md](BENCHMARKS.md)** - Fair benchmarking methodology (renamed from FAIR_BENCHMARK_RESULTS.md)
- **[data/README.md](data/README.md)** - Data generation module

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
