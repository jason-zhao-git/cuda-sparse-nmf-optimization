# CSE587 Final Project: GPU-Accelerated Sparse vs Dense NMF

## Comparative Analysis of Dense and Sparse Matrix Factorization on GPU

**Timeline:** 2 weeks
**Platform:** CUDA on UMich Great Lakes (single GPU)
**Goal:** Learn CUDA + cuBLAS + cuSPARSE, understand sparse vs dense trade-offs

---

## Table of Contents
- [Project Overview](#project-overview)
- [Motivation](#motivation)
- [Algorithm: Multiplicative Update NMF](#algorithm-multiplicative-update-nmf)
- [Implementation Timeline](#implementation-timeline)
- [Technical Specifications](#technical-specifications)
- [Deliverables](#deliverables)
- [Analysis Plan](#analysis-plan)
- [Success Criteria](#success-criteria)
- [Resources & References](#resources--references)

---

## Project Overview

### Core Question
**When should you use sparse matrix operations vs dense matrix operations for GPU-accelerated NMF?**

This project implements the same NMF algorithm in two ways:
1. **Dense GPU** using cuBLAS (optimized dense linear algebra)
2. **Sparse GPU** using cuSPARSE (CSR format sparse operations)

Then comprehensively compares:
- **Performance** (time, memory)
- **Crossover points** (at what sparsity does sparse win?)
- **Practical guidelines** for real applications

### What Makes This Project Valuable

Unlike the simpler "just learn CUDA" approach, this project:
- ✅ Compares two fundamentally different approaches
- ✅ Provides actionable insights for practitioners
- ✅ Explores real trade-offs in scientific computing
- ✅ Has applications in single-cell genomics, topic modeling, etc.
- ✅ Teaches both cuBLAS AND cuSPARSE libraries

---

## Motivation

### Why NMF?
Non-negative Matrix Factorization is widely used in:
- **Bioinformatics:** Single-cell RNA sequencing (90-95% sparse!)
- **Computer Vision:** Image decomposition, facial recognition
- **Natural Language Processing:** Topic modeling, document clustering
- **Audio:** Source separation, music analysis

### Why Compare Sparse vs Dense?

Real-world data is often sparse, but:
- **Dense operations** are highly optimized (cuBLAS is FAST)
- **Sparse operations** save memory but have overhead
- **Crossover point** depends on sparsity level, matrix size, GPU architecture

**Nobody knows the answer without testing!** This project will provide concrete guidelines.

### Example: Single-Cell Genomics
A typical dataset:
- Matrix size: 20,000 genes × 10,000 cells
- Sparsity: ~95% (only 5% non-zero)
- Question: Should we use sparse or dense GPU operations?

**You'll answer this by the end of Week 2!**

---

## Algorithm: Multiplicative Update NMF

### Problem Statement
Given non-negative matrix **X** (m×n), find **W** (m×k) and **H** (k×n) such that:

```
X ≈ W × H
```

Where:
- **X:** Input data matrix (can be sparse!)
- **W:** Basis matrix (m×k, always dense)
- **H:** Coefficient matrix (k×n, always dense)
- **k:** Rank (typically k << min(m,n))

### Update Rules (Multiplicative Update)

```python
For each iteration:
    # Update H
    H = H .* (W^T × X) ./ (W^T × W × H + eps)

    # Update W
    W = W .* (X × H^T) ./ (W × H × H^T + eps)
```

**Notation:**
- `.*` = element-wise multiplication
- `./` = element-wise division
- `eps = 1e-10` = small constant to prevent division by zero

### Key Insight for Sparse Implementation
**Only X can be sparse!** W and H must stay dense because:
- They're much smaller (k << min(m, n))
- They quickly become dense during iteration
- Dense operations on small matrices are faster

**Operations that can benefit from sparsity:**
- ✅ `W^T × X` (sparse matrix on right)
- ✅ `X × H^T` (sparse matrix on left)

**Operations that stay dense:**
- `W^T × W` (small dense × dense)
- `H × H^T` (small dense × dense)
- Element-wise operations on W and H

---

## Implementation Timeline

### Week 1: Implementation (Days 1-7)

#### Day 1: CPU Baseline & Environment Setup
**Goal:** Reference implementation for correctness

**Tasks:**
- [ ] Set up Great Lakes environment
- [ ] Implement `nmf_cpu.py` using NumPy
- [ ] Test on small matrix (100×100)
- [ ] Validate convergence

**Deliverable:** Working CPU baseline

---

#### Days 2-3: Dense GPU Implementation
**Goal:** Learn cuBLAS, basic CUDA programming

**File:** `nmf_dense_gpu.cu`

**Key Components:**
1. Memory management (cudaMalloc, cudaMemcpy)
2. cuBLAS integration (cublasSgemm)
3. Custom kernels for element-wise operations
4. Main iteration loop

**Learning Focus:**
- CUDA memory model (host vs device)
- Kernel launch syntax `<<<grid, block>>>`
- Thread indexing
- cuBLAS API

**Deliverable:** Working dense GPU version matching CPU results

---

#### Days 4-7: Sparse GPU Implementation
**Goal:** Learn cuSPARSE, CSR format, sparse operations

**File:** `nmf_sparse_gpu.cu`

**Key Components:**
1. CSR format utilities (conversion, loading)
2. cuSPARSE setup (handles, descriptors)
3. Sparse-dense matrix multiply
4. Hybrid algorithm (sparse for X, dense for W/H)

**Learning Focus:**
- CSR (Compressed Sparse Row) format
- cuSPARSE API
- Sparse matrix descriptors
- Buffer management

**Deliverable:** Working sparse GPU version

---

### Week 2: Analysis & Benchmarking (Days 8-14)

#### Days 8-9: Data Generation & Initial Benchmarking
**Goal:** Create comprehensive test suite

**Tasks:**
- [ ] Generate sparse matrices at various sparsity levels
- [ ] Implement automated benchmarking script
- [ ] Run initial performance tests
- [ ] Verify correctness across all test cases

**Deliverable:** Benchmark data for multiple configurations

---

#### Days 10-11: Comprehensive Analysis
**Goal:** Identify crossover points and bottlenecks

**Tasks:**
- [ ] Run full benchmark suite
- [ ] Profile with Nsight Compute
- [ ] Analyze time vs sparsity
- [ ] Analyze memory usage
- [ ] Identify bottlenecks

**Deliverable:** Complete performance data + profiling results

---

#### Days 12-14: Report Writing & Visualization
**Goal:** Communicate findings clearly

**Tasks:**
- [ ] Create performance plots
- [ ] Write analysis sections
- [ ] Document insights and recommendations
- [ ] Prepare presentation (if required)

**Deliverable:** Final report (8-10 pages)

---

## Technical Specifications

### Dense Implementation (cuBLAS)

**Data Structure:**
```c
// All matrices in row-major format
float *d_X;     // m × n (dense)
float *d_W;     // m × k (dense)
float *d_H;     // k × n (dense)
```

**Matrix Operations:**
```c
// W^T × X using cuBLAS
cublasSgemm(handle,
            CUBLAS_OP_T,           // Transpose W
            CUBLAS_OP_N,           // Don't transpose X
            k, n, m,               // Dimensions
            &alpha, d_W, m,        // W matrix
            d_X, m,                // X matrix
            &beta, d_WtX, k);      // Output
```

**Element-wise Operations:**
```c
__global__ void elementwise_multiply(float* A, float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * B[idx];
    }
}

__global__ void elementwise_divide_eps(float* A, float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] / (B[idx] + 1e-10f);
    }
}
```

---

### Sparse Implementation (cuSPARSE)

**CSR Format:**
```c
// Sparse matrix X in CSR format
int *d_csrRowPtr;    // Size: m+1
int *d_csrColInd;    // Size: nnz
float *d_csrVal;     // Size: nnz

// W and H remain dense!
float *d_W;          // m × k (dense)
float *d_H;          // k × n (dense)
```

**CSR Format Example:**
```
Dense matrix (3×3):
[1.0  0.0  2.0]
[0.0  3.0  0.0]
[4.0  0.0  5.0]

CSR representation:
values   = [1.0, 2.0, 3.0, 4.0, 5.0]
colInd   = [0,   2,   1,   0,   2  ]
rowPtr   = [0,   2,   3,   5      ]

Interpretation:
- Row 0: values[0:2] = [1.0, 2.0] at columns [0, 2]
- Row 1: values[2:3] = [3.0] at column [1]
- Row 2: values[3:5] = [4.0, 5.0] at columns [0, 2]
```

**Sparse-Dense Multiply:**
```c
// Create sparse matrix descriptor
cusparseSpMatDescr_t matX;
cusparseCreateCsr(&matX, m, n, nnz,
                  d_csrRowPtr, d_csrColInd, d_csrVal,
                  CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                  CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

// Create dense matrix descriptor
cusparseDnMatDescr_t matW;
cusparseCreateDnMat(&matW, m, k, m, d_W, CUDA_R_32F, CUSPARSE_ORDER_COL);

// Allocate buffer
size_t bufferSize;
cusparseSpMM_bufferSize(sp_handle,
                        CUSPARSE_OPERATION_TRANSPOSE,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, matW, matX, &beta, matWtX,
                        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                        &bufferSize);
cudaMalloc(&d_buffer, bufferSize);

// Perform W^T × X (sparse)
cusparseSpMM(sp_handle,
             CUSPARSE_OPERATION_TRANSPOSE,
             CUSPARSE_OPERATION_NON_TRANSPOSE,
             &alpha, matW, matX, &beta, matWtX,
             CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, d_buffer);
```

---

## Deliverables

### Code (Week 1)

#### 1. CPU Baseline: `nmf_cpu.py`
Simple NumPy implementation for validation
```bash
python nmf_cpu.py --input data/X_1000x1000_sp90.npz --k 20 --iters 100
```

#### 2. Dense GPU: `nmf_dense_gpu.cu`
cuBLAS-based implementation
```bash
./nmf_dense_gpu data/X_1000x1000_sp90.npz 20 100
```

#### 3. Sparse GPU: `nmf_sparse_gpu.cu`
cuSPARSE-based implementation
```bash
./nmf_sparse_gpu data/X_1000x1000_sp90.npz 20 100
```

#### 4. Utilities: `utils.cu/utils.h`
Shared functionality:
- Error checking macros (CUDA_CHECK, CUBLAS_CHECK, CUSPARSE_CHECK)
- Timing utilities (cudaEvent)
- Memory monitoring
- Data loading/generation

#### 5. Build System: `Makefile`
```makefile
NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_70 -lcublas -lcusparse

all: nmf_dense_gpu nmf_sparse_gpu

nmf_dense_gpu: src/nmf_dense_gpu.cu src/utils.cu
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

nmf_sparse_gpu: src/nmf_sparse_gpu.cu src/utils.cu
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

clean:
	rm -f nmf_dense_gpu nmf_sparse_gpu *.o
```

---

### Analysis (Week 2)

#### 1. Data Generation: `scripts/generate_data.py`
```python
import numpy as np
from scipy.sparse import random, save_npz

def generate_sparse_matrix(m, n, sparsity, seed=42):
    """Generate sparse non-negative matrix"""
    np.random.seed(seed)
    density = 1.0 - sparsity
    X = random(m, n, density=density, format='csr', dtype=np.float32)
    X.data = np.abs(X.data)  # Ensure non-negative
    return X

# Test configurations
sizes = [(1000, 1000), (5000, 5000), (10000, 10000)]
sparsities = [0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]

for m, n in sizes:
    for sp in sparsities:
        X = generate_sparse_matrix(m, n, sp)
        filename = f"data/X_{m}x{n}_sp{int(sp*100)}.npz"
        save_npz(filename, X)
        print(f"Generated {filename}: {X.nnz:,} non-zeros ({sp*100}% sparse)")
```

#### 2. Benchmarking: `scripts/benchmark.sh`
```bash
#!/bin/bash
#SBATCH --account=cse587f25s001_class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --mem=16GB

module load cuda/12.2.0 gcc/11.2.0

SIZES="1000 5000 10000"
SPARSITIES="50 70 80 85 90 95 99"
K=20
ITERS=100

echo "size,sparsity,method,time_ms,memory_mb,final_error,nnz" > results.csv

for size in $SIZES; do
    for sp in $SPARSITIES; do
        input="data/X_${size}x${size}_sp${sp}.npz"

        # Dense GPU
        echo "Running dense GPU: $input"
        ./nmf_dense_gpu $input $K $ITERS >> results.csv

        # Sparse GPU
        echo "Running sparse GPU: $input"
        ./nmf_sparse_gpu $input $K $ITERS >> results.csv

        # CPU baseline (only for smaller sizes)
        if [ $size -le 1000 ]; then
            python src/nmf_cpu.py $input $K $ITERS >> results.csv
        fi
    done
done

echo "Benchmarking complete! Results in results.csv"
```

#### 3. Visualization: `scripts/plot_results.py`
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('results.csv')

# Plot 1: Time vs Sparsity
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, size in enumerate([1000, 5000, 10000]):
    data = df[df['size'] == size]

    for method in ['dense_gpu', 'sparse_gpu']:
        subset = data[data['method'] == method]
        axes[i].plot(subset['sparsity'], subset['time_ms'],
                    marker='o', label=method, linewidth=2)

    axes[i].set_xlabel('Sparsity Level', fontsize=12)
    axes[i].set_ylabel('Time (ms)', fontsize=12)
    axes[i].set_title(f'Matrix Size: {size}×{size}', fontsize=14)
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/time_vs_sparsity.png', dpi=300)

# Plot 2: Speedup (dense/sparse ratio)
# Plot 3: Memory comparison
# Plot 4: Crossover analysis
```

#### 4. Report: `docs/final_report.md` (8-10 pages)

**Structure:**

**1. Introduction (1 page)**
- Motivation: Why compare sparse vs dense?
- Applications (single-cell genomics, NLP, etc.)
- Research question and goals

**2. Background (1-2 pages)**
- NMF algorithm overview
- Multiplicative Update derivation
- CSR sparse matrix format
- cuBLAS vs cuSPARSE overview

**3. Implementation (2-3 pages)**
- **Dense approach:**
  - Memory layout
  - cuBLAS operations
  - Custom kernels
- **Sparse approach:**
  - CSR format handling
  - cuSPARSE API usage
  - Hybrid algorithm design
- **Code snippets showing key differences**

**4. Performance Analysis (2-3 pages)** ⭐ **MOST IMPORTANT SECTION**
- **Time vs Sparsity:**
  - Curves for different matrix sizes
  - Crossover point identification
  - Interpretation of results
- **Memory Usage:**
  - Dense: O(mn) storage
  - Sparse: O(nnz) storage
  - Break-even analysis
- **Bottleneck Analysis:**
  - Profiling results (Nsight Compute)
  - Where does each approach spend time?
  - Memory bandwidth utilization
- **Practical Guidelines:**
  - When to use dense
  - When to use sparse
  - Decision flowchart

**5. Learning Outcomes (1 page)**
- CUDA concepts mastered
- Surprises and challenges
- What would you do differently?

**6. Conclusions & Future Work (1 page)**
- Key findings summary
- Guidelines for practitioners
- Extensions: multi-GPU, mixed precision, etc.

---

## Analysis Plan

### Key Questions to Answer

#### 1. **Crossover Point**
At what sparsity level does sparse beat dense?

**Hypothesis:**
- Low sparsity (< 70%): Dense is faster (overhead dominates)
- High sparsity (> 90%): Sparse is faster (memory savings dominate)
- Middle range: Depends on matrix size

**Analysis:**
- Plot time vs sparsity for each size
- Identify crossover points
- Explain why (memory bandwidth, arithmetic intensity)

#### 2. **Memory Scaling**
How does memory usage scale?

**Dense:** `Memory = m * n * sizeof(float) = 4mn bytes`
**Sparse:** `Memory = (nnz * 8 + (m+1) * 4) bytes`
(8 bytes per non-zero: 4 for value + 4 for column index)

**Break-even:** `4mn = 8 * nnz + 4(m+1)`

Solve for sparsity where they're equal.

#### 3. **Performance Bottlenecks**
Where does each approach spend time?

**Use Nsight Compute to measure:**
- Kernel execution time
- Memory throughput
- Occupancy
- Cache hit rates

**Expected findings:**
- Dense: Memory bandwidth limited
- Sparse: Irregular memory access patterns

#### 4. **Practical Guidelines**
Given a real dataset, which should you use?

**Create decision flowchart:**
```
Is sparsity > 90%?
  Yes → Is matrix size > 5000×5000?
    Yes → Use sparse
    No → Benchmark both
  No → Use dense
```

---

## Success Criteria

### Minimum Viable Project (Pass: C/B)
- ✅ Dense GPU implementation works
- ✅ Sparse GPU implementation works
- ✅ Both match CPU baseline
- ✅ Basic performance comparison (1-2 plots)
- ✅ Report documents implementation

### Target Achievement (B+/A-)
- ✅ All above requirements
- ✅ Comprehensive benchmarking (multiple sizes + sparsities)
- ✅ Crossover point clearly identified
- ✅ Memory analysis included
- ✅ Well-written report with clear insights
- ✅ Code is clean and well-documented

### Excellent Achievement (A/A+)
- ✅ All above requirements
- ✅ Profiling analysis (Nsight Compute)
- ✅ Deep performance understanding (bottlenecks explained)
- ✅ Practical guidelines for practitioners
- ✅ Production-quality code
- ✅ Novel insights or unexpected findings
- ✅ Clear, publication-quality visualizations
- ✅ Potential for conference poster/paper

**Bonus (A+ territory):**
- Hybrid approach explored (switch between sparse/dense at runtime)
- Comparison with other libraries (Eigen, cuTENSOR)
- Application to real dataset (e.g., single-cell data)

---

## Resources & References

### CUDA Libraries Documentation
- **cuBLAS:** https://docs.nvidia.com/cuda/cublas/
- **cuSPARSE:** https://docs.nvidia.com/cuda/cusparse/
- **CUDA Programming Guide:** https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **Nsight Compute:** https://docs.nvidia.com/nsight-compute/

### Great Lakes HPC
- **GPU Guide:** https://arc.umich.edu/greatlakes/gpus/
- **SLURM Documentation:** https://arc.umich.edu/greatlakes/slurm-user-guide/
- **Account:** `cse587f25s001_class`

### NMF Algorithm References
- **Lee & Seung (1999):** "Learning the parts of objects by non-negative matrix factorization" (Nature)
- **Gillis (2014):** "The Why and How of Nonnegative Matrix Factorization"
- **scikit-learn NMF:** https://scikit-learn.org/stable/modules/decomposition.html#nmf

### Sparse Matrix Formats
- **CSR Format Tutorial:** http://netlib.org/linalg/html_templates/node91.html
- **SciPy Sparse:** https://docs.scipy.org/doc/scipy/reference/sparse.html

### Example Datasets (for validation)
- **Single-cell RNA-seq:** https://www.10xgenomics.com/datasets
- **Text corpus:** 20 Newsgroups, Reuters

---

## Utilities & Code Templates

### Error Checking Macros

```c
// utils.h
#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA error checking
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// cuBLAS error checking
#define CUBLAS_CHECK(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error at %s:%d - code %d\n", \
                __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}

// cuSPARSE error checking
#define CUSPARSE_CHECK(call) { \
    cusparseStatus_t status = call; \
    if (status != CUSPARSE_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSPARSE Error at %s:%d - code %d\n", \
                __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}

#endif // UTILS_H
```

### Timing Utilities

```c
// utils.cu
#include "utils.h"

class CudaTimer {
private:
    cudaEvent_t start, stop;

public:
    CudaTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void startTimer() {
        cudaEventRecord(start);
    }

    float stopTimer() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        return milliseconds;
    }
};

// Usage:
// CudaTimer timer;
// timer.startTimer();
// ... computation ...
// float time_ms = timer.stopTimer();
```

### Memory Monitoring

```c
void print_gpu_memory_usage() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t used_mem = total_mem - free_mem;

    printf("GPU Memory Usage:\n");
    printf("  Used:  %.2f MB\n", used_mem / 1e6);
    printf("  Free:  %.2f MB\n", free_mem / 1e6);
    printf("  Total: %.2f MB\n", total_mem / 1e6);
}
```

### Data Loading (for .npz files)

```python
# save_as_txt.py - Convert .npz to text format for C++
import numpy as np
from scipy.sparse import load_npz
import sys

if len(sys.argv) != 2:
    print("Usage: python save_as_txt.py input.npz")
    sys.exit(1)

X = load_npz(sys.argv[1])
X_csr = X.tocsr()

# Save CSR format
basename = sys.argv[1].replace('.npz', '')
np.savetxt(f"{basename}_values.txt", X_csr.data)
np.savetxt(f"{basename}_colind.txt", X_csr.indices, fmt='%d')
np.savetxt(f"{basename}_rowptr.txt", X_csr.indptr, fmt='%d')

# Save metadata
with open(f"{basename}_meta.txt", 'w') as f:
    f.write(f"{X_csr.shape[0]} {X_csr.shape[1]} {X_csr.nnz}\n")

print(f"Saved: {basename}_*.txt")
```

---

## Project Structure

```
nmf-sparse-dense/
├── README.md                      # This file
├── Makefile                       # Build system
│
├── src/
│   ├── nmf_cpu.py                # CPU baseline (NumPy)
│   ├── nmf_dense_gpu.cu          # Dense GPU (cuBLAS)
│   ├── nmf_sparse_gpu.cu         # Sparse GPU (cuSPARSE)
│   ├── utils.h                   # Utility headers
│   └── utils.cu                  # Utility implementations
│
├── scripts/
│   ├── generate_data.py          # Generate test matrices
│   ├── benchmark.sh              # SLURM batch script
│   ├── plot_results.py           # Visualization
│   └── save_as_txt.py            # Convert .npz to text
│
├── data/
│   ├── X_1000x1000_sp50.npz     # Test data
│   ├── X_1000x1000_sp70.npz
│   └── ...
│
├── results/
│   ├── results.csv               # Benchmark data
│   ├── plots/
│   │   ├── time_vs_sparsity.png
│   │   ├── memory_comparison.png
│   │   └── speedup_analysis.png
│   └── profiles/
│       ├── dense_profile.ncu-rep
│       └── sparse_profile.ncu-rep
│
└── docs/
    └── final_report.md           # Final report
```

---

## Quick Start Guide

### 1. Environment Setup (Great Lakes)

```bash
# Login to Great Lakes
ssh uniqname@greatlakes.arc-ts.umich.edu

# Request interactive GPU node
salloc --account=cse587f25s001_class --partition=gpu --gres=gpu:1 --time=02:00:00 --mem=8GB

# Load modules
module load cuda/12.2.0
module load gcc/11.2.0
module load python/3.10

# Verify GPU
nvidia-smi
nvcc --version
```

### 2. Create Project Structure

```bash
cd ~
mkdir -p nmf-sparse-dense/{src,scripts,data,results/{plots,profiles},docs}
cd nmf-sparse-dense
```

### 3. Day 1: CPU Baseline

```bash
cd src
# Create nmf_cpu.py (see template in full README)
python nmf_cpu.py
```

### 4. Day 3: Dense GPU

```bash
# Create nmf_dense_gpu.cu
cd ..
make nmf_dense_gpu
./nmf_dense_gpu
```

### 5. Day 7: Sparse GPU

```bash
# Create nmf_sparse_gpu.cu
make nmf_sparse_gpu
./nmf_sparse_gpu
```

### 6. Week 2: Benchmarking

```bash
# Generate test data
python scripts/generate_data.py

# Run benchmarks
sbatch scripts/benchmark.sh

# Visualize results
python scripts/plot_results.py
```

---

## Debugging Tips

### Common Issues

**1. cuSPARSE buffer size errors:**
```c
// Always query buffer size first!
size_t bufferSize = 0;
cusparseSpMM_bufferSize(..., &bufferSize);
cudaMalloc(&d_buffer, bufferSize);
```

**2. Matrix dimension mismatches:**
```c
// Add assertions in debug mode
#ifdef DEBUG
assert(m > 0 && n > 0 && k > 0);
assert(nnz <= m * n);
#endif
```

**3. Memory leaks:**
```bash
# Use cuda-memcheck
cuda-memcheck --leak-check full ./nmf_dense_gpu
```

**4. Incorrect results:**
```c
// Compare with CPU at each iteration
#ifdef DEBUG
cudaMemcpy(h_H, d_H, k*n*sizeof(float), cudaMemcpyDeviceToHost);
float cpu_error = compare_with_cpu(h_H, h_H_cpu, k*n);
printf("Error vs CPU: %.6e\n", cpu_error);
#endif
```

### Profiling Commands

```bash
# Basic profiling
nvprof ./nmf_dense_gpu

# Detailed analysis
ncu --set full -o dense_profile ./nmf_dense_gpu
ncu-ui dense_profile.ncu-rep

# Specific metrics
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed ./nmf_dense_gpu
```

---

## Learning Checkpoints

### After Day 3 (Dense GPU working)
- [ ] Understand CUDA memory transfers
- [ ] Can explain cuBLAS matrix multiply
- [ ] Know how to write element-wise kernels
- [ ] Can calculate thread indices correctly

**Self-test:** "What does `cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, ...)` compute?"

### After Day 7 (Sparse GPU working)
- [ ] Understand CSR format
- [ ] Can convert dense to CSR
- [ ] Know cuSPARSE API for SpMM
- [ ] Understand sparse-dense hybrid approach

**Self-test:** "Why do we keep W and H dense?"

### After Day 14 (Project complete)
- [ ] Can identify crossover points
- [ ] Understand performance trade-offs
- [ ] Can give practical recommendations
- [ ] Mastered cuBLAS + cuSPARSE

**Self-test:** "When would you recommend sparse vs dense for a 10000×10000 matrix?"

---

## Expected Outcomes

### Performance Predictions

**Dense GPU:**
- ✅ Consistent performance across sparsity levels
- ✅ High memory bandwidth utilization
- ✅ Fast for low sparsity (< 80%)

**Sparse GPU:**
- ✅ Performance improves with sparsity
- ✅ Lower memory usage
- ✅ Fast for high sparsity (> 90%)

**Crossover point hypothesis:** ~85-90% sparsity for 5000×5000 matrices

### Key Insights to Discover

1. **Overhead matters:** Sparse format has overhead (indices storage, irregular access)
2. **Size matters:** Crossover point shifts with matrix size
3. **Real-world wins:** For typical single-cell data (95% sparse), sparse should win by 2-5×

---

## Immediate Action Items

### Today (Day 1)
1. ✅ Set up Great Lakes environment
2. ✅ Create project directory structure
3. ✅ Implement CPU baseline
4. ✅ Test on small matrix

### This Week (Days 2-7)
1. ✅ Dense GPU implementation
2. ✅ Sparse GPU implementation
3. ✅ Correctness validation

### Next Week (Days 8-14)
1. ✅ Generate test data
2. ✅ Run benchmarks
3. ✅ Analyze results
4. ✅ Write report

---

## Final Thoughts

### Focus Areas

**Week 1:** Get both implementations working correctly
- Don't optimize prematurely
- Focus on correctness first
- Use small test cases

**Week 2:** Understand the trade-offs
- Comprehensive benchmarking
- Thoughtful analysis
- Clear communication

### What Makes This Project Successful

Not just "does it run fast?" but:
- ✅ **Understanding:** Can you explain WHY one approach wins?
- ✅ **Analysis:** Can you identify bottlenecks?
- ✅ **Communication:** Can you provide actionable guidelines?
- ✅ **Rigor:** Did you test thoroughly?

---

**Remember: This is about comparing approaches, not achieving peak performance. Focus on learning, understanding trade-offs, and providing practical insights!**

Good luck! 🚀

---

## Contact & Resources

**Student:** [Your Name]
**Email:** [Your Email]
**Instructor:** [Instructor Name]
**Course:** CSE587 - Parallel Computing
**Semester:** Fall 2025

**Help Resources:**
- Office hours: [Time/Location]
- Great Lakes support: hpc-support@umich.edu
- CUDA forums: https://forums.developer.nvidia.com/
