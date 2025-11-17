# Test Data Generation Module

This module generates test matrices for fair NMF benchmarking across all implementations.

## Purpose

**Problem:** Previously, each NMF implementation generated its own test data, making comparisons unfair.

**Solution:** Generate test matrices ONCE, then all NMF implementations use the SAME input data.

## Usage

### Generate Test Matrices

```bash
# 1. Dense matrix (1000×1000, no sparsity)
python3 data/generate_matrix.py --size 1000 --sparsity 0.0 --output data/dense_1000.bin

# 2. Sparse matrix (1000×1000, 90% zeros)
python3 data/generate_matrix.py --size 1000 --sparsity 0.9 --output data/sparse_1000_90.bin

# 3. Small test matrix (100×100, 80% sparse)
python3 data/generate_matrix.py --size 100 --sparsity 0.8 --output data/test_100_80.bin
```

### Use in NMF Benchmarks

```bash
# All implementations now take a matrix file as input
./nmf_naive data/sparse_1000_90.bin 20 50
./nmf_memory_opt data/sparse_1000_90.bin 20 50
./nmf_sparse data/sparse_1000_90.bin 20 50
./nmf_sparse_transpose data/sparse_1000_90.bin 20 50

# This ensures ALL methods factorize the SAME matrix!
```

## File Format

Binary format for efficient C/CUDA loading:

```
[int32]    rows
[int32]    cols
[float32]  data[0]
[float32]  data[1]
...
[float32]  data[rows*cols-1]
```

Data is stored in **row-major** order (C-style).

## Command-Line Arguments

```
--size N          Matrix size (creates N×N square matrix)
--sparsity F      Fraction of zeros (0.0=dense, 0.9=90% zeros)
--output FILE     Output filename (.bin or .npz)
--seed N          Random seed for reproducibility (default: 42)
--verify          Also save .npz file for verification
```

## Examples

### 1. Generate Standard Test Cases

```bash
# Small test (quick)
python3 data/generate_matrix.py --size 100 --sparsity 0.9 --output data/test_small.bin

# Medium benchmark (1000×1000)
python3 data/generate_matrix.py --size 1000 --sparsity 0.9 --output data/bench_1k_sparse.bin
python3 data/generate_matrix.py --size 1000 --sparsity 0.0 --output data/bench_1k_dense.bin

# Large benchmark (2000×2000)
python3 data/generate_matrix.py --size 2000 --sparsity 0.9 --output data/bench_2k_sparse.bin
```

### 2. Verify Matrix Properties

```bash
# Generate with .npz for inspection
python3 data/generate_matrix.py --size 1000 --sparsity 0.9 \
    --output data/test.bin --verify

# Inspect in Python
python3 -c "import numpy as np; m = np.load('data/test.npz')['matrix']; print(m.shape, (m==0).sum()/m.size)"
```

## Test Matrix Statistics

The generator prints statistics about each generated matrix:

```
Matrix Statistics:
  Dimensions: 1000×1000
  Total elements: 1000000
  Non-zero elements: 100000
  Zero elements: 900000
  Actual sparsity: 90.00%
  Min value: 0.000003
  Max value: 0.999998
  Mean (non-zero): 0.499823
```

## Reproducibility

- All matrices use **seed=42** by default
- Same seed + same parameters = identical matrix
- Critical for reproducible benchmarks!

```bash
# These will generate IDENTICAL matrices
python3 data/generate_matrix.py --size 1000 --sparsity 0.9 --seed 42 --output m1.bin
python3 data/generate_matrix.py --size 1000 --sparsity 0.9 --seed 42 --output m2.bin
```

## Fair Benchmarking Workflow

```bash
# 1. Generate test matrix ONCE
python3 data/generate_matrix.py --size 1000 --sparsity 0.9 --output data/test.bin

# 2. Benchmark ALL methods on SAME matrix
echo "Dense Naive:"
./nmf_naive data/test.bin 20 50

echo "Dense Optimized:"
./nmf_memory_opt data/test.bin 20 50

echo "Sparse Hybrid:"
./nmf_sparse data/test.bin 20 50

echo "Sparse Transpose:"
./nmf_sparse_transpose data/test.bin 20 50

# 3. All methods factorize SAME input → fair comparison!
#    - Errors should be similar (same input)
#    - Time differences show TRUE performance gaps
```

## Why This Matters

### Before (UNFAIR):
```
Dense:  generates random dense matrix → factorizes it → error = 0.538
Sparse: generates random matrix → sparsifies to 90% → factorizes → error = 0.973
        └─ Different inputs! Not comparable!
```

### After (FAIR):
```
Both:   load same pre-generated sparse matrix → factorize → compare results
        └─ Same input → errors should match → time difference is REAL
```

## Real-World Dataset Recommendations

For more realistic NMF benchmarks, consider using:

1. **Images**: Convert grayscale images to matrices
   - Download: [ORL Faces](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html)
   - Naturally sparse (many dark pixels = zeros)

2. **Text**: Term-document matrices
   - Download: [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/)
   - Very sparse (90-99% typically)

3. **Audio**: Spectrograms
   - Generate from audio files using librosa
   - Moderate sparsity with structure

## Utility Functions (Future)

TODO: Add utilities for:
- Converting real datasets to .bin format
- Generating matrices with specific condition numbers
- Creating synthetic low-rank matrices
- Adding structured sparsity patterns
