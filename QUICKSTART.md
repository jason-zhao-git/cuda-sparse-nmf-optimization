# Quick Start Guide

Get your NMF project running in 15 minutes!

## Step 1: Environment Setup (Great Lakes)

```bash
# Login to Great Lakes
ssh your_uniqname@greatlakes.arc-ts.umich.edu

# Navigate to project directory
cd ~/nmf-sparse-dense

# Load required modules
module load cuda/12.2.0
module load gcc/11.2.0
module load python/3.10

# Verify GPU is available
nvidia-smi
```

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.2    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-PCIE...  Off  | 00000000:3B:00.0 Off |                    0 |
| N/A   30C    P0    24W / 250W |      0MiB / 16384MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```

## Step 2: Test CPU Baseline

```bash
# Test the CPU version
python3 src/nmf_cpu.py --size 100 --k 10 --iters 50
```

Expected output:
```
Running NMF-MU: m=100, n=100, k=10, max_iter=50
------------------------------------------------------------
Iter   0: Relative error = 0.458921
Iter  10: Relative error = 0.234156
Iter  20: Relative error = 0.178432
Iter  30: Relative error = 0.145678
Iter  40: Relative error = 0.123456
------------------------------------------------------------
Final relative error: 1.123456e-01
Time: 0.234s (234.5ms)
```

✅ If this works, your Python environment is good!

## Step 3: Build GPU Versions

```bash
# Build both implementations
make all
```

Expected output:
```
nvcc -O3 -arch=sm_70 -Xcompiler -Wall src/nmf_dense_gpu.cu src/utils.cu -o nmf_dense_gpu -lcublas -lcusparse
Built: nmf_dense_gpu
nvcc -O3 -arch=sm_70 -Xcompiler -Wall src/nmf_sparse_gpu.cu src/utils.cu -o nmf_sparse_gpu -lcublas -lcusparse
Built: nmf_sparse_gpu
```

✅ If you see these messages, compilation succeeded!

### Troubleshooting Build Issues

**Error: `nvcc: command not found`**
```bash
# Make sure CUDA module is loaded
module load cuda/12.2.0
nvcc --version
```

**Error: `unsupported GPU architecture 'compute_70'`**
```bash
# Check your GPU architecture
nvidia-smi --query-gpu=compute_cap --format=csv

# Edit Makefile and change -arch=sm_70 to match your GPU:
# V100: sm_70
# A100: sm_80
```

## Step 4: Test GPU Versions

### Test Dense GPU
```bash
./nmf_dense_gpu 100 10 50
```

Expected output:
```
Generating random 100x100 matrix...
Running Dense GPU NMF: m=100, n=100, k=10, max_iter=50
------------------------------------------------------------
Iteration 0
Iteration 10
Iteration 20
Iteration 30
Iteration 40
------------------------------------------------------------
Final relative error: 1.234567e-01
Time: 12.34 ms
```

✅ If you see timing output, dense GPU works!

### Test Sparse GPU
```bash
./nmf_sparse_gpu 100 10 0.7 50
```

Expected output:
```
Generating random 100x100 matrix with 70.0% sparsity...
Converted to CSR: 3000 non-zeros (70.0% sparse)
Running Sparse GPU NMF: m=100, n=100, k=10, nnz=3000, max_iter=50
------------------------------------------------------------
Iteration 0
Iteration 10
Iteration 20
Iteration 30
Iteration 40
------------------------------------------------------------
Time: 15.67 ms
```

✅ If you see timing output, sparse GPU works!

## Step 5: Generate Test Data

```bash
# Generate test matrices
python3 scripts/generate_data.py

# This creates matrices in data/ directory
ls -lh data/
```

Expected output:
```
X_1000x1000_sp50.npz
X_1000x1000_sp70.npz
X_1000x1000_sp80.npz
X_1000x1000_sp85.npz
X_1000x1000_sp90.npz
X_1000x1000_sp95.npz
X_1000x1000_sp99.npz
X_5000x5000_sp50.npz
...
```

## Step 6: Run Benchmarks

### Option A: Interactive Testing (Quick)
```bash
# Test different sizes manually
./nmf_dense_gpu 1000 20 100
./nmf_sparse_gpu 1000 20 0.9 100

./nmf_dense_gpu 5000 20 100
./nmf_sparse_gpu 5000 20 0.9 100
```

### Option B: Full Benchmark Suite (Recommended)
```bash
# Submit batch job
sbatch scripts/benchmark.sh

# Check job status
squeue -u $USER

# View output when done
cat results/benchmark_*.log
```

## Step 7: Visualize Results

```bash
# After benchmarks complete, create plots
python3 scripts/plot_results.py

# View generated plots
ls results/plots/
```

Expected files:
```
time_vs_sparsity.png
speedup_analysis.png
memory_comparison.png
crossover_analysis.png
summary_table.png
```

## Step 8: Profile Performance (Optional)

```bash
# Profile dense version
make profile-dense

# Profile sparse version
make profile-sparse

# View profiles (on Great Lakes login node)
ncu-ui results/profiles/dense_profile.ncu-rep
```

---

## Common Commands Reference

### Building
```bash
make all           # Build everything
make clean         # Remove executables
make test          # Run quick tests
```

### Testing
```bash
# CPU baseline
python3 src/nmf_cpu.py --size 1000 --k 20 --iters 100

# Dense GPU
./nmf_dense_gpu <size> <rank> <iterations>
./nmf_dense_gpu 1000 20 100

# Sparse GPU
./nmf_sparse_gpu <size> <rank> <sparsity> <iterations>
./nmf_sparse_gpu 1000 20 0.9 100
```

### Data Generation
```bash
# Default configuration
python3 scripts/generate_data.py

# Custom sizes and sparsities
python3 scripts/generate_data.py --sizes 2000 4000 --sparsities 0.8 0.9 0.95
```

### Benchmarking
```bash
# Interactive
make benchmark

# Batch job
sbatch scripts/benchmark.sh

# Check results
head -20 results/results.csv
```

---

## Troubleshooting

### GPU Not Found
```bash
# Check if GPU is allocated
nvidia-smi

# If not, request GPU node
salloc --account=cse587f25s001_class --partition=gpu --gres=gpu:1 --time=02:00:00
```

### Out of Memory
```bash
# Start with smaller matrices
./nmf_dense_gpu 500 10 50

# Or request more memory
salloc --mem=32GB --gres=gpu:1
```

### Compilation Errors
```bash
# Check CUDA version
nvcc --version

# Check loaded modules
module list

# Reload if needed
module purge
module load cuda/12.2.0 gcc/11.2.0 python/3.10
```

### Results Don't Match CPU
```bash
# This is normal! GPU uses single precision (float32)
# Small numerical differences are expected
# Large differences (>1e-3) indicate a bug

# Compare:
python3 src/nmf_cpu.py --size 100 --k 10 --iters 50
./nmf_dense_gpu 100 10 50
```

---

## Next Steps

1. **Week 1 Goal:** Get both implementations working correctly
   - ✅ CPU baseline validates
   - ✅ Dense GPU matches CPU (within tolerance)
   - ✅ Sparse GPU works on sparse matrices

2. **Week 2 Goal:** Comprehensive analysis
   - Run full benchmark suite
   - Generate all plots
   - Identify crossover points
   - Write report

3. **Key Questions to Answer:**
   - At what sparsity does sparse beat dense?
   - How does this change with matrix size?
   - What are the bottlenecks in each approach?
   - When should practitioners use which method?

---

## Getting Help

- **Great Lakes Support:** hpc-support@umich.edu
- **CUDA Documentation:** https://docs.nvidia.com/cuda/
- **Office Hours:** [Check Canvas]

---

## Success Checklist

Day 1:
- [ ] Environment set up
- [ ] CPU baseline working
- [ ] GPU versions compile
- [ ] Basic tests pass

Day 3:
- [ ] Dense GPU fully working
- [ ] Results match CPU baseline
- [ ] Understand cuBLAS operations

Day 7:
- [ ] Sparse GPU fully working
- [ ] CSR format understood
- [ ] cuSPARSE operations working

Day 14:
- [ ] All benchmarks complete
- [ ] Plots generated
- [ ] Crossover points identified
- [ ] Report drafted

Good luck! 🚀
