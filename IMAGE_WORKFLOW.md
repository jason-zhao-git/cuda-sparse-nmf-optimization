# NMF on Real Images - Complete Workflow

This guide shows how to test your NMF implementations on real images (instead of random matrices) to demonstrate the algorithm actually works.

---

## Quick Start

```bash
# 1. Prepare image datasets
python3 data/prepare_image_datasets.py --sizes 128 256 512

# 2. Build NMF implementations (if not done)
make all

# 3. Run end-to-end pipeline
bash scripts/test_nmf_on_images.sh --size 256 --rank 10
```

**Result:** Visualizations showing original vs reconstructed images + learned basis images

---

## Step-by-Step Guide

### Step 1: Prepare Image Datasets

The script downloads/creates 3 standard test images and prepares them at multiple sizes:

```bash
python3 data/prepare_image_datasets.py --sizes 128 256 512 --num-images 3
```

**Output:**
```
data/images_128.bin   (16,384 × 3 matrix)
data/images_256.bin   (65,536 × 3 matrix)
data/images_512.bin   (262,144 × 3 matrix)
```

**Each file contains:**
- 3 grayscale images stacked as columns
- Normalized to [0, 1] range (NMF requires non-negative)
- Binary format compatible with your existing C++ code

### Step 2: Modify NMF Code to Save Results

**Add to your NMF implementations** (before cleanup section):

```cpp
// Save W, H matrices for visualization
printf("\nSaving results for visualization...\n");
save_matrix_binary("results/W_matrix.bin", h_W, m, k);
save_matrix_binary("results/H_matrix.bin", h_H, k, n);
```

**Example: Update `src/nmf_dense_gpu_v2_memory.cu`**

Find the section after `cudaMemcpy(h_W, d_W, ...)` and before `free(h_W)`, add:

```cpp
    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_W, d_W, m * k * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_H, d_H, k * n * sizeof(float), cudaMemcpyDeviceToHost));

    // +++ ADD THESE LINES +++
    // Save results for visualization
    printf("\nSaving results...\n");
    save_matrix_binary("results/W_matrix.bin", h_W, m, k);
    save_matrix_binary("results/H_matrix.bin", h_H, k, n);
    // +++ END ADD +++

    // Calculate metrics
    // ... rest of code ...
```

**Then rebuild:**
```bash
make memory-opt
```

### Step 3: Run NMF on Images

```bash
./nmf_memory_opt data/images_256.bin 10 100
```

**What happens:**
- Loads 256×256 images (65,536 × 3 matrix)
- Factorizes into W (65,536 × 10) and H (10 × 3)
- Saves W, H to results/ for visualization

**Matrix interpretation:**
- **W:** 10 basis images (columns are 256×256 images when reshaped)
- **H:** Weights to combine basis images for each of 3 original images
- **Reconstruction:** Image i ≈ W × H[:, i]

### Step 4: Visualize Results

```bash
python3 scripts/visualize_nmf_results.py \
    --original data/images_256.bin \
    --size 256 \
    --images 3 \
    --rank 10
```

**Creates two visualizations:**

1. **Reconstruction:** `results/reconstruction_256.png`
   ```
   [Original 1] [Original 2] [Original 3]
   [Recon 1]    [Recon 2]    [Recon 3]
   [Error 1]    [Error 2]    [Error 3]

   Metrics: PSNR, MSE for each image
   ```

2. **Basis Images:** `results/basis_256_k10.png`
   ```
   [Basis 1] [Basis 2] [Basis 3] ... [Basis 10]

   These are the learned components that combine
   to form all 3 original images
   ```

---

## Automated Pipeline

Use the end-to-end script:

```bash
# Default: 256×256, rank 10, 100 iterations
bash scripts/test_nmf_on_images.sh

# Custom parameters
bash scripts/test_nmf_on_images.sh --size 512 --rank 20 --iters 200
```

**The script:**
1. Prepares images (if needed)
2. Runs NMF
3. Visualizes results
4. Reports quality metrics

---

## Testing Different Configurations

### Small Images (Fast Testing)
```bash
python3 data/prepare_image_datasets.py --sizes 128
./nmf_memory_opt data/images_128.bin 8 50
python3 scripts/visualize_nmf_results.py --size 128 --rank 8
```

**Time:** ~10-20ms per run
**Use for:** Quick iteration, debugging

### Medium Images (Main Demo)
```bash
python3 data/prepare_image_datasets.py --sizes 256
./nmf_memory_opt data/images_256.bin 10 100
python3 scripts/visualize_nmf_results.py --size 256 --rank 10
```

**Time:** ~50-100ms per run
**Use for:** Main demonstration, paper figures

### Large Images (Performance)
```bash
python3 data/prepare_image_datasets.py --sizes 512
./nmf_memory_opt data/images_512.bin 20 100
python3 scripts/visualize_nmf_results.py --size 512 --rank 20
```

**Time:** ~200-400ms per run
**Use for:** Performance benchmarking

---

## Compare All Optimization Levels

```bash
# Prepare images once
python3 data/prepare_image_datasets.py --sizes 256

# Run all levels
./nmf_naive data/images_256.bin 10 100
cp results/W_matrix.bin results/W_naive.bin
cp results/H_matrix.bin results/H_naive.bin

./nmf_memory_opt data/images_256.bin 10 100
cp results/W_matrix.bin results/W_memory.bin
cp results/H_matrix.bin results/H_memory.bin

./nmf_compute_opt data/images_256.bin 10 100
cp results/W_matrix.bin results/W_compute.bin
cp results/H_matrix.bin results/H_compute.bin

# Compare quality
for level in naive memory compute; do
    echo "=== Level: $level ==="
    python3 scripts/visualize_nmf_results.py \
        --original data/images_256.bin \
        --W results/W_${level}.bin \
        --H results/H_${level}.bin \
        --size 256 --rank 10 --no-display
done
```

**Expected:** All levels produce similar quality (different speeds)

---

## Understanding the Results

### Good Reconstruction (PSNR > 25 dB)
- Images are clearly recognizable
- Main features preserved
- Shows NMF is working correctly

### Basis Images Show Structure
- Each basis image captures different features
- Edges, gradients, textures
- Combined with positive weights to form originals

### What Affects Quality?

**Rank k:**
- **Too low (k < 5):** Poor reconstruction, blurry
- **Optimal (k ≈ 10-20):** Good balance
- **Too high (k > 50):** Overfitting, diminishing returns

**Iterations:**
- **Too few (< 50):** May not converge
- **Optimal (50-100):** Good convergence
- **Too many (> 200):** Minimal improvement, waste time

**Image content:**
- **Simple (gradients, patterns):** Easy to reconstruct
- **Complex (natural photos):** Requires higher k

---

## Troubleshooting

### "File not found: data/images_256.bin"
```bash
# Run preparation script
python3 data/prepare_image_datasets.py --sizes 256
```

### "NMF result files not found"
```bash
# Add save_matrix_binary calls to your NMF code
# See Step 2 above
# Rebuild: make memory-opt
# Run again: ./nmf_memory_opt data/images_256.bin 10 100
```

### "Module not found: PIL"
```bash
pip install pillow scipy matplotlib
```

### "Poor reconstruction quality"
- Try higher rank: `--rank 20`
- More iterations: `--iters 200`
- Check convergence in output

---

## Example Output

```
$ bash scripts/test_nmf_on_images.sh

========================================
NMF on Images - End-to-End Pipeline
========================================

Configuration:
  Image size: 256×256
  Number of images: 3
  NMF rank: 10
  Iterations: 100

Step 1: Preparing Image Datasets
  ✓ Image dataset already exists: data/images_256.bin

Step 2: Running NMF (Memory-Optimized Version)
  Matrix: 65536×3, Rank: 10, Iterations: 100
  Time: 52.34 ms
  GFLOPS: 182.45
  Final error: 0.215

Step 3: Checking for NMF Results
  ✓ Found NMF results:
    • results/W_matrix.bin
    • results/H_matrix.bin

Step 4: Visualizing Reconstruction

  Quality Metrics:
  ──────────────────────────────────────────────
    Image 1: PSNR =  28.45 dB, MSE = 0.0123
    Image 2: PSNR =  29.12 dB, MSE = 0.0108
    Image 3: PSNR =  27.89 dB, MSE = 0.0142
  ──────────────────────────────────────────────
    Average: PSNR =  28.49 dB, MSE = 0.0124

  ✓ Saved visualization: results/reconstruction_256.png
  ✓ Saved basis images: results/basis_256_k10.png

========================================
✓ Pipeline Complete!
========================================

Results saved to:
  • results/reconstruction_256.png
  • results/basis_256_k10.png
```

---

## Files Created by This Workflow

```
data/
├── image_originals/        # Source images (downloaded)
│   ├── lena.png
│   ├── peppers.png
│   └── cameraman.png
├── images_128.bin          # 128×128 × 3 dataset
├── images_256.bin          # 256×256 × 3 dataset
├── images_512.bin          # 512×512 × 3 dataset
└── images_*_meta.txt       # Metadata files

results/
├── W_matrix.bin            # Basis images (from NMF)
├── H_matrix.bin            # Coefficients (from NMF)
├── reconstruction_*.png    # Original vs reconstructed
└── basis_*_k*.png          # Learned basis images
```

---

## Next Steps

1. **For your project report:**
   - Include reconstruction visualizations
   - Show basis images to demonstrate learning
   - Compare quality across optimization levels

2. **For presentations:**
   - Use 256×256 images (clear and fast)
   - Show k=10 basis images (interpretable)
   - Display PSNR metrics (quantitative proof)

3. **For further experiments:**
   - Try different ranks (5, 10, 20, 50)
   - Test on different image types
   - Compare dense vs sparse on images

---

**Document Version:** 1.0
**Last Updated:** 2025-01-23
