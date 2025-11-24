#!/bin/bash
#
# End-to-End NMF Image Testing Pipeline
#
# This script:
# 1. Prepares image datasets (if not already done)
# 2. Runs NMF on images
# 3. Visualizes results
#
# Usage:
#   bash scripts/test_nmf_on_images.sh
#   bash scripts/test_nmf_on_images.sh --size 256 --rank 10
#

set -e  # Exit on error

# Default parameters
SIZE=256
RANK=10
ITERS=100
NUM_IMAGES=3

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --size)
            SIZE="$2"
            shift 2
            ;;
        --rank)
            RANK="$2"
            shift 2
            ;;
        --iters)
            ITERS="$2"
            shift 2
            ;;
        --images)
            NUM_IMAGES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--size SIZE] [--rank RANK] [--iters ITERS] [--images NUM_IMAGES]"
            exit 1
            ;;
    esac
done

echo "========================================================================"
echo "NMF on Images - End-to-End Pipeline"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Image size: ${SIZE}×${SIZE}"
echo "  Number of images: ${NUM_IMAGES}"
echo "  NMF rank: ${RANK}"
echo "  Iterations: ${ITERS}"
echo ""

# Create results directory
mkdir -p results

# Step 1: Prepare image datasets (if not already done)
echo "========================================================================"
echo "Step 1: Preparing Image Datasets"
echo "========================================================================"
echo ""

if [ ! -f "data/images_${SIZE}.bin" ]; then
    echo "Image dataset not found. Generating..."
    python3 data/prepare_image_datasets.py --sizes ${SIZE} --num-images ${NUM_IMAGES}
else
    echo "✓ Image dataset already exists: data/images_${SIZE}.bin"
fi

echo ""

# Step 2: Run NMF
echo "========================================================================"
echo "Step 2: Running NMF (Memory-Optimized Version)"
echo "========================================================================"
echo ""

if [ ! -f "./nmf_memory_opt" ]; then
    echo "Error: nmf_memory_opt not found. Build with:"
    echo "  make memory-opt"
    exit 1
fi

# Note: The user needs to modify nmf_memory_opt to save W, H matrices
# For now, we'll just run it and expect results to be saved
./nmf_memory_opt data/images_${SIZE}.bin ${RANK} ${ITERS}

echo ""

# Step 3: Check if W, H matrices were saved
echo "========================================================================"
echo "Step 3: Checking for NMF Results"
echo "========================================================================"
echo ""

if [ ! -f "results/W_matrix.bin" ] || [ ! -f "results/H_matrix.bin" ]; then
    echo "⚠ W, H matrices not found in results/"
    echo ""
    echo "The NMF implementations need to be updated to save results."
    echo "Add these lines before the cleanup section in nmf_memory_opt.cu:"
    echo ""
    echo "    // Save results for visualization"
    echo "    save_matrix_binary(\"results/W_matrix.bin\", h_W, m, k);"
    echo "    save_matrix_binary(\"results/H_matrix.bin\", h_H, k, n);"
    echo ""
    echo "Then rebuild with: make memory-opt"
    echo ""
    echo "Skipping visualization..."
    exit 0
fi

echo "✓ Found NMF results:"
echo "  • results/W_matrix.bin"
echo "  • results/H_matrix.bin"
echo ""

# Step 4: Visualize results
echo "========================================================================"
echo "Step 4: Visualizing Reconstruction"
echo "========================================================================"
echo ""

python3 scripts/visualize_nmf_results.py \
    --original data/images_${SIZE}.bin \
    --W results/W_matrix.bin \
    --H results/H_matrix.bin \
    --size ${SIZE} \
    --images ${NUM_IMAGES} \
    --rank ${RANK} \
    --no-display

echo ""
echo "========================================================================"
echo "✓ Pipeline Complete!"
echo "========================================================================"
echo ""
echo "Results saved to:"
echo "  • results/reconstruction_${SIZE}.png"
echo "  • results/basis_${SIZE}_k${RANK}.png"
echo ""
echo "To view:"
echo "  open results/reconstruction_${SIZE}.png"
echo "  open results/basis_${SIZE}_k${RANK}.png"
echo ""
