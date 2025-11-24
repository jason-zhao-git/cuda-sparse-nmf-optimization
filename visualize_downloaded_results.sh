#!/bin/bash
#
# Visualize NMF Results Downloaded from Cluster
#
# After running NMF on cluster and downloading results:
#   scp cluster:~/project/results/*.bin results/
#
# Then run this script to create visualizations locally
#

set -e

echo "========================================================================"
echo "Visualizing NMF Results"
echo "========================================================================"
echo ""

# Check if results exist
if [ ! -f "results/W_256_k10.bin" ] || [ ! -f "results/H_256_k10.bin" ]; then
    echo "Error: Results not found in results/"
    echo ""
    echo "Download them from cluster first:"
    echo "  scp cluster:~/path/to/project/results/*.bin results/"
    echo ""
    exit 1
fi

# Visualize 128×128 results
if [ -f "results/W_128_k8.bin" ]; then
    echo "Visualizing 128×128 results..."
    python3 scripts/visualize_nmf_results.py \
        --original data/images_128.bin \
        --W results/W_128_k8.bin \
        --H results/H_128_k8.bin \
        --size 128 \
        --images 3 \
        --rank 8 \
        --no-display
    echo "  ✓ Saved: results/reconstruction_128.png"
    echo "  ✓ Saved: results/basis_128_k8.png"
    echo ""
fi

# Visualize 256×256 results
if [ -f "results/W_256_k10.bin" ]; then
    echo "Visualizing 256×256 results..."
    python3 scripts/visualize_nmf_results.py \
        --original data/images_256.bin \
        --W results/W_256_k10.bin \
        --H results/H_256_k10.bin \
        --size 256 \
        --images 3 \
        --rank 10 \
        --no-display
    echo "  ✓ Saved: results/reconstruction_256.png"
    echo "  ✓ Saved: results/basis_256_k10.png"
    echo ""
fi

# Visualize 512×512 results
if [ -f "results/W_512_k20.bin" ]; then
    echo "Visualizing 512×512 results..."
    python3 scripts/visualize_nmf_results.py \
        --original data/images_512.bin \
        --W results/W_512_k20.bin \
        --H results/H_512_k20.bin \
        --size 512 \
        --images 3 \
        --rank 20 \
        --no-display
    echo "  ✓ Saved: results/reconstruction_512.png"
    echo "  ✓ Saved: results/basis_512_k20.png"
    echo ""
fi

echo "========================================================================"
echo "✓ Visualization complete!"
echo "========================================================================"
echo ""
echo "View results:"
echo "  open results/reconstruction_256.png"
echo "  open results/basis_256_k10.png"
echo ""
