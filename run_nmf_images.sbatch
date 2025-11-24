#!/bin/bash
#SBATCH --job-name=nmf_images
#SBATCH --output=logs/nmf_images_%j.log
#SBATCH --error=logs/nmf_images_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Load required modules
module load cuda/12.2.0
module load gcc/11.2.0

# Create directories
mkdir -p results
mkdir -p logs

echo "========================================================================"
echo "NMF on Images - Cluster Run"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Check GPU
nvidia-smi

echo ""
echo "========================================================================"
echo "Step 1: Building NMF implementations"
echo "========================================================================"
make memory-opt

echo ""
echo "========================================================================"
echo "Step 2: Running NMF on 128×128 images (quick test)"
echo "========================================================================"
./nmf_memory_opt data/images_128.bin 8 50
mv results/W_matrix.bin results/W_128_k8.bin
mv results/H_matrix.bin results/H_128_k8.bin

echo ""
echo "========================================================================"
echo "Step 3: Running NMF on 256×256 images (main demo)"
echo "========================================================================"
./nmf_memory_opt data/images_256.bin 10 100
mv results/W_matrix.bin results/W_256_k10.bin
mv results/H_matrix.bin results/H_256_k10.bin

echo ""
echo "========================================================================"
echo "Step 4: Running NMF on 512×512 images (performance test)"
echo "========================================================================"
./nmf_memory_opt data/images_512.bin 20 100
mv results/W_matrix.bin results/W_512_k20.bin
mv results/H_matrix.bin results/H_512_k20.bin

echo ""
echo "========================================================================"
echo "✓ All runs complete!"
echo "========================================================================"
echo "End time: $(date)"
echo ""
echo "Results saved:"
echo "  • results/W_128_k8.bin, results/H_128_k8.bin"
echo "  • results/W_256_k10.bin, results/H_256_k10.bin"
echo "  • results/W_512_k20.bin, results/H_512_k20.bin"
echo ""
echo "Next: Download results and run visualization locally:"
echo "  scp cluster:path/results/*.bin ."
echo "  python3 scripts/visualize_nmf_results.py --size 256 --rank 10"
