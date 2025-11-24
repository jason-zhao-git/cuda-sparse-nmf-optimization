#!/usr/bin/env python3
"""
Visualize NMF Results on Images

Loads original images and NMF results (W, H matrices) and creates
visualizations showing:
- Original images
- Reconstructed images (W × H)
- Reconstruction error
- Learned basis images

Usage:
    python3 visualize_nmf_results.py --size 256 --images 3 --rank 10
    python3 visualize_nmf_results.py --original data/images_256.bin --size 256
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys


def load_matrix_binary(filename):
    """
    Load matrix from binary format.

    Format:
        - 4 bytes: m (rows) as int32
        - 4 bytes: n (cols) as int32
        - m*n*4 bytes: data as float32 (row-major)

    Returns:
        Matrix of shape (m, n)
    """
    with open(filename, 'rb') as f:
        # Read dimensions
        m, n = np.fromfile(f, dtype=np.int32, count=2)

        # Read data
        data = np.fromfile(f, dtype=np.float32, count=m*n)

        # Reshape to matrix
        X = data.reshape(m, n)

    return X


def calculate_psnr(original, reconstructed):
    """Calculate Peak Signal-to-Noise Ratio."""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')

    max_pixel = 1.0  # Our images are normalized to [0, 1]
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def calculate_mse(original, reconstructed):
    """Calculate Mean Squared Error."""
    return np.mean((original - reconstructed) ** 2)


def visualize_reconstruction(X_orig, X_recon, img_size, num_images, output_file=None):
    """
    Visualize original vs reconstructed images.

    Args:
        X_orig: Original image matrix (pixels × num_images)
        X_recon: Reconstructed image matrix (pixels × num_images)
        img_size: Image dimension (assumes square images)
        num_images: Number of images
        output_file: Path to save figure (None = display)
    """
    fig, axes = plt.subplots(3, num_images, figsize=(4*num_images, 10))

    if num_images == 1:
        axes = axes.reshape(3, 1)

    for i in range(num_images):
        # Original
        img_orig = X_orig[:, i].reshape(img_size, img_size)
        axes[0, i].imshow(img_orig, cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Original {i+1}', fontsize=12, fontweight='bold')
        axes[0, i].axis('off')

        # Reconstructed
        img_recon = X_recon[:, i].reshape(img_size, img_size)
        axes[1, i].imshow(img_recon, cmap='gray', vmin=0, vmax=1)

        # Calculate metrics
        psnr = calculate_psnr(img_orig, img_recon)
        mse = calculate_mse(img_orig, img_recon)

        axes[1, i].set_title(f'Reconstructed {i+1}\nPSNR: {psnr:.2f} dB',
                             fontsize=12, fontweight='bold')
        axes[1, i].axis('off')

        # Error map
        error = np.abs(img_orig - img_recon)
        im = axes[2, i].imshow(error, cmap='hot', vmin=0, vmax=0.5)
        axes[2, i].set_title(f'Error Map {i+1}\nMSE: {mse:.4f}',
                             fontsize=12, fontweight='bold')
        axes[2, i].axis('off')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved visualization: {output_file}")
    else:
        plt.show()

    plt.close()


def visualize_basis_images(W, img_size, num_basis=None, output_file=None):
    """
    Visualize learned basis images (columns of W).

    Args:
        W: Basis matrix (pixels × rank)
        img_size: Image dimension (assumes square)
        num_basis: Number of basis images to show (None = all)
        output_file: Path to save figure (None = display)
    """
    rank = W.shape[1]

    if num_basis is None:
        num_basis = rank
    else:
        num_basis = min(num_basis, rank)

    # Determine grid size
    cols = min(5, num_basis)
    rows = (num_basis + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))

    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)

    for i in range(num_basis):
        row = i // cols
        col = i % cols

        # Get basis image
        basis = W[:, i].reshape(img_size, img_size)

        # Normalize for visualization
        basis_norm = (basis - basis.min()) / (basis.max() - basis.min() + 1e-10)

        axes[row, col].imshow(basis_norm, cmap='gray')
        axes[row, col].set_title(f'Basis {i+1}', fontsize=10)
        axes[row, col].axis('off')

    # Hide unused subplots
    for i in range(num_basis, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')

    plt.suptitle(f'Learned Basis Images (Rank k={rank})', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved basis images: {output_file}")
    else:
        plt.show()

    plt.close()


def print_quality_metrics(X_orig, X_recon, num_images):
    """Print detailed quality metrics."""
    print("\n  Quality Metrics:")
    print("  " + "─" * 60)

    total_psnr = 0
    total_mse = 0

    for i in range(num_images):
        img_orig = X_orig[:, i]
        img_recon = X_recon[:, i]

        psnr = calculate_psnr(img_orig, img_recon)
        mse = calculate_mse(img_orig, img_recon)

        total_psnr += psnr
        total_mse += mse

        print(f"    Image {i+1}: PSNR = {psnr:6.2f} dB, MSE = {mse:.6f}")

    print("  " + "─" * 60)
    print(f"    Average: PSNR = {total_psnr/num_images:6.2f} dB, MSE = {total_mse/num_images:.6f}")
    print()


def main():
    parser = argparse.ArgumentParser(description='Visualize NMF results on images')
    parser.add_argument('--original', type=str, default='data/images_256.bin',
                        help='Original image matrix file')
    parser.add_argument('--W', type=str, default='results/W_matrix.bin',
                        help='W matrix file (basis images)')
    parser.add_argument('--H', type=str, default='results/H_matrix.bin',
                        help='H matrix file (coefficients)')
    parser.add_argument('--size', type=int, required=True,
                        help='Image size (e.g., 256 for 256×256)')
    parser.add_argument('--images', type=int, default=3,
                        help='Number of images (default: 3)')
    parser.add_argument('--rank', type=int, default=10,
                        help='Rank k used in NMF (default: 10)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for visualizations')
    parser.add_argument('--no-display', action='store_true',
                        help='Save figures without displaying')

    args = parser.parse_args()

    print("=" * 70)
    print("NMF Results Visualization")
    print("=" * 70)
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load original images
    print("Step 1: Loading data...")
    try:
        X_orig = load_matrix_binary(args.original)
        print(f"  ✓ Original images: {X_orig.shape}")
    except FileNotFoundError:
        print(f"  ✗ Original image file not found: {args.original}")
        print("    Run: python3 data/prepare_image_datasets.py --sizes {args.size}")
        sys.exit(1)

    # Load NMF results
    try:
        W = load_matrix_binary(args.W)
        H = load_matrix_binary(args.H)
        print(f"  ✓ W matrix: {W.shape}")
        print(f"  ✓ H matrix: {H.shape}")
    except FileNotFoundError:
        print(f"  ✗ NMF result files not found")
        print("    Run NMF first: ./nmf_memory_opt {args.original} {args.rank} 100")
        sys.exit(1)

    # Verify dimensions
    m, n = X_orig.shape
    expected_m = args.size * args.size

    if m != expected_m:
        print(f"  ✗ Size mismatch: Expected {expected_m} pixels ({args.size}×{args.size}), got {m}")
        sys.exit(1)

    if n != args.images:
        print(f"  ⚠ Image count mismatch: Expected {args.images}, got {n}. Using {n}.")
        args.images = n

    # Reconstruct images
    print("\nStep 2: Reconstructing images...")
    X_recon = W @ H
    print(f"  ✓ Reconstruction: {X_recon.shape}")

    # Print quality metrics
    print_quality_metrics(X_orig, X_recon, args.images)

    # Visualize reconstruction
    print("Step 3: Creating visualizations...")
    recon_output = output_dir / f"reconstruction_{args.size}.png" if args.no_display else None
    visualize_reconstruction(X_orig, X_recon, args.size, args.images, recon_output)

    # Visualize basis images
    basis_output = output_dir / f"basis_{args.size}_k{args.rank}.png" if args.no_display else None
    visualize_basis_images(W, args.size, num_basis=min(20, args.rank), output_file=basis_output)

    print()
    print("=" * 70)
    print("✓ Visualization complete!")
    print("=" * 70)

    if args.no_display:
        print(f"\nOutput files saved to: {output_dir}/")
        print(f"  • reconstruction_{args.size}.png")
        print(f"  • basis_{args.size}_k{args.rank}.png")

    print()


if __name__ == '__main__':
    main()
