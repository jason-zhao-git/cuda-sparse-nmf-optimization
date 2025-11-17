#!/usr/bin/env python3
"""
Matrix Generation Module for NMF Benchmarking

Generates test matrices for fair comparison across all NMF implementations.
All NMF methods will use the SAME pre-generated matrix for fair benchmarking.

Usage:
    python3 generate_matrix.py --size 1000 --sparsity 0.9 --output sparse_1000.bin
    python3 generate_matrix.py --size 1000 --sparsity 0.0 --output dense_1000.bin
"""

import numpy as np
import argparse
import struct
import os


def generate_random_matrix(rows, cols, seed=42):
    """Generate random non-negative matrix."""
    np.random.seed(seed)
    return np.random.rand(rows, cols).astype(np.float32)


def sparsify_matrix(matrix, sparsity):
    """
    Zero out a fraction of the matrix elements randomly.

    Args:
        matrix: Input matrix
        sparsity: Fraction of elements to zero (0.0 = dense, 0.9 = 90% zeros)

    Returns:
        Sparsified matrix
    """
    if sparsity <= 0:
        return matrix

    rows, cols = matrix.shape
    total_elements = rows * cols
    num_zeros = int(total_elements * sparsity)

    # Fisher-Yates shuffle for unique random indices
    indices = np.arange(total_elements)
    np.random.shuffle(indices)

    # Zero out first num_zeros indices
    flat_matrix = matrix.flatten()
    flat_matrix[indices[:num_zeros]] = 0.0

    return flat_matrix.reshape(rows, cols)


def save_matrix_binary(matrix, filename):
    """
    Save matrix in binary format for C/CUDA consumption.

    Format:
        [int32: rows]
        [int32: cols]
        [float32 × rows × cols: data in row-major order]
    """
    rows, cols = matrix.shape

    with open(filename, 'wb') as f:
        # Write dimensions
        f.write(struct.pack('i', rows))
        f.write(struct.pack('i', cols))

        # Write data (row-major)
        matrix.astype(np.float32).tofile(f)

    print(f"✓ Saved matrix to {filename}")
    print(f"  Dimensions: {rows}×{cols}")
    print(f"  Size: {os.path.getsize(filename) / 1024:.2f} KB")


def save_matrix_npz(matrix, filename):
    """Save matrix in NumPy format for verification."""
    np.savez_compressed(filename, matrix=matrix)
    print(f"✓ Saved NumPy archive to {filename}")


def analyze_matrix(matrix):
    """Print statistics about the matrix."""
    rows, cols = matrix.shape
    total = rows * cols
    zeros = np.sum(matrix == 0)
    nonzeros = total - zeros

    print("\nMatrix Statistics:")
    print(f"  Dimensions: {rows}×{cols}")
    print(f"  Total elements: {total}")
    print(f"  Non-zero elements: {nonzeros}")
    print(f"  Zero elements: {zeros}")
    print(f"  Actual sparsity: {zeros/total*100:.2f}%")
    print(f"  Min value: {matrix[matrix > 0].min() if nonzeros > 0 else 0:.6f}")
    print(f"  Max value: {matrix.max():.6f}")
    print(f"  Mean (non-zero): {matrix[matrix > 0].mean() if nonzeros > 0 else 0:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate test matrices for NMF benchmarking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 90% sparse matrix for sparse NMF testing
  python3 generate_matrix.py --size 1000 --sparsity 0.9 --output data/sparse_1000_90.bin

  # Generate dense matrix for dense NMF testing
  python3 generate_matrix.py --size 1000 --sparsity 0.0 --output data/dense_1000.bin

  # Generate with custom seed
  python3 generate_matrix.py --size 500 --sparsity 0.8 --seed 123 --output data/test.bin
        """
    )

    parser.add_argument('--size', type=int, required=True,
                        help='Matrix size (will create size×size square matrix)')
    parser.add_argument('--sparsity', type=float, default=0.0,
                        help='Sparsity level (0.0=dense, 0.9=90%% zeros)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output filename (.bin for binary, .npz for NumPy)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--verify', action='store_true',
                        help='Also save .npz file for verification')

    args = parser.parse_args()

    # Validation
    if args.sparsity < 0 or args.sparsity >= 1.0:
        print("Error: Sparsity must be in [0, 1)")
        return 1

    print("════════════════════════════════════════════════════════════════")
    print("  NMF Matrix Generator")
    print("════════════════════════════════════════════════════════════════")
    print(f"Size: {args.size}×{args.size}")
    print(f"Sparsity: {args.sparsity*100:.1f}%")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output}")
    print("────────────────────────────────────────────────────────────────")

    # Generate matrix
    print("\nGenerating random matrix...")
    np.random.seed(args.seed)
    matrix = generate_random_matrix(args.size, args.size, seed=args.seed)

    # Sparsify if needed
    if args.sparsity > 0:
        print(f"Applying {args.sparsity*100:.1f}% sparsity...")
        matrix = sparsify_matrix(matrix, args.sparsity)

    # Analyze
    analyze_matrix(matrix)

    # Save
    print("\nSaving matrix...")
    if args.output.endswith('.npz'):
        save_matrix_npz(matrix, args.output)
    else:
        save_matrix_binary(matrix, args.output)

        # Also save .npz for verification if requested
        if args.verify:
            npz_file = args.output.replace('.bin', '.npz')
            save_matrix_npz(matrix, npz_file)

    print("\n✓ Done!")
    print("════════════════════════════════════════════════════════════════")

    return 0


if __name__ == '__main__':
    exit(main())
