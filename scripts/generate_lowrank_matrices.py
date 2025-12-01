#!/usr/bin/env python3
"""
Generate low-rank test matrices for NMF benchmarking.

Creates matrices X = W_true @ H_true + noise with true low-rank structure.
This ensures NMF algorithms can actually converge (unlike random noise).

Output format: Binary file with [int32 rows][int32 cols][float32 data in column-major]
"""

import numpy as np
import struct
import argparse
import os

def generate_lowrank_matrix(m, n, rank, noise_level=0.01, seed=42):
    """
    Generate a low-rank non-negative matrix.

    X = W_true @ H_true + noise

    Parameters:
    - m, n: Matrix dimensions
    - rank: True rank of the matrix
    - noise_level: Standard deviation of additive noise
    - seed: Random seed for reproducibility

    Returns:
    - X: m x n non-negative matrix
    """
    np.random.seed(seed)

    # Generate true factors
    W_true = np.random.rand(m, rank).astype(np.float32)
    H_true = np.random.rand(rank, n).astype(np.float32)

    # Create low-rank matrix
    X = W_true @ H_true

    # Add small noise
    X += noise_level * np.random.rand(m, n).astype(np.float32)

    # Ensure non-negative
    X = np.maximum(X, 0).astype(np.float32)

    return X

def save_matrix_binary(filename, X):
    """
    Save matrix in binary format compatible with CUDA implementations.
    Format: [int32 rows][int32 cols][float32 data in column-major order]
    """
    m, n = X.shape

    # Convert to column-major (Fortran order)
    X_col = np.asfortranarray(X)

    with open(filename, 'wb') as f:
        f.write(struct.pack('i', m))
        f.write(struct.pack('i', n))
        f.write(X_col.tobytes())

    print(f"Saved {m}x{n} matrix to {filename}")
    print(f"  File size: {os.path.getsize(filename):,} bytes")
    print(f"  Min: {X.min():.4f}, Max: {X.max():.4f}, Mean: {X.mean():.4f}")

def verify_lowrank(X, rank):
    """Verify the matrix has expected low-rank structure."""
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    energy_ratio = (S[:rank]**2).sum() / (S**2).sum()
    print(f"  Rank-{rank} energy: {energy_ratio*100:.2f}%")
    print(f"  Top 5 singular values: {S[:5].round(1)}")
    return energy_ratio

def main():
    parser = argparse.ArgumentParser(description='Generate low-rank test matrices for NMF')
    parser.add_argument('--sizes', nargs='+', type=int, default=[500, 1000, 2000, 4000],
                        help='Matrix sizes to generate (default: 500 1000 2000 4000)')
    parser.add_argument('--rank', type=int, default=20,
                        help='True rank of matrices (default: 20)')
    parser.add_argument('--noise', type=float, default=0.01,
                        help='Noise level (default: 0.01)')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Output directory (default: data)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Low-Rank Matrix Generator for NMF Benchmarking")
    print("=" * 60)
    print(f"Sizes: {args.sizes}")
    print(f"True rank: {args.rank}")
    print(f"Noise level: {args.noise}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)

    for size in args.sizes:
        print(f"\nGenerating {size}x{size} matrix with rank {args.rank}...")

        X = generate_lowrank_matrix(size, size, args.rank, args.noise, args.seed)

        filename = os.path.join(args.output_dir, f"lowrank_{size}.bin")
        save_matrix_binary(filename, X)
        verify_lowrank(X, args.rank)

    print("\n" + "=" * 60)
    print("All matrices generated successfully!")
    print("=" * 60)
    print("\nUsage examples:")
    for size in args.sizes:
        print(f"  ./nmf_memory_opt data/lowrank_{size}.bin 20 100 --log-convergence 10")

if __name__ == "__main__":
    main()
