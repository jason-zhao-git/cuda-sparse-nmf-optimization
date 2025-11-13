#!/usr/bin/env python3
"""
Generate sparse test matrices for benchmarking.
Creates matrices at various sizes and sparsity levels.
"""

import numpy as np
from scipy.sparse import random, save_npz
import os
import argparse


def generate_sparse_matrix(m, n, sparsity, seed=42):
    """
    Generate a sparse non-negative matrix.

    Args:
        m, n: Matrix dimensions
        sparsity: Fraction of zeros (0.0 to 1.0)
        seed: Random seed for reproducibility

    Returns:
        Sparse matrix in CSR format
    """
    np.random.seed(seed)
    density = 1.0 - sparsity

    # Generate random sparse matrix
    X = random(m, n, density=density, format='csr', dtype=np.float32, random_state=seed)

    # Make all values non-negative
    X.data = np.abs(X.data)

    return X


def save_as_text(X, base_filename):
    """
    Save sparse matrix in text format for C++ loading.

    Args:
        X: Sparse matrix in CSR format
        base_filename: Base name for output files (without extension)
    """
    X_csr = X.tocsr()

    # Save values
    np.savetxt(f"{base_filename}_values.txt", X_csr.data, fmt='%.6f')

    # Save column indices
    np.savetxt(f"{base_filename}_colind.txt", X_csr.indices, fmt='%d')

    # Save row pointers
    np.savetxt(f"{base_filename}_rowptr.txt", X_csr.indptr, fmt='%d')

    # Save metadata (m, n, nnz)
    with open(f"{base_filename}_meta.txt", 'w') as f:
        f.write(f"{X_csr.shape[0]} {X_csr.shape[1]} {X_csr.nnz}\n")


def main():
    parser = argparse.ArgumentParser(description='Generate test matrices')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Output directory for generated matrices')
    parser.add_argument('--sizes', type=int, nargs='+',
                        default=[1000, 5000, 10000],
                        help='Matrix sizes to generate')
    parser.add_argument('--sparsities', type=float, nargs='+',
                        default=[0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99],
                        help='Sparsity levels (fraction of zeros)')
    parser.add_argument('--text-format', action='store_true',
                        help='Also save in text format for C++')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("Generating Test Matrices for NMF Benchmarking")
    print("=" * 70)

    total_matrices = len(args.sizes) * len(args.sparsities)
    count = 0

    for size in args.sizes:
        m, n = size, size

        for sparsity in args.sparsities:
            count += 1
            print(f"\n[{count}/{total_matrices}] Generating {m}×{n} matrix "
                  f"with {sparsity*100:.0f}% sparsity...")

            # Generate matrix
            X = generate_sparse_matrix(m, n, sparsity)

            # Base filename
            base_filename = f"{args.output_dir}/X_{m}x{n}_sp{int(sparsity*100)}"

            # Save as .npz (for Python)
            npz_filename = f"{base_filename}.npz"
            save_npz(npz_filename, X)

            print(f"  ✓ Saved: {npz_filename}")
            print(f"    Non-zeros: {X.nnz:,}")
            print(f"    Memory: {(X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / 1e6:.2f} MB")
            print(f"    vs Dense: {(m * n * 4) / 1e6:.2f} MB "
                  f"({100 * X.nnz / (m * n):.1f}% of dense)")

            # Optionally save as text files
            if args.text_format:
                save_as_text(X, base_filename)
                print(f"  ✓ Saved text format: {base_filename}_*.txt")

    print("\n" + "=" * 70)
    print(f"✓ Generated {total_matrices} matrices")
    print(f"✓ Output directory: {args.output_dir}/")
    print("=" * 70)

    # Print summary table
    print("\nSummary:")
    print(f"{'Size':<12} {'Sparsity':<10} {'Non-zeros':<12} {'File'}")
    print("-" * 70)

    for size in args.sizes:
        for sparsity in args.sparsities:
            filename = f"X_{size}x{size}_sp{int(sparsity*100)}.npz"
            filepath = os.path.join(args.output_dir, filename)

            if os.path.exists(filepath):
                X = np.load(filepath, allow_pickle=True)['arr_0'].item().tocsr()
                print(f"{size}×{size:<6} {sparsity*100:>5.0f}%     "
                      f"{X.nnz:>10,}   {filename}")


if __name__ == "__main__":
    main()
