#!/usr/bin/env python3
"""
CPU Baseline for NMF using Multiplicative Update (MU) algorithm
This serves as the reference implementation for correctness validation.
"""

import numpy as np
import time
import argparse
from scipy.sparse import load_npz, issparse
import sys


def nmf_mu(X, k, max_iter=100, tol=1e-4, verbose=True):
    """
    Non-negative Matrix Factorization using Multiplicative Update.

    Args:
        X: Input matrix (m×n), can be dense or sparse
        k: Rank (number of components)
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        verbose: Print progress

    Returns:
        W: Basis matrix (m×k)
        H: Coefficient matrix (k×n)
    """
    # Convert sparse to dense for CPU baseline
    if issparse(X):
        X = X.toarray()

    m, n = X.shape

    # Random initialization
    np.random.seed(42)
    W = np.random.rand(m, k).astype(np.float32)
    H = np.random.rand(k, n).astype(np.float32)

    eps = 1e-10  # Small constant to prevent division by zero

    if verbose:
        print(f"Running NMF-MU: m={m}, n={n}, k={k}, max_iter={max_iter}")
        print("-" * 60)

    start_time = time.time()

    for iter in range(max_iter):
        # Update H: H = H .* (W^T × X) ./ (W^T × W × H + eps)
        WtW = W.T @ W                    # k×k
        WtX = W.T @ X                    # k×n
        WtWH = WtW @ H                   # k×n
        H *= WtX / (WtWH + eps)

        # Update W: W = W .* (X × H^T) ./ (W × H × H^T + eps)
        HHt = H @ H.T                    # k×k
        XHt = X @ H.T                    # m×k
        WHHt = W @ HHt                   # m×k
        W *= XHt / (WHHt + eps)

        # Check convergence
        if iter % 10 == 0:
            error = np.linalg.norm(X - W @ H) / np.linalg.norm(X)
            if verbose:
                print(f"Iter {iter:3d}: Relative error = {error:.6f}")

            if error < tol:
                if verbose:
                    print(f"Converged at iteration {iter}")
                break

    elapsed = time.time() - start_time

    if verbose:
        print("-" * 60)
        final_error = np.linalg.norm(X - W @ H) / np.linalg.norm(X)
        print(f"Final relative error: {final_error:.6e}")
        print(f"Time: {elapsed:.3f}s ({elapsed*1000:.1f}ms)")

    return W, H, elapsed


def main():
    parser = argparse.ArgumentParser(description='CPU Baseline NMF')
    parser.add_argument('--input', type=str, default=None,
                        help='Input matrix file (.npz format)')
    parser.add_argument('--size', type=int, default=100,
                        help='Matrix size for random test (if no input file)')
    parser.add_argument('--k', type=int, default=10,
                        help='Rank (number of components)')
    parser.add_argument('--iters', type=int, default=100,
                        help='Maximum iterations')
    parser.add_argument('--tol', type=float, default=1e-4,
                        help='Convergence tolerance')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress output')

    args = parser.parse_args()

    # Load or generate data
    if args.input:
        print(f"Loading data from {args.input}...")
        X = load_npz(args.input)
        if issparse(X):
            print(f"Sparse matrix: {X.shape[0]}×{X.shape[1]}, "
                  f"{X.nnz:,} non-zeros ({100*(1-X.nnz/(X.shape[0]*X.shape[1])):.1f}% sparse)")
            X_dense = X.toarray()
        else:
            X_dense = X
    else:
        print(f"Generating random {args.size}×{args.size} matrix...")
        np.random.seed(42)
        X_dense = np.random.rand(args.size, args.size).astype(np.float32)

    # Run NMF
    W, H, elapsed = nmf_mu(X_dense, args.k, max_iter=args.iters,
                           tol=args.tol, verbose=not args.quiet)

    # Final reconstruction error
    error = np.linalg.norm(X_dense - W @ H) / np.linalg.norm(X_dense)

    # Output CSV format for benchmarking
    if args.input:
        size = X_dense.shape[0]
        nnz = np.count_nonzero(X_dense)
        sparsity = 100 * (1 - nnz / (size * size))
        print(f"\n{size},{sparsity:.0f},cpu,{elapsed*1000:.2f},0,{error:.6e},{nnz}")


if __name__ == "__main__":
    main()
