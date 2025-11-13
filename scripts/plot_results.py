#!/usr/bin/env python3
"""
Visualization script for NMF benchmark results.
Creates plots comparing sparse vs dense performance.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os


def load_results(csv_file):
    """Load benchmark results from CSV file."""
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} results from {csv_file}")
        return df
    except FileNotFoundError:
        print(f"Error: Results file {csv_file} not found!")
        print("Run benchmarks first: make benchmark")
        exit(1)


def plot_time_vs_sparsity(df, output_dir):
    """Plot execution time vs sparsity level."""
    sizes = sorted(df['size'].unique())
    n_sizes = len(sizes)

    fig, axes = plt.subplots(1, n_sizes, figsize=(5*n_sizes, 4))
    if n_sizes == 1:
        axes = [axes]

    for i, size in enumerate(sizes):
        data = df[df['size'] == size]

        for method in ['dense_gpu', 'sparse_gpu']:
            subset = data[data['method'] == method]
            if len(subset) > 0:
                label = method.replace('_', ' ').title()
                axes[i].plot(subset['sparsity'], subset['time_ms'],
                           marker='o', linewidth=2, markersize=8, label=label)

        axes[i].set_xlabel('Sparsity (%)', fontsize=12)
        axes[i].set_ylabel('Time (ms)', fontsize=12)
        axes[i].set_title(f'Matrix Size: {size}×{size}', fontsize=14, fontweight='bold')
        axes[i].legend(fontsize=10)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(45, 100)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'time_vs_sparsity.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_speedup(df, output_dir):
    """Plot speedup of sparse over dense."""
    sizes = sorted(df['size'].unique())

    fig, ax = plt.subplots(figsize=(10, 6))

    for size in sizes:
        data = df[df['size'] == size]

        speedups = []
        sparsities = []

        for sp in sorted(data['sparsity'].unique()):
            dense = data[(data['method'] == 'dense_gpu') & (data['sparsity'] == sp)]
            sparse = data[(data['method'] == 'sparse_gpu') & (data['sparsity'] == sp)]

            if len(dense) > 0 and len(sparse) > 0:
                speedup = dense['time_ms'].values[0] / sparse['time_ms'].values[0]
                speedups.append(speedup)
                sparsities.append(sp)

        if speedups:
            ax.plot(sparsities, speedups, marker='o', linewidth=2,
                   markersize=8, label=f'{size}×{size}')

    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7,
              label='Break-even (1.0x)')
    ax.set_xlabel('Sparsity (%)', fontsize=12)
    ax.set_ylabel('Speedup (Dense Time / Sparse Time)', fontsize=12)
    ax.set_title('Sparse vs Dense Speedup', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'speedup_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_memory_comparison(df, output_dir):
    """Plot memory usage comparison."""
    sizes = sorted(df['size'].unique())

    fig, ax = plt.subplots(figsize=(10, 6))

    for size in sizes:
        # Calculate theoretical memory usage
        sparsities = np.linspace(0, 100, 100)

        # Dense memory: m * n * 4 bytes (float32)
        dense_memory = (size * size * 4) / 1e6  # MB

        # Sparse memory: nnz * 8 + (m+1) * 4 bytes
        # 8 bytes per non-zero (4 for value + 4 for column index)
        sparse_memory = []
        for sp in sparsities:
            nnz = int(size * size * (1 - sp/100))
            mem = (nnz * 8 + (size + 1) * 4) / 1e6
            sparse_memory.append(mem)

        ax.plot(sparsities, [dense_memory] * len(sparsities),
               linewidth=2, label=f'Dense {size}×{size}')
        ax.plot(sparsities, sparse_memory, linestyle='--',
               linewidth=2, label=f'Sparse {size}×{size}')

    ax.set_xlabel('Sparsity (%)', fontsize=12)
    ax.set_ylabel('Memory Usage (MB)', fontsize=12)
    ax.set_title('Theoretical Memory Usage', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'memory_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_crossover_analysis(df, output_dir):
    """Identify and visualize crossover points."""
    sizes = sorted(df['size'].unique())

    crossover_points = []

    fig, ax = plt.subplots(figsize=(10, 6))

    for size in sizes:
        data = df[df['size'] == size]

        speedups = []
        sparsities = []

        for sp in sorted(data['sparsity'].unique()):
            dense = data[(data['method'] == 'dense_gpu') & (data['sparsity'] == sp)]
            sparse = data[(data['method'] == 'sparse_gpu') & (data['sparsity'] == sp)]

            if len(dense) > 0 and len(sparse) > 0:
                speedup = dense['time_ms'].values[0] / sparse['time_ms'].values[0]
                speedups.append(speedup)
                sparsities.append(sp)

        if speedups:
            # Find crossover point (where speedup crosses 1.0)
            for i in range(len(speedups) - 1):
                if (speedups[i] < 1.0 and speedups[i+1] >= 1.0) or \
                   (speedups[i] >= 1.0 and speedups[i+1] < 1.0):
                    # Linear interpolation
                    crossover = sparsities[i] + (1.0 - speedups[i]) * \
                               (sparsities[i+1] - sparsities[i]) / (speedups[i+1] - speedups[i])
                    crossover_points.append((size, crossover))
                    ax.scatter(crossover, size, s=200, marker='*', color='red', zorder=5)
                    ax.text(crossover + 2, size, f'{crossover:.1f}%',
                           fontsize=10, va='center')

    # Plot crossover points
    if crossover_points:
        sizes_cross = [p[0] for p in crossover_points]
        sparsity_cross = [p[1] for p in crossover_points]
        ax.plot(sparsity_cross, sizes_cross, 'ro-', linewidth=2, markersize=10,
               label='Crossover Points')

    ax.set_xlabel('Crossover Sparsity (%)', fontsize=12)
    ax.set_ylabel('Matrix Size', fontsize=12)
    ax.set_title('Crossover Point Analysis\n(Where Sparse Becomes Faster)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'crossover_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

    # Print crossover summary
    print("\nCrossover Point Summary:")
    print("=" * 50)
    print(f"{'Matrix Size':<15} {'Crossover Sparsity'}")
    print("-" * 50)
    for size, sparsity in crossover_points:
        print(f"{size}×{size:<10} {sparsity:>6.1f}%")
    print("=" * 50)


def plot_summary_table(df, output_dir):
    """Create a summary table visualization."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')

    # Prepare summary data
    summary_data = []
    sizes = sorted(df['size'].unique())

    for size in sizes:
        for method in ['dense_gpu', 'sparse_gpu']:
            data = df[(df['size'] == size) & (df['method'] == method)]
            if len(data) > 0:
                avg_time = data['time_ms'].mean()
                min_time = data['time_ms'].min()
                max_time = data['time_ms'].max()

                summary_data.append([
                    f"{size}×{size}",
                    method.replace('_', ' ').title(),
                    f"{avg_time:.2f}",
                    f"{min_time:.2f}",
                    f"{max_time:.2f}"
                ])

    table = ax.table(cellText=summary_data,
                    colLabels=['Size', 'Method', 'Avg Time (ms)', 'Min Time (ms)', 'Max Time (ms)'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(summary_data) + 1):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    plt.title('Performance Summary Table', fontsize=14, fontweight='bold', pad=20)

    output_file = os.path.join(output_dir, 'summary_table.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize NMF benchmark results')
    parser.add_argument('--input', type=str, default='results/results.csv',
                        help='Input CSV file with results')
    parser.add_argument('--output-dir', type=str, default='results/plots',
                        help='Output directory for plots')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set plotting style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'

    print("=" * 70)
    print("NMF Benchmark Visualization")
    print("=" * 70)

    # Load results
    df = load_results(args.input)

    print("\nGenerating plots...")

    # Create all plots
    plot_time_vs_sparsity(df, args.output_dir)
    plot_speedup(df, args.output_dir)
    plot_memory_comparison(df, args.output_dir)
    plot_crossover_analysis(df, args.output_dir)
    plot_summary_table(df, args.output_dir)

    print("\n" + "=" * 70)
    print(f"✓ All plots saved to {args.output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
