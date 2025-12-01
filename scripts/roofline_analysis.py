#!/usr/bin/env python3
"""
Roofline Model Analysis for NMF Implementations

The Roofline Model helps identify performance bottlenecks by plotting:
- Operational Intensity (FLOPS/byte) on x-axis
- Performance (GFLOPS) on y-axis

Performance is bounded by:
1. Memory bandwidth (diagonal line)
2. Peak compute (horizontal line)

If your point is near the diagonal: memory-bound
If your point is near the horizontal: compute-bound
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os


class RooflineModel:
    """
    Roofline model for GPU performance analysis.
    """

    def __init__(self, gpu_name="V100"):
        """
        Initialize roofline model with GPU specifications.

        GPU Specs:
        - V100: 7.8 TFLOPS (FP32), 900 GB/s bandwidth
        - A100: 19.5 TFLOPS (FP32), 1555 GB/s bandwidth
        """
        self.specs = {
            "V100": {
                "peak_flops": 7.8e12,      # 7.8 TFLOPS
                "peak_bandwidth": 900e9,    # 900 GB/s
                "name": "NVIDIA V100"
            },
            "A100": {
                "peak_flops": 19.5e12,     # 19.5 TFLOPS
                "peak_bandwidth": 1555e9,   # 1555 GB/s
                "name": "NVIDIA A100"
            },
            "RTX3050": {
                "peak_flops": 9.1e12,      # 9.1 TFLOPS
                "peak_bandwidth": 224e9,    # 224 GB/s
                "name": "NVIDIA RTX 3050"
            }
        }

        if gpu_name not in self.specs:
            print(f"Warning: Unknown GPU {gpu_name}, defaulting to V100")
            gpu_name = "V100"

        self.gpu = self.specs[gpu_name]
        self.peak_flops = self.gpu["peak_flops"]
        self.peak_bandwidth = self.gpu["peak_bandwidth"]
        self.gpu_name = self.gpu["name"]

    def calculate_nmf_metrics(self, m, n, k):
        """
        Calculate theoretical FLOPS and bytes for one NMF iteration.

        NMF Update Equations:
        H = H .* (W^T × X) ./ (W^T × W × H + eps)
        W = W .* (X × H^T) ./ (W × H × H^T + eps)

        Args:
            m, n: Matrix dimensions of X
            k: Rank

        Returns:
            flops: Total FLOPS
            bytes: Total bytes transferred
            operational_intensity: FLOPS/byte
        """
        # FLOPS calculation
        flops = 0

        # H update
        flops += 2 * k * k * m        # W^T × W
        flops += 2 * k * n * m        # W^T × X
        flops += 2 * k * n * k        # (W^T × W) × H
        flops += 4 * k * n            # Element-wise multiply + divide

        # W update
        flops += 2 * k * k * n        # H × H^T
        flops += 2 * m * k * n        # X × H^T
        flops += 2 * m * k * k        # W × (H × H^T)
        flops += 4 * m * k            # Element-wise multiply + divide

        # Bytes transferred (minimum, assuming perfect reuse)
        bytes_transferred = 0

        # Read X, W, H (multiple times per iteration)
        bytes_transferred += m * n * 4  # Read X
        bytes_transferred += m * k * 4  # Read W
        bytes_transferred += k * n * 4  # Read H

        # Write W and H
        bytes_transferred += m * k * 4  # Write W
        bytes_transferred += k * n * 4  # Write H

        # Intermediate results (may be cached in L2)
        bytes_transferred += (k * k + k * n + k * k + m * k) * 4

        operational_intensity = flops / bytes_transferred

        return flops, bytes_transferred, operational_intensity

    def plot_roofline(self, results, output_file="results/plots/roofline.png"):
        """
        Plot roofline model with actual results.

        Args:
            results: List of dicts with keys:
                - name: Implementation name
                - flops: Measured FLOPS
                - bytes: Measured bytes transferred
                - time_ms: Execution time
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Operational intensity range
        oi_range = np.logspace(-2, 3, 1000)

        # Peak bandwidth line (diagonal)
        bandwidth_limited = oi_range * self.peak_bandwidth

        # Peak compute line (horizontal)
        compute_limited = np.full_like(oi_range, self.peak_flops)

        # Roofline (minimum of bandwidth and compute)
        roofline = np.minimum(bandwidth_limited, compute_limited)

        # Plot roofline
        ax.loglog(oi_range, roofline / 1e9, 'k-', linewidth=3, label='Roofline')
        ax.loglog(oi_range, bandwidth_limited / 1e9, 'k--', linewidth=1.5,
                 alpha=0.5, label='Bandwidth Bound')
        ax.axhline(self.peak_flops / 1e9, color='k', linestyle='--',
                  linewidth=1.5, alpha=0.5, label='Compute Bound')

        # Plot results
        colors = sns.color_palette("husl", len(results))

        for i, result in enumerate(results):
            oi = result['flops'] / result['bytes']
            achieved_flops = result['flops'] / (result['time_ms'] / 1000.0)  # FLOPS/sec

            ax.loglog(oi, achieved_flops / 1e9, 'o', markersize=12,
                     color=colors[i], label=result['name'],
                     markeredgecolor='black', markeredgewidth=1.5)

            # Add annotation
            ax.annotate(result['name'],
                       xy=(oi, achieved_flops / 1e9),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.3))

        # Ridge point (where bandwidth meets compute)
        ridge_oi = self.peak_flops / self.peak_bandwidth
        ax.axvline(ridge_oi, color='red', linestyle=':', linewidth=2,
                  alpha=0.7, label=f'Ridge Point ({ridge_oi:.2f} FLOP/byte)')

        # Labels and formatting
        ax.set_xlabel('Operational Intensity (FLOPS/byte)', fontsize=14)
        ax.set_ylabel('Performance (GFLOPS)', fontsize=14)
        ax.set_title(f'Roofline Model - {self.gpu_name}\n'
                    f'Peak: {self.peak_flops/1e12:.1f} TFLOPS, '
                    f'Bandwidth: {self.peak_bandwidth/1e9:.0f} GB/s',
                    fontsize=16, fontweight='bold')

        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=10, loc='lower right')

        # Set reasonable axis limits
        ax.set_xlim(0.01, 100)
        ax.set_ylim(1, self.peak_flops / 1e9 * 1.2)

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved roofline plot: {output_file}")

        plt.close()

    def analyze_result(self, name, flops, bytes_transferred, time_ms):
        """
        Analyze a single result and print insights.
        """
        oi = flops / bytes_transferred
        achieved_flops = flops / (time_ms / 1000.0)
        achieved_gflops = achieved_flops / 1e9

        # Theoretical max at this OI
        theoretical_max = min(oi * self.peak_bandwidth, self.peak_flops)
        theoretical_gflops = theoretical_max / 1e9

        # Efficiency
        efficiency = (achieved_flops / theoretical_max) * 100

        # Ridge point
        ridge_oi = self.peak_flops / self.peak_bandwidth

        # Determine bottleneck
        if oi < ridge_oi * 0.8:
            bottleneck = "Memory-bound"
            advice = "Focus on reducing memory traffic (shared memory, better caching)"
        elif oi > ridge_oi * 1.2:
            bottleneck = "Compute-bound"
            advice = "Focus on compute optimizations (ILP, better instruction mix)"
        else:
            bottleneck = "Balanced"
            advice = "Optimize both memory and compute"

        print(f"\n{'='*70}")
        print(f"Analysis: {name}")
        print(f"{'='*70}")
        print(f"Operational Intensity: {oi:.2f} FLOPS/byte")
        print(f"Ridge Point: {ridge_oi:.2f} FLOPS/byte")
        print(f"Achieved Performance: {achieved_gflops:.2f} GFLOPS")
        print(f"Theoretical Max (at this OI): {theoretical_gflops:.2f} GFLOPS")
        print(f"Efficiency: {efficiency:.1f}%")
        print(f"Bottleneck: {bottleneck}")
        print(f"Recommendation: {advice}")
        print(f"{'='*70}\n")


def load_benchmark_results(csv_file, k=20):
    """
    Load results from benchmark CSV and convert to roofline format.
    Computes theoretical FLOPS/bytes based on matrix size and k.
    """
    import pandas as pd

    df = pd.read_csv(csv_file)
    results = []
    roof = RooflineModel("RTX3050")  # For FLOPS/bytes calculation

    # Group by method and size
    for _, row in df.iterrows():
        method = row['method']
        size = row['size']
        time_ms = row['time_ms']
        iterations = row.get('iterations', 100)

        # Calculate theoretical FLOPS and bytes for this size
        flops, bytes_transferred, _ = roof.calculate_nmf_metrics(size, size, k)

        # Scale by iterations
        total_flops = flops * iterations
        total_bytes = bytes_transferred * iterations

        results.append({
            'name': f"{method} ({size})",
            'method': method,
            'size': size,
            'flops': total_flops,
            'bytes': total_bytes,
            'time_ms': time_ms
        })

    return results


def main():
    parser = argparse.ArgumentParser(description='Roofline Model Analysis')
    parser.add_argument('--gpu', type=str, default='RTX3050', choices=['V100', 'A100', 'RTX3050'],
                       help='GPU model')
    parser.add_argument('--size', type=int, default=1000,
                       help='Matrix size for theoretical analysis')
    parser.add_argument('--rank', type=int, default=20,
                       help='Rank k')
    parser.add_argument('--results', type=str, default='results/benchmark_timing.csv',
                       help='CSV file with benchmark results')
    parser.add_argument('--output', type=str, default='results/figures/roofline.png',
                       help='Output plot file')

    args = parser.parse_args()

    # Create roofline model
    roof = RooflineModel(args.gpu)

    print("="*70)
    print("ROOFLINE MODEL ANALYSIS")
    print("="*70)
    print(f"GPU: {roof.gpu_name}")
    print(f"Peak Compute: {roof.peak_flops/1e12:.2f} TFLOPS")
    print(f"Peak Bandwidth: {roof.peak_bandwidth/1e9:.0f} GB/s")
    print("="*70)

    # Calculate theoretical metrics for NMF
    m, n, k = args.size, args.size, args.rank
    flops, bytes_transferred, oi = roof.calculate_nmf_metrics(m, n, k)

    print(f"\nTheoretical Analysis (m={m}, n={n}, k={k}):")
    print(f"  FLOPS per iteration: {flops/1e9:.2f} GFLOPS")
    print(f"  Bytes transferred: {bytes_transferred/1e6:.2f} MB")
    print(f"  Operational Intensity: {oi:.2f} FLOPS/byte")
    print(f"  Ridge Point: {roof.peak_flops/roof.peak_bandwidth:.2f} FLOPS/byte")

    if oi < roof.peak_flops / roof.peak_bandwidth:
        print(f"  → Memory-bound (OI < ridge point)")
        print(f"  → Max achievable: {oi * roof.peak_bandwidth / 1e12:.2f} TFLOPS")
    else:
        print(f"  → Compute-bound (OI > ridge point)")
        print(f"  → Max achievable: {roof.peak_flops / 1e12:.2f} TFLOPS")

    # If results provided, analyze them
    if args.results and os.path.exists(args.results):
        print(f"\nLoading benchmark results from {args.results}...")
        results = load_benchmark_results(args.results)

        # Analyze each result
        for result in results:
            roof.analyze_result(result['name'], result['flops'],
                              result['bytes'], result['time_ms'])

        # Plot roofline
        roof.plot_roofline(results, args.output)
    else:
        # Create example plot with theoretical points
        example_results = [
            {
                'name': 'Theoretical Peak',
                'flops': flops * 100,  # 100 iterations
                'bytes': bytes_transferred * 100,
                'time_ms': 10.0  # Hypothetical
            }
        ]
        roof.plot_roofline(example_results, args.output)
        print("\nNo benchmark results provided. Created example plot.")
        print("Run benchmarks and use --results flag to analyze real data.")


if __name__ == "__main__":
    main()
