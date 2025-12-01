# NMF Parallelization Project - Context Summary

## Project Overview
Graduate-level final project comparing GPU parallelization strategies for NMF algorithms.

## Key Findings
1. **HALS converges ~7x better** than MU at same iteration count (3.4% vs 23.6% error)
2. **Block-parallel HALS** with random shuffling preserves Gauss-Seidel convergence
3. MU optimization limited by **Amdahl's Law** (cuBLAS dominates 85% of runtime)

## Implementations

| Binary | Source | Description |
|--------|--------|-------------|
| `nmf_memory_opt` | `src/mu/nmf_dense_gpu_v2_memory.cu` | MU L2: Memory-optimized |
| `nmf_compute_opt` | `src/mu/nmf_dense_gpu_v3_compute.cu` | MU L3: Compute-optimized |
| `nmf_multigpu` | `src/mu/nmf_dense_gpu_v4_multigpu.cu` | MU L4: Multi-GPU |
| `nmf_hals_gpu_block` | `src/hals/nmf_hals_gpu_v2_block.cu` | HALS Block-Parallel |

## Convergence Logging
All implementations support `--log-convergence [interval]` flag:
- Output: `results/convergence_*.csv`
- Columns: `iteration, error, time_ms` (MU L4 adds `compute_ms, comm_ms`)

## Test Data
- **DO NOT USE** `data/dense_*.bin` - random noise, no low-rank structure
- **USE** `data/lowrank_*.bin` - proper rank-20 matrices that converge

## Workflow (All in Jupyter)
```bash
jupyter notebook notebooks/nmf_report_analysis.ipynb
```

The notebook does everything:
1. Generates low-rank test matrices (500, 1000, 2000)
2. Runs all benchmarks via subprocess
3. Loads convergence CSVs
4. Generates publication-ready figures
5. Exports summary tables

## For Multi-GPU (Cluster)
Uncomment the multi-GPU cell in the notebook when running on a machine with 2+ GPUs.

## Output Files
- `results/figures/convergence_comparison.png`
- `results/figures/scaling_plot.png`
- `results/figures/multigpu_breakdown.png`
- `results/figures/summary.md`
