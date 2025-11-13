# CSE587 Enhanced Project: Progressive GPU Optimization for NMF

## Project Evolution: From Comparison to Comprehensive Optimization Study

**Original:** Compare sparse vs dense NMF
**Enhanced:** Progressive optimization levels + comprehensive performance analysis

---

## PROGRESSIVE OPTIMIZATION LEVELS

### Level 0: CPU Baseline
**File:** `src/nmf_cpu.py`
**Purpose:** Correctness validation and serial performance baseline
**Status:** ✅ Complete

---

### Level 1: Naive GPU (Days 1-2)
**File:** `src/nmf_dense_gpu_v1_naive.cu`

**Implementation:**
- Basic cuBLAS calls (no custom kernels)
- Simple element-wise operations
- No optimization whatsoever
- Establish GPU baseline

**Learning Goals:**
- CUDA memory transfers
- cuBLAS API basics
- Basic kernel launches

**Expected Performance:**
- 10-30x speedup over CPU
- Low memory bandwidth utilization (< 40%)
- Low occupancy

**Metrics to Collect:**
- Execution time
- Memory bandwidth used
- Kernel occupancy
- FLOPS achieved

---

### Level 2: Memory Optimized (Days 3-4)
**File:** `src/nmf_dense_gpu_v2_memory.cu`

**Optimizations:**
1. **Shared Memory Tiling**
   - Tile-based matrix multiply
   - Reduce global memory accesses
   - Block size tuning (16×16, 32×32)

2. **Memory Coalescing**
   - Ensure contiguous memory access
   - Proper thread-to-data mapping
   - Avoid strided access patterns

3. **Optimal Thread Configuration**
   - Maximize occupancy
   - Balance threads per block
   - Consider register usage

**Learning Goals:**
- Shared memory bank conflicts
- Memory coalescing patterns
- Occupancy optimization
- Thread block sizing

**Expected Performance:**
- 1.5-3x speedup over Level 1
- Memory bandwidth utilization (60-80%)
- Higher occupancy (> 70%)

**Demonstrable Understanding:**
- Explain shared memory bank conflicts
- Show before/after memory access patterns
- Justify thread block dimensions
- Measure occupancy improvements

**Analysis Required:**
```
Before vs After:
- Global memory transactions
- Shared memory bank conflicts
- L1/L2 cache hit rates
- Occupancy percentage
```

---

### Level 3: Compute Optimized (Days 5-6)
**File:** `src/nmf_dense_gpu_v3_compute.cu`

**Optimizations:**
1. **Kernel Fusion**
   - Combine element-wise operations
   - Reduce kernel launch overhead
   - Minimize intermediate memory writes

2. **Warp-Level Primitives**
   - Use `__shfl_down_sync()` for reductions
   - Warp-level reduce operations
   - Avoid unnecessary synchronization

3. **ILP (Instruction-Level Parallelism)**
   - Unroll loops where beneficial
   - Multiple operations per thread
   - Hide memory latency

**Learning Goals:**
- Kernel fusion benefits
- Warp-level programming
- ILP and latency hiding

**Expected Performance:**
- 1.2-2x speedup over Level 2
- Reduced kernel launch overhead
- Better compute utilization

---

### Level 4: Sparse Implementation (Days 7-9)
**File:** `src/nmf_sparse_gpu_v1.cu` + `src/nmf_sparse_gpu_v2_optimized.cu`

**Implementation:**
1. **Naive Sparse (v1)**
   - Basic cuSPARSE integration
   - CSR format handling
   - Hybrid sparse-dense operations

2. **Optimized Sparse (v2)**
   - Format-specific kernels
   - Vectorized loads for CSR
   - Efficient sparse-dense multiply

**Learning Goals:**
- cuSPARSE API
- CSR format intricacies
- Sparse-dense hybrid algorithms
- Irregular memory access handling

**Expected Performance:**
- Crossover point identification
- Memory savings quantified
- Performance vs sparsity curve

---

### Level 5: Advanced (Days 10-12) - OPTIONAL
**File:** `src/nmf_advanced.cu`

**Advanced Techniques:**
1. **Adaptive Sparse/Dense Switching**
   - Runtime sparsity detection
   - Automatic format selection
   - Hybrid execution

2. **Mixed Precision**
   - FP16 for storage
   - FP32 for computation
   - Tensor Core utilization (A100)

3. **Persistent Kernels**
   - Reduce launch overhead
   - Thread block reuse
   - Custom scheduling

**Learning Goals:**
- Production optimization techniques
- Tensor Core programming
- Advanced CUDA features

---

## REQUIRED ANALYSIS COMPONENTS

### 1. Roofline Model Analysis

**Theory:**
```
Roofline Model identifies performance bottlenecks:
- Memory-bound: Performance limited by bandwidth
- Compute-bound: Performance limited by FLOPS

Operational Intensity = FLOPS / Bytes Transferred
```

**For NMF Update Step:**
```
W update: W = W .* (X × H^T) ./ (W × H × H^T + eps)

FLOPS:
- X × H^T: 2 × m × n × k
- H × H^T: 2 × k × k × n
- W × (H×H^T): 2 × m × k × k
- Element-wise ops: 3 × m × k
Total: ~2mnk + 2k²n + 2mk² + 3mk

Bytes Transferred:
- Read X: m × n × 4 bytes
- Read H: k × n × 4 bytes
- Read/Write W: 2 × m × k × 4 bytes
Total: 4(mn + kn + 2mk) bytes

Operational Intensity = FLOPS / Bytes
```

**Implementation:**
**File:** `scripts/roofline_analysis.py`

```python
def calculate_roofline_nmf(m, n, k):
    # FLOPS for one iteration
    flops_WtW = 2 * k * k * m
    flops_WtX = 2 * k * n * m
    flops_HHt = 2 * k * k * n
    flops_XHt = 2 * m * k * n
    flops_updates = 4 * (m*k + k*n)  # multiply + divide
    total_flops = flops_WtW + flops_WtX + flops_HHt + flops_XHt + flops_updates

    # Bytes transferred (assuming perfect caching)
    bytes_X = m * n * 4
    bytes_W = m * k * 4 * 2  # read + write
    bytes_H = k * n * 4 * 2  # read + write
    total_bytes = bytes_X + bytes_W + bytes_H

    operational_intensity = total_flops / total_bytes

    return total_flops, total_bytes, operational_intensity

# For V100 GPU:
peak_flops = 7.8e12      # 7.8 TFLOPS (FP32)
peak_bandwidth = 900e9    # 900 GB/s

def plot_roofline(results):
    # Plot theoretical roofline
    # Plot achieved performance for each optimization level
    # Identify if memory-bound or compute-bound
```

**Deliverables:**
- Roofline plot with all optimization levels
- Identification of bottlenecks
- Gap analysis (theoretical vs achieved)

---

### 2. Memory Bandwidth Analysis

**File:** `scripts/bandwidth_analysis.py`

**Metrics to Measure:**
```python
# For each kernel:
bandwidth_achieved = (bytes_read + bytes_written) / time_seconds

# Theoretical peak (V100): 900 GB/s
bandwidth_efficiency = (bandwidth_achieved / peak_bandwidth) * 100

# Analyze:
- Per-kernel bandwidth
- Total bandwidth utilization
- Impact of optimizations
- Bottleneck identification
```

**Required Measurements:**
1. **Naive Version:**
   - Achieved bandwidth: ~200-400 GB/s (20-40%)
   - Lots of redundant memory transfers

2. **Memory Optimized:**
   - Achieved bandwidth: ~600-800 GB/s (60-80%)
   - Shared memory reduces global accesses

3. **Compute Optimized:**
   - Achieved bandwidth: May decrease (more compute)
   - But overall throughput improves

**Profiling Command:**
```bash
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed \
    --metrics l1tex__throughput.avg.pct_of_peak_sustained_elapsed \
    ./nmf_gpu_v2
```

**Deliverables:**
- Bandwidth utilization table (per version)
- Bandwidth vs optimization level plot
- Memory access pattern analysis

---

### 3. Nsight Compute Profiling

**File:** `scripts/profile_all_versions.sh`

**Metrics to Collect:**

**A. Occupancy Metrics**
```bash
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active
```
- Theoretical occupancy
- Achieved occupancy
- Limiting factors (registers, shared memory)

**B. Memory Metrics**
```bash
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed \
    --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
    --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum
```
- Global memory load/store throughput
- L1/L2 cache hit rates
- Shared memory bank conflicts

**C. Compute Metrics**
```bash
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
    --metrics smsp__sass_thread_inst_executed_op_fp32_pred_on.sum
```
- SM throughput
- FLOPS achieved
- Instruction mix

**D. Warp Efficiency**
```bash
ncu --metrics smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio \
    --metrics smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio
```
- Warp execution efficiency
- Stall reasons
- Branch efficiency

**Deliverables:**
- Profiling report for each version
- Comparative analysis table
- Bottleneck identification

---

### 4. Optimization Impact Table

**File:** `results/optimization_impact.csv`

**Format:**
```csv
Version,Time_ms,Speedup_vs_Naive,Speedup_vs_CPU,Bandwidth_GBps,Bandwidth_Pct,FLOPS,FLOPS_Pct,Occupancy_Pct,Notes
CPU,5000.0,0.04x,1.0x,N/A,N/A,10 GFLOPS,N/A,N/A,Baseline
GPU_Naive,200.0,1.0x,25.0x,300,33%,100 GFLOPS,1.3%,45%,Basic cuBLAS
GPU_MemOpt,80.0,2.5x,62.5x,700,78%,250 GFLOPS,3.2%,85%,Shared memory + coalescing
GPU_CompOpt,50.0,4.0x,100.0x,650,72%,400 GFLOPS,5.1%,90%,Kernel fusion + warp ops
Sparse_90pct,30.0,6.7x,166.7x,200,22%,450 GFLOPS,5.8%,70%,90% sparsity
```

**Visualization:**
```python
# Create multi-metric comparison
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Execution Time
# Plot 2: Speedup
# Plot 3: Bandwidth Utilization
# Plot 4: FLOPS Achieved
# Plot 5: Occupancy
# Plot 6: Overall Efficiency
```

**Deliverables:**
- Complete optimization impact table
- Multi-metric comparison plots
- Analysis of each optimization's contribution

---

## ENHANCED PROJECT STRUCTURE

```
MU_Parallel/
├── src/
│   ├── nmf_cpu.py                      # ✅ Level 0: CPU baseline
│   ├── nmf_dense_gpu_v1_naive.cu       # 🔨 Level 1: Naive GPU
│   ├── nmf_dense_gpu_v2_memory.cu      # 🔨 Level 2: Memory optimized
│   ├── nmf_dense_gpu_v3_compute.cu     # 🔨 Level 3: Compute optimized
│   ├── nmf_sparse_gpu_v1.cu            # ✅ Level 4: Basic sparse
│   ├── nmf_sparse_gpu_v2_optimized.cu  # 🔨 Level 4: Optimized sparse
│   ├── nmf_advanced.cu                 # 🔨 Level 5: Advanced (optional)
│   ├── utils.h                         # ✅ Utilities
│   └── utils.cu                        # ✅ Utilities
│
├── scripts/
│   ├── generate_data.py                # ✅ Data generation
│   ├── benchmark_all_versions.sh       # 🔨 Benchmark all versions
│   ├── profile_all_versions.sh         # 🔨 Profile each version
│   ├── roofline_analysis.py            # 🔨 Roofline model
│   ├── bandwidth_analysis.py           # 🔨 Bandwidth analysis
│   ├── compare_optimizations.py        # 🔨 Compare all versions
│   └── plot_results.py                 # ✅ Visualization
│
├── results/
│   ├── optimization_impact.csv         # Impact table
│   ├── roofline_data.csv              # Roofline metrics
│   ├── bandwidth_data.csv             # Bandwidth metrics
│   ├── profiling/                     # Nsight Compute reports
│   │   ├── v1_naive.ncu-rep
│   │   ├── v2_memory.ncu-rep
│   │   ├── v3_compute.ncu-rep
│   │   └── sparse_v1.ncu-rep
│   └── plots/
│       ├── roofline_plot.png
│       ├── optimization_comparison.png
│       ├── bandwidth_utilization.png
│       └── performance_breakdown.png
│
└── docs/
    └── final_report.md
```

---

## REVISED TIMELINE (2 weeks)

### Week 1: Progressive Implementation

**Days 1-2: Naive GPU (Level 1)**
- [ ] Implement basic cuBLAS version
- [ ] Measure baseline GPU performance
- [ ] Profile with Nsight Compute
- [ ] Calculate initial roofline metrics
- **Deliverable:** Working naive GPU version + baseline metrics

**Days 3-4: Memory Optimization (Level 2)**
- [ ] Implement shared memory tiling
- [ ] Optimize memory coalescing
- [ ] Tune thread block size
- [ ] Profile and compare with Level 1
- [ ] Measure bandwidth improvements
- **Deliverable:** Memory-optimized version + comparative analysis

**Days 5-6: Compute Optimization (Level 3)**
- [ ] Implement kernel fusion
- [ ] Add warp-level primitives
- [ ] Explore ILP opportunities
- [ ] Profile and compare with Level 2
- **Deliverable:** Compute-optimized version + analysis

**Days 7: Integration & Testing**
- [ ] Verify all versions produce correct results
- [ ] Run comprehensive benchmarks
- [ ] Collect all profiling data
- [ ] Generate initial plots
- **Checkpoint:** All dense versions working + profiled

### Week 2: Sparse + Analysis

**Days 8-9: Sparse Implementation (Level 4)**
- [ ] Implement naive sparse version
- [ ] Optimize sparse operations
- [ ] Compare with dense at various sparsities
- [ ] Profile sparse-specific metrics
- **Deliverable:** Sparse versions + comparison data

**Days 10-11: Comprehensive Analysis**
- [ ] Generate roofline plots
- [ ] Analyze bandwidth utilization
- [ ] Create optimization impact table
- [ ] Identify bottlenecks in each version
- [ ] Crossover point analysis
- **Deliverable:** Complete analysis + all plots

**Days 12-14: Report Writing**
- [ ] Introduction & background
- [ ] Progressive optimization methodology
- [ ] Results for each optimization level
- [ ] Roofline analysis section
- [ ] Bandwidth analysis section
- [ ] Sparse vs dense comparison
- [ ] Practical guidelines
- [ ] Conclusions
- **Deliverable:** Final report (10-12 pages)

---

## ENHANCED REPORT STRUCTURE

### 1. Introduction (1 page)
- NMF application areas
- GPU optimization motivation
- Progressive optimization approach

### 2. Background (1.5 pages)
- NMF algorithm
- GPU architecture (V100/A100)
- Roofline model theory
- Memory hierarchy

### 3. Progressive Optimization Methodology (3 pages)

**3.1 Level 1: Naive GPU Implementation**
- cuBLAS integration
- Initial performance
- Identified bottlenecks

**3.2 Level 2: Memory Optimization**
- Shared memory tiling
- Coalescing strategies
- Performance improvements
- Roofline analysis

**3.3 Level 3: Compute Optimization**
- Kernel fusion
- Warp-level operations
- ILP techniques
- Performance gains

**3.4 Level 4: Sparse Implementation**
- cuSPARSE integration
- CSR format handling
- Sparse vs dense trade-offs

### 4. Performance Analysis (3 pages) ⭐ CORE SECTION

**4.1 Roofline Analysis**
- Operational intensity calculations
- Roofline plot with all versions
- Memory-bound vs compute-bound identification
- Gap analysis

**4.2 Memory Bandwidth Analysis**
- Bandwidth utilization per version
- Impact of memory optimizations
- Comparison with theoretical peak
- Memory access pattern analysis

**4.3 Profiling Data Analysis**
- Occupancy trends
- Warp efficiency
- Instruction mix
- Bottleneck identification

**4.4 Optimization Impact**
- Speedup breakdown
- Cumulative impact
- Cost-benefit analysis

### 5. Sparse vs Dense Comparison (2 pages)
- Crossover point analysis
- Memory usage comparison
- Performance vs sparsity
- Practical guidelines

### 6. Learning Outcomes & Insights (1 page)
- Key optimization principles
- Surprising findings
- Practical lessons

### 7. Conclusions (0.5 pages)
- Summary of findings
- Guidelines for practitioners
- Future work

---

## SUCCESS CRITERIA

### Minimum (B)
- ✅ All 4 levels implemented and working
- ✅ Basic profiling data collected
- ✅ Roofline model calculated
- ✅ Sparse vs dense comparison
- ✅ Report documents findings

### Target (A-)
- ✅ All above requirements
- ✅ Comprehensive profiling with Nsight Compute
- ✅ Detailed roofline analysis
- ✅ Memory bandwidth analysis
- ✅ Optimization impact quantified
- ✅ Clear insights and explanations

### Excellent (A/A+)
- ✅ All above requirements
- ✅ Deep understanding demonstrated
- ✅ Novel insights discovered
- ✅ Production-quality optimizations
- ✅ Publication-worthy analysis
- ✅ Advanced techniques explored (Level 5)

---

## KEY LEARNING OBJECTIVES

By completing this enhanced project, you will master:

**GPU Programming:**
- [ ] CUDA memory hierarchy
- [ ] Shared memory optimization
- [ ] Memory coalescing
- [ ] Kernel fusion techniques
- [ ] Warp-level programming
- [ ] cuBLAS and cuSPARSE APIs

**Performance Analysis:**
- [ ] Roofline model application
- [ ] Memory bandwidth analysis
- [ ] Nsight Compute profiling
- [ ] Bottleneck identification
- [ ] Optimization impact quantification

**Sparse Computing:**
- [ ] CSR format intricacies
- [ ] Sparse-dense hybrid algorithms
- [ ] Crossover point analysis
- [ ] Format-specific optimizations

**Professional Skills:**
- [ ] Systematic optimization methodology
- [ ] Performance modeling
- [ ] Technical communication
- [ ] Production code quality

---

## IMMEDIATE NEXT STEPS

1. **Review this plan** - Understand the full scope
2. **Start with Level 1** - Implement naive GPU version
3. **Profile immediately** - Establish baseline metrics
4. **Iterate progressively** - One optimization at a time

**Remember:** The goal is deep understanding, not just speed!

🚀 Let's build something publication-worthy!
