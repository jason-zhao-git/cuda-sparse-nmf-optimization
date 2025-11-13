#!/bin/bash
#
# Comprehensive Profiling Script for All NMF Optimization Levels
#
# This script profiles each optimization level with Nsight Compute
# and collects detailed performance metrics.
#

# ============================================================================
# Configuration
# ============================================================================

# Matrix size for profiling (smaller for detailed analysis)
SIZE=1000
RANK=20
ITERS=10  # Fewer iterations for profiling

# Output directory
PROFILE_DIR="results/profiles"
mkdir -p "$PROFILE_DIR"

# Nsight Compute executable
NCU="ncu"

# ============================================================================
# Color output
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  NMF Profiling Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Matrix size: ${SIZE}x${SIZE}"
echo "Rank: ${RANK}"
echo "Iterations: ${ITERS}"
echo "Output: ${PROFILE_DIR}/"
echo ""

# ============================================================================
# Check if executables exist
# ============================================================================

VERSIONS=(
    "nmf_dense_gpu_v1_naive"
    "nmf_dense_gpu_v2_memory"
    "nmf_dense_gpu_v3_compute"
    "nmf_sparse_gpu"
)

echo "Checking executables..."
for version in "${VERSIONS[@]}"; do
    if [ ! -f "./$version" ]; then
        echo -e "${YELLOW}Warning: $version not found. Skipping.${NC}"
    else
        echo -e "${GREEN}✓${NC} Found $version"
    fi
done
echo ""

# ============================================================================
# Metric Sets
# ============================================================================

# Basic metrics (fast)
METRICS_BASIC="gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed"

# Memory metrics
METRICS_MEMORY="dram__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum"

# Occupancy metrics
METRICS_OCCUPANCY="sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__maximum_warps_per_active_cycle_pct"

# Compute metrics
METRICS_COMPUTE="sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum"

# ============================================================================
# Profile Function
# ============================================================================

profile_version() {
    local exe=$1
    local name=$2
    local extra_args=$3

    if [ ! -f "./$exe" ]; then
        echo -e "${YELLOW}Skipping $name (executable not found)${NC}"
        return
    fi

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Profiling: $name${NC}"
    echo -e "${BLUE}========================================${NC}"

    # Full profile (takes longer)
    echo -e "${GREEN}Running full profile...${NC}"
    $NCU --set full \
        -o "${PROFILE_DIR}/${name}_full" \
        "./$exe" $SIZE $RANK $extra_args $ITERS \
        2>&1 | grep -E "(Profiling|Collecting|Writing)"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Full profile complete${NC}"
        echo "  Output: ${PROFILE_DIR}/${name}_full.ncu-rep"
    else
        echo -e "${RED}✗ Full profile failed${NC}"
    fi

    # Memory-focused profile
    echo -e "${GREEN}Running memory profile...${NC}"
    $NCU --metrics $METRICS_MEMORY \
        -o "${PROFILE_DIR}/${name}_memory" \
        "./$exe" $SIZE $RANK $extra_args $ITERS \
        2>&1 | grep -E "(Profiling|Metrics)"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Memory profile complete${NC}"
    fi

    # Occupancy profile
    echo -e "${GREEN}Running occupancy profile...${NC}"
    $NCU --metrics $METRICS_OCCUPANCY \
        -o "${PROFILE_DIR}/${name}_occupancy" \
        "./$exe" $SIZE $RANK $extra_args $ITERS \
        2>&1 | grep -E "(Profiling|Metrics)"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Occupancy profile complete${NC}"
    fi

    echo ""
}

# ============================================================================
# Profile Each Version
# ============================================================================

echo -e "${YELLOW}Starting profiling (this will take several minutes)...${NC}"
echo ""

# Level 1: Naive
profile_version "nmf_dense_gpu_v1_naive" "v1_naive" ""

# Level 2: Memory Optimized (if exists)
profile_version "nmf_dense_gpu_v2_memory" "v2_memory" ""

# Level 3: Compute Optimized (if exists)
profile_version "nmf_dense_gpu_v3_compute" "v3_compute" ""

# Sparse version
profile_version "nmf_sparse_gpu" "sparse_90pct" "0.9"

# ============================================================================
# Generate Summary Report
# ============================================================================

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Generating Summary Report${NC}"
echo -e "${BLUE}========================================${NC}"

REPORT_FILE="${PROFILE_DIR}/profiling_summary.txt"

cat > "$REPORT_FILE" << EOF
NSIGHT COMPUTE PROFILING SUMMARY
Generated: $(date)
Configuration: ${SIZE}x${SIZE}, rank=${RANK}, iters=${ITERS}

========================================
FILES GENERATED
========================================

EOF

# List all generated files
for profile in ${PROFILE_DIR}/*.ncu-rep; do
    if [ -f "$profile" ]; then
        size=$(du -h "$profile" | cut -f1)
        echo "  $(basename $profile) - $size" >> "$REPORT_FILE"
    fi
done

cat >> "$REPORT_FILE" << EOF

========================================
HOW TO VIEW PROFILES
========================================

1. On Great Lakes (text mode):
   ncu --import <profile>.ncu-rep --page details

2. On Great Lakes (web browser):
   ncu --import <profile>.ncu-rep --page details --html > report.html
   # Then download report.html and open in browser

3. On local machine with GUI:
   # Download .ncu-rep files
   scp greatlakes:~/path/to/results/profiles/*.ncu-rep .
   # Open with Nsight Compute UI
   ncu-ui <profile>.ncu-rep

========================================
KEY METRICS TO ANALYZE
========================================

1. Memory Bandwidth Utilization
   - Target: >70% for memory-optimized versions
   - Look at: dram__throughput.avg.pct_of_peak_sustained_elapsed

2. Occupancy
   - Target: >70% for optimized versions
   - Look at: sm__warps_active.avg.pct_of_peak_sustained_active

3. Compute Utilization
   - Look at: sm__throughput.avg.pct_of_peak_sustained_elapsed
   - Compare FP32 instruction counts

4. Memory Access Patterns
   - Check for coalesced vs uncoalesced accesses
   - Look at L1/L2 cache hit rates

5. Warp Execution Efficiency
   - Check for stalls
   - Look at divergence

========================================
COMPARISON WORKFLOW
========================================

1. Compare v1_naive vs v2_memory:
   - Should see higher bandwidth utilization
   - Should see higher occupancy
   - Identify impact of shared memory

2. Compare v2_memory vs v3_compute:
   - Should see higher compute utilization
   - May see slightly lower bandwidth (expected)
   - Check kernel fusion impact

3. Compare dense vs sparse:
   - Different memory access patterns
   - Sparse may have lower occupancy
   - Check bandwidth savings at high sparsity

========================================
EXPORT METRICS TO CSV
========================================

Use this command to export specific metrics:

ncu --import <profile>.ncu-rep \\
    --csv \\
    --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\\
             sm__warps_active.avg.pct_of_peak_sustained_active,\\
             sm__throughput.avg.pct_of_peak_sustained_elapsed \\
    > metrics.csv

EOF

echo -e "${GREEN}✓ Summary report generated: $REPORT_FILE${NC}"
echo ""

# ============================================================================
# Extract Quick Stats (if ncu supports it)
# ============================================================================

echo -e "${BLUE}Quick Statistics:${NC}"
echo ""

for profile in ${PROFILE_DIR}/*_full.ncu-rep; do
    if [ -f "$profile" ]; then
        name=$(basename "$profile" _full.ncu-rep)
        echo -e "${YELLOW}$name:${NC}"

        # Try to extract key metrics (this requires ncu CLI)
        # Note: This may not work on all systems
        $NCU --import "$profile" --page raw 2>/dev/null | \
            grep -E "(Duration|Throughput|Occupancy)" | head -5 || \
            echo "  (Use ncu-ui or web interface to view details)"

        echo ""
    fi
done

# ============================================================================
# Finalize
# ============================================================================

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Profiling Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Results location: $PROFILE_DIR/"
echo ""
echo "Next steps:"
echo "  1. View summary: cat $REPORT_FILE"
echo "  2. Analyze profiles: ncu-ui $PROFILE_DIR/<profile>.ncu-rep"
echo "  3. Compare versions side-by-side"
echo "  4. Document findings in report"
echo ""
echo -e "${BLUE}Suggested comparison order:${NC}"
echo "  1. v1_naive → v2_memory (memory optimization impact)"
echo "  2. v2_memory → v3_compute (compute optimization impact)"
echo "  3. v3_compute → sparse_90pct (sparse benefit at high sparsity)"
echo ""
