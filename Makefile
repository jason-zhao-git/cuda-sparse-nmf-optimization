# Makefile for Progressive NMF Optimization Project

# ============================================================================
# Compiler and Flags
# ============================================================================

# Use nvcc from PATH (works with module load cuda)
NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_86 -Xcompiler -Wall
LIBS = -lcublas
LIBS_SPARSE = -lcublas -lcusparse

# GPU Architecture:
# RTX 3050 (local): -arch=sm_86
# V100 (Great Lakes): -arch=sm_70
# A100 (Great Lakes): -arch=sm_80
# Check with: nvidia-smi --query-gpu=compute_cap --format=csv

# ============================================================================
# Source Files
# ============================================================================

SRC_DIR = src
UTILS = $(SRC_DIR)/utils.cu

# ============================================================================
# Progressive Optimization Levels
# ============================================================================

.PHONY: all clean test help levels naive memory-opt compute-opt sparse advanced

# Build all optimization levels (MU + HALS)
all: naive memory-opt compute-opt multigpu async-multigpu hals
	@echo ""
	@echo "╔════════════════════════════════════════════════════╗"
	@echo "║         Build Complete!                            ║"
	@echo "╚════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Built implementations:"
	@echo "  ✓ MU L1 Naive: Custom naive GEMM baseline"
	@echo "  ✓ MU L2 Memory: cuBLAS + fused kernels"
	@echo "  ✓ MU L3 Compute: cuBLAS + 8-way ILP"
	@echo "  ✓ MU L4 MultiGPU: Data parallel, sync every iter"
	@echo "  ✓ MU L5 Async: Data parallel, configurable sync"
	@echo "  ✓ HALS CPU: Sequential baseline"
	@echo "  ✓ HALS GPU Strict: Gauss-Seidel parallelism"
	@echo "  ✓ HALS GPU Block: Block-parallel with shuffling"
	@echo ""

# Build levels individually
levels: all

# ============================================================================
# Level 1: Naive Baseline
# ============================================================================

naive: nmf_naive
	@echo "✓ Level 1: Naive baseline built"

nmf_naive: $(SRC_DIR)/mu/nmf_dense_gpu_v1_naive.cu $(UTILS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS)

# Backwards compatibility
level1: naive
nmf_dense_gpu_v1_naive: naive

# ============================================================================
# Level 2: Memory Optimization (TO BE IMPLEMENTED)
# ============================================================================

memory-opt: nmf_memory_opt
	@echo "✓ Level 2: Memory-optimized version built"

nmf_memory_opt: $(SRC_DIR)/mu/nmf_dense_gpu_v2_memory.cu $(UTILS)
	@echo "Building Level 2: Memory optimization..."
	@if [ ! -f $(SRC_DIR)/mu/nmf_dense_gpu_v2_memory.cu ]; then \
		echo "⚠ Level 2 source not yet implemented"; \
		echo "  Copy v1_naive.cu to v2_memory.cu and add:"; \
		echo "  - Tiled matrix multiply with shared memory"; \
		echo "  - Memory coalescing optimization"; \
		echo "  - Optimal thread block configuration"; \
		exit 1; \
	fi
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS)

# Backwards compatibility
level2: memory-opt
nmf_dense_gpu_v2_memory: memory-opt

# ============================================================================
# Level 3: Compute Optimization
# ============================================================================

compute-opt: nmf_compute_opt
	@echo "✓ Level 3: Compute-optimized version built"

nmf_compute_opt: $(SRC_DIR)/mu/nmf_dense_gpu_v3_compute.cu $(UTILS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS)

# Backwards compatibility
level3: compute-opt
nmf_dense_gpu_v3_compute: compute-opt

# ============================================================================
# Sparse Implementations
# ============================================================================

sparse: nmf_sparse nmf_sparse_transpose nmf_sparse_hybrid
	@echo "✓ All sparse variants built"

nmf_sparse: $(SRC_DIR)/nmf_sparse_gpu.cu $(UTILS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS_SPARSE)

nmf_sparse_transpose: $(SRC_DIR)/nmf_sparse_gpu_v3_transpose.cu $(UTILS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS_SPARSE)

nmf_sparse_hybrid: $(SRC_DIR)/nmf_sparse_gpu_v3.cu $(UTILS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS_SPARSE)

# Backwards compatibility
nmf_sparse_gpu: nmf_sparse

# ============================================================================
# Level 4: Multi-GPU with OpenMP
# ============================================================================

multigpu: nmf_multigpu
	@echo "✓ Level 4: Multi-GPU version built"

nmf_multigpu: $(SRC_DIR)/mu/nmf_dense_gpu_v4_multigpu.cu $(UTILS)
	@echo "Building Level 4: Multi-GPU with OpenMP..."
	$(NVCC) $(NVCC_FLAGS) -Xcompiler -fopenmp $^ -o $@ $(LIBS)

# Backwards compatibility
level4: multigpu
advanced: multigpu
nmf_advanced: multigpu

# ============================================================================
# Level 5: Async Multi-GPU with Configurable Sync Interval
# ============================================================================

async-multigpu: nmf_async_multigpu
	@echo "✓ Level 5: Async Multi-GPU version built"

nmf_async_multigpu: $(SRC_DIR)/mu/nmf_dense_gpu_v5_async.cu $(UTILS)
	@echo "Building Level 5: Async Multi-GPU with sync interval..."
	$(NVCC) $(NVCC_FLAGS) -Xcompiler -fopenmp $^ -o $@ $(LIBS)

# Backwards compatibility
level5: async-multigpu

nmf_sparse_gpu_v2: $(SRC_DIR)/nmf_sparse_gpu_v2_optimized.cu $(UTILS)
	@if [ -f $(SRC_DIR)/nmf_sparse_gpu_v2_optimized.cu ]; then \
		$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS); \
	else \
		echo "⚠ Optimized sparse version not yet implemented"; \
	fi

# ============================================================================
# HALS Implementations
# ============================================================================

hals-cpu: nmf_hals_cpu
	@echo "✓ HALS CPU: Sequential baseline built"

nmf_hals_cpu: $(SRC_DIR)/hals/nmf_hals_cpu.cpp
	@echo "Building HALS CPU baseline (C++ with column-major indexing)..."
	g++ -std=c++11 -O3 -o $@ $(SRC_DIR)/hals/nmf_hals_cpu.cpp -lm

hals-gpu-strict: nmf_hals_gpu_strict
	@echo "✓ HALS GPU Level 1: Strict single-column parallelism built"

nmf_hals_gpu_strict: $(SRC_DIR)/hals/nmf_hals_gpu_v1_strict.cu $(UTILS)
	@echo "Building HALS GPU Level 1 (strict Gauss-Seidel)..."
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS)

hals-gpu-block: nmf_hals_gpu_block
	@echo "✓ HALS GPU Level 2: Block-parallel with random shuffling built"

nmf_hals_gpu_block: $(SRC_DIR)/hals/nmf_hals_gpu_v2_block.cu $(UTILS)
	@echo "Building HALS GPU Level 2 (block-parallel)..."
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS)

# Backwards compatibility
hals: hals-cpu
hals-gpu: hals-gpu-strict
hals-all: hals-cpu hals-gpu-strict hals-gpu-block

# ============================================================================
# Original simple targets (for backward compatibility)
# ============================================================================

nmf_dense_gpu: nmf_dense_gpu_v1_naive
	@cp nmf_dense_gpu_v1_naive nmf_dense_gpu
	@echo "✓ Created nmf_dense_gpu (alias for v1_naive)"

# ============================================================================
# Testing
# ============================================================================

test: test-cpu test-naive

test-cpu:
	@echo "Testing CPU baseline..."
	python3 $(SRC_DIR)/nmf_cpu.py --size 100 --k 10 --iters 50

test-naive: nmf_naive
	@echo "Testing Level 1: Naive GPU..."
	./nmf_naive 100 10 50

test-memory-opt: nmf_memory_opt
	@echo "Testing Level 2: Memory-Optimized GPU..."
	./nmf_memory_opt 100 10 50

test-sparse: nmf_sparse
	@echo "Testing Level 3: Sparse GPU..."
	./nmf_sparse 100 10 0.9 50

test-advanced: nmf_advanced
	@echo "Testing Level 4: Advanced GPU..."
	./nmf_advanced 100 10 50

test-all: test-cpu test-naive test-sparse

# Backwards compatibility
test-level1: test-naive
test-level2: test-memory-opt
test-level3: test-sparse
test-level4: test-advanced
	@echo ""
	@echo "╔════════════════════════════════════════════════════╗"
	@echo "║         All Tests Complete!                        ║"
	@echo "╚════════════════════════════════════════════════════╝"
	@echo ""

# ============================================================================
# Benchmarking
# ============================================================================

benchmark-all:
	@echo "Running comprehensive benchmarks..."
	bash scripts/benchmark_all_versions.sh

# ============================================================================
# Profiling
# ============================================================================

profile-all: naive sparse
	@echo "Profiling all versions..."
	bash scripts/profile_all_versions.sh

profile-naive: nmf_naive
	@mkdir -p results/profiles
	ncu --set full -o results/profiles/level1_naive ./nmf_naive 1000 20 10
	@echo "✓ Level 1 profile saved to results/profiles/level1_naive.ncu-rep"

profile-memory-opt: nmf_memory_opt
	@mkdir -p results/profiles
	ncu --set full -o results/profiles/level2_memory_opt ./nmf_memory_opt 1000 20 10
	@echo "✓ Level 2 profile saved to results/profiles/level2_memory_opt.ncu-rep"

profile-sparse: nmf_sparse
	@mkdir -p results/profiles
	ncu --set full -o results/profiles/level3_sparse ./nmf_sparse 1000 20 0.9 10
	@echo "✓ Level 3 profile saved to results/profiles/level3_sparse.ncu-rep"

profile-advanced: nmf_advanced
	@mkdir -p results/profiles
	ncu --set full -o results/profiles/level4_advanced ./nmf_advanced 1000 20 10
	@echo "✓ Level 4 profile saved to results/profiles/level4_advanced.ncu-rep"

# Backwards compatibility
profile-level1: profile-naive
profile-level2: profile-memory-opt
profile-level3: profile-sparse
profile-level4: profile-advanced

# ============================================================================
# Analysis
# ============================================================================

roofline:
	@echo "Generating roofline analysis..."
	python3 scripts/roofline_analysis.py --size 1000 --rank 20 --results results/optimization_impact.csv

analyze:
	@echo "Running comprehensive analysis..."
	@if [ -f scripts/compare_optimizations.py ]; then \
		python3 scripts/compare_optimizations.py; \
	else \
		echo "Analysis script not yet created"; \
	fi

# ============================================================================
# Cleanup
# ============================================================================

clean:
	rm -f nmf_naive nmf_memory_opt nmf_compute_opt nmf_multigpu nmf_async_multigpu nmf_sparse nmf_advanced
	rm -f nmf_dense_gpu_v1_naive nmf_dense_gpu_v2_memory nmf_sparse_gpu  # Old names
	rm -f nmf_dense_gpu nmf_sparse_transpose nmf_sparse_hybrid  # Aliases
	rm -f nmf_hals_cpu nmf_hals_gpu_strict nmf_hals_gpu_block  # HALS implementations
	rm -f *.o
	@echo "✓ Cleaned executables"

clean-results:
	rm -f results/*.csv
	rm -f results/plots/*.png
	rm -f results/profiles/*.ncu-rep
	rm -f results/*.txt
	@echo "✓ Cleaned results"

clean-data:
	rm -f data/*.npz
	rm -f data/*.txt
	@echo "✓ Cleaned data files"

clean-all: clean clean-results clean-data
	@echo "✓ Cleaned everything"

# ============================================================================
# Help
# ============================================================================

help:
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║         CSE587 Progressive NMF Optimization                    ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "BUILD TARGETS:"
	@echo "  all             - Build Level 1 (naive) and Level 4 (sparse)"
	@echo "  level1          - Build naive GPU baseline"
	@echo "  level2          - Build memory-optimized version"
	@echo "  level3          - Build compute-optimized version"
	@echo "  level4          - Build sparse implementations"
	@echo ""
	@echo "TESTING:"
	@echo "  test            - Run basic tests (CPU + Level 1)"
	@echo "  test-level1     - Test naive GPU"
	@echo "  test-level2     - Test memory-optimized"
	@echo "  test-level3     - Test compute-optimized"
	@echo "  test-sparse     - Test sparse GPU"
	@echo "  test-all        - Test everything"
	@echo ""
	@echo "PROFILING:"
	@echo "  profile-all     - Profile all versions with Nsight Compute"
	@echo "  profile-level1  - Profile naive GPU"
	@echo "  profile-level2  - Profile memory-optimized"
	@echo "  profile-level3  - Profile compute-optimized"
	@echo "  profile-sparse  - Profile sparse GPU"
	@echo ""
	@echo "ANALYSIS:"
	@echo "  roofline        - Generate roofline model plot"
	@echo "  analyze         - Run comprehensive analysis"
	@echo "  benchmark-all   - Run full benchmark suite"
	@echo ""
	@echo "CLEANUP:"
	@echo "  clean           - Remove executables"
	@echo "  clean-results   - Remove results files"
	@echo "  clean-data      - Remove generated data"
	@echo "  clean-all       - Remove everything"
	@echo ""
	@echo "EXAMPLES:"
	@echo "  make level1 && make test-level1"
	@echo "  make profile-level1"
	@echo "  make roofline"
	@echo ""
	@echo "PROGRESSIVE WORKFLOW:"
	@echo "  1. make level1 && make test-level1     # Establish baseline"
	@echo "  2. make profile-level1                 # Profile naive version"
	@echo "  3. [Implement Level 2 optimizations]   # Add shared memory, etc."
	@echo "  4. make level2 && make profile-level2  # Compare improvements"
	@echo "  5. make roofline                       # Analyze with roofline"
	@echo ""
