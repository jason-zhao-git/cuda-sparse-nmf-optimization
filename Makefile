# Makefile for Progressive NMF Optimization Project

# ============================================================================
# Compiler and Flags
# ============================================================================

NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_86 -Xcompiler -Wall
LIBS = -lcublas -lcusparse

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

.PHONY: all clean test help levels naive memory-opt sparse advanced

# Build all optimization levels
all: naive sparse
	@echo ""
	@echo "╔════════════════════════════════════════════════════╗"
	@echo "║         Build Complete!                            ║"
	@echo "╚════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Built optimization levels:"
	@echo "  ✓ naive: Level 1 - Naive baseline"
	@echo "  ✓ sparse: Level 3 - Sparse implementation"
	@echo ""
	@echo "TODO (implement these):"
	@echo "  ○ memory-opt: Level 2 - Shared memory + coalescing"
	@echo "  ○ advanced: Level 4 - Kernel fusion + adaptive"
	@echo ""
	@echo "Quick test: make test-naive"
	@echo ""

# Build levels individually
levels: all

# ============================================================================
# Level 1: Naive Baseline
# ============================================================================

naive: nmf_naive
	@echo "✓ Level 1: Naive baseline built"

nmf_naive: $(SRC_DIR)/nmf_dense_gpu_v1_naive.cu $(UTILS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS)

# Backwards compatibility
level1: naive
nmf_dense_gpu_v1_naive: naive

# ============================================================================
# Level 2: Memory Optimization (TO BE IMPLEMENTED)
# ============================================================================

memory-opt: nmf_memory_opt
	@echo "✓ Level 2: Memory-optimized version built"

nmf_memory_opt: $(SRC_DIR)/nmf_dense_gpu_v2_memory.cu $(UTILS)
	@echo "Building Level 2: Memory optimization..."
	@if [ ! -f $(SRC_DIR)/nmf_dense_gpu_v2_memory.cu ]; then \
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
# Level 3: Sparse Implementation
# ============================================================================

sparse: nmf_sparse
	@echo "✓ Level 3: Sparse implementation built"

nmf_sparse: $(SRC_DIR)/nmf_sparse_gpu.cu $(UTILS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS)

# Backwards compatibility
level3: sparse
nmf_sparse_gpu: sparse

# ============================================================================
# Level 4: Advanced (TO BE IMPLEMENTED)
# ============================================================================

advanced: nmf_advanced
	@echo "✓ Level 4: Advanced version built"

nmf_advanced: $(SRC_DIR)/nmf_advanced.cu $(UTILS)
	@echo "Building Level 4: Advanced optimizations..."
	@if [ ! -f $(SRC_DIR)/nmf_advanced.cu ]; then \
		echo "⚠ Level 4 source not yet implemented"; \
		echo "  Implement advanced optimizations:"; \
		echo "  - Kernel fusion"; \
		echo "  - Adaptive sparse/dense switching"; \
		echo "  - Mixed precision exploration"; \
		exit 1; \
	fi
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS)

# Backwards compatibility
level4: advanced
nmf_advanced_gpu: advanced

nmf_sparse_gpu_v2: $(SRC_DIR)/nmf_sparse_gpu_v2_optimized.cu $(UTILS)
	@if [ -f $(SRC_DIR)/nmf_sparse_gpu_v2_optimized.cu ]; then \
		$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS); \
	else \
		echo "⚠ Optimized sparse version not yet implemented"; \
	fi

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
	rm -f nmf_naive nmf_memory_opt nmf_sparse nmf_advanced
	rm -f nmf_dense_gpu_v1_naive nmf_dense_gpu_v2_memory nmf_sparse_gpu  # Old names
	rm -f nmf_dense_gpu  # Alias
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
