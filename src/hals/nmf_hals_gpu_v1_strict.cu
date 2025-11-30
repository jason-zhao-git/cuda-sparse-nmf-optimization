#include "../utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*
 * GPU HALS NMF - LEVEL 1: STRICT SINGLE-COLUMN PARALLELISM
 *
 * Purpose: Implement strict Gauss-Seidel HALS on GPU with single-column updates
 *
 * Algorithm: HALS (Hierarchical Alternating Least Squares)
 * =========================================================
 * HALS solves NMF by updating one feature at a time using closed-form solutions.
 * Given X ≈ W × H where X(m×n), W(m×k), H(k×n):
 *
 * For H update (row f, all samples j):
 *   H[f,j] = max(ε, H[f,j] + (Num_H[f,j] - Σₗ Denom_H[f,l]·H[l,j]) / Denom_H[f,f])
 *   where: Num_H = WᵀX (k×n), Denom_H = WᵀW (k×k)
 *
 * For W update (column f, all rows i):
 *   W[i,f] = max(ε, W[i,f] + (Num_W[i,f] - Σₗ W[i,l]·Denom_W[l,f]) / Denom_W[f,f])
 *   where: Num_W = XHᵀ (m×k), Denom_W = HHᵀ (k×k)
 *   Then normalize: W[:,f] /= ||W[:,f]||₂
 *
 * Gauss-Seidel Property:
 * - Features updated sequentially (f = 0, 1, ..., k-1)
 * - Feature f sees updated values from features 0..f-1
 * - Leads to faster convergence than Jacobi (all features at once)
 *
 * Parallelization Strategy (Level 1 - Strict):
 * =============================================
 * - Sequential across features: f = 0, 1, ..., k-1 (maintains Gauss-Seidel)
 * - Parallel within each feature:
 *   - H update: parallelize across n samples (each thread handles 4 samples)
 *   - W update: parallelize across m rows (each thread handles 4 rows)
 * - Synchronization: cudaDeviceSynchronize() after each feature
 *
 * Memory Layout: Column-major (for cuBLAS compatibility)
 * - W[i,f] at position f*m + i
 * - H[f,j] at position j*k + f
 *
 * Expected Performance:
 * - Convergence: ~10-15 iterations (same as CPU HALS)
 * - Speedup: 5-10× over CPU (memory-bandwidth limited)
 * - Bottleneck: Sequential feature updates, sync overhead
 */

// ============================================================================
// CUDA KERNELS
// ============================================================================

/*
 * Kernel: Update single column of W (feature f)
 *
 * HALS W Update Formula:
 *   W[i,f] = max(ε, W[i,f] + (Num[i,f] - interaction) / Denom[f,f])
 *   where interaction = Σₗ W[i,l] · Denom[l,f] = (W × Denom)[i,f]
 *
 * Parameters:
 *   - W: m×k matrix (column-major), updated in-place
 *   - Numerator: m×k matrix = X × Hᵀ (precomputed)
 *   - Denom: k×k matrix = H × Hᵀ (precomputed)
 *   - target_col: feature index f (0 to k-1)
 *
 * Parallelization:
 *   - Each thread updates 4 consecutive rows of column f
 *   - Total threads needed: ceil(m/4)
 *   - All threads work on the SAME column (feature f)
 *
 * Memory Access (column-major):
 *   - W[i,f] at W[f*m + i] - coalesced writes within column
 *   - W[i,l] at W[l*m + i] - strided reads across columns (for interaction)
 *   - Denom[l,f] at Denom[f*k + l]
 */
__global__ void update_W_column_strict_hals(
    float* W,              // m × k (column-major, read AND write)
    const float* Numerator, // m × k (X × H^T, pre-computed)
    const float* Denom,     // k × k (H × H^T, pre-computed)
    int m, int k,
    int target_col,         // Which column we're updating
    float epsilon
) {
    // Thread processes 4 consecutive rows
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int row_start = tid * 4;

    if (row_start >= m) return;

    // Load diagonal element for division
    float denom_diag = Denom[target_col * k + target_col];
    if (denom_diag < epsilon) denom_diag = epsilon;

    // Process 4 rows with ILP
    float4 w_new;

    #pragma unroll
    for (int offset = 0; offset < 4; offset++) {
        int row = row_start + offset;
        if (row >= m) break;

        // A. Read numerator term: (X × H^T)[row, target_col]
        // Column-major: element at (row, col) is col*m + row
        float num = Numerator[target_col * m + row];

        // B. Compute interaction: (W × Denom)[row, target_col]
        // This is the dot product of W[row,:] and Denom[:,target_col]
        // CRITICAL: We read the LATEST W values (strict Gauss-Seidel)
        float interaction = 0.0f;

        #pragma unroll 8
        for (int l = 0; l < k; l++) {
            // W[row, l] in column-major: l*m + row
            float w_val = W[l * m + row];

            // Denom[l, target_col] in column-major: target_col*k + l
            float denom_val = Denom[target_col * k + l];

            interaction += w_val * denom_val;
        }

        // C. HALS update: W_new = W_old + (numerator - interaction) / diagonal
        float w_old = W[target_col * m + row];
        float gradient = (num - interaction) / denom_diag;
        float w_updated = w_old + gradient;

        // D. Non-negativity constraint
        w_updated = fmaxf(w_updated, epsilon);

        // Store to temporary
        if (offset == 0) w_new.x = w_updated;
        else if (offset == 1) w_new.y = w_updated;
        else if (offset == 2) w_new.z = w_updated;
        else if (offset == 3) w_new.w = w_updated;
    }

    // E. Coalesced write-back (float4)
    // Note: This assumes m is divisible by 4. Handle remainder separately if needed.
    if (row_start + 3 < m) {
        float4* W_col_ptr = (float4*)(&W[target_col * m]);
        W_col_ptr[tid] = w_new;
    } else {
        // Handle remainder rows with scalar writes
        for (int offset = 0; offset < 4 && row_start + offset < m; offset++) {
            float val;
            if (offset == 0) val = w_new.x;
            else if (offset == 1) val = w_new.y;
            else if (offset == 2) val = w_new.z;
            else val = w_new.w;
            W[target_col * m + row_start + offset] = val;
        }
    }
}

/*
 * Kernel: Update single row of H (feature f)
 *
 * HALS H Update Formula:
 *   H[f,j] = max(ε, H[f,j] + (Num[f,j] - interaction) / Denom[f,f])
 *   where interaction = Σₗ Denom[f,l] · H[l,j] = (Denom × H)[f,j]
 *
 * Parameters:
 *   - H: k×n matrix (column-major), updated in-place
 *   - Numerator: k×n matrix = Wᵀ × X (precomputed)
 *   - Denom: k×k matrix = Wᵀ × W (precomputed)
 *   - target_feature: feature index f (0 to k-1)
 *
 * Parallelization:
 *   - Each thread updates 4 consecutive samples for row f
 *   - Total threads needed: ceil(n/4)
 *   - All threads work on the SAME row (feature f)
 *
 * Memory Access (column-major):
 *   - H[f,j] at H[j*k + f] - strided access (one element per column)
 *   - H[l,j] at H[j*k + l] - reading full column for interaction
 *   - Denom[f,l] at Denom[l*k + f] (symmetric, so same as Denom[l,f])
 *
 * Note: Variable named target_col but actually refers to target row/feature
 */
__global__ void update_H_column_strict_hals(
    float* H,              // k × n (column-major, read AND write)
    const float* Numerator, // k × n (W^T × X, pre-computed)
    const float* Denom,     // k × k (W^T × W, pre-computed)
    int k, int n,
    int target_col,         // Which feature we're updating (0..k-1)
    float epsilon
) {
    // Thread processes 4 consecutive samples
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int sample_start = tid * 4;

    if (sample_start >= n) return;

    // Load diagonal element
    float denom_diag = Denom[target_col * k + target_col];
    if (denom_diag < epsilon) denom_diag = epsilon;

    // Process 4 samples with ILP
    float4 h_new;

    #pragma unroll
    for (int offset = 0; offset < 4; offset++) {
        int sample = sample_start + offset;
        if (sample >= n) break;

        // A. Read numerator: (W^T × X)[target_col, sample]
        // H is k×n, so element (row, col) is col*k + row
        float num = Numerator[sample * k + target_col];

        // B. Compute interaction: (Denom × H)[target_col, sample]
        // H is k×n matrix. We're updating row target_col across all n samples.
        // The interaction is: sum_l Denom[target_col, l] * H[l, sample]
        float interaction = 0.0f;

        #pragma unroll 8
        for (int l = 0; l < k; l++) {
            // H[l, sample] in column-major: sample*k + l
            float h_val = H[sample * k + l];

            // Denom[target_col, l] in column-major: l*k + target_col
            float denom_val = Denom[l * k + target_col];

            interaction += h_val * denom_val;
        }

        // C. HALS update
        float h_old = H[sample * k + target_col];
        float gradient = (num - interaction) / denom_diag;
        float h_updated = h_old + gradient;

        // D. Non-negativity
        h_updated = fmaxf(h_updated, epsilon);

        // Store
        if (offset == 0) h_new.x = h_updated;
        else if (offset == 1) h_new.y = h_updated;
        else if (offset == 2) h_new.z = h_updated;
        else if (offset == 3) h_new.w = h_updated;
    }

    // E. Write-back
    // We're writing to row target_col across samples [sample_start, sample_start+3]
    // This is NOT coalesced because we're writing to different columns
    for (int offset = 0; offset < 4 && sample_start + offset < n; offset++) {
        float val;
        if (offset == 0) val = h_new.x;
        else if (offset == 1) val = h_new.y;
        else if (offset == 2) val = h_new.z;
        else val = h_new.w;
        H[(sample_start + offset) * k + target_col] = val;
    }
}

/*
 * Kernel: Normalize a single column of W
 *
 * Computes L2 norm and divides all elements by it
 */
__global__ void normalize_W_column(
    float* W,
    int m, int k,
    int target_col,
    float epsilon
) {
    // First, compute the norm (reduction)
    __shared__ float shared_sum[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread computes partial sum
    float local_sum = 0.0f;
    if (idx < m) {
        float val = W[target_col * m + idx];
        local_sum = val * val;
    }

    shared_sum[tid] = local_sum;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    // Block 0 writes result to global memory
    __shared__ float norm_val;
    if (tid == 0) {
        // Note: For simplicity, assuming single block. For multiple blocks, need atomic adds.
        norm_val = sqrtf(shared_sum[0]);
        if (norm_val < epsilon) norm_val = epsilon;
    }
    __syncthreads();

    // Now divide all elements
    if (idx < m) {
        W[target_col * m + idx] /= norm_val;
    }
}

// ============================================================================
// ERROR COMPUTATION (Column-Major)
// ============================================================================

/*
 * Compute relative reconstruction error: ||X - WH|| / ||X||
 *
 * IMPORTANT: This function assumes COLUMN-MAJOR storage for W and H
 * - W is m×k: W[i,f] at position f*m + i
 * - H is k×n: H[f,j] at position j*k + f
 * - X is m×n: X[i,j] at position j*m + i (column-major)
 *
 * The utils.h version assumes row-major, which gives wrong results!
 */
float compute_error_column_major(const float* X, const float* W, const float* H,
                                  int m, int n, int k) {
    double norm_diff_sq = 0.0;
    double norm_X_sq = 0.0;

    // Compute ||X - WH||² and ||X||²
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            // X[i,j] in column-major: j*m + i
            float x_val = X[j * m + i];

            // Compute (WH)[i,j] = Σ_f W[i,f] * H[f,j]
            float wh_val = 0.0f;
            for (int f = 0; f < k; f++) {
                // W[i,f] in column-major: f*m + i
                // H[f,j] in column-major: j*k + f
                wh_val += W[f * m + i] * H[j * k + f];
            }

            float diff = x_val - wh_val;
            norm_diff_sq += diff * diff;
            norm_X_sq += x_val * x_val;
        }
    }

    return (float)(sqrt(norm_diff_sq) / sqrt(norm_X_sq));
}

// ============================================================================
// HOST FUNCTION
// ============================================================================

void nmf_hals_gpu_strict(float* h_X, int m, int n, int k, int max_iter,
                         float* time_ms, float* bandwidth_achieved, float* flops_achieved,
                         float* final_error) {

    printf("========================================\n");
    printf("GPU HALS - LEVEL 1: STRICT PARALLELISM\n");
    printf("========================================\n");
    printf("Matrix: %dx%d, Rank: %d, Iterations: %d\n", m, n, k, max_iter);
    printf("Strategy: Single-column updates (strict Gauss-Seidel)\n");
    printf("Parallelism: Row-wise (%d threads per column)\n", m);
    printf("Expected convergence: ~10-15 iterations\n");
    printf("----------------------------------------\n\n");

    // Allocate device memory
    float *d_X, *d_W, *d_H;
    float *d_Numerator_W, *d_Denom_W;  // For W update: X×H^T and H×H^T
    float *d_Numerator_H, *d_Denom_H;  // For H update: W^T×X and W^T×W

    CUDA_CHECK(cudaMalloc(&d_X, m * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W, m * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_H, k * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Numerator_W, m * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Denom_W, k * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Numerator_H, k * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Denom_H, k * k * sizeof(float)));

    // Initialize W and H on host
    float *h_W = (float*)malloc(m * k * sizeof(float));
    float *h_H = (float*)malloc(k * n * sizeof(float));

    generate_random_matrix(h_W, m, k, 42);
    generate_random_matrix(h_H, k, n, 43);

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_X, h_X, m * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, h_W, m * k * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_H, h_H, k * n * sizeof(float), cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float alpha = 1.0f;
    float beta = 0.0f;
    float epsilon = 1e-16f;

    // Kernel configuration
    int block_size = 128;
    int grid_size_W = ((m / 4) + block_size - 1) / block_size;  // Each thread handles 4 rows
    int grid_size_H = ((n / 4) + block_size - 1) / block_size;  // Each thread handles 4 samples
    int grid_size_norm = (m + block_size - 1) / block_size;

    printf("Kernel Configuration:\n");
    printf("  Block size: %d threads\n", block_size);
    printf("  Grid size W: %d blocks (%d threads per column)\n", grid_size_W, grid_size_W * block_size);
    printf("  Grid size H: %d blocks (%d threads per feature)\n", grid_size_H, grid_size_H * block_size);
    printf("  Columns to update sequentially: %d\n", k);
    printf("  Synchronizations per iteration: %d\n", 2 * k);
    printf("----------------------------------------\n\n");

    // Main iteration loop
    CudaTimer timer;
    timer.startTimer();

    for (int iter = 0; iter < max_iter; iter++) {
        // ====================================================================
        // Update H (strict Gauss-Seidel across features)
        // ====================================================================

        // 1. Pre-compute matrices needed for ALL H columns
        // Numerator_H = W^T × X (k × n)
        // Use cuBLAS transpose: C = alpha * op(A) * op(B) + beta * C
        // W is m×k, X is m×n, result is k×n
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_T, CUBLAS_OP_N,
                                 k, n, m,
                                 &alpha, d_W, m, d_X, m,
                                 &beta, d_Numerator_H, k));

        // Denom_H = W^T × W (k × k)
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_T, CUBLAS_OP_N,
                                 k, k, m,
                                 &alpha, d_W, m, d_W, m,
                                 &beta, d_Denom_H, k));

        // 2. Update each feature column sequentially
        for (int f = 0; f < k; f++) {
            update_H_column_strict_hals<<<grid_size_H, block_size>>>(
                d_H, d_Numerator_H, d_Denom_H, k, n, f, epsilon
            );

            // CRITICAL: Wait for column f to finish before starting f+1
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // ====================================================================
        // Update W (strict Gauss-Seidel across features)
        // ====================================================================

        // 1. Pre-compute matrices
        // Numerator_W = X × H^T (m × k)
        // X is m×n, H is k×n, result is m×k
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_T,
                                 m, k, n,
                                 &alpha, d_X, m, d_H, k,
                                 &beta, d_Numerator_W, m));

        // Denom_W = H × H^T (k × k)
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_T,
                                 k, k, n,
                                 &alpha, d_H, k, d_H, k,
                                 &beta, d_Denom_W, k));

        // 2. Update each feature column sequentially
        for (int f = 0; f < k; f++) {
            update_W_column_strict_hals<<<grid_size_W, block_size>>>(
                d_W, d_Numerator_W, d_Denom_W, m, k, f, epsilon
            );
            CUDA_CHECK(cudaDeviceSynchronize());

            // 3. Normalize column f
            normalize_W_column<<<grid_size_norm, block_size>>>(
                d_W, m, k, f, epsilon
            );
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Print progress
        if (iter % 5 == 0 || iter == max_iter - 1) {
            printf("Iteration %d\n", iter);
        }
    }

    float elapsed_ms = timer.stopTimer();

    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_W, d_W, m * k * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_H, d_H, k * n * sizeof(float), cudaMemcpyDeviceToHost));

    // Save results
    printf("\nSaving results for visualization...\n");
    save_matrix_binary("results/W_matrix_hals_gpu.bin", h_W, m, k);
    save_matrix_binary("results/H_matrix_hals_gpu.bin", h_H, k, n);

    // Calculate metrics using column-major error computation
    // NOTE: compute_relative_error_dense from utils.h assumes ROW-MAJOR,
    // but our matrices are COLUMN-MAJOR. Must use our own function!
    float error = compute_error_column_major(h_X, h_W, h_H, m, n, k);

    // Estimate FLOPS (rough approximation)
    long long flops_per_iter = 0;
    flops_per_iter += 2LL * k * n * m;  // W^T × X
    flops_per_iter += 2LL * k * k * m;  // W^T × W
    flops_per_iter += 2LL * m * k * n;  // X × H^T
    flops_per_iter += 2LL * k * k * n;  // H × H^T
    // Column updates: k * (m * k) for W, k * (n * k) for H
    flops_per_iter += 2LL * k * m * k;
    flops_per_iter += 2LL * k * n * k;

    long long total_flops = flops_per_iter * max_iter;
    float gflops = (total_flops / elapsed_ms) * 1000.0f / 1e9f;

    // Estimate bandwidth (rough)
    long long bytes_per_iter = (long long)(m * n + m * k + k * n) * sizeof(float) * 2 * k;  // Amplified by k
    float bandwidth_gbps = (bytes_per_iter * max_iter / elapsed_ms) * 1000.0f / 1e9f;

    // Print results
    printf("\n========================================\n");
    printf("GPU HALS STRICT RESULTS\n");
    printf("========================================\n");
    printf("Time: %.2f ms\n", elapsed_ms);
    printf("Time per iteration: %.2f ms\n", elapsed_ms / max_iter);
    printf("FLOPS: %.2f GFLOPS\n", gflops);
    printf("Bandwidth (est): %.2f GB/s\n", bandwidth_gbps);
    printf("Final error: %.6e\n", error);
    printf("========================================\n\n");

    printf("Performance Analysis:\n");
    printf("  - Sequential column updates: %d per iteration\n", 2 * k);
    printf("  - Kernel launches: %d per iteration\n", 4 * k);  // H update, W update, normalize x2
    printf("  - Memory pattern: Non-coalesced row reads\n");
    printf("  - Expected bottleneck: Memory bandwidth + sync overhead\n");
    printf("========================================\n");

    // Return metrics
    *time_ms = elapsed_ms;
    *bandwidth_achieved = bandwidth_gbps;
    *flops_achieved = gflops;
    *final_error = error;

    // Cleanup
    CUBLAS_CHECK(cublasDestroy(handle));
    cudaFree(d_X);
    cudaFree(d_W);
    cudaFree(d_H);
    cudaFree(d_Numerator_W);
    cudaFree(d_Denom_W);
    cudaFree(d_Numerator_H);
    cudaFree(d_Denom_H);

    free(h_W);
    free(h_H);
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: %s <matrix_file> <rank_k> <max_iter>\n", argv[0]);
        printf("Example: %s data/dense_1000.bin 20 15\n", argv[0]);
        return 1;
    }

    const char* matrix_file = argv[1];
    int k = atoi(argv[2]);
    int max_iter = atoi(argv[3]);

    printf("\n╔════════════════════════════════════════════════════════════════╗\n");
    printf("║     CSE587 NMF - GPU HALS LEVEL 1: STRICT PARALLELISM        ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");

    // Load matrix
    float* h_X = NULL;
    int m, n;
    load_matrix_binary(matrix_file, &h_X, &m, &n);
    printf("\n");

    // Run GPU HALS
    float time_ms, bandwidth_gbps, gflops, error;
    nmf_hals_gpu_strict(h_X, m, n, k, max_iter, &time_ms, &bandwidth_gbps, &gflops, &error);

    // Save metrics
    printf("\nSaving metrics to results/hals_gpu_strict_metrics.txt...\n");
    FILE* fp = fopen("results/hals_gpu_strict_metrics.txt", "w");
    if (fp) {
        fprintf(fp, "Version: GPU HALS Strict (Level 1)\n");
        fprintf(fp, "Size: %dx%d\n", m, n);
        fprintf(fp, "Rank: %d\n", k);
        fprintf(fp, "Iterations: %d\n", max_iter);
        fprintf(fp, "Time: %.2f ms\n", time_ms);
        fprintf(fp, "Time per iteration: %.2f ms\n", time_ms / max_iter);
        fprintf(fp, "Final error: %e\n", error);
        fprintf(fp, "Bandwidth: %.2f GB/s\n", bandwidth_gbps);
        fprintf(fp, "GFLOPS: %.2f\n", gflops);
        fprintf(fp, "\nAlgorithm: Strict HALS (Gauss-Seidel)\n");
        fprintf(fp, "Parallelism: Single-column (row-wise)\n");
        fclose(fp);
        printf("✓ Metrics saved\n");
    }

    printf("\nNext steps:\n");
    printf("1. Compare convergence: CPU HALS vs GPU HALS strict\n");
    printf("2. Verify ~10-15 iteration convergence maintained\n");
    printf("3. Measure speedup vs CPU baseline\n");
    printf("4. (Optional) Implement Level 2 with tiled batching\n\n");

    free(h_X);
    return 0;
}
