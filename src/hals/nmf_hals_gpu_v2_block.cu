#include "../utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <random>

/*
 * GPU HALS NMF - LEVEL 2: BLOCK-PARALLEL WITH RANDOM SHUFFLING
 *
 * Purpose: Increase parallelism by updating multiple features in parallel
 *          while maintaining local Gauss-Seidel ordering within blocks.
 *
 * Algorithm: Block-Parallel HALS
 * ==============================
 * - Partition k features into blocks of size B (e.g., B=5)
 * - Each iteration: randomly shuffle feature-to-block assignments
 * - Within each block: sequential Gauss-Seidel (feature i sees i-1's update)
 * - Between blocks: parallel via CUDA streams (Jacobi-style)
 *
 * Why Random Shuffling?
 * =====================
 * - Fixed grouping creates systematic bias (some pairs never interact correctly)
 * - Random shuffling ensures every feature pair eventually sees each other's updates
 * - Similar to stochastic mini-batches in SGD - helps convergence
 *
 * Architecture:
 * ============
 * Iteration i (random shuffle):
 *   Block 0: features [3, 17, 8, 1, 12]  ──┐
 *   Block 1: features [5, 19, 0, 14, 7]  ──┼── Run in PARALLEL (streams)
 *   Block 2: features [11, 6, 16, 2, 18] ──┤
 *   Block 3: features [9, 4, 13, 10, 15] ──┘
 *
 *   Within Block 0: Sequential (3→17→8→1→12)
 *   Feature 12 sees updated 3,17,8,1 before updating
 *
 * Performance:
 * ============
 * - Sync reduction: 2k → 2 per iteration
 * - May need ~10-20% more iterations (partial Jacobi)
 * - But each iteration is much faster
 *
 * Memory Layout: Column-major (for cuBLAS compatibility)
 * - W[i,f] at position f*m + i
 * - H[f,j] at position j*k + f
 */

#define DEFAULT_BLOCK_SIZE 5

// ============================================================================
// SHUFFLING INFRASTRUCTURE
// ============================================================================

/*
 * Fisher-Yates shuffle: randomly permute array of k integers
 * Called once per iteration on CPU (negligible overhead for small k)
 */
void shuffle_features(int* perm, int k, unsigned int seed) {
    // Initialize: perm[i] = i
    for (int i = 0; i < k; i++) {
        perm[i] = i;
    }

    // Fisher-Yates shuffle
    std::mt19937 rng(seed);
    for (int i = k - 1; i > 0; i--) {
        std::uniform_int_distribution<int> dist(0, i);
        int j = dist(rng);
        std::swap(perm[i], perm[j]);
    }
}

/*
 * Get feature index for given block and local position
 * Returns -1 for padding (when k is not divisible by block_size)
 */
inline int get_feature(int* perm, int block_id, int local_idx, int block_size, int k) {
    int flat_idx = block_id * block_size + local_idx;
    return (flat_idx < k) ? perm[flat_idx] : -1;
}

// ============================================================================
// CUDA KERNELS (same as Level 1)
// ============================================================================

/*
 * Kernel: Update single column of W (feature f)
 */
__global__ void update_W_column_block_hals(
    float* W,
    const float* Numerator,
    const float* Denom,
    int m, int k,
    int target_col,
    float epsilon
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int row_start = tid * 4;

    if (row_start >= m) return;

    float denom_diag = Denom[target_col * k + target_col];
    if (denom_diag < epsilon) denom_diag = epsilon;

    float4 w_new;

    #pragma unroll
    for (int offset = 0; offset < 4; offset++) {
        int row = row_start + offset;
        if (row >= m) break;

        float num = Numerator[target_col * m + row];

        float interaction = 0.0f;
        #pragma unroll 8
        for (int l = 0; l < k; l++) {
            float w_val = W[l * m + row];
            float denom_val = Denom[target_col * k + l];
            interaction += w_val * denom_val;
        }

        float w_old = W[target_col * m + row];
        float gradient = (num - interaction) / denom_diag;
        float w_updated = fmaxf(w_old + gradient, epsilon);

        if (offset == 0) w_new.x = w_updated;
        else if (offset == 1) w_new.y = w_updated;
        else if (offset == 2) w_new.z = w_updated;
        else if (offset == 3) w_new.w = w_updated;
    }

    if (row_start + 3 < m) {
        float4* W_col_ptr = (float4*)(&W[target_col * m]);
        W_col_ptr[tid] = w_new;
    } else {
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
 */
__global__ void update_H_column_block_hals(
    float* H,
    const float* Numerator,
    const float* Denom,
    int k, int n,
    int target_col,
    float epsilon
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int sample_start = tid * 4;

    if (sample_start >= n) return;

    float denom_diag = Denom[target_col * k + target_col];
    if (denom_diag < epsilon) denom_diag = epsilon;

    float4 h_new;

    #pragma unroll
    for (int offset = 0; offset < 4; offset++) {
        int sample = sample_start + offset;
        if (sample >= n) break;

        float num = Numerator[sample * k + target_col];

        float interaction = 0.0f;
        #pragma unroll 8
        for (int l = 0; l < k; l++) {
            float h_val = H[sample * k + l];
            float denom_val = Denom[l * k + target_col];
            interaction += h_val * denom_val;
        }

        float h_old = H[sample * k + target_col];
        float gradient = (num - interaction) / denom_diag;
        float h_updated = fmaxf(h_old + gradient, epsilon);

        if (offset == 0) h_new.x = h_updated;
        else if (offset == 1) h_new.y = h_updated;
        else if (offset == 2) h_new.z = h_updated;
        else if (offset == 3) h_new.w = h_updated;
    }

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
 */
__global__ void normalize_W_column_block(
    float* W,
    int m, int k,
    int target_col,
    float epsilon
) {
    __shared__ float shared_sum[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float local_sum = 0.0f;
    if (idx < m) {
        float val = W[target_col * m + idx];
        local_sum = val * val;
    }

    shared_sum[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    __shared__ float norm_val;
    if (tid == 0) {
        norm_val = sqrtf(shared_sum[0]);
        if (norm_val < epsilon) norm_val = epsilon;
    }
    __syncthreads();

    if (idx < m) {
        W[target_col * m + idx] /= norm_val;
    }
}

// ============================================================================
// ERROR COMPUTATION (Column-Major)
// ============================================================================

float compute_error_column_major(const float* X, const float* W, const float* H,
                                  int m, int n, int k) {
    double norm_diff_sq = 0.0;
    double norm_X_sq = 0.0;

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            float x_val = X[j * m + i];

            float wh_val = 0.0f;
            for (int f = 0; f < k; f++) {
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

void nmf_hals_gpu_block(float* h_X, int m, int n, int k, int max_iter,
                        int block_size,
                        float* time_ms, float* bandwidth_achieved,
                        float* flops_achieved, float* final_error,
                        bool log_convergence, int log_interval) {

    int num_blocks = (k + block_size - 1) / block_size;

    printf("========================================\n");
    printf("GPU HALS - LEVEL 2: BLOCK-PARALLEL\n");
    printf("========================================\n");
    printf("Matrix: %dx%d, Rank: %d, Iterations: %d\n", m, n, k, max_iter);
    printf("Block size: %d, Number of blocks: %d\n", block_size, num_blocks);
    printf("Strategy: Random shuffle + block-parallel streams\n");
    printf("Within-block: Gauss-Seidel (sequential)\n");
    printf("Between-blocks: Jacobi (parallel via streams)\n");
    printf("----------------------------------------\n\n");

    // Allocate device memory
    float *d_X, *d_W, *d_H;
    float *d_Numerator_W, *d_Denom_W;
    float *d_Numerator_H, *d_Denom_H;

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

    // Create CUDA streams (one per block)
    cudaStream_t* streams = new cudaStream_t[num_blocks];
    for (int i = 0; i < num_blocks; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    // Allocate permutation array for random shuffling
    int* perm = new int[k];

    float alpha = 1.0f;
    float beta = 0.0f;
    float epsilon = 1e-16f;

    // Kernel configuration
    int threads_per_block = 128;
    int grid_size_W = ((m / 4) + threads_per_block - 1) / threads_per_block;
    int grid_size_H = ((n / 4) + threads_per_block - 1) / threads_per_block;
    int grid_size_norm = (m + threads_per_block - 1) / threads_per_block;

    printf("Kernel Configuration:\n");
    printf("  Threads per block: %d\n", threads_per_block);
    printf("  Grid size W: %d blocks\n", grid_size_W);
    printf("  Grid size H: %d blocks\n", grid_size_H);
    printf("  CUDA streams: %d (one per feature block)\n", num_blocks);
    printf("  Syncs per iteration: 2 (vs %d in strict)\n", 2 * k);
    printf("----------------------------------------\n\n");

    // Convergence logging setup
    FILE* csv_fp = NULL;
    if (log_convergence) {
        csv_fp = fopen("results/convergence_hals_block.csv", "w");
        if (csv_fp) {
            fprintf(csv_fp, "iteration,error,time_ms\n");
        }
    }

    // Main iteration loop
    CudaTimer timer;
    timer.startTimer();

    for (int iter = 0; iter < max_iter; iter++) {

        // ================================================================
        // RANDOM SHUFFLE each iteration
        // Different seed ensures different permutations
        // ================================================================
        shuffle_features(perm, k, 42 + iter);

        // ================================================================
        // Update H (block-parallel)
        // ================================================================

        // 1. Pre-compute matrices (same as strict)
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_T, CUBLAS_OP_N,
                                 k, n, m,
                                 &alpha, d_W, m, d_X, m,
                                 &beta, d_Numerator_H, k));

        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_T, CUBLAS_OP_N,
                                 k, k, m,
                                 &alpha, d_W, m, d_W, m,
                                 &beta, d_Denom_H, k));

        // 2. Update features in block-parallel fashion
        // Each block runs in its own stream
        // Within a block, features are sequential (Gauss-Seidel)
        for (int b = 0; b < num_blocks; b++) {
            for (int local_f = 0; local_f < block_size; local_f++) {
                int f = get_feature(perm, b, local_f, block_size, k);
                if (f >= 0) {  // Valid feature (not padding)
                    update_H_column_block_hals<<<grid_size_H, threads_per_block, 0, streams[b]>>>(
                        d_H, d_Numerator_H, d_Denom_H, k, n, f, epsilon
                    );
                }
            }
        }

        // 3. Barrier: wait for all blocks to complete H update
        for (int b = 0; b < num_blocks; b++) {
            CUDA_CHECK(cudaStreamSynchronize(streams[b]));
        }

        // ================================================================
        // Update W (block-parallel)
        // ================================================================

        // 1. Pre-compute matrices
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_T,
                                 m, k, n,
                                 &alpha, d_X, m, d_H, k,
                                 &beta, d_Numerator_W, m));

        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_T,
                                 k, k, n,
                                 &alpha, d_H, k, d_H, k,
                                 &beta, d_Denom_W, k));

        // 2. Update features in block-parallel fashion (same shuffle as H)
        for (int b = 0; b < num_blocks; b++) {
            for (int local_f = 0; local_f < block_size; local_f++) {
                int f = get_feature(perm, b, local_f, block_size, k);
                if (f >= 0) {
                    update_W_column_block_hals<<<grid_size_W, threads_per_block, 0, streams[b]>>>(
                        d_W, d_Numerator_W, d_Denom_W, m, k, f, epsilon
                    );
                    // Normalize in same stream (maintains order within block)
                    normalize_W_column_block<<<grid_size_norm, threads_per_block, 0, streams[b]>>>(
                        d_W, m, k, f, epsilon
                    );
                }
            }
        }

        // 3. Barrier: wait for all blocks to complete W update
        for (int b = 0; b < num_blocks; b++) {
            CUDA_CHECK(cudaStreamSynchronize(streams[b]));
        }

        // Per-iteration convergence logging
        if (log_convergence && (iter % log_interval == 0 || iter == max_iter - 1)) {
            float iter_time_ms = timer.stopTimer();

            // Copy current W, H to host for error computation
            CUDA_CHECK(cudaMemcpy(h_W, d_W, m * k * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_H, d_H, k * n * sizeof(float), cudaMemcpyDeviceToHost));

            float error = compute_error_column_major(h_X, h_W, h_H, m, n, k);

            if (csv_fp) {
                fprintf(csv_fp, "%d,%.6e,%.2f\n", iter, error, iter_time_ms);
                fflush(csv_fp);
            }
            printf("Iteration %d: error=%.6e, time=%.2f ms\n", iter, error, iter_time_ms);

            // Restart timer for next interval
            timer.startTimer();
        } else if (iter % 5 == 0 || iter == max_iter - 1) {
            printf("Iteration %d\n", iter);
        }
    }

    float elapsed_ms = timer.stopTimer();

    // Close convergence log
    if (csv_fp) {
        fclose(csv_fp);
        printf("Convergence log saved to results/convergence_hals_block.csv\n");
    }

    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_W, d_W, m * k * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_H, d_H, k * n * sizeof(float), cudaMemcpyDeviceToHost));

    // Save results
    printf("\nSaving results for visualization...\n");
    save_matrix_binary("results/W_matrix_hals_gpu_block.bin", h_W, m, k);
    save_matrix_binary("results/H_matrix_hals_gpu_block.bin", h_H, k, n);

    // Calculate error
    float error = compute_error_column_major(h_X, h_W, h_H, m, n, k);

    // Estimate FLOPS
    long long flops_per_iter = 0;
    flops_per_iter += 2LL * k * n * m;  // W^T × X
    flops_per_iter += 2LL * k * k * m;  // W^T × W
    flops_per_iter += 2LL * m * k * n;  // X × H^T
    flops_per_iter += 2LL * k * k * n;  // H × H^T
    flops_per_iter += 2LL * k * m * k;  // W updates
    flops_per_iter += 2LL * k * n * k;  // H updates

    long long total_flops = flops_per_iter * max_iter;
    float gflops = (total_flops / elapsed_ms) * 1000.0f / 1e9f;

    // Estimate bandwidth
    long long bytes_per_iter = (long long)(m * n + m * k + k * n) * sizeof(float) * 2 * k;
    float bandwidth_gbps = (bytes_per_iter * max_iter / elapsed_ms) * 1000.0f / 1e9f;

    // Print results
    printf("\n========================================\n");
    printf("GPU HALS BLOCK-PARALLEL RESULTS\n");
    printf("========================================\n");
    printf("Time: %.2f ms\n", elapsed_ms);
    printf("Time per iteration: %.2f ms\n", elapsed_ms / max_iter);
    printf("FLOPS: %.2f GFLOPS\n", gflops);
    printf("Bandwidth (est): %.2f GB/s\n", bandwidth_gbps);
    printf("Final error: %.6e\n", error);
    printf("========================================\n\n");

    printf("Performance Analysis:\n");
    printf("  - Block size: %d features\n", block_size);
    printf("  - Parallel blocks: %d\n", num_blocks);
    printf("  - Syncs per iteration: 2 (vs %d in strict)\n", 2 * k);
    printf("  - Random shuffle: yes (different each iteration)\n");
    printf("========================================\n");

    // Return metrics
    *time_ms = elapsed_ms;
    *bandwidth_achieved = bandwidth_gbps;
    *flops_achieved = gflops;
    *final_error = error;

    // Cleanup
    delete[] perm;
    for (int i = 0; i < num_blocks; i++) {
        cudaStreamDestroy(streams[i]);
    }
    delete[] streams;

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
        printf("Usage: %s <matrix_file> <rank_k> <max_iter> [block_size] [--log-convergence [interval]]\n", argv[0]);
        printf("Example: %s data/dense_1000.bin 20 15 5\n", argv[0]);
        printf("         %s data/dense_1000.bin 20 50 5 --log-convergence 5\n", argv[0]);
        printf("\nblock_size: features per block (default: %d)\n", DEFAULT_BLOCK_SIZE);
        return 1;
    }

    const char* matrix_file = argv[1];
    int k = atoi(argv[2]);
    int max_iter = atoi(argv[3]);
    int block_size = DEFAULT_BLOCK_SIZE;
    bool log_convergence = false;
    int log_interval = 1;

    // Parse optional arguments
    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "--log-convergence") == 0) {
            log_convergence = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                log_interval = atoi(argv[i + 1]);
                i++;
            }
        } else if (argv[i][0] != '-') {
            block_size = atoi(argv[i]);
        }
    }

    printf("\n╔════════════════════════════════════════════════════════════════╗\n");
    printf("║     CSE587 NMF - GPU HALS LEVEL 2: BLOCK-PARALLEL             ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");

    // Load matrix
    float* h_X = NULL;
    int m, n;
    load_matrix_binary(matrix_file, &h_X, &m, &n);
    printf("\n");

    // Run GPU HALS Block-Parallel
    float time_ms, bandwidth_gbps, gflops, error;
    nmf_hals_gpu_block(h_X, m, n, k, max_iter, block_size,
                       &time_ms, &bandwidth_gbps, &gflops, &error,
                       log_convergence, log_interval);

    // Save metrics
    printf("\nSaving metrics to results/hals_gpu_block_metrics.txt...\n");
    FILE* fp = fopen("results/hals_gpu_block_metrics.txt", "w");
    if (fp) {
        fprintf(fp, "Version: GPU HALS Block-Parallel (Level 2)\n");
        fprintf(fp, "Size: %dx%d\n", m, n);
        fprintf(fp, "Rank: %d\n", k);
        fprintf(fp, "Block size: %d\n", block_size);
        fprintf(fp, "Iterations: %d\n", max_iter);
        fprintf(fp, "Time: %.2f ms\n", time_ms);
        fprintf(fp, "Time per iteration: %.2f ms\n", time_ms / max_iter);
        fprintf(fp, "Final_Error: %.6e\n", error);
        fprintf(fp, "Bandwidth: %.2f GB/s\n", bandwidth_gbps);
        fprintf(fp, "GFLOPS: %.2f\n", gflops);
        fprintf(fp, "\nAlgorithm: Block-Parallel HALS\n");
        fprintf(fp, "Within-block: Gauss-Seidel (sequential)\n");
        fprintf(fp, "Between-blocks: Jacobi (parallel streams)\n");
        fprintf(fp, "Shuffling: Random each iteration\n");
        fclose(fp);
        printf("✓ Metrics saved\n");
    }

    if (log_convergence) {
        printf("✓ Convergence log saved to results/convergence_hals_block.csv\n");
    }

    printf("\nComparison with Level 1 (Strict):\n");
    printf("  Run: ./nmf_hals_gpu_strict %s %d %d\n", matrix_file, k, max_iter);
    printf("  Then compare: diff results/hals_gpu_strict_metrics.txt results/hals_gpu_block_metrics.txt\n\n");

    free(h_X);
    return 0;
}
