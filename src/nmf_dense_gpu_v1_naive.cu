#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

/*
 * LEVEL 1: NAIVE GPU IMPLEMENTATION
 *
 * Purpose: Establish GPU baseline with minimal optimization
 *
 * Characteristics:
 * - Uses cuBLAS for all matrix operations
 * - Simple element-wise kernels (no optimization)
 * - No shared memory usage
 * - No memory coalescing considerations
 * - Basic thread block size
 *
 * Expected Performance:
 * - 10-30x speedup over CPU
 * - Low memory bandwidth utilization (~30-40%)
 * - Low occupancy (~40-60%)
 *
 * Learning Focus:
 * - CUDA memory model basics
 * - cuBLAS API usage
 * - Kernel launch mechanics
 * - Performance baseline establishment
 */

// ============================================================================
// Naive Element-wise Kernels (No optimization)
// ============================================================================

__global__ void elementwise_multiply_naive(float* A, float* B, float* C, int size) {
    // Simple 1D indexing, no optimization
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        C[idx] = A[idx] * B[idx];
    }
}

__global__ void elementwise_divide_eps_naive(float* A, float* B, float* C, int size, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        C[idx] = A[idx] / (B[idx] + eps);
    }
}

// ============================================================================
// Naive NMF Implementation
// ============================================================================

void nmf_naive_gpu(float* h_X, int m, int n, int k, int max_iter,
                   float* time_ms, float* bandwidth_achieved, float* flops_achieved) {

    printf("========================================\n");
    printf("LEVEL 1: NAIVE GPU IMPLEMENTATION\n");
    printf("========================================\n");
    printf("Matrix: %dx%d, Rank: %d, Iterations: %d\n", m, n, k, max_iter);
    printf("Optimization Level: NONE (Baseline)\n");
    printf("----------------------------------------\n");

    // ========================================================================
    // Allocate device memory (no optimization)
    // ========================================================================

    float *d_X, *d_W, *d_H;
    float *d_WtW, *d_WtX, *d_HHt, *d_XHt;
    float *d_temp_H, *d_temp_W;

    CUDA_CHECK(cudaMalloc(&d_X, m * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W, m * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_H, k * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_WtW, k * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_WtX, k * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_HHt, k * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_XHt, m * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_temp_H, k * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_temp_W, m * k * sizeof(float)));

    // ========================================================================
    // Initialize W and H
    // ========================================================================

    float *h_W = (float*)malloc(m * k * sizeof(float));
    float *h_H = (float*)malloc(k * n * sizeof(float));

    generate_random_matrix(h_W, m, k, 42);
    generate_random_matrix(h_H, k, n, 43);

    CUDA_CHECK(cudaMemcpy(d_X, h_X, m * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, h_W, m * k * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_H, h_H, k * n * sizeof(float), cudaMemcpyHostToDevice));

    // ========================================================================
    // Create cuBLAS handle
    // ========================================================================

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float alpha = 1.0f;
    float beta = 0.0f;

    // ========================================================================
    // Naive kernel configuration (not optimized)
    // ========================================================================

    // Using fixed 128 threads per block (not optimized for occupancy)
    int block_size = 128;
    int grid_size_H = (k * n + block_size - 1) / block_size;
    int grid_size_W = (m * k + block_size - 1) / block_size;

    printf("\nKernel Configuration (Naive):\n");
    printf("  Block size: %d threads\n", block_size);
    printf("  Grid size H: %d blocks\n", grid_size_H);
    printf("  Grid size W: %d blocks\n", grid_size_W);
    printf("  Total threads H: %d\n", grid_size_H * block_size);
    printf("  Total threads W: %d\n", grid_size_W * block_size);
    printf("----------------------------------------\n\n");

    // ========================================================================
    // Main iteration loop (with timing)
    // ========================================================================

    CudaTimer timer;
    timer.startTimer();

    for (int iter = 0; iter < max_iter; iter++) {
        // ====================================================================
        // Update H: H = H .* (W^T × X) ./ (W^T × W × H + eps)
        // ====================================================================

        // 1. WtW = W^T × W (using cuBLAS, no custom optimization)
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_T, CUBLAS_OP_N,
                                 k, k, m,
                                 &alpha, d_W, m, d_W, m,
                                 &beta, d_WtW, k));

        // 2. WtX = W^T × X (using cuBLAS)
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_T, CUBLAS_OP_N,
                                 k, n, m,
                                 &alpha, d_W, m, d_X, m,
                                 &beta, d_WtX, k));

        // 3. temp_H = WtW × H (using cuBLAS)
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 k, n, k,
                                 &alpha, d_WtW, k, d_H, k,
                                 &beta, d_temp_H, k));

        // 4. H = H .* WtX (naive kernel)
        elementwise_multiply_naive<<<grid_size_H, block_size>>>(d_H, d_WtX, d_H, k * n);

        // 5. H = H ./ (temp_H + eps) (naive kernel)
        elementwise_divide_eps_naive<<<grid_size_H, block_size>>>(d_H, d_temp_H, d_H, k * n, 1e-10f);

        // ====================================================================
        // Update W: W = W .* (X × H^T) ./ (W × H × H^T + eps)
        // ====================================================================

        // 1. HHt = H × H^T
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_T,
                                 k, k, n,
                                 &alpha, d_H, k, d_H, k,
                                 &beta, d_HHt, k));

        // 2. XHt = X × H^T
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_T,
                                 m, k, n,
                                 &alpha, d_X, m, d_H, k,
                                 &beta, d_XHt, m));

        // 3. temp_W = W × HHt
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 m, k, k,
                                 &alpha, d_W, m, d_HHt, k,
                                 &beta, d_temp_W, m));

        // 4. W = W .* XHt (naive kernel)
        elementwise_multiply_naive<<<grid_size_W, block_size>>>(d_W, d_XHt, d_W, m * k);

        // 5. W = W ./ (temp_W + eps) (naive kernel)
        elementwise_divide_eps_naive<<<grid_size_W, block_size>>>(d_W, d_temp_W, d_W, m * k, 1e-10f);

        if (iter % 10 == 0) {
            printf("Iteration %d\n", iter);
        }
    }

    float elapsed_ms = timer.stopTimer();

    // ========================================================================
    // Copy results back
    // ========================================================================

    CUDA_CHECK(cudaMemcpy(h_W, d_W, m * k * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_H, d_H, k * n * sizeof(float), cudaMemcpyDeviceToHost));

    // ========================================================================
    // Calculate metrics
    // ========================================================================

    // FLOPS calculation (per iteration)
    long long flops_per_iter = 0;
    flops_per_iter += 2LL * k * k * m;        // WtW
    flops_per_iter += 2LL * k * n * m;        // WtX
    flops_per_iter += 2LL * k * n * k;        // WtW × H
    flops_per_iter += 2LL * k * k * n;        // HHt
    flops_per_iter += 2LL * m * k * n;        // XHt
    flops_per_iter += 2LL * m * k * k;        // W × HHt
    flops_per_iter += 4LL * (m * k + k * n);  // Element-wise ops

    long long total_flops = flops_per_iter * max_iter;
    float flops_per_second = (total_flops / elapsed_ms) * 1000.0f;
    float gflops = flops_per_second / 1e9f;

    // Bandwidth calculation (rough estimate)
    long long bytes_per_iter = 0;
    bytes_per_iter += (long long)(m * n + m * k + k * n) * 4 * 2;  // Read X, W, H (multiple times)
    bytes_per_iter += (long long)(m * k + k * n) * 4 * 2;          // Write W, H

    long long total_bytes = bytes_per_iter * max_iter;
    float bandwidth_gbps = (total_bytes / elapsed_ms) * 1000.0f / 1e9f;

    // Compute final error
    float error = compute_relative_error_dense(h_X, h_W, h_H, m, n, k);

    // ========================================================================
    // Print results
    // ========================================================================

    printf("========================================\n");
    printf("NAIVE GPU RESULTS\n");
    printf("========================================\n");
    printf("Time: %.2f ms\n", elapsed_ms);
    printf("FLOPS: %.2f GFLOPS\n", gflops);
    printf("Bandwidth: %.2f GB/s\n", bandwidth_gbps);
    printf("Final error: %.6e\n", error);
    printf("========================================\n\n");

    printf("Performance Baseline Established:\n");
    printf("  This is your starting point!\n");
    printf("  Next optimization goal: 1.5-3x speedup\n");
    printf("  Target bandwidth: 60-80%% of peak\n");
    printf("  Target occupancy: >70%%\n");
    printf("========================================\n");

    // Return metrics
    *time_ms = elapsed_ms;
    *bandwidth_achieved = bandwidth_gbps;
    *flops_achieved = gflops;

    // ========================================================================
    // Cleanup
    // ========================================================================

    CUBLAS_CHECK(cublasDestroy(handle));

    cudaFree(d_X);
    cudaFree(d_W);
    cudaFree(d_H);
    cudaFree(d_WtW);
    cudaFree(d_WtX);
    cudaFree(d_HHt);
    cudaFree(d_XHt);
    cudaFree(d_temp_H);
    cudaFree(d_temp_W);

    free(h_W);
    free(h_H);
}

// ============================================================================
// Main Function
// ============================================================================

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: %s <matrix_size> <rank_k> <max_iter>\n", argv[0]);
        printf("Example: %s 1000 20 100\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);
    int k = atoi(argv[2]);
    int max_iter = atoi(argv[3]);

    int m = size, n = size;

    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║         CSE587 NMF - LEVEL 1: NAIVE GPU BASELINE              ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    // Generate random test matrix
    printf("Generating random %dx%d matrix...\n\n", m, n);
    float* h_X = (float*)malloc(m * n * sizeof(float));
    generate_random_matrix(h_X, m, n, 42);

    // Run naive GPU NMF
    float time_ms, bandwidth_gbps, gflops;
    nmf_naive_gpu(h_X, m, n, k, max_iter, &time_ms, &bandwidth_gbps, &gflops);

    // Save metrics for comparison
    printf("\nSaving metrics to results/naive_metrics.txt...\n");
    FILE* fp = fopen("results/naive_metrics.txt", "w");
    if (fp) {
        fprintf(fp, "Version: Naive GPU (Level 1)\n");
        fprintf(fp, "Size: %dx%d\n", m, n);
        fprintf(fp, "Rank: %d\n", k);
        fprintf(fp, "Iterations: %d\n", max_iter);
        fprintf(fp, "Time: %.2f ms\n", time_ms);
        fprintf(fp, "Bandwidth: %.2f GB/s\n", bandwidth_gbps);
        fprintf(fp, "GFLOPS: %.2f\n", gflops);
        fclose(fp);
        printf("✓ Metrics saved\n");
    }

    printf("\n");
    printf("Next steps:\n");
    printf("1. Profile with: ncu --set full -o results/profiles/v1_naive ./nmf_dense_gpu_v1_naive %d %d %d\n",
           size, k, max_iter);
    printf("2. Analyze occupancy and bandwidth utilization\n");
    printf("3. Implement Level 2 (memory optimizations)\n");
    printf("\n");

    free(h_X);
    return 0;
}
