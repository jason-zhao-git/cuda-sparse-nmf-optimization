#include "../utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * LEVEL 2: cuBLAS GPU IMPLEMENTATION (NO ILP)
 *
 * Purpose: Show speedup from using optimized BLAS library (vs L1 naive GEMM)
 *
 * Optimizations:
 * 1. cuBLAS for GEMM operations (vs custom naive GEMM in L1)
 * 2. Simple fused element-wise kernel (NO ILP)
 *
 * Expected Performance:
 * - 10-50x speedup over L1 naive GEMM
 * - Demonstrates impact of optimized libraries
 * - GEMM is now fast, element-wise is the bottleneck
 *
 */


// Simple fused element-wise kernel (NO ILP)
// Each thread processes one element


__global__ void elementwise_multiply_divide_fused_simple(
    float* input,       // Input array to be updated
    float* numerator,   // Multiply by this
    float* denominator, // Divide by this
    int size,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input[idx] = input[idx] * numerator[idx] / (denominator[idx] + eps);
    }
}


// Memory-Optimized NMF Implementation


void nmf_memory_opt_gpu(float* h_X, int m, int n, int k, int max_iter,
                        float* time_ms, float* bandwidth_achieved, float* flops_achieved,
                        float* final_error, bool log_convergence, int log_interval) {

    printf("========================================\n");
    printf("LEVEL 2: cuBLAS GPU (NO ILP)\n");
    printf("========================================\n");
    printf("Matrix: %dx%d, Rank: %d, Iterations: %d\n", m, n, k, max_iter);
    printf("Optimizations:\n");
    printf("  - cuBLAS for GEMM (vs naive GEMM in L1)\n");
    printf("  - Simple fused element-wise kernel\n");
    printf("  - NO ILP (1 element per thread)\n");
    printf("----------------------------------------\n");


    // Allocate device memory


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


    // Initialize W and H


    float *h_W = (float*)malloc(m * k * sizeof(float));
    float *h_H = (float*)malloc(k * n * sizeof(float));

    generate_random_matrix(h_W, m, k, 42);
    generate_random_matrix(h_H, k, n, 43);

    CUDA_CHECK(cudaMemcpy(d_X, h_X, m * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, h_W, m * k * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_H, h_H, k * n * sizeof(float), cudaMemcpyHostToDevice));


    // Create cuBLAS handle


    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float alpha = 1.0f;
    float beta = 0.0f;


    // Kernel configuration (NO ILP - 1 element per thread)


    int block_size = 128;
    int grid_size_H = (k * n + block_size - 1) / block_size;
    int grid_size_W = (m * k + block_size - 1) / block_size;

    printf("\nKernel Configuration:\n");
    printf("  Block size: %d threads\n", block_size);
    printf("  Grid size H: %d blocks\n", grid_size_H);
    printf("  Grid size W: %d blocks\n", grid_size_W);
    printf("  Elements per thread: 1 (no ILP)\n");
    printf("  Kernel launches per iter: 2 (fused multiply/divide)\n");
    printf("----------------------------------------\n\n");


    // Main iteration loop

    // Convergence logging setup
    FILE* csv_fp = NULL;
    if (log_convergence) {
        csv_fp = fopen("results/convergence_mu_l2.csv", "w");
        if (csv_fp) {
            fprintf(csv_fp, "iteration,error,time_ms\n");
        }
    }

    CudaTimer timer;
    timer.startTimer();

    for (int iter = 0; iter < max_iter; iter++) {
        // ====================================================================
        // Update H: H = H .* (W^T × X) ./ (W^T × W × H + eps)
        // ====================================================================

        // 1. WtW = W^T × W
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_T, CUBLAS_OP_N,
                                 k, k, m,
                                 &alpha, d_W, m, d_W, m,
                                 &beta, d_WtW, k));

        // 2. WtX = W^T × X
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_T, CUBLAS_OP_N,
                                 k, n, m,
                                 &alpha, d_W, m, d_X, m,
                                 &beta, d_WtX, k));

        // 3. temp_H = WtW × H
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 k, n, k,
                                 &alpha, d_WtW, k, d_H, k,
                                 &beta, d_temp_H, k));

        // 4. FUSED: H = H .* WtX ./ (temp_H + eps) in ONE kernel (no ILP)
        elementwise_multiply_divide_fused_simple<<<grid_size_H, block_size>>>(
            d_H, d_WtX, d_temp_H, k * n, 1e-10f
        );

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

        // 4. FUSED: W = W .* XHt ./ (temp_W + eps) in ONE kernel (no ILP)
        elementwise_multiply_divide_fused_simple<<<grid_size_W, block_size>>>(
            d_W, d_XHt, d_temp_W, m * k, 1e-10f
        );

        // Per-iteration convergence logging
        if (log_convergence && (iter % log_interval == 0 || iter == max_iter - 1)) {
            cudaDeviceSynchronize();
            float iter_time_ms = timer.stopTimer();

            // Copy current W, H to host for error computation
            CUDA_CHECK(cudaMemcpy(h_W, d_W, m * k * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_H, d_H, k * n * sizeof(float), cudaMemcpyDeviceToHost));

            float error = compute_relative_error_dense(h_X, h_W, h_H, m, n, k);

            if (csv_fp) {
                fprintf(csv_fp, "%d,%.6e,%.2f\n", iter, error, iter_time_ms);
                fflush(csv_fp);
            }
            printf("Iteration %d: error=%.6e, time=%.2f ms\n", iter, error, iter_time_ms);

            // Restart timer for next interval
            timer.startTimer();
        } else if (iter % 10 == 0) {
            printf("Iteration %d\n", iter);
        }
    }

    float elapsed_ms = timer.stopTimer();

    // Close convergence log
    if (csv_fp) {
        fclose(csv_fp);
        printf("Convergence log saved to results/convergence_mu_l2.csv\n");
    }

    // Copy results back


    CUDA_CHECK(cudaMemcpy(h_W, d_W, m * k * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_H, d_H, k * n * sizeof(float), cudaMemcpyDeviceToHost));

    // Save W, H matrices for visualization
    printf("\nSaving results for visualization...\n");
    save_matrix_binary("results/W_matrix.bin", h_W, m, k);
    save_matrix_binary("results/H_matrix.bin", h_H, k, n);


    // Calculate metrics


    // FLOPS calculation
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

    // Bandwidth calculation (reduced due to kernel fusion)
    long long bytes_per_iter = 0;
    bytes_per_iter += (long long)(m * n + m * k + k * n) * 4 * 2;  // Read X, W, H
    bytes_per_iter += (long long)(m * k + k * n) * 4;              // Write W, H (once)

    long long total_bytes = bytes_per_iter * max_iter;
    float bandwidth_gbps = (total_bytes / elapsed_ms) * 1000.0f / 1e9f;

    // Compute final error
    float error = compute_relative_error_dense(h_X, h_W, h_H, m, n, k);


    // Print results


    printf("========================================\n");
    printf("cuBLAS GPU RESULTS (NO ILP)\n");
    printf("========================================\n");
    printf("Time: %.2f ms\n", elapsed_ms);
    printf("FLOPS: %.2f GFLOPS\n", gflops);
    printf("Bandwidth: %.2f GB/s\n", bandwidth_gbps);
    printf("Final error: %.6e\n", error);
    printf("========================================\n\n");

    printf("cuBLAS Impact:\n");
    printf("  - GEMM is now highly optimized\n");
    printf("  - Element-wise kernels are simple (no ILP)\n");
    printf("  - Most time spent in cuBLAS GEMM calls\n");
    printf("  - Next: L3 adds ILP to element-wise kernels\n");
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
    cudaFree(d_WtW);
    cudaFree(d_WtX);
    cudaFree(d_HHt);
    cudaFree(d_XHt);
    cudaFree(d_temp_H);
    cudaFree(d_temp_W);

    free(h_W);
    free(h_H);
}


// Main Function


int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: %s <matrix_file> <rank_k> <max_iter> [--log-convergence [interval]]\n", argv[0]);
        printf("Example: %s data/dense_1000.bin 20 100\n", argv[0]);
        printf("         %s data/dense_1000.bin 20 100 --log-convergence 5\n", argv[0]);
        printf("\nGenerate matrix with: python3 data/generate_matrix.py --size 1000 --output data/dense_1000.bin\n");
        return 1;
    }

    const char* matrix_file = argv[1];
    int k = atoi(argv[2]);
    int max_iter = atoi(argv[3]);

    // Parse optional --log-convergence flag
    bool log_convergence = false;
    int log_interval = 1;  // Default: log every iteration
    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "--log-convergence") == 0) {
            log_convergence = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                log_interval = atoi(argv[i + 1]);
                i++;
            }
        }
    }

    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║       CSE587 NMF - LEVEL 2: cuBLAS GPU (NO ILP)               ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    // Load matrix from file
    float* h_X = NULL;
    int m, n;
    load_matrix_binary(matrix_file, &h_X, &m, &n);
    printf("\n");

    // Run memory-optimized GPU NMF
    float time_ms, bandwidth_gbps, gflops, final_error;
    nmf_memory_opt_gpu(h_X, m, n, k, max_iter, &time_ms, &bandwidth_gbps, &gflops,
                       &final_error, log_convergence, log_interval);

    // Save metrics for comparison
    printf("\nSaving metrics to results/memory_opt_metrics.txt...\n");
    FILE* fp = fopen("results/memory_opt_metrics.txt", "w");
    if (fp) {
        fprintf(fp, "Version: Memory-Optimized GPU (Level 2)\n");
        fprintf(fp, "Size: %dx%d\n", m, n);
        fprintf(fp, "Rank: %d\n", k);
        fprintf(fp, "Iterations: %d\n", max_iter);
        fprintf(fp, "Time: %.2f ms\n", time_ms);
        fprintf(fp, "Bandwidth: %.2f GB/s\n", bandwidth_gbps);
        fprintf(fp, "GFLOPS: %.2f\n", gflops);
        fprintf(fp, "Final_Error: %.6e\n", final_error);
        fclose(fp);
        printf("✓ Metrics saved\n");
    }

    if (log_convergence) {
        printf("✓ Convergence log saved to results/convergence_mu_l2.csv\n");
    }

    printf("\n");
    printf("Next steps:\n");
    printf("1. Compare with naive: diff results/naive_metrics.txt results/memory_opt_metrics.txt\n");
    printf("2. Move to Level 3: Warp-level optimizations\n");
    printf("\n");

    free(h_X);
    return 0;
}
