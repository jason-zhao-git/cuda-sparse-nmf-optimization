#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

// ============================================================================
// Element-wise CUDA Kernels
// ============================================================================

__global__ void elementwise_multiply(float* A, float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * B[idx];
    }
}

__global__ void elementwise_divide_eps(float* A, float* B, float* C, int size, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] / (B[idx] + eps);
    }
}

// ============================================================================
// Dense NMF using cuBLAS
// ============================================================================

void nmf_dense_gpu(float* h_X, int m, int n, int k, int max_iter) {
    printf("Running Dense GPU NMF: m=%d, n=%d, k=%d, max_iter=%d\n", m, n, k, max_iter);
    printf("------------------------------------------------------------\n");

    // ========================================================================
    // Step 1: Allocate device memory
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
    // Step 2: Initialize W and H randomly on host, then copy to device
    // ========================================================================

    float *h_W = (float*)malloc(m * k * sizeof(float));
    float *h_H = (float*)malloc(k * n * sizeof(float));

    generate_random_matrix(h_W, m, k, 42);
    generate_random_matrix(h_H, k, n, 43);

    CUDA_CHECK(cudaMemcpy(d_X, h_X, m * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, h_W, m * k * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_H, h_H, k * n * sizeof(float), cudaMemcpyHostToDevice));

    // ========================================================================
    // Step 3: Create cuBLAS handle
    // ========================================================================

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // cuBLAS constants
    float alpha = 1.0f;
    float beta = 0.0f;

    // ========================================================================
    // Step 4: Setup kernel launch parameters
    // ========================================================================

    int block_size = 256;
    int grid_size_H = (k * n + block_size - 1) / block_size;
    int grid_size_W = (m * k + block_size - 1) / block_size;

    // ========================================================================
    // Step 5: Main iteration loop
    // ========================================================================

    CudaTimer timer;
    timer.startTimer();

    for (int iter = 0; iter < max_iter; iter++) {
        // ====================================================================
        // Update H: H = H .* (W^T × X) ./ (W^T × W × H + eps)
        // ====================================================================

        // 1. WtW = W^T × W (k×k)
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_T,        // Transpose W
                                 CUBLAS_OP_N,        // Don't transpose W
                                 k, k, m,
                                 &alpha,
                                 d_W, m,
                                 d_W, m,
                                 &beta,
                                 d_WtW, k));

        // 2. WtX = W^T × X (k×n)
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_T,        // Transpose W
                                 CUBLAS_OP_N,        // Don't transpose X
                                 k, n, m,
                                 &alpha,
                                 d_W, m,
                                 d_X, m,
                                 &beta,
                                 d_WtX, k));

        // 3. temp_H = WtW × H (k×n)
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_N,        // Don't transpose WtW
                                 CUBLAS_OP_N,        // Don't transpose H
                                 k, n, k,
                                 &alpha,
                                 d_WtW, k,
                                 d_H, k,
                                 &beta,
                                 d_temp_H, k));

        // 4. H = H .* WtX
        elementwise_multiply<<<grid_size_H, block_size>>>(d_H, d_WtX, d_H, k * n);

        // 5. H = H ./ (temp_H + eps)
        elementwise_divide_eps<<<grid_size_H, block_size>>>(d_H, d_temp_H, d_H, k * n, 1e-10f);

        // ====================================================================
        // Update W: W = W .* (X × H^T) ./ (W × H × H^T + eps)
        // ====================================================================

        // 1. HHt = H × H^T (k×k)
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_N,        // Don't transpose H
                                 CUBLAS_OP_T,        // Transpose H
                                 k, k, n,
                                 &alpha,
                                 d_H, k,
                                 d_H, k,
                                 &beta,
                                 d_HHt, k));

        // 2. XHt = X × H^T (m×k)
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_N,        // Don't transpose X
                                 CUBLAS_OP_T,        // Transpose H
                                 m, k, n,
                                 &alpha,
                                 d_X, m,
                                 d_H, k,
                                 &beta,
                                 d_XHt, m));

        // 3. temp_W = W × HHt (m×k)
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_N,        // Don't transpose W
                                 CUBLAS_OP_N,        // Don't transpose HHt
                                 m, k, k,
                                 &alpha,
                                 d_W, m,
                                 d_HHt, k,
                                 &beta,
                                 d_temp_W, m));

        // 4. W = W .* XHt
        elementwise_multiply<<<grid_size_W, block_size>>>(d_W, d_XHt, d_W, m * k);

        // 5. W = W ./ (temp_W + eps)
        elementwise_divide_eps<<<grid_size_W, block_size>>>(d_W, d_temp_W, d_W, m * k, 1e-10f);

        // Print progress
        if (iter % 10 == 0) {
            printf("Iteration %d\n", iter);
        }
    }

    float elapsed_ms = timer.stopTimer();

    // ========================================================================
    // Step 6: Copy results back to host
    // ========================================================================

    CUDA_CHECK(cudaMemcpy(h_W, d_W, m * k * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_H, d_H, k * n * sizeof(float), cudaMemcpyDeviceToHost));

    // ========================================================================
    // Step 7: Compute final error
    // ========================================================================

    float error = compute_relative_error_dense(h_X, h_W, h_H, m, n, k);

    printf("------------------------------------------------------------\n");
    printf("Final relative error: %.6e\n", error);
    printf("Time: %.2f ms\n", elapsed_ms);

    // ========================================================================
    // Step 8: Cleanup
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

    // Generate random test matrix
    printf("Generating random %dx%d matrix...\n", m, n);
    float* h_X = (float*)malloc(m * n * sizeof(float));
    generate_random_matrix(h_X, m, n, 42);

    // Run NMF
    nmf_dense_gpu(h_X, m, n, k, max_iter);

    free(h_X);

    return 0;
}
