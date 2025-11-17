#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

/*
 * LEVEL 3 VARIANT: SPARSE NMF WITH TRANSPOSE TRICK
 *
 * Uses transpose trick to avoid storing X twice:
 * W^T × X = (X^T × W)^T
 *
 * Approach:
 * - Store X in CSR format (for X × H^T)
 * - Store X^T in CSC format (for X^T × W)
 * - Both operations use cuSPARSE
 * - Memory savings: Only store sparse format (no dense copy)
 */


// Transpose kernel for column-major matrices
// Input: rows×cols in column-major (stored as input[col*rows + row])
// Output: cols×rows in column-major (stored as output[row*cols + col])
__global__ void transpose_kernel(float* input, float* output, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // row index
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // col index

    if (i < rows && j < cols) {
        // Column-major: input[i,j] = input[j*rows + i]
        // After transpose: output[j,i] = input[i,j]
        // Column-major: output[j,i] = output[i*cols + j]
        output[i * cols + j] = input[j * rows + i];
    }
}

// Fused element-wise kernel (same as before)
__global__ void elementwise_multiply_fused_ilp(
    float* input, float* numerator, float* denominator,
    int size, float eps
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx + 3 < size) {
        float in0 = input[idx];
        float in1 = input[idx + 1];
        float in2 = input[idx + 2];
        float in3 = input[idx + 3];

        float num0 = numerator[idx];
        float num1 = numerator[idx + 1];
        float num2 = numerator[idx + 2];
        float num3 = numerator[idx + 3];

        float den0 = denominator[idx];
        float den1 = denominator[idx + 1];
        float den2 = denominator[idx + 2];
        float den3 = denominator[idx + 3];

        in0 = in0 * num0 / (den0 + eps);
        in1 = in1 * num1 / (den1 + eps);
        in2 = in2 * num2 / (den2 + eps);
        in3 = in3 * num3 / (den3 + eps);

        input[idx] = in0;
        input[idx + 1] = in1;
        input[idx + 2] = in2;
        input[idx + 3] = in3;
    } else {
        for (int i = idx; i < size && i < idx + 4; i++) {
            input[i] = input[i] * numerator[i] / (denominator[i] + eps);
        }
    }
}


// Convert CSR to CSC (transpose sparse matrix)
void csr_to_csc(int m, int n, int nnz,
                int* csr_rowPtr, int* csr_colInd, float* csr_values,
                int** csc_colPtr, int** csc_rowInd, float** csc_values) {

    *csc_values = (float*)malloc(nnz * sizeof(float));
    *csc_rowInd = (int*)malloc(nnz * sizeof(int));
    *csc_colPtr = (int*)malloc((n + 1) * sizeof(int));

    // Count nnz per column
    int* col_count = (int*)calloc(n, sizeof(int));
    for (int i = 0; i < nnz; i++) {
        col_count[csr_colInd[i]]++;
    }

    // Compute column pointers
    (*csc_colPtr)[0] = 0;
    for (int j = 0; j < n; j++) {
        (*csc_colPtr)[j + 1] = (*csc_colPtr)[j] + col_count[j];
    }

    // Fill CSC arrays
    int* current_pos = (int*)calloc(n, sizeof(int));
    for (int i = 0; i < m; i++) {
        for (int idx = csr_rowPtr[i]; idx < csr_rowPtr[i + 1]; idx++) {
            int col = csr_colInd[idx];
            int dest = (*csc_colPtr)[col] + current_pos[col];
            (*csc_values)[dest] = csr_values[idx];
            (*csc_rowInd)[dest] = i;
            current_pos[col]++;
        }
    }

    free(col_count);
    free(current_pos);
}


// Dense to CSR conversion
void dense_to_csr(const float* dense, int m, int n, float threshold,
                  int** rowPtr, int** colInd, float** values, int* nnz) {
    *nnz = 0;
    for (int i = 0; i < m * n; i++) {
        if (fabs(dense[i]) > threshold) {
            (*nnz)++;
        }
    }

    *values = (float*)malloc(*nnz * sizeof(float));
    *colInd = (int*)malloc(*nnz * sizeof(int));
    *rowPtr = (int*)malloc((m + 1) * sizeof(int));

    int idx = 0;
    (*rowPtr)[0] = 0;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float val = dense[i * n + j];
            if (fabs(val) > threshold) {
                (*values)[idx] = val;
                (*colInd)[idx] = j;
                idx++;
            }
        }
        (*rowPtr)[i + 1] = idx;
    }
}


void nmf_sparse_transpose_trick(float* h_X, int m, int n, int k, int max_iter,
                                 float* time_ms, float* bandwidth_achieved, float* flops_achieved) {

    printf("========================================\n");
    printf("SPARSE NMF - TRANSPOSE TRICK\n");
    printf("========================================\n");
    printf("Matrix: %dx%d, Rank: %d, Iterations: %d\n", m, n, k, max_iter);
    printf("Approach: Pure sparse (transpose trick for both ops)\n");
    printf("----------------------------------------\n");

    // Convert to CSR (for X × H^T) - no artificial sparsification
    int *h_csr_rowPtr, *h_csr_colInd;
    float *h_csr_values;
    int nnz;
    dense_to_csr(h_X, m, n, 0.0f, &h_csr_rowPtr, &h_csr_colInd, &h_csr_values, &nnz);

    // Convert to CSC (for X^T × W)
    int *h_csc_colPtr, *h_csc_rowInd;
    float *h_csc_values;
    csr_to_csc(m, n, nnz, h_csr_rowPtr, h_csr_colInd, h_csr_values,
               &h_csc_colPtr, &h_csc_rowInd, &h_csc_values);

    float actual_sparsity = 1.0f - ((float)nnz / (m * n));
    printf("Actual sparsity: %.1f%% (%d non-zeros)\n", actual_sparsity * 100, nnz);
    printf("Memory: CSR + CSC = %.2f MB (vs %.2f MB dense)\n",
           2.0f * (nnz * sizeof(float) + nnz * sizeof(int)) / 1e6f,
           (m * n * sizeof(float)) / 1e6f);
    printf("----------------------------------------\n");

    // Allocate device memory for CSR (X)
    int *d_csr_rowPtr, *d_csr_colInd;
    float *d_csr_values;
    CUDA_CHECK(cudaMalloc(&d_csr_rowPtr, (m + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csr_colInd, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csr_values, nnz * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_csr_rowPtr, h_csr_rowPtr, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csr_colInd, h_csr_colInd, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csr_values, h_csr_values, nnz * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate device memory for CSC (X^T)
    int *d_csc_colPtr, *d_csc_rowInd;
    float *d_csc_values;
    CUDA_CHECK(cudaMalloc(&d_csc_colPtr, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csc_rowInd, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csc_values, nnz * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_csc_colPtr, h_csc_colPtr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csc_rowInd, h_csc_rowInd, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csc_values, h_csc_values, nnz * sizeof(float), cudaMemcpyHostToDevice));

    // Dense W and H
    float *d_W, *d_H;
    float *d_WtW, *d_WtX, *d_WtX_T, *d_HHt, *d_XHt;
    float *d_temp_H, *d_temp_W;

    CUDA_CHECK(cudaMalloc(&d_W, m * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_H, k * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_WtW, k * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_WtX, k * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_WtX_T, n * k * sizeof(float)));  // Transposed result
    CUDA_CHECK(cudaMalloc(&d_HHt, k * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_XHt, m * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_temp_H, k * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_temp_W, m * k * sizeof(float)));

    // Initialize W and H
    float *h_W = (float*)malloc(m * k * sizeof(float));
    float *h_H = (float*)malloc(k * n * sizeof(float));
    generate_random_matrix(h_W, m, k, 42);
    generate_random_matrix(h_H, k, n, 43);
    CUDA_CHECK(cudaMemcpy(d_W, h_W, m * k * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_H, h_H, k * n * sizeof(float), cudaMemcpyHostToDevice));

    // Create handles
    cublasHandle_t cublas_handle;
    cusparseHandle_t cusparse_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUSPARSE_CHECK(cusparseCreate(&cusparse_handle));

    float alpha = 1.0f, beta = 0.0f;

    // Create sparse matrix descriptors
    cusparseSpMatDescr_t matX_csr;  // For X × H^T
    CUSPARSE_CHECK(cusparseCreateCsr(&matX_csr, m, n, nnz,
                                     d_csr_rowPtr, d_csr_colInd, d_csr_values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    cusparseSpMatDescr_t matX_csc;  // For X^T × W
    CUSPARSE_CHECK(cusparseCreateCsc(&matX_csc, n, m, nnz,
                                     d_csc_colPtr, d_csc_rowInd, d_csc_values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // Dense matrix descriptors (use COL order to match cuBLAS column-major)
    // For column-major: leading dimension = number of rows
    cusparseDnMatDescr_t matW, matH, matXHt, matWtX_T;
    CUSPARSE_CHECK(cusparseCreateDnMat(&matW, m, k, m, d_W, CUDA_R_32F, CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(&matH, k, n, k, d_H, CUDA_R_32F, CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(&matXHt, m, k, m, d_XHt, CUDA_R_32F, CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(&matWtX_T, n, k, n, d_WtX_T, CUDA_R_32F, CUSPARSE_ORDER_COL));

    // Allocate workspace
    size_t bufferSize1 = 0, bufferSize2 = 0;
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
        &alpha, matX_csr, matH, &beta, matXHt,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize1));

    CUSPARSE_CHECK(cusparseSpMM_bufferSize(cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matX_csc, matW, &beta, matWtX_T,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize2));

    size_t bufferSize = (bufferSize1 > bufferSize2) ? bufferSize1 : bufferSize2;
    void* d_buffer = NULL;
    if (bufferSize > 0) {
        CUDA_CHECK(cudaMalloc(&d_buffer, bufferSize));
    }

    // Kernel configuration
    int block_size = 128;
    int grid_size_H = ((k * n) + (block_size * 4) - 1) / (block_size * 4);
    int grid_size_W = ((m * k) + (block_size * 4) - 1) / (block_size * 4);

    dim3 transpose_block(16, 16);
    // Grid dimensions should match INPUT matrix dimensions (n×k)
    // blockIdx.x maps to rows, blockIdx.y maps to cols
    dim3 transpose_grid((n + 15) / 16, (k + 15) / 16);

    printf("\nRunning sparse NMF with transpose trick...\n");

    CudaTimer timer;
    timer.startTimer();

    for (int iter = 0; iter < max_iter; iter++) {
        // ====================================================================
        // Update H: H = H .* (W^T × X) ./ (W^T × W × H + eps)
        // ====================================================================

        // 1. WtW = W^T × W
        CUBLAS_CHECK(cublasSgemm(cublas_handle,
                                 CUBLAS_OP_T, CUBLAS_OP_N,
                                 k, k, m,
                                 &alpha, d_W, m, d_W, m,
                                 &beta, d_WtW, k));

        // 2. WtX = (X^T × W)^T using transpose trick
        // First: X^T × W (n×m sparse × m×k dense = n×k)
        CUSPARSE_CHECK(cusparseSpMM(cusparse_handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matX_csc, matW, &beta, matWtX_T,
                                    CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                                    d_buffer));

        // Transpose result: (n×k)^T = k×n
        transpose_kernel<<<transpose_grid, transpose_block>>>(d_WtX_T, d_WtX, n, k);

        // 3. temp_H = WtW × H
        CUBLAS_CHECK(cublasSgemm(cublas_handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 k, n, k,
                                 &alpha, d_WtW, k, d_H, k,
                                 &beta, d_temp_H, k));

        // 4. H = H .* WtX ./ (temp_H + eps)
        elementwise_multiply_fused_ilp<<<grid_size_H, block_size>>>(
            d_H, d_WtX, d_temp_H, k * n, 1e-10f
        );

        // ====================================================================
        // Update W: W = W .* (X × H^T) ./ (W × H × H^T + eps)
        // ====================================================================

        // 1. HHt = H × H^T
        CUBLAS_CHECK(cublasSgemm(cublas_handle,
                                 CUBLAS_OP_N, CUBLAS_OP_T,
                                 k, k, n,
                                 &alpha, d_H, k, d_H, k,
                                 &beta, d_HHt, k));

        // 2. XHt = X × H^T (sparse × dense)
        CUSPARSE_CHECK(cusparseSpMM(cusparse_handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_TRANSPOSE,
                                    &alpha, matX_csr, matH, &beta, matXHt,
                                    CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                                    d_buffer));

        // 3. temp_W = W × HHt
        CUBLAS_CHECK(cublasSgemm(cublas_handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 m, k, k,
                                 &alpha, d_W, m, d_HHt, k,
                                 &beta, d_temp_W, m));

        // 4. W = W .* XHt ./ (temp_W + eps)
        elementwise_multiply_fused_ilp<<<grid_size_W, block_size>>>(
            d_W, d_XHt, d_temp_W, m * k, 1e-10f
        );

        if (iter % 10 == 0) {
            printf("Iteration %d\n", iter);
        }
    }

    float elapsed_ms = timer.stopTimer();

    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_W, d_W, m * k * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_H, d_H, k * n * sizeof(float), cudaMemcpyDeviceToHost));

    // Calculate metrics
    long long flops_per_iter = 0;
    flops_per_iter += 2LL * k * k * m;
    flops_per_iter += 2LL * nnz * k;  // Sparse X^T × W
    flops_per_iter += 2LL * k * n * k;
    flops_per_iter += 2LL * k * k * n;
    flops_per_iter += 2LL * nnz * k;  // Sparse X × H^T
    flops_per_iter += 2LL * m * k * k;
    flops_per_iter += 4LL * (m * k + k * n);

    long long total_flops = flops_per_iter * max_iter;
    float gflops = (total_flops / elapsed_ms) * 1000.0f / 1e9f;

    long long bytes_per_iter = 0;
    bytes_per_iter += (long long)(nnz * sizeof(float)) * 2;
    bytes_per_iter += (long long)(m * k + k * n) * 4 * 2;

    long long total_bytes = bytes_per_iter * max_iter;
    float bandwidth_gbps = (total_bytes / elapsed_ms) * 1000.0f / 1e9f;

    // Compute error
    float* h_X_reconstructed = (float*)calloc(m * n, sizeof(float));
    for (int i = 0; i < m; i++) {
        for (int j = h_csr_rowPtr[i]; j < h_csr_rowPtr[i+1]; j++) {
            h_X_reconstructed[i * n + h_csr_colInd[j]] = h_csr_values[j];
        }
    }
    float error = compute_relative_error_dense(h_X_reconstructed, h_W, h_H, m, n, k);

    printf("========================================\n");
    printf("TRANSPOSE TRICK RESULTS\n");
    printf("========================================\n");
    printf("Time: %.2f ms\n", elapsed_ms);
    printf("GFLOPS: %.2f\n", gflops);
    printf("Bandwidth: %.2f GB/s\n", bandwidth_gbps);
    printf("Final error: %.6e\n", error);
    printf("========================================\n");

    *time_ms = elapsed_ms;
    *bandwidth_achieved = bandwidth_gbps;
    *flops_achieved = gflops;

    // Cleanup
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUSPARSE_CHECK(cusparseDestroy(cusparse_handle));
    CUSPARSE_CHECK(cusparseDestroySpMat(matX_csr));
    CUSPARSE_CHECK(cusparseDestroySpMat(matX_csc));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matW));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matH));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matXHt));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matWtX_T));

    cudaFree(d_csr_rowPtr); cudaFree(d_csr_colInd); cudaFree(d_csr_values);
    cudaFree(d_csc_colPtr); cudaFree(d_csc_rowInd); cudaFree(d_csc_values);
    cudaFree(d_W); cudaFree(d_H); cudaFree(d_WtW); cudaFree(d_WtX); cudaFree(d_WtX_T);
    cudaFree(d_HHt); cudaFree(d_XHt); cudaFree(d_temp_H); cudaFree(d_temp_W);
    if (d_buffer) cudaFree(d_buffer);

    free(h_csr_rowPtr); free(h_csr_colInd); free(h_csr_values);
    free(h_csc_colPtr); free(h_csc_rowInd); free(h_csc_values);
    free(h_W); free(h_H); free(h_X_reconstructed);
}


int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: %s <matrix_file> <rank_k> <max_iter>\n", argv[0]);
        return 1;
    }

    const char* matrix_file = argv[1];
    int k = atoi(argv[2]);
    int max_iter = atoi(argv[3]);

    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║     SPARSE NMF - TRANSPOSE TRICK (Pure Sparse Approach)       ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    float* h_X = NULL;
    int m, n;
    load_matrix_binary(matrix_file, &h_X, &m, &n);

    float time_ms, bandwidth_gbps, gflops;
    nmf_sparse_transpose_trick(h_X, m, n, k, max_iter, &time_ms, &bandwidth_gbps, &gflops);

    free(h_X);
    return 0;
}
