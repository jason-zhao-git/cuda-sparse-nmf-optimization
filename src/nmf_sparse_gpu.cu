#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

// ============================================================================
// Element-wise CUDA Kernels (same as dense version)
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
// CSR Format Conversion: Dense to CSR
// ============================================================================

void dense_to_csr(const float* dense, int m, int n, float threshold,
                  int** rowPtr, int** colInd, float** values, int* nnz) {
    // First pass: count non-zeros
    *nnz = 0;
    for (int i = 0; i < m * n; i++) {
        if (dense[i] > threshold) {
            (*nnz)++;
        }
    }

    // Allocate CSR arrays
    *values = (float*)malloc(*nnz * sizeof(float));
    *colInd = (int*)malloc(*nnz * sizeof(int));
    *rowPtr = (int*)malloc((m + 1) * sizeof(int));

    // Second pass: fill CSR arrays
    int idx = 0;
    (*rowPtr)[0] = 0;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float val = dense[i * n + j];
            if (val > threshold) {
                (*values)[idx] = val;
                (*colInd)[idx] = j;
                idx++;
            }
        }
        (*rowPtr)[i + 1] = idx;
    }

    printf("Converted to CSR: %d non-zeros (%.1f%% sparse)\n",
           *nnz, 100.0 * (1.0 - (float)*nnz / (m * n)));
}

// ============================================================================
// Sparse NMF using cuSPARSE
// ============================================================================

void nmf_sparse_gpu(const float* h_csrVal, const int* h_csrColInd, const int* h_csrRowPtr,
                    int m, int n, int k, int nnz, int max_iter) {

    printf("Running Sparse GPU NMF: m=%d, n=%d, k=%d, nnz=%d, max_iter=%d\n",
           m, n, k, nnz, max_iter);
    printf("------------------------------------------------------------\n");

    // ========================================================================
    // Step 1: Allocate device memory
    // ========================================================================

    // Sparse matrix X in CSR format
    float *d_csrVal;
    int *d_csrColInd, *d_csrRowPtr;

    CUDA_CHECK(cudaMalloc(&d_csrVal, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_csrColInd, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csrRowPtr, (m + 1) * sizeof(int)));

    // Dense matrices W and H
    float *d_W, *d_H;
    CUDA_CHECK(cudaMalloc(&d_W, m * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_H, k * n * sizeof(float)));

    // Intermediate results (all dense)
    float *d_WtW, *d_WtX, *d_HHt, *d_XHt;
    float *d_temp_H, *d_temp_W;

    CUDA_CHECK(cudaMalloc(&d_WtW, k * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_WtX, k * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_HHt, k * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_XHt, m * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_temp_H, k * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_temp_W, m * k * sizeof(float)));

    // ========================================================================
    // Step 2: Copy data to device
    // ========================================================================

    CUDA_CHECK(cudaMemcpy(d_csrVal, h_csrVal, nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csrColInd, h_csrColInd, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csrRowPtr, h_csrRowPtr, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));

    // Initialize W and H randomly
    float *h_W = (float*)malloc(m * k * sizeof(float));
    float *h_H = (float*)malloc(k * n * sizeof(float));

    generate_random_matrix(h_W, m, k, 42);
    generate_random_matrix(h_H, k, n, 43);

    CUDA_CHECK(cudaMemcpy(d_W, h_W, m * k * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_H, h_H, k * n * sizeof(float), cudaMemcpyHostToDevice));

    // ========================================================================
    // Step 3: Create cuBLAS and cuSPARSE handles
    // ========================================================================

    cublasHandle_t cublas_handle;
    cusparseHandle_t cusparse_handle;

    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUSPARSE_CHECK(cusparseCreate(&cusparse_handle));

    float alpha = 1.0f;
    float beta = 0.0f;

    // ========================================================================
    // Step 4: Create sparse matrix descriptor for X (CSR format)
    // ========================================================================

    cusparseSpMatDescr_t matX;
    CUSPARSE_CHECK(cusparseCreateCsr(&matX, m, n, nnz,
                                     d_csrRowPtr, d_csrColInd, d_csrVal,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // ========================================================================
    // Step 5: Create dense matrix descriptors for W and H
    // ========================================================================

    // NOTE: cuSPARSE uses column-major order, but we need to be careful
    // We'll create descriptors as needed for each operation

    cusparseDnMatDescr_t matW, matH, matWtX, matXHt;

    // W is m×k (column-major leading dimension = m)
    CUSPARSE_CHECK(cusparseCreateDnMat(&matW, m, k, m, d_W,
                                       CUDA_R_32F, CUSPARSE_ORDER_COL));

    // H is k×n (column-major leading dimension = k)
    CUSPARSE_CHECK(cusparseCreateDnMat(&matH, k, n, k, d_H,
                                       CUDA_R_32F, CUSPARSE_ORDER_COL));

    // WtX is k×n
    CUSPARSE_CHECK(cusparseCreateDnMat(&matWtX, k, n, k, d_WtX,
                                       CUDA_R_32F, CUSPARSE_ORDER_COL));

    // XHt is m×k
    CUSPARSE_CHECK(cusparseCreateDnMat(&matXHt, m, k, m, d_XHt,
                                       CUDA_R_32F, CUSPARSE_ORDER_COL));

    // ========================================================================
    // Step 6: Allocate buffer for cuSPARSE operations
    // ========================================================================

    // Query buffer size for W^T × X (sparse × dense)
    size_t bufferSize1 = 0;
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        cusparse_handle,
        CUSPARSE_OPERATION_TRANSPOSE,       // W^T
        CUSPARSE_OPERATION_NON_TRANSPOSE,   // X
        &alpha, matW, matX, &beta, matWtX,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
        &bufferSize1));

    // Query buffer size for X × H^T (sparse × dense)
    size_t bufferSize2 = 0;
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,   // X
        CUSPARSE_OPERATION_TRANSPOSE,       // H^T
        &alpha, matX, matH, &beta, matXHt,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
        &bufferSize2));

    size_t bufferSize = (bufferSize1 > bufferSize2) ? bufferSize1 : bufferSize2;

    void* d_buffer = NULL;
    if (bufferSize > 0) {
        CUDA_CHECK(cudaMalloc(&d_buffer, bufferSize));
    }

    // ========================================================================
    // Step 7: Setup kernel launch parameters
    // ========================================================================

    int block_size = 256;
    int grid_size_H = (k * n + block_size - 1) / block_size;
    int grid_size_W = (m * k + block_size - 1) / block_size;

    // ========================================================================
    // Step 8: Main iteration loop
    // ========================================================================

    CudaTimer timer;
    timer.startTimer();

    for (int iter = 0; iter < max_iter; iter++) {
        // ====================================================================
        // Update H: H = H .* (W^T × X) ./ (W^T × W × H + eps)
        // ====================================================================

        // 1. WtW = W^T × W (dense × dense, use cuBLAS)
        CUBLAS_CHECK(cublasSgemm(cublas_handle,
                                 CUBLAS_OP_T, CUBLAS_OP_N,
                                 k, k, m,
                                 &alpha, d_W, m, d_W, m,
                                 &beta, d_WtW, k));

        // 2. WtX = W^T × X (dense × sparse, use cuSPARSE)
        CUSPARSE_CHECK(cusparseSpMM(cusparse_handle,
                                    CUSPARSE_OPERATION_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matW, matX, &beta, matWtX,
                                    CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                                    d_buffer));

        // 3. temp_H = WtW × H (dense × dense, use cuBLAS)
        CUBLAS_CHECK(cublasSgemm(cublas_handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 k, n, k,
                                 &alpha, d_WtW, k, d_H, k,
                                 &beta, d_temp_H, k));

        // 4. H = H .* WtX
        elementwise_multiply<<<grid_size_H, block_size>>>(d_H, d_WtX, d_H, k * n);

        // 5. H = H ./ (temp_H + eps)
        elementwise_divide_eps<<<grid_size_H, block_size>>>(d_H, d_temp_H, d_H, k * n, 1e-10f);

        // ====================================================================
        // Update W: W = W .* (X × H^T) ./ (W × H × H^T + eps)
        // ====================================================================

        // 1. HHt = H × H^T (dense × dense, use cuBLAS)
        CUBLAS_CHECK(cublasSgemm(cublas_handle,
                                 CUBLAS_OP_N, CUBLAS_OP_T,
                                 k, k, n,
                                 &alpha, d_H, k, d_H, k,
                                 &beta, d_HHt, k));

        // 2. XHt = X × H^T (sparse × dense, use cuSPARSE)
        CUSPARSE_CHECK(cusparseSpMM(cusparse_handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_TRANSPOSE,
                                    &alpha, matX, matH, &beta, matXHt,
                                    CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                                    d_buffer));

        // 3. temp_W = W × HHt (dense × dense, use cuBLAS)
        CUBLAS_CHECK(cublasSgemm(cublas_handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 m, k, k,
                                 &alpha, d_W, m, d_HHt, k,
                                 &beta, d_temp_W, m));

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
    // Step 9: Copy results back to host
    // ========================================================================

    CUDA_CHECK(cudaMemcpy(h_W, d_W, m * k * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_H, d_H, k * n * sizeof(float), cudaMemcpyDeviceToHost));

    printf("------------------------------------------------------------\n");
    printf("Time: %.2f ms\n", elapsed_ms);

    // ========================================================================
    // Step 10: Cleanup
    // ========================================================================

    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUSPARSE_CHECK(cusparseDestroy(cusparse_handle));

    CUSPARSE_CHECK(cusparseDestroySpMat(matX));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matW));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matH));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matWtX));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matXHt));

    cudaFree(d_csrVal);
    cudaFree(d_csrColInd);
    cudaFree(d_csrRowPtr);
    cudaFree(d_W);
    cudaFree(d_H);
    cudaFree(d_WtW);
    cudaFree(d_WtX);
    cudaFree(d_HHt);
    cudaFree(d_XHt);
    cudaFree(d_temp_H);
    cudaFree(d_temp_W);
    if (d_buffer) cudaFree(d_buffer);

    free(h_W);
    free(h_H);
}

// ============================================================================
// Main Function
// ============================================================================

int main(int argc, char** argv) {
    if (argc < 5) {
        printf("Usage: %s <matrix_size> <rank_k> <sparsity> <max_iter>\n", argv[0]);
        printf("Example: %s 1000 20 0.9 100\n", argv[0]);
        printf("  sparsity: fraction of zeros (0.0 to 1.0)\n");
        return 1;
    }

    int size = atoi(argv[1]);
    int k = atoi(argv[2]);
    float sparsity = atof(argv[3]);
    int max_iter = atoi(argv[4]);

    int m = size, n = size;

    // Generate random test matrix
    printf("Generating random %dx%d matrix with %.1f%% sparsity...\n",
           m, n, sparsity * 100);

    float* h_dense = (float*)malloc(m * n * sizeof(float));
    generate_random_matrix(h_dense, m, n, 42);

    // Make it sparse by zeroing out elements
    int target_zeros = (int)(sparsity * m * n);
    int zeros_added = 0;
    srand(43);
    while (zeros_added < target_zeros) {
        int idx = rand() % (m * n);
        if (h_dense[idx] != 0.0f) {
            h_dense[idx] = 0.0f;
            zeros_added++;
        }
    }

    // Convert to CSR
    int *h_csrRowPtr, *h_csrColInd;
    float *h_csrVal;
    int nnz;

    dense_to_csr(h_dense, m, n, 0.0f, &h_csrRowPtr, &h_csrColInd, &h_csrVal, &nnz);

    // Run NMF
    nmf_sparse_gpu(h_csrVal, h_csrColInd, h_csrRowPtr, m, n, k, nnz, max_iter);

    free(h_dense);
    free(h_csrVal);
    free(h_csrColInd);
    free(h_csrRowPtr);

    return 0;
}
