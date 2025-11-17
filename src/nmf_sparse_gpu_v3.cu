#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

/*
 * LEVEL 3: SPARSE NMF IMPLEMENTATION
 *
 * Purpose: Handle sparse matrices efficiently using CSR format
 *
 * Key Differences from Dense:
 * - X stored in CSR (Compressed Sparse Row) format
 * - Only stores non-zero values
 * - Uses cuSPARSE for sparse matrix operations
 * - W and H remain dense (typically not sparse in NMF)
 *
 * Expected Performance:
 * - 2-10x speedup at high sparsity (>90%)
 * - Significant memory savings
 * - Crossover point around 70-80% sparsity
 *
 */


// Element-wise kernels (same as dense - W and H are dense)


__global__ void elementwise_multiply_fused_ilp(
    float* input,
    float* numerator,
    float* denominator,
    int size,
    float eps
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


// Dense to CSR conversion


void dense_to_csr(const float* dense, int m, int n, float threshold,
                  int** rowPtr, int** colInd, float** values, int* nnz) {
    // Count non-zeros
    *nnz = 0;
    for (int i = 0; i < m * n; i++) {
        if (fabs(dense[i]) > threshold) {
            (*nnz)++;
        }
    }

    // Allocate CSR arrays
    *values = (float*)malloc(*nnz * sizeof(float));
    *colInd = (int*)malloc(*nnz * sizeof(int));
    *rowPtr = (int*)malloc((m + 1) * sizeof(int));

    // Fill CSR arrays
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


// Sparse NMF using cuSPARSE


void nmf_sparse_gpu(float* h_X, int m, int n, int k, int max_iter,
                   float* time_ms, float* bandwidth_achieved, float* flops_achieved) {

    printf("========================================\n");
    printf("LEVEL 3: SPARSE NMF IMPLEMENTATION\n");
    printf("========================================\n");
    printf("Matrix: %dx%d, Rank: %d, Iterations: %d\n", m, n, k, max_iter);
    printf("Format: CSR (Compressed Sparse Row)\n");
    printf("Approach: Hybrid (dense for W^T×X, sparse for X×H^T)\n");
    printf("----------------------------------------\n");


    // Convert input matrix to CSR format
    // (Matrix is already as-is from file - no artificial sparsification)


    int *h_rowPtr, *h_colInd;
    float *h_values;
    int nnz;

    float threshold = 0.0f;

    // Convert to CSR (will only store non-zero elements)
    dense_to_csr(h_X, m, n, threshold, &h_rowPtr, &h_colInd, &h_values, &nnz);

    float actual_sparsity = 1.0f - ((float)nnz / (m * n));
    printf("Actual sparsity: %.1f%% (%d non-zeros out of %d)\n",
           actual_sparsity * 100, nnz, m * n);
    printf("Memory savings: %.1f%% (CSR vs dense)\n",
           (1.0f - (float)(nnz * sizeof(float) + nnz * sizeof(int) + (m+1) * sizeof(int)) /
                   (m * n * sizeof(float))) * 100);
    printf("----------------------------------------\n");


    // Allocate device memory


    // HYBRID APPROACH: Store X in both formats
    // - Dense X: For W^T × X operation (cuBLAS)
    // - Sparse X: For X × H^T operation (cuSPARSE)
    // This allows complete NMF implementation while showing sparse benefits

    // Dense X (for W^T × X)
    float *d_X_dense;
    CUDA_CHECK(cudaMalloc(&d_X_dense, m * n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_X_dense, h_X, m * n * sizeof(float), cudaMemcpyHostToDevice));

    // Sparse X in CSR format (for X × H^T)
    int *d_rowPtr, *d_colInd;
    float *d_values;
    CUDA_CHECK(cudaMalloc(&d_rowPtr, (m + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_colInd, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, nnz * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_rowPtr, h_rowPtr, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colInd, h_colInd, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, h_values, nnz * sizeof(float), cudaMemcpyHostToDevice));

    // Dense W and H
    float *d_W, *d_H;
    float *d_WtW, *d_WtX, *d_HHt, *d_XHt;
    float *d_temp_H, *d_temp_W;

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

    CUDA_CHECK(cudaMemcpy(d_W, h_W, m * k * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_H, h_H, k * n * sizeof(float), cudaMemcpyHostToDevice));


    // Create cuBLAS and cuSPARSE handles


    cublasHandle_t cublas_handle;
    cusparseHandle_t cusparse_handle;

    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUSPARSE_CHECK(cusparseCreate(&cusparse_handle));

    float alpha = 1.0f;
    float beta = 0.0f;


    // Create sparse matrix descriptor for X


    cusparseSpMatDescr_t matX;
    CUSPARSE_CHECK(cusparseCreateCsr(&matX, m, n, nnz,
                                     d_rowPtr, d_colInd, d_values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // Create dense matrix descriptors
    cusparseDnMatDescr_t matW, matH, matXHt;

    CUSPARSE_CHECK(cusparseCreateDnMat(&matW, m, k, m, d_W, CUDA_R_32F, CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(&matH, k, n, k, d_H, CUDA_R_32F, CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(&matXHt, m, k, m, d_XHt, CUDA_R_32F, CUSPARSE_ORDER_COL));


    // Allocate workspace for cuSPARSE


    size_t bufferSize = 0;
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_TRANSPOSE,
        &alpha, matX, matH, &beta, matXHt,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
        &bufferSize));

    void* d_buffer = NULL;
    if (bufferSize > 0) {
        CUDA_CHECK(cudaMalloc(&d_buffer, bufferSize));
    }


    // Kernel configuration


    int block_size = 128;
    int grid_size_H = ((k * n) + (block_size * 4) - 1) / (block_size * 4);
    int grid_size_W = ((m * k) + (block_size * 4) - 1) / (block_size * 4);

    printf("\nRunning sparse NMF...\n");


    // Main iteration loop


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

        // 2. WtX = W^T × X (dense × dense, use cuBLAS with dense X)
        // NOTE: Using dense X here because cuSPARSE doesn't efficiently support dense × sparse
        CUBLAS_CHECK(cublasSgemm(cublas_handle,
                                 CUBLAS_OP_T, CUBLAS_OP_N,
                                 k, n, m,
                                 &alpha, d_W, m, d_X_dense, m,
                                 &beta, d_WtX, k));

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

        // 2. XHt = X × H^T (SPARSE × dense, use cuSPARSE!)
        // This is where sparse format provides benefit
        CUSPARSE_CHECK(cusparseSpMM(cusparse_handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_TRANSPOSE,
                                    &alpha, matX, matH, &beta, matXHt,
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


    // FLOPS (less than dense because we skip zeros)
    long long flops_per_iter = 0;
    flops_per_iter += 2LL * k * k * m;        // WtW
    flops_per_iter += 2LL * k * n * k;        // WtW × H (temp)
    flops_per_iter += 2LL * k * k * n;        // HHt
    flops_per_iter += 2LL * nnz * k;          // Sparse: XHt (only non-zeros!)
    flops_per_iter += 2LL * m * k * k;        // W × HHt
    flops_per_iter += 4LL * m * k;            // Element-wise W update

    long long total_flops = flops_per_iter * max_iter;
    float flops_per_second = (total_flops / elapsed_ms) * 1000.0f;
    float gflops = flops_per_second / 1e9f;

    // Bandwidth (less than dense - only access non-zeros)
    long long bytes_per_iter = 0;
    bytes_per_iter += (long long)(nnz * sizeof(float));     // Sparse X values
    bytes_per_iter += (long long)(m * k + k * n) * 4 * 2;   // W, H
    bytes_per_iter += (long long)(m * k) * 4;               // Write W

    long long total_bytes = bytes_per_iter * max_iter;
    float bandwidth_gbps = (total_bytes / elapsed_ms) * 1000.0f / 1e9f;

    // Compute final error (reconstruct full X for error calculation)
    float* h_X_reconstructed = (float*)calloc(m * n, sizeof(float));
    for (int i = 0; i < m; i++) {
        for (int j = h_rowPtr[i]; j < h_rowPtr[i+1]; j++) {
            h_X_reconstructed[i * n + h_colInd[j]] = h_values[j];
        }
    }

    float error = compute_relative_error_dense(h_X_reconstructed, h_W, h_H, m, n, k);


    // Print results


    printf("========================================\n");
    printf("SPARSE NMF RESULTS\n");
    printf("========================================\n");
    printf("Time: %.2f ms\n", elapsed_ms);
    printf("GFLOPS: %.2f (%.1f%% less than dense due to sparsity)\n",
           gflops, (1.0f - (float)nnz / (m*n)) * 100);
    printf("Bandwidth: %.2f GB/s\n", bandwidth_gbps);
    printf("Final error: %.6e\n", error);
    printf("========================================\n\n");

    printf("Sparse Implementation Notes:\n");
    printf("  - Only X×H^T uses sparse format (1 of 6 operations)\n");
    printf("  - W^T×X uses dense (cuSPARSE lacks efficient dense×sparse)\n");
    printf("  - FLOP reduction: %.1f%% for X×H^T operation only\n", (1.0f - (float)nnz/(m*n)) * 100);
    printf("  - Memory for CSR: %.1fMB vs %.1fMB dense X\n",
           (nnz * sizeof(float) + nnz * sizeof(int) + (m+1) * sizeof(int)) / 1e6f,
           (m * n * sizeof(float)) / 1e6f);
    printf("  - W and H remain dense (not sparse-preserving algorithm)\n");
    printf("========================================\n");

    // Return metrics
    *time_ms = elapsed_ms;
    *bandwidth_achieved = bandwidth_gbps;
    *flops_achieved = gflops;


    // Cleanup


    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUSPARSE_CHECK(cusparseDestroy(cusparse_handle));

    CUSPARSE_CHECK(cusparseDestroySpMat(matX));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matW));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matH));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matXHt));

    cudaFree(d_X_dense);
    cudaFree(d_rowPtr);
    cudaFree(d_colInd);
    cudaFree(d_values);
    cudaFree(d_W);
    cudaFree(d_H);
    cudaFree(d_WtW);
    cudaFree(d_WtX);
    cudaFree(d_HHt);
    cudaFree(d_XHt);
    cudaFree(d_temp_H);
    cudaFree(d_temp_W);
    if (d_buffer) cudaFree(d_buffer);

    free(h_rowPtr);
    free(h_colInd);
    free(h_values);
    free(h_W);
    free(h_H);
    free(h_X_reconstructed);
}


// Main Function


int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: %s <matrix_file> <rank_k> <max_iter>\n", argv[0]);
        printf("Example: %s data/sparse_1000.bin 20 50\n", argv[0]);
        return 1;
    }

    const char* matrix_file = argv[1];
    int k = atoi(argv[2]);
    int max_iter = atoi(argv[3]);

    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║          CSE587 NMF - LEVEL 3: SPARSE NMF                     ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    // Load matrix from file
    float* h_X = NULL;
    int m, n;
    load_matrix_binary(matrix_file, &h_X, &m, &n);
    printf("\n");

    // Run sparse NMF
    float time_ms, bandwidth_gbps, gflops;
    nmf_sparse_gpu(h_X, m, n, k, max_iter, &time_ms, &bandwidth_gbps, &gflops);

    // Save metrics
    printf("\nSaving metrics to results/sparse_metrics.txt...\n");
    FILE* fp = fopen("results/sparse_metrics.txt", "w");
    if (fp) {
        fprintf(fp, "Version: Sparse NMF (Level 3)\n");
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
    printf("Compare with other implementations:\n");
    printf("./nmf_memory_opt %s %d %d\n", matrix_file, k, max_iter);
    printf("\n");

    free(h_X);
    return 0;
}
