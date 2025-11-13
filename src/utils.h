#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ============================================================================
// Error Checking Macros
// ============================================================================

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CUBLAS_CHECK(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error at %s:%d - code %d\n", \
                __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}

#define CUSPARSE_CHECK(call) { \
    cusparseStatus_t status = call; \
    if (status != CUSPARSE_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSPARSE Error at %s:%d - code %d\n", \
                __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}

// ============================================================================
// CUDA Timer Class
// ============================================================================

class CudaTimer {
private:
    cudaEvent_t start, stop;

public:
    CudaTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void startTimer() {
        cudaEventRecord(start);
    }

    float stopTimer() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        return milliseconds;
    }
};

// ============================================================================
// Utility Functions (implemented in utils.cu)
// ============================================================================

// Print GPU memory usage
void print_gpu_memory_usage();

// Load dense matrix from text file
void load_dense_matrix(const char* filename, float** data, int* m, int* n);

// Load sparse matrix in CSR format from text files
void load_sparse_matrix_csr(const char* base_filename,
                             float** values, int** colInd, int** rowPtr,
                             int* m, int* n, int* nnz);

// Compute relative error: ||X - WH|| / ||X||
float compute_relative_error_dense(const float* h_X, const float* h_W, const float* h_H,
                                    int m, int n, int k);

// Generate random non-negative matrix
void generate_random_matrix(float* data, int rows, int cols, unsigned int seed);

#endif // UTILS_H
