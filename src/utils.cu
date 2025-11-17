#include "utils.h"
#include <string.h>

// ============================================================================
// GPU Memory Usage
// ============================================================================

void print_gpu_memory_usage() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t used_mem = total_mem - free_mem;

    printf("GPU Memory Usage:\n");
    printf("  Used:  %.2f MB\n", used_mem / 1e6);
    printf("  Free:  %.2f MB\n", free_mem / 1e6);
    printf("  Total: %.2f MB\n", total_mem / 1e6);
}

// ============================================================================
// Matrix Loading Functions
// ============================================================================

void load_dense_matrix(const char* filename, float** data, int* m, int* n) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Read dimensions
    fscanf(fp, "%d %d", m, n);

    // Allocate memory
    *data = (float*)malloc((*m) * (*n) * sizeof(float));
    if (!*data) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Read data
    for (int i = 0; i < (*m) * (*n); i++) {
        fscanf(fp, "%f", &((*data)[i]));
    }

    fclose(fp);
}

void load_matrix_binary(const char* filename, float** data, int* m, int* n) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Read dimensions (int32)
    int rows, cols;
    if (fread(&rows, sizeof(int), 1, fp) != 1 ||
        fread(&cols, sizeof(int), 1, fp) != 1) {
        fprintf(stderr, "Error: Failed to read matrix dimensions from %s\n", filename);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    *m = rows;
    *n = cols;

    // Allocate memory
    size_t total_elements = (size_t)rows * cols;
    *data = (float*)malloc(total_elements * sizeof(float));
    if (!*data) {
        fprintf(stderr, "Error: Memory allocation failed for %dx%d matrix\n", rows, cols);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    // Read data (float32 in row-major order)
    size_t elements_read = fread(*data, sizeof(float), total_elements, fp);
    if (elements_read != total_elements) {
        fprintf(stderr, "Error: Expected %zu elements, read %zu from %s\n",
                total_elements, elements_read, filename);
        free(*data);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    fclose(fp);

    printf("✓ Loaded matrix from %s\n", filename);
    printf("  Dimensions: %d×%d\n", rows, cols);
    printf("  Elements: %zu\n", total_elements);

    // Analyze sparsity
    size_t zeros = 0;
    for (size_t i = 0; i < total_elements; i++) {
        if ((*data)[i] == 0.0f) zeros++;
    }
    if (zeros > 0) {
        printf("  Sparsity: %.2f%% (%zu zeros)\n",
               (float)zeros / total_elements * 100.0f, zeros);
    }
}

void load_sparse_matrix_csr(const char* base_filename,
                             float** values, int** colInd, int** rowPtr,
                             int* m, int* n, int* nnz) {
    char meta_file[256], val_file[256], col_file[256], row_file[256];

    // Construct filenames
    snprintf(meta_file, 256, "%s_meta.txt", base_filename);
    snprintf(val_file, 256, "%s_values.txt", base_filename);
    snprintf(col_file, 256, "%s_colind.txt", base_filename);
    snprintf(row_file, 256, "%s_rowptr.txt", base_filename);

    // Read metadata
    FILE* fp = fopen(meta_file, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open metadata file %s\n", meta_file);
        exit(EXIT_FAILURE);
    }
    fscanf(fp, "%d %d %d", m, n, nnz);
    fclose(fp);

    // Allocate memory
    *values = (float*)malloc((*nnz) * sizeof(float));
    *colInd = (int*)malloc((*nnz) * sizeof(int));
    *rowPtr = (int*)malloc(((*m) + 1) * sizeof(int));

    // Read values
    fp = fopen(val_file, "r");
    for (int i = 0; i < *nnz; i++) {
        fscanf(fp, "%f", &((*values)[i]));
    }
    fclose(fp);

    // Read column indices
    fp = fopen(col_file, "r");
    for (int i = 0; i < *nnz; i++) {
        fscanf(fp, "%d", &((*colInd)[i]));
    }
    fclose(fp);

    // Read row pointers
    fp = fopen(row_file, "r");
    for (int i = 0; i < *m + 1; i++) {
        fscanf(fp, "%d", &((*rowPtr)[i]));
    }
    fclose(fp);
}

// ============================================================================
// Error Computation
// ============================================================================

float compute_relative_error_dense(const float* h_X, const float* h_W, const float* h_H,
                                    int m, int n, int k) {
    // Compute WH
    float* WH = (float*)malloc(m * n * sizeof(float));
    memset(WH, 0, m * n * sizeof(float));

    // Matrix multiply: WH = W × H
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += h_W[i * k + p] * h_H[p * n + j];
            }
            WH[i * n + j] = sum;
        }
    }

    // Compute ||X - WH||
    float norm_diff = 0.0f;
    float norm_X = 0.0f;

    for (int i = 0; i < m * n; i++) {
        float diff = h_X[i] - WH[i];
        norm_diff += diff * diff;
        norm_X += h_X[i] * h_X[i];
    }

    free(WH);

    return sqrtf(norm_diff) / sqrtf(norm_X);
}

// ============================================================================
// Random Matrix Generation
// ============================================================================

void generate_random_matrix(float* data, int rows, int cols, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < rows * cols; i++) {
        data[i] = (float)rand() / (float)RAND_MAX;
    }
}
