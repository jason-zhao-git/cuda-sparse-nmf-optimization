#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <cstring>

/*
 * CPU HALS NMF - CORRECT COLUMN-MAJOR IMPLEMENTATION
 *
 * Purpose: Sequential HALS baseline with proper column-major indexing
 *
 * Key Fix: Uses consistent column-major indexing IDX(row, col, num_rows)
 *          instead of mixing row-major and column-major access
 */

// ============================================================================
// CONFIGURATION & CONSTANTS
// ============================================================================
const float EPSILON = 1e-16f;

// ============================================================================
// MATRIX UTILITIES (COLUMN-MAJOR)
// ============================================================================
// Access macro: M(row, col) with 'rows' as the leading dimension
#define IDX(row, col, num_rows) ((col) * (num_rows) + (row))

// Load matrix from binary file
void load_matrix_binary(const char* filename, std::vector<float>& matrix, int& m, int& n) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        exit(1);
    }

    // Read dimensions
    fread(&m, sizeof(int), 1, fp);
    fread(&n, sizeof(int), 1, fp);

    std::cout << "Loading matrix from " << filename << std::endl;
    std::cout << "  Dimensions: " << m << "x" << n << std::endl;

    // Allocate and read data
    matrix.resize(m * n);
    size_t elements_read = fread(matrix.data(), sizeof(float), m * n, fp);

    if (elements_read != (size_t)(m * n)) {
        std::cerr << "Error: Expected " << m * n << " elements, read " << elements_read << std::endl;
        exit(1);
    }

    fclose(fp);
    std::cout << "✓ Matrix loaded successfully" << std::endl;
}

// Save matrix to binary file
void save_matrix_binary(const char* filename, const std::vector<float>& matrix, int m, int n) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        std::cerr << "Error: Cannot create file " << filename << std::endl;
        return;
    }

    fwrite(&m, sizeof(int), 1, fp);
    fwrite(&n, sizeof(int), 1, fp);
    fwrite(matrix.data(), sizeof(float), m * n, fp);

    fclose(fp);
}

// Initialize random matrix
void randomize(std::vector<float>& M, int rows, int cols, unsigned int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& val : M) val = dist(rng);
}

// Matrix Multiply: C = A * B
// A: (m x k), B: (k x n), C: (m x n)
void matmul(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
            int m, int k, int n) {
    std::fill(C.begin(), C.end(), 0.0f);
    // J-K-I loop order for cache efficiency with column-major
    for (int j = 0; j < n; ++j) {
        for (int l = 0; l < k; ++l) {
            float b_val = B[IDX(l, j, k)];
            for (int i = 0; i < m; ++i) {
                C[IDX(i, j, m)] += A[IDX(i, l, m)] * b_val;
            }
        }
    }
}

// Matrix Transpose: B = A^T
// A: (m x n), B: (n x m)
void transpose(const std::vector<float>& A, std::vector<float>& B, int m, int n) {
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            B[IDX(j, i, n)] = A[IDX(i, j, m)];
        }
    }
}

// Frobenius Norm for error calculation
float get_reconstruction_error(const std::vector<float>& X,
                               const std::vector<float>& W,
                               const std::vector<float>& H,
                               int m, int n, int k) {
    // Temp = W * H
    std::vector<float> WH(m * n);
    matmul(W, H, WH, m, k, n);

    double error_sum = 0.0;
    double x_norm_sum = 0.0;

    for (int i = 0; i < m * n; ++i) {
        float diff = X[i] - WH[i];
        error_sum += diff * diff;
        x_norm_sum += X[i] * X[i];
    }

    return std::sqrt(error_sum) / std::sqrt(x_norm_sum);
}

// ============================================================================
// HALS ALGORITHM
// ============================================================================
void nmf_hals(const std::vector<float>& X,
              std::vector<float>& W,
              std::vector<float>& H,
              int m, int n, int k, int max_iter) {

    // Pre-allocate temp buffers to avoid malloc inside loop
    std::vector<float> Wt(k * m);
    std::vector<float> Ht(n * k);
    std::vector<float> Numerator_H(k * n);   // W^T * X
    std::vector<float> Denom_Part_H(k * k);  // W^T * W
    std::vector<float> Numerator_W(m * k);   // X * H^T
    std::vector<float> Denom_Part_W(k * k);  // H * H^T

    std::cout << "\n========================================" << std::endl;
    std::cout << "CPU HALS NMF - SEQUENTIAL BASELINE" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Matrix: " << m << "x" << n << ", Rank: " << k << ", Iterations: " << max_iter << std::endl;
    std::cout << "Algorithm: Gauss-Seidel column-wise updates" << std::endl;
    std::cout << "Expected convergence: ~40 iterations" << std::endl;
    std::cout << "----------------------------------------\n" << std::endl;

    for (int iter = 0; iter < max_iter; ++iter) {

        // --- UPDATE H ---------------------------------------------
        // H_fj <- H_fj + [ (W^T X)_fj - (W^T W H)_fj ] / (W^T W)_ff

        // 1. Prepare Numerator: W^T * X  (Size: k x n)
        transpose(W, Wt, m, k);
        matmul(Wt, X, Numerator_H, k, m, n);

        // 2. Prepare Partial Denominator: W^T * W (Size: k x k)
        matmul(Wt, W, Denom_Part_H, k, m, k);

        // 3. Solve column-by-column (Feature f)
        for (int f = 0; f < k; ++f) {
            float wtw_val = Denom_Part_H[IDX(f, f, k)]; // Diagonal element (W^T W)_ff
            if (wtw_val < EPSILON) wtw_val = EPSILON;

            for (int j = 0; j < n; ++j) {
                // Numerator element (f, j)
                float num = Numerator_H[IDX(f, j, k)];

                // Interaction: sum of (W^T W)_fl * H_lj (includes self-term)
                float interaction = 0.0f;
                for (int l = 0; l < k; ++l) {
                   interaction += Denom_Part_H[IDX(f, l, k)] * H[IDX(l, j, k)];
                }

                // H_new = H_old + (Numerator - Interaction) / Diagonal
                float current_h = H[IDX(f, j, k)];
                float grad = num - interaction;

                H[IDX(f, j, k)] = std::max(EPSILON, current_h + grad / wtw_val);
            }
        }

        // --- UPDATE W ---------------------------------------------
        // W_if <- W_if + [ (X H^T)_if - (W H H^T)_if ] / (H H^T)_ff

        // 1. Prepare Numerator: X * H^T (Size: m x k)
        transpose(H, Ht, k, n);
        matmul(X, Ht, Numerator_W, m, n, k);

        // 2. Prepare Partial Denominator: H * H^T (Size: k x k)
        matmul(H, Ht, Denom_Part_W, k, n, k);

        // 3. Solve column-by-column
        for (int f = 0; f < k; ++f) {
            float hht_val = Denom_Part_W[IDX(f, f, k)];
            if (hht_val < EPSILON) hht_val = EPSILON;

            for (int i = 0; i < m; ++i) {
                float num = Numerator_W[IDX(i, f, m)];

                float interaction = 0.0f;
                for (int l = 0; l < k; ++l) {
                    interaction += W[IDX(i, l, m)] * Denom_Part_W[IDX(l, f, k)];
                }

                float current_w = W[IDX(i, f, m)];
                float grad = num - interaction;

                W[IDX(i, f, m)] = std::max(EPSILON, current_w + grad / hht_val);
            }

            // 4. Normalize W column 'f'
            float norm = 0.0f;
            for (int i = 0; i < m; ++i) norm += W[IDX(i, f, m)] * W[IDX(i, f, m)];
            norm = std::sqrt(norm);
            if (norm < EPSILON) norm = EPSILON;
            for (int i = 0; i < m; ++i) W[IDX(i, f, m)] /= norm;
        }

        // Print Progress
        if (iter % 10 == 0 || iter == max_iter - 1) {
            float error = get_reconstruction_error(X, W, H, m, n, k);
            std::cout << "Iteration " << std::setw(3) << iter
                      << ": Error = " << std::scientific << std::setprecision(6) << error << std::endl;
        }
    }
}

// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <matrix_file> <rank_k> <max_iter>" << std::endl;
        std::cout << "Example: " << argv[0] << " data/dense_1000.bin 20 50" << std::endl;
        std::cout << "\nGenerate matrix with: python3 data/generate_matrix.py --size 1000 --sparsity 0.0 --output data/dense_1000.bin" << std::endl;
        return 1;
    }

    const char* matrix_file = argv[1];
    int k = atoi(argv[2]);
    int max_iter = atoi(argv[3]);

    std::cout << "\n╔════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║         CSE587 NMF - CPU HALS SEQUENTIAL BASELINE             ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n" << std::endl;

    // 1. Load Data
    std::vector<float> X;
    int m, n;
    load_matrix_binary(matrix_file, X, m, n);

    // 2. Initialize W and H
    std::vector<float> W(m * k);
    std::vector<float> H(k * n);
    randomize(W, m, k, 42);
    randomize(H, k, n, 43);

    // 3. Run HALS
    auto start = std::chrono::high_resolution_clock::now();
    nmf_hals(X, W, H, m, n, k, max_iter);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    float elapsed_ms = duration.count();
    float error = get_reconstruction_error(X, W, H, m, n, k);

    // 4. Print Results
    std::cout << "\n========================================" << std::endl;
    std::cout << "CPU HALS RESULTS" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Time: " << std::fixed << std::setprecision(2) << elapsed_ms << " ms" << std::endl;
    std::cout << "Time per iteration: " << elapsed_ms / max_iter << " ms" << std::endl;
    std::cout << "Final error: " << std::scientific << std::setprecision(6) << error << std::endl;
    std::cout << "========================================\n" << std::endl;

    std::cout << "Convergence Analysis:" << std::endl;
    std::cout << "  - HALS uses Gauss-Seidel updates" << std::endl;
    std::cout << "  - Each feature depends on previously updated features" << std::endl;
    std::cout << "  - Faster convergence than MU (~40 vs 200 iters)" << std::endl;
    std::cout << "  - But inherently sequential (hard to parallelize)" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // 5. Save Results
    std::cout << "Saving results for visualization..." << std::endl;
    save_matrix_binary("results/W_matrix_hals.bin", W, m, k);
    save_matrix_binary("results/H_matrix_hals.bin", H, k, n);

    // Save metrics
    std::cout << "\nSaving metrics to results/hals_cpu_metrics.txt..." << std::endl;
    std::ofstream fp("results/hals_cpu_metrics.txt");
    if (fp) {
        fp << "Version: CPU HALS Sequential Baseline (C++)" << std::endl;
        fp << "Size: " << m << "x" << n << std::endl;
        fp << "Rank: " << k << std::endl;
        fp << "Iterations: " << max_iter << std::endl;
        fp << "Time: " << elapsed_ms << " ms" << std::endl;
        fp << "Time per iteration: " << elapsed_ms / max_iter << " ms" << std::endl;
        fp << "Final error: " << error << std::endl;
        fp << "\nAlgorithm: HALS (Hierarchical Alternating Least Squares)" << std::endl;
        fp << "Update pattern: Gauss-Seidel (sequential feature-wise)" << std::endl;
        fp << "Expected convergence: ~40 iterations" << std::endl;
        fp.close();
        std::cout << "✓ Metrics saved" << std::endl;
    }

    std::cout << "\nNext steps:" << std::endl;
    std::cout << "1. Compare convergence with MU: diff results/memory_opt_metrics.txt results/hals_cpu_metrics.txt" << std::endl;
    std::cout << "2. Verify ~40 iteration convergence (vs ~200 for MU)" << std::endl;
    std::cout << "3. Move to GPU HALS with tiled batching (T=4-10)\n" << std::endl;

    return 0;
}
