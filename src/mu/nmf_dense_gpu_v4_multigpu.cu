#include "../utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/*
 * LEVEL 4: MULTI-GPU IMPLEMENTATION WITH OPENMP
 *
 * Purpose: Distributed parallel computing across multiple GPUs
 *
 * Key Features:
 * 1. Column-wise data parallelism - Split X and H across GPUs
 * 2. W matrix replication - Each GPU maintains full W
 * 3. CPU-mediated AllReduce - Synchronize partial results
 * 4. OpenMP for multi-threading - One thread per GPU
 * 5. Pinned memory for fast transfers
 *
 * Data Distribution:
 * - GPU 0: X[:, 0:n/2], H[:, 0:n/2], W[m×k]
 * - GPU 1: X[:, n/2:n], H[:, n/2:n], W[m×k]
 *
 * Communication Pattern:
 * - H update: Independent (no communication)
 * - W update: AllReduce for HHt and XHt
 *
 * Expected Performance:
 * - 1.7-1.9x speedup for 2 GPUs (91% efficiency)
 * - Crossover point: ~800×800 matrices
 * - Communication overhead: ~9% for typical sizes
 */


// Per-GPU context structure
struct GPUContext {
    int gpu_id;
    int n_local;           // Number of columns on this GPU
    int col_offset;        // Column offset in global matrix

    // Device pointers
    float *d_X;            // Local portion of X [m × n_local]
    float *d_W;            // Full W matrix [m × k]
    float *d_H;            // Local portion of H [k × n_local]
    float *d_WtW;          // W^T × W [k × k]
    float *d_WtX;          // W^T × X_local [k × n_local]
    float *d_HHt;          // H_local × H_local^T [k × k]
    float *d_XHt;          // X_local × H_local^T [m × k]
    float *d_temp_H;       // Temporary for H update [k × n_local]
    float *d_temp_W;       // Temporary for W update [m × k]

    // Pre-allocated buffers for global reduced results (avoid malloc in loop!)
    float *d_HHt_global;   // Global reduced HHt [k × k]
    float *d_XHt_global;   // Global reduced XHt [m × k]

    // cuBLAS handle
    cublasHandle_t cublas_handle;

    // Host pinned memory for AllReduce
    float *h_HHt_pinned;   // Pinned memory for HHt
    float *h_XHt_pinned;   // Pinned memory for XHt
};


// 8-way ILP fused kernel (from Level 3)
__global__ void elementwise_multiply_divide_fused_ilp8(
    float* input,
    float* numerator,
    float* denominator,
    int size,
    float eps
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

    if (idx + 7 < size) {
        // Load 8 elements
        float in0 = input[idx];
        float in1 = input[idx + 1];
        float in2 = input[idx + 2];
        float in3 = input[idx + 3];
        float in4 = input[idx + 4];
        float in5 = input[idx + 5];
        float in6 = input[idx + 6];
        float in7 = input[idx + 7];

        float num0 = numerator[idx];
        float num1 = numerator[idx + 1];
        float num2 = numerator[idx + 2];
        float num3 = numerator[idx + 3];
        float num4 = numerator[idx + 4];
        float num5 = numerator[idx + 5];
        float num6 = numerator[idx + 6];
        float num7 = numerator[idx + 7];

        float den0 = denominator[idx];
        float den1 = denominator[idx + 1];
        float den2 = denominator[idx + 2];
        float den3 = denominator[idx + 3];
        float den4 = denominator[idx + 4];
        float den5 = denominator[idx + 5];
        float den6 = denominator[idx + 6];
        float den7 = denominator[idx + 7];

        // Compute all 8
        in0 = in0 * num0 / (den0 + eps);
        in1 = in1 * num1 / (den1 + eps);
        in2 = in2 * num2 / (den2 + eps);
        in3 = in3 * num3 / (den3 + eps);
        in4 = in4 * num4 / (den4 + eps);
        in5 = in5 * num5 / (den5 + eps);
        in6 = in6 * num6 / (den6 + eps);
        in7 = in7 * num7 / (den7 + eps);

        // Store 8 elements
        input[idx] = in0;
        input[idx + 1] = in1;
        input[idx + 2] = in2;
        input[idx + 3] = in3;
        input[idx + 4] = in4;
        input[idx + 5] = in5;
        input[idx + 6] = in6;
        input[idx + 7] = in7;
    } else {
        for (int i = idx; i < size && i < idx + 8; i++) {
            input[i] = input[i] * numerator[i] / (denominator[i] + eps);
        }
    }
}


// Initialize GPU context
void init_gpu_context(GPUContext* ctx, int gpu_id, int m, int n, int k,
                      int n_local, int col_offset, float* h_X, float* h_W, float* h_H) {
    ctx->gpu_id = gpu_id;
    ctx->n_local = n_local;
    ctx->col_offset = col_offset;

    // Set device
    CUDA_CHECK(cudaSetDevice(gpu_id));

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&ctx->d_X, m * n_local * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_W, m * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_H, k * n_local * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_WtW, k * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_WtX, k * n_local * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_HHt, k * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_XHt, m * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_temp_H, k * n_local * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_temp_W, m * k * sizeof(float)));

    // Pre-allocate buffers for global reduced results (avoid malloc in iteration loop!)
    CUDA_CHECK(cudaMalloc(&ctx->d_HHt_global, k * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_XHt_global, m * k * sizeof(float)));

    // Allocate pinned memory for fast transfers
    CUDA_CHECK(cudaMallocHost(&ctx->h_HHt_pinned, k * k * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&ctx->h_XHt_pinned, m * k * sizeof(float)));

    // Copy data to device (column-wise distribution)
    // Copy local columns of X
    for (int col = 0; col < n_local; col++) {
        CUDA_CHECK(cudaMemcpy(ctx->d_X + col * m,
                              h_X + (col_offset + col) * m,
                              m * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    // Copy full W (replicated on all GPUs)
    CUDA_CHECK(cudaMemcpy(ctx->d_W, h_W, m * k * sizeof(float), cudaMemcpyHostToDevice));

    // Copy local columns of H
    for (int col = 0; col < n_local; col++) {
        CUDA_CHECK(cudaMemcpy(ctx->d_H + col * k,
                              h_H + (col_offset + col) * k,
                              k * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    // Create cuBLAS handle
    CUBLAS_CHECK(cublasCreate(&ctx->cublas_handle));
}


// Cleanup GPU context
void cleanup_gpu_context(GPUContext* ctx) {
    CUDA_CHECK(cudaSetDevice(ctx->gpu_id));

    CUBLAS_CHECK(cublasDestroy(ctx->cublas_handle));

    cudaFree(ctx->d_X);
    cudaFree(ctx->d_W);
    cudaFree(ctx->d_H);
    cudaFree(ctx->d_WtW);
    cudaFree(ctx->d_WtX);
    cudaFree(ctx->d_HHt);
    cudaFree(ctx->d_XHt);
    cudaFree(ctx->d_temp_H);
    cudaFree(ctx->d_temp_W);
    cudaFree(ctx->d_HHt_global);
    cudaFree(ctx->d_XHt_global);

    cudaFreeHost(ctx->h_HHt_pinned);
    cudaFreeHost(ctx->h_XHt_pinned);
}


// Multi-GPU NMF Implementation
void nmf_multigpu(float* h_X, int m, int n, int k, int max_iter, int num_gpus,
                  float* time_ms, float* bandwidth_achieved, float* flops_achieved,
                  float* final_error, bool log_convergence, int log_interval) {

    printf("========================================\n");
    printf("LEVEL 4: MULTI-GPU IMPLEMENTATION\n");
    printf("========================================\n");
    printf("Matrix: %dx%d, Rank: %d, Iterations: %d\n", m, n, k, max_iter);
    printf("Number of GPUs: %d\n", num_gpus);
    printf("Parallelization: Column-wise data distribution\n");
    printf("Communication: CPU-mediated AllReduce\n");
    printf("Expected: 1.7-1.9x speedup for 2 GPUs\n");
    printf("----------------------------------------\n");

    // Check GPU availability
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (num_gpus > device_count) {
        printf("ERROR: Requested %d GPUs but only %d available\n", num_gpus, device_count);
        return;
    }

    // Initialize W and H on host
    float *h_W = (float*)malloc(m * k * sizeof(float));
    float *h_H = (float*)malloc(k * n * sizeof(float));
    generate_random_matrix(h_W, m, k, 42);
    generate_random_matrix(h_H, k, n, 43);

    // Calculate column distribution
    int n_per_gpu = n / num_gpus;
    int n_remainder = n % num_gpus;

    printf("\nColumn Distribution:\n");
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        int n_local = n_per_gpu + (gpu < n_remainder ? 1 : 0);
        int offset = gpu * n_per_gpu + (gpu < n_remainder ? gpu : n_remainder);
        printf("  GPU %d: columns %d-%d (%d columns)\n", gpu, offset, offset + n_local - 1, n_local);
    }
    printf("----------------------------------------\n\n");

    // Initialize GPU contexts
    GPUContext* contexts = new GPUContext[num_gpus];

    #pragma omp parallel for num_threads(num_gpus)
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        int n_local = n_per_gpu + (gpu < n_remainder ? 1 : 0);
        int offset = gpu * n_per_gpu + (gpu < n_remainder ? gpu : n_remainder);
        init_gpu_context(&contexts[gpu], gpu, m, n, k, n_local, offset, h_X, h_W, h_H);
    }

    printf("✓ Initialized %d GPUs\n\n", num_gpus);

    // Host buffers for AllReduce
    float *h_HHt_global = (float*)malloc(k * k * sizeof(float));
    float *h_XHt_global = (float*)malloc(m * k * sizeof(float));

    float alpha = 1.0f;
    float beta = 0.0f;

    // Convergence logging setup with communication breakdown
    FILE* csv_fp = NULL;
    if (log_convergence) {
        csv_fp = fopen("results/convergence_mu_l4.csv", "w");
        if (csv_fp) {
            fprintf(csv_fp, "iteration,error,compute_ms,comm_ms,total_ms\n");
        }
    }

    // Timing accumulators for breakdown
    float total_compute_ms = 0.0f;
    float total_comm_ms = 0.0f;

    // Main iteration loop
    CudaTimer timer;
    CudaTimer comm_timer;  // For communication timing
    timer.startTimer();

    for (int iter = 0; iter < max_iter; iter++) {
        float iter_compute_ms = 0.0f;
        float iter_comm_ms = 0.0f;
        // ====================================================================
        // Update H: H = H .* (W^T × X) ./ (W^T × W × H + eps)
        // This is INDEPENDENT across GPUs (no communication needed)
        // ====================================================================

        #pragma omp parallel for num_threads(num_gpus)
        for (int gpu = 0; gpu < num_gpus; gpu++) {
            GPUContext* ctx = &contexts[gpu];
            CUDA_CHECK(cudaSetDevice(ctx->gpu_id));

            int n_local = ctx->n_local;
            int grid_size_H = ((k * n_local) + (128 * 8) - 1) / (128 * 8);

            // WtW = W^T × W (same on all GPUs)
            CUBLAS_CHECK(cublasSgemm(ctx->cublas_handle,
                                     CUBLAS_OP_T, CUBLAS_OP_N,
                                     k, k, m,
                                     &alpha, ctx->d_W, m, ctx->d_W, m,
                                     &beta, ctx->d_WtW, k));

            // WtX = W^T × X_local
            CUBLAS_CHECK(cublasSgemm(ctx->cublas_handle,
                                     CUBLAS_OP_T, CUBLAS_OP_N,
                                     k, n_local, m,
                                     &alpha, ctx->d_W, m, ctx->d_X, m,
                                     &beta, ctx->d_WtX, k));

            // temp_H = WtW × H_local
            CUBLAS_CHECK(cublasSgemm(ctx->cublas_handle,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     k, n_local, k,
                                     &alpha, ctx->d_WtW, k, ctx->d_H, k,
                                     &beta, ctx->d_temp_H, k));

            // H_local = H_local .* WtX ./ (temp_H + eps)
            elementwise_multiply_divide_fused_ilp8<<<grid_size_H, 128>>>(
                ctx->d_H, ctx->d_WtX, ctx->d_temp_H, k * n_local, 1e-10f
            );
        }

        // Synchronize all GPUs after H update
        for (int gpu = 0; gpu < num_gpus; gpu++) {
            CUDA_CHECK(cudaSetDevice(gpu));
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // ====================================================================
        // Update W: W = W .* (X × H^T) ./ (W × H × H^T + eps)
        // Requires AllReduce for HHt and XHt
        // ====================================================================

        // Start communication timing (includes D2H, CPU sum, H2D)
        comm_timer.startTimer();

        // Step 1: Each GPU computes local HHt and XHt and copies to host
        #pragma omp parallel for num_threads(num_gpus)
        for (int gpu = 0; gpu < num_gpus; gpu++) {
            GPUContext* ctx = &contexts[gpu];
            CUDA_CHECK(cudaSetDevice(ctx->gpu_id));

            int n_local = ctx->n_local;

            // HHt_local = H_local × H_local^T
            CUBLAS_CHECK(cublasSgemm(ctx->cublas_handle,
                                     CUBLAS_OP_N, CUBLAS_OP_T,
                                     k, k, n_local,
                                     &alpha, ctx->d_H, k, ctx->d_H, k,
                                     &beta, ctx->d_HHt, k));

            // XHt_local = X_local × H_local^T
            CUBLAS_CHECK(cublasSgemm(ctx->cublas_handle,
                                     CUBLAS_OP_N, CUBLAS_OP_T,
                                     m, k, n_local,
                                     &alpha, ctx->d_X, m, ctx->d_H, k,
                                     &beta, ctx->d_XHt, m));

            // Copy to pinned memory for AllReduce (D2H)
            CUDA_CHECK(cudaMemcpy(ctx->h_HHt_pinned, ctx->d_HHt, k * k * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(ctx->h_XHt_pinned, ctx->d_XHt, m * k * sizeof(float), cudaMemcpyDeviceToHost));
        }

        // Step 2: CPU-mediated AllReduce (sum reduction)
        // Initialize with zeros
        for (int i = 0; i < k * k; i++) h_HHt_global[i] = 0.0f;
        for (int i = 0; i < m * k; i++) h_XHt_global[i] = 0.0f;

        // Sum all partial results
        for (int gpu = 0; gpu < num_gpus; gpu++) {
            for (int i = 0; i < k * k; i++) {
                h_HHt_global[i] += contexts[gpu].h_HHt_pinned[i];
            }
            for (int i = 0; i < m * k; i++) {
                h_XHt_global[i] += contexts[gpu].h_XHt_pinned[i];
            }
        }

        // Step 3: Broadcast reduced results to each GPU (H2D)
        #pragma omp parallel for num_threads(num_gpus)
        for (int gpu = 0; gpu < num_gpus; gpu++) {
            GPUContext* ctx = &contexts[gpu];
            CUDA_CHECK(cudaSetDevice(ctx->gpu_id));

            // Copy reduced HHt and XHt to pre-allocated device buffers (no malloc in loop!)
            CUDA_CHECK(cudaMemcpy(ctx->d_HHt_global, h_HHt_global, k * k * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(ctx->d_XHt_global, h_XHt_global, m * k * sizeof(float), cudaMemcpyHostToDevice));
        }

        // End communication timing
        for (int gpu = 0; gpu < num_gpus; gpu++) {
            CUDA_CHECK(cudaSetDevice(gpu));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        iter_comm_ms = comm_timer.stopTimer();
        total_comm_ms += iter_comm_ms;

        // Step 4: Update W on each GPU (pure compute)
        #pragma omp parallel for num_threads(num_gpus)
        for (int gpu = 0; gpu < num_gpus; gpu++) {
            GPUContext* ctx = &contexts[gpu];
            CUDA_CHECK(cudaSetDevice(ctx->gpu_id));

            int grid_size_W = ((m * k) + (128 * 8) - 1) / (128 * 8);

            // temp_W = W × HHt_global
            CUBLAS_CHECK(cublasSgemm(ctx->cublas_handle,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     m, k, k,
                                     &alpha, ctx->d_W, m, ctx->d_HHt_global, k,
                                     &beta, ctx->d_temp_W, m));

            // W = W .* XHt_global ./ (temp_W + eps)
            elementwise_multiply_divide_fused_ilp8<<<grid_size_W, 128>>>(
                ctx->d_W, ctx->d_XHt_global, ctx->d_temp_W, m * k, 1e-10f
            );
        }

        // Synchronize all GPUs after W update
        for (int gpu = 0; gpu < num_gpus; gpu++) {
            CUDA_CHECK(cudaSetDevice(gpu));
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Calculate compute time as total iter time minus communication time
        float iter_total_ms = timer.stopTimer();
        iter_compute_ms = iter_total_ms - iter_comm_ms;
        total_compute_ms += iter_compute_ms;

        // Per-iteration convergence logging with communication breakdown
        if (log_convergence && (iter % log_interval == 0 || iter == max_iter - 1)) {
            // Gather current W and H for error computation
            CUDA_CHECK(cudaSetDevice(0));
            CUDA_CHECK(cudaMemcpy(h_W, contexts[0].d_W, m * k * sizeof(float), cudaMemcpyDeviceToHost));

            // Gather H from all GPUs
            for (int gpu = 0; gpu < num_gpus; gpu++) {
                GPUContext* ctx = &contexts[gpu];
                CUDA_CHECK(cudaSetDevice(ctx->gpu_id));
                for (int col = 0; col < ctx->n_local; col++) {
                    CUDA_CHECK(cudaMemcpy(h_H + (ctx->col_offset + col) * k,
                                          ctx->d_H + col * k,
                                          k * sizeof(float),
                                          cudaMemcpyDeviceToHost));
                }
            }

            float error = compute_relative_error_dense(h_X, h_W, h_H, m, n, k);

            if (csv_fp) {
                fprintf(csv_fp, "%d,%.6e,%.4f,%.4f,%.4f\n", iter, error,
                        total_compute_ms, total_comm_ms, total_compute_ms + total_comm_ms);
                fflush(csv_fp);
            }
            printf("Iteration %d: error=%.6e, compute=%.2f ms, comm=%.2f ms (%.1f%%)\n",
                   iter, error, total_compute_ms, total_comm_ms,
                   100.0f * total_comm_ms / (total_compute_ms + total_comm_ms));
        } else if (iter % 10 == 0) {
            printf("Iteration %d\n", iter);
        }

        // Restart timer for next iteration
        timer.startTimer();
    }

    // Final synchronization
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    float elapsed_ms = total_compute_ms + total_comm_ms;

    // Close convergence log
    if (csv_fp) {
        fclose(csv_fp);
        printf("\nConvergence log saved to results/convergence_mu_l4.csv\n");
        printf("Communication breakdown: compute=%.2f ms (%.1f%%), comm=%.2f ms (%.1f%%)\n",
               total_compute_ms, 100.0f * total_compute_ms / elapsed_ms,
               total_comm_ms, 100.0f * total_comm_ms / elapsed_ms);
    }

    // Copy results back from GPU 0
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMemcpy(h_W, contexts[0].d_W, m * k * sizeof(float), cudaMemcpyDeviceToHost));

    // Gather H from all GPUs
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        GPUContext* ctx = &contexts[gpu];
        CUDA_CHECK(cudaSetDevice(ctx->gpu_id));
        for (int col = 0; col < ctx->n_local; col++) {
            CUDA_CHECK(cudaMemcpy(h_H + (ctx->col_offset + col) * k,
                                  ctx->d_H + col * k,
                                  k * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        }
    }

    // Calculate metrics
    long long flops_per_iter = 0;
    flops_per_iter += 2LL * k * k * m;
    flops_per_iter += 2LL * k * n * m;
    flops_per_iter += 2LL * k * n * k;
    flops_per_iter += 2LL * k * k * n;
    flops_per_iter += 2LL * m * k * n;
    flops_per_iter += 2LL * m * k * k;
    flops_per_iter += 4LL * (m * k + k * n);

    long long total_flops = flops_per_iter * max_iter;
    float gflops = (total_flops / elapsed_ms) * 1000.0f / 1e9f;

    long long bytes_per_iter = (long long)(m * n + m * k + k * n) * 4 * 2;
    bytes_per_iter += (long long)(m * k + k * n) * 4;
    long long total_bytes = bytes_per_iter * max_iter;
    float bandwidth_gbps = (total_bytes / elapsed_ms) * 1000.0f / 1e9f;

    float error = compute_relative_error_dense(h_X, h_W, h_H, m, n, k);

    // Print results
    printf("========================================\n");
    printf("MULTI-GPU RESULTS\n");
    printf("========================================\n");
    printf("Time: %.2f ms\n", elapsed_ms);
    printf("FLOPS: %.2f GFLOPS\n", gflops);
    printf("Bandwidth: %.2f GB/s\n", bandwidth_gbps);
    printf("Final error: %.6e\n", error);
    printf("========================================\n\n");

    printf("Multi-GPU Analysis:\n");
    printf("  GPUs used: %d\n", num_gpus);
    printf("  Communication: CPU-mediated AllReduce\n");
    printf("  H update: Independent (no communication)\n");
    printf("  W update: Synchronized (AllReduce overhead)\n");
    printf("========================================\n");

    *time_ms = elapsed_ms;
    *bandwidth_achieved = bandwidth_gbps;
    *flops_achieved = gflops;
    *final_error = error;

    // Cleanup
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        cleanup_gpu_context(&contexts[gpu]);
    }

    delete[] contexts;
    free(h_W);
    free(h_H);
    free(h_HHt_global);
    free(h_XHt_global);
}


// Main Function
int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: %s <matrix_file> <rank_k> <max_iter> [num_gpus] [--log-convergence [interval]]\n", argv[0]);
        printf("Example: %s data/dense_1000.bin 20 50 2\n", argv[0]);
        printf("         %s data/dense_1000.bin 20 100 2 --log-convergence 5\n", argv[0]);
        return 1;
    }

    const char* matrix_file = argv[1];
    int k = atoi(argv[2]);
    int max_iter = atoi(argv[3]);
    int num_gpus = 2;  // Default: 2 GPUs
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
            num_gpus = atoi(argv[i]);
        }
    }

    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║       CSE587 NMF - LEVEL 4: MULTI-GPU IMPLEMENTATION          ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    // Load matrix
    float* h_X = NULL;
    int m, n;
    load_matrix_binary(matrix_file, &h_X, &m, &n);
    printf("\n");

    // Run multi-GPU NMF
    float time_ms, bandwidth_gbps, gflops, error;
    nmf_multigpu(h_X, m, n, k, max_iter, num_gpus, &time_ms, &bandwidth_gbps, &gflops, &error,
                 log_convergence, log_interval);

    // Save metrics
    printf("\nSaving metrics to results/multigpu_metrics.txt...\n");
    FILE* fp = fopen("results/multigpu_metrics.txt", "w");
    if (fp) {
        fprintf(fp, "Version: Multi-GPU (Level 4)\n");
        fprintf(fp, "Size: %dx%d\n", m, n);
        fprintf(fp, "Rank: %d\n", k);
        fprintf(fp, "Iterations: %d\n", max_iter);
        fprintf(fp, "GPUs: %d\n", num_gpus);
        fprintf(fp, "Time: %.2f ms\n", time_ms);
        fprintf(fp, "Final_Error: %.6e\n", error);
        fprintf(fp, "Bandwidth: %.2f GB/s\n", bandwidth_gbps);
        fprintf(fp, "GFLOPS: %.2f\n", gflops);
        fclose(fp);
        printf("✓ Metrics saved\n");
    }

    if (log_convergence) {
        printf("✓ Convergence log saved to results/convergence_mu_l4.csv\n");
    }

    printf("\n");
    printf("Next steps:\n");
    printf("1. Compare with single GPU: ./nmf_memory_opt %s %d %d\n", matrix_file, k, max_iter);
    printf("2. Test scalability with larger matrices\n");
    printf("3. Analyze communication overhead\n");
    printf("\n");

    free(h_X);
    return 0;
}
