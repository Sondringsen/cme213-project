// ===========================================================================
// test_gemm.cu
//
// Correctness + performance harness for the tiled GEMM kernel.
//
// For each problem size we:
//   1. Fill host A, B with deterministic random values.
//   2. Copy to device, run the GPU kernel, copy result back.
//   3. (Small sizes only) Run the CPU reference and compare.
//   4. Time the GPU kernel with cudaEvent and report GFLOPS.
//
// "Pass" criterion: max relative error < 1e-3. With FP32 inputs in [-1, 1]
// and K up to a few hundred, the accumulated rounding stays well below
// this. We'll have to widen the tolerance when we go to BF16.
// ===========================================================================

#include "kernels/gemm.cuh"
#include "kernels/gemm_cpu.hpp"
#include "utils/cuda_check.hpp"
#include "utils/tensor.hpp"
#include "test_utils.hpp"

#include <chrono>
#include <cstdio>
#include <vector>

#ifdef WITH_CUBLAS
#include <cublas_v2.h>
#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t s = (call);                                            \
        if (s != CUBLAS_STATUS_SUCCESS) {                                     \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n",               \
                         (int)s, __FILE__, __LINE__);                         \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)
#endif

// Run one (M, N, K) case. `check_correctness` runs the CPU oracle; turn it
// off for large sizes where the CPU triple loop would take ages.
// Returns 0 on success, 1 on correctness failure.
static int run_case(int M, int N, int K, bool check_correctness) {
    std::printf("=== M=%d N=%d K=%d ===\n", M, N, K);

    std::vector<float> hA(static_cast<size_t>(M) * K);
    std::vector<float> hB(static_cast<size_t>(K) * N);
    std::vector<float> hC_gpu(static_cast<size_t>(M) * N);
    std::vector<float> hC_cpu(static_cast<size_t>(M) * N);
    fill_random(hA, 1);
    fill_random(hB, 2);

    Tensor<float> dA({M, K});
    Tensor<float> dB({K, N});
    Tensor<float> dC({M, N});
    dA.copy_from_host(hA.data());
    dB.copy_from_host(hB.data());

    auto launch = [&]() {
        launch_gemm_tiled(dA.data(), dB.data(), dC.data(), M, N, K);
    };

    constexpr int N_ITER = 10;
    float ms = time_kernel(launch, N_ITER);
    // GEMM does 2*M*N*K flops (one multiply + one add per inner-loop step).
    double gflops = (2.0 * M * N * K) / (ms * 1e-3) / 1e9;
    std::printf("  GPU: %7.3f ms/iter, %7.1f GFLOPS (avg of %d)\n",
                ms, gflops, N_ITER);
    std::printf("PERF: kernel=gemm M=%d N=%d K=%d ms=%.4f gflops=%.1f\n",
                M, N, K, ms, gflops);

    if (check_correctness) {
        dC.copy_to_host(hC_gpu.data());

        auto t0 = std::chrono::high_resolution_clock::now();
        gemm_cpu(hA.data(), hB.data(), hC_cpu.data(), M, N, K);
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("  CPU: %7.3f ms\n", cpu_ms);

        float abs_err, rel_err;
        compare(hC_cpu, hC_gpu, abs_err, rel_err);
        std::printf("  max abs err: %.3e, max rel err: %.3e\n",
                    abs_err, rel_err);

        if (rel_err > 2e-3f) {
            std::printf("  FAIL\n");
            return 1;
        }
        std::printf("  PASS\n");
    }
    return 0;
}

#ifdef WITH_CUBLAS
// Compare our kernel against cuBLAS for correctness (when check_correctness=true)
// and report GFLOPS for both.
//
// cuBLAS uses column-major storage. For row-major C = A*B we use the identity
// C^T = B^T * A^T, calling cublasSgemm with the pointers swapped and leading
// dimensions set to the column counts of the original matrices.
static int run_case_cublas(cublasHandle_t handle,
                           int M, int N, int K,
                           bool check_correctness) {
    std::printf("--- cuBLAS M=%d N=%d K=%d ---\n", M, N, K);

    std::vector<float> hA(static_cast<size_t>(M) * K);
    std::vector<float> hB(static_cast<size_t>(K) * N);
    fill_random(hA, 1);
    fill_random(hB, 2);

    Tensor<float> dA({M, K}), dB({K, N}), dC_ours({M, N}), dC_ref({M, N});
    dA.copy_from_host(hA.data());
    dB.copy_from_host(hB.data());

    constexpr int N_ITER = 10;

    auto launch_ours = [&]() {
        launch_gemm_tiled(dA.data(), dB.data(), dC_ours.data(), M, N, K);
    };
    float ms_ours = time_kernel(launch_ours, N_ITER);

    const float one = 1.0f, zero = 0.0f;
    auto launch_cublas = [&]() {
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K,
                                 &one,
                                 dB.data(), N,
                                 dA.data(), K,
                                 &zero,
                                 dC_ref.data(), N));
    };
    float ms_cublas = time_kernel(launch_cublas, N_ITER);

    double gflops_ours   = (2.0 * M * N * K) / (ms_ours   * 1e-3) / 1e9;
    double gflops_cublas = (2.0 * M * N * K) / (ms_cublas * 1e-3) / 1e9;
    std::printf("  Ours:   %7.3f ms, %7.1f GFLOPS\n", ms_ours,   gflops_ours);
    std::printf("  cuBLAS: %7.3f ms, %7.1f GFLOPS\n", ms_cublas, gflops_cublas);
    std::printf("  Ratio ours/cuBLAS: %.1f%%\n", 100.0 * gflops_ours / gflops_cublas);
    std::printf("PERF: kernel=gemm    M=%d N=%d K=%d ms=%.4f gflops=%.1f\n",
                M, N, K, ms_ours, gflops_ours);
    std::printf("PERF: kernel=cublas  M=%d N=%d K=%d ms=%.4f gflops=%.1f\n",
                M, N, K, ms_cublas, gflops_cublas);

    if (check_correctness) {
        std::vector<float> hC_ours(static_cast<size_t>(M) * N);
        std::vector<float> hC_ref(static_cast<size_t>(M) * N);
        dC_ours.copy_to_host(hC_ours.data());
        dC_ref.copy_to_host(hC_ref.data());

        float abs_err, rel_err;
        compare(hC_ref, hC_ours, abs_err, rel_err);
        std::printf("  vs cuBLAS: max abs err=%.3e, max rel err=%.3e", abs_err, rel_err);

        if (rel_err > 2e-3f) {
            std::printf("  FAIL\n");
            return 1;
        }
        std::printf("  PASS\n");
    }
    return 0;
}
#endif  // WITH_CUBLAS

int main() {
    int fails = 0;

    // Small sizes: correctness + timing vs CPU reference.
    fails += run_case(64,  64,  64,  /*check=*/true);
    fails += run_case(128, 128, 128, /*check=*/true);
    fails += run_case(256, 256, 64,  /*check=*/true);

    // Non-multiples of tile dimensions — exercises boundary masking.
    fails += run_case(127, 65, 99, /*check=*/true);
    fails += run_case(33,  33, 33, /*check=*/true);

    // Larger sizes: timing only (CPU oracle is too slow).
    run_case(1024, 1024, 1024, /*check=*/false);
    run_case(2048, 2048, 2048, /*check=*/false);
    run_case(4096, 4096, 4096, /*check=*/false);

#ifdef WITH_CUBLAS
    std::printf("\n=== cuBLAS comparison ===\n");
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Small: correctness vs cuBLAS + timing.
    fails += run_case_cublas(handle, 128,  128,  128,  /*check=*/true);
    fails += run_case_cublas(handle, 256,  256,  256,  /*check=*/true);

    // GPT-2 realistic shapes: timing + ratio only.
    run_case_cublas(handle, 1024, 1024, 1024, /*check=*/false);
    run_case_cublas(handle, 2048, 2048, 2048, /*check=*/false);
    run_case_cublas(handle, 4096, 4096, 4096, /*check=*/false);

    CUBLAS_CHECK(cublasDestroy(handle));
#endif

    if (fails) {
        std::printf("\n%d case(s) FAILED\n", fails);
        return 1;
    }
    std::printf("\nAll correctness cases passed.\n");
    return 0;
}
