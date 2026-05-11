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

        if (rel_err > 1e-3f) {
            std::printf("  FAIL\n");
            return 1;
        }
        std::printf("  PASS\n");
    }
    return 0;
}

int main() {
    int fails = 0;

    // Small sizes: correctness + timing. CPU reference is O(M*N*K) so we
    // can't run it on huge matrices.
    fails += run_case(64,  64,  64,  /*check=*/true);
    fails += run_case(128, 128, 128, /*check=*/true);
    fails += run_case(256, 256, 256, /*check=*/true);

    // Awkward, non-multiple-of-TILE shapes -- exercises the boundary-mask
    // logic in the kernel.
    fails += run_case(127, 65, 99, /*check=*/true);
    fails += run_case(33,  33, 33, /*check=*/true);

    // Larger sizes: timing only. These reflect realistic GPT-2 GEMM shapes.
    // E.g. the FFN-up projection at d_model=768 is (batch*seq) x 768 -> 3072.
    run_case(1024, 1024, 1024, /*check=*/false);
    run_case(2048, 2048, 2048, /*check=*/false);
    run_case(4096, 4096, 4096, /*check=*/false);

    if (fails) {
        std::printf("\n%d case(s) FAILED\n", fails);
        return 1;
    }
    std::printf("\nAll correctness cases passed.\n");
    return 0;
}
