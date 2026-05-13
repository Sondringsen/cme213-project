// ===========================================================================
// test_gelu.cu -- correctness + performance for GELU forward.
//
// GELU is purely pointwise so the "interesting" test cases are:
//   - Exact matches vs the CPU reference (erff identity)
//   - Non-multiple-of-block-size lengths (boundary guard)
//   - Large sizes to measure effective memory bandwidth
// ===========================================================================

#include "kernels/gelu.cuh"
#include "kernels/gelu_cpu.hpp"
#include "utils/tensor.hpp"
#include "test_utils.hpp"

#include <cstdio>
#include <vector>

static int run_case(int N, bool check_correctness) {
    std::printf("=== N=%d ===\n", N);

    std::vector<float> hx(N), hy_gpu(N), hy_cpu(N);
    fill_random(hx, /*seed=*/42);

    Tensor<float> dx({N});
    Tensor<float> dy({N});
    dx.copy_from_host(hx.data());

    auto launch = [&]() {
        launch_gelu_forward(dx.data(), dy.data(), N);
    };

    constexpr int N_ITER = 20;
    float ms = time_kernel(launch, N_ITER);

    // GELU reads N floats and writes N floats = 2*N*4 bytes of memory traffic.
    // The kernel is bandwidth-bound, so GB/s is more informative than GFLOPS.
    double gb_s = (2.0 * N * sizeof(float)) / (ms * 1e-3) / 1e9;
    std::printf("  GPU: %7.3f ms/iter, %6.1f GB/s (avg of %d)\n",
                ms, gb_s, N_ITER);

    if (check_correctness) {
        dy.copy_to_host(hy_gpu.data());
        gelu_cpu(hx.data(), hy_cpu.data(), N);

        float abs_err, rel_err;
        compare(hy_cpu, hy_gpu, abs_err, rel_err);
        std::printf("  max abs err: %.3e, max rel err: %.3e\n",
                    abs_err, rel_err);

        // GPU erff() and CPU erff() are both IEEE-754 compliant to ~1 ULP,
        // so differences should be negligible. 1e-5 gives plenty of margin.
        if (rel_err > 1e-5f) {
            std::printf("  FAIL\n");
            return 1;
        }
        std::printf("  PASS\n");
    }
    return 0;
}

int main() {
    int fails = 0;

    // Correctness: exact sizes and non-multiples of GELU_BLOCK (256).
    fails += run_case(64,   true);
    fails += run_case(256,  true);
    fails += run_case(1024, true);
    fails += run_case(4096, true);
    fails += run_case(100,  true);   // non-multiple of 256
    fails += run_case(777,  true);   // non-multiple of 256

    // Bandwidth benchmark at realistic FFN sizes.
    // GPT-2 small FFN: hidden 768 -> 3072, batch*seq up to 8*512 = 4096 rows
    // -> GELU applies to 4096 * 3072 = 12.6 M elements.
    run_case(1 << 20, false);  // ~4 M elements
    run_case(1 << 22, false);  // ~16 M elements (covers realistic FFN)

    if (fails) {
        std::printf("\n%d case(s) FAILED\n", fails);
        return 1;
    }
    std::printf("\nAll correctness cases passed.\n");
    return 0;
}
