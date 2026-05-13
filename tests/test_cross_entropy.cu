// ===========================================================================
// test_cross_entropy.cu
//
// Correctness + perf for per-token cross-entropy loss. Shapes mirror the
// LM head: N = batch*seq, V = vocab. M2 config: batch=8, seq=512, vocab=30k
// -> N=4096, V=30000.
// ===========================================================================

#include "kernels/cross_entropy.cuh"
#include "kernels/cross_entropy_cpu.hpp"
#include "utils/cuda_check.hpp"
#include "utils/tensor.hpp"
#include "test_utils.hpp"

#include <cstdio>
#include <vector>

static int run_case(int N, int V, bool check_correctness) {
    std::printf("=== N=%d V=%d ===\n", N, V);

    std::vector<float> hlogits(static_cast<size_t>(N) * V);
    std::vector<int>   htargets(N);
    std::vector<float> hlosses_gpu(N);
    std::vector<float> hlosses_cpu(N);

    fill_random(hlogits, 1);
    fill_random_int(htargets, V, 2);

    Tensor<float> dlogits({N, V});
    Tensor<int>   dtargets({N});
    Tensor<float> dlosses({N});

    dlogits.copy_from_host(hlogits.data());
    dtargets.copy_from_host(htargets.data());

    auto launch = [&]() {
        launch_cross_entropy_forward(dlogits.data(), dtargets.data(),
                                     dlosses.data(), N, V);
    };

    constexpr int N_ITER = 20;
    float ms = time_kernel(launch, N_ITER);

    // We read the full logits matrix (4*N*V bytes) and write one float per
    // token (negligible). The kernel reads logits twice (for max, then for
    // sum-exp), so 8*N*V is the right number for bandwidth.
    double bytes = 8.0 * static_cast<double>(N) * V;
    double gbps  = bytes / (ms * 1e-3) / 1e9;
    std::printf("  GPU: %7.3f ms/iter, %7.1f GB/s (avg of %d)\n",
                ms, gbps, N_ITER);
    std::printf("PERF: kernel=cross_entropy N=%d V=%d ms=%.4f bandwidth_gbs=%.1f\n",
                N, V, ms, gbps);

    if (check_correctness) {
        dlosses.copy_to_host(hlosses_gpu.data());
        cross_entropy_cpu(hlogits.data(), htargets.data(),
                          hlosses_cpu.data(), N, V);

        float abs_err, rel_err;
        compare(hlosses_cpu, hlosses_gpu, abs_err, rel_err);
        std::printf("  max abs err: %.3e, max rel err: %.3e\n",
                    abs_err, rel_err);

        if (rel_err > 5e-4f) {
            std::printf("  FAIL\n");
            return 1;
        }
        std::printf("  PASS\n");
    }
    return 0;
}

int main() {
    int fails = 0;

    fails += run_case(4,   64,    true);
    fails += run_case(64,  1024,  true);
    fails += run_case(128, 30000, true);

    // Realistic LM-head shape.
    fails += run_case(4096, 30000, true);

    if (fails) {
        std::printf("\n%d case(s) FAILED\n", fails);
        return 1;
    }
    std::printf("\nAll correctness cases passed.\n");
    return 0;
}
