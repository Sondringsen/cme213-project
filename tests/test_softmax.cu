// ===========================================================================
// test_softmax.cu
//
// Correctness + perf harness for the softmax-forward kernel. Shapes mirror
// the two places softmax actually appears in the transformer:
//   - LM head:    (batch*seq, vocab)   e.g. (4096, 30000)
//   - (legacy) plain attention: (batch*heads*seq, seq)
//     -- but our model uses Flash Attention which folds the softmax inside,
//     so this is just a building block / sanity check.
// ===========================================================================

#include "kernels/softmax.cuh"
#include "kernels/softmax_cpu.hpp"
#include "utils/cuda_check.hpp"
#include "utils/tensor.hpp"
#include "test_utils.hpp"

#include <cstdio>
#include <vector>

static int run_case(int N, int V, bool check_correctness) {
    std::printf("=== N=%d V=%d ===\n", N, V);

    std::vector<float> hx(static_cast<size_t>(N) * V);
    std::vector<float> hy_gpu(static_cast<size_t>(N) * V);
    std::vector<float> hy_cpu(static_cast<size_t>(N) * V);
    fill_random(hx, 1);

    Tensor<float> dx({N, V});
    Tensor<float> dy({N, V});
    dx.copy_from_host(hx.data());

    auto launch = [&]() {
        launch_softmax_forward(dx.data(), dy.data(), N, V);
    };

    constexpr int N_ITER = 20;
    float ms = time_kernel(launch, N_ITER);

    // Bytes: 3 reads of x (passes 1/2/3) + 1 write of y. Roughly 16*N*V.
    double bytes = 4.0 * 4.0 * static_cast<double>(N) * V;
    double gbps  = bytes / (ms * 1e-3) / 1e9;
    std::printf("  GPU: %7.3f ms/iter, %7.1f GB/s (avg of %d)\n",
                ms, gbps, N_ITER);
    std::printf("PERF: kernel=softmax N=%d V=%d ms=%.4f bandwidth_gbs=%.1f\n",
                N, V, ms, gbps);

    if (check_correctness) {
        dy.copy_to_host(hy_gpu.data());
        softmax_cpu(hx.data(), hy_cpu.data(), N, V);

        float abs_err, rel_err;
        compare(hy_cpu, hy_gpu, abs_err, rel_err);
        std::printf("  max abs err: %.3e, max rel err: %.3e\n",
                    abs_err, rel_err);

        // Tolerance is a bit looser than GEMM because __expf is a fast
        // intrinsic that trades a couple of bits of accuracy for speed.
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

    // Small correctness sweeps. The 257 row width exercises non-multiple
    // boundary handling.
    fails += run_case(4,    64,    true);
    fails += run_case(16,   257,   true);
    fails += run_case(64,   1024,  true);

    // LM-head shape: (batch*seq, vocab). Vocab 30k from M2.
    fails += run_case(512,  30000, true);

    // Larger timing-only cases.
    run_case(4096, 30000, false);

    if (fails) {
        std::printf("\n%d case(s) FAILED\n", fails);
        return 1;
    }
    std::printf("\nAll correctness cases passed.\n");
    return 0;
}
