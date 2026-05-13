// ===========================================================================
// test_attention.cu
//
// Correctness + perf for Flash Attention forward. Shapes are picked to
// match our M2 config (D = 96 for hidden 768 / 8 heads).
//
// The CPU oracle is O(B*H*S^2*D), so we only run it on small S. Larger S
// is timing-only.
// ===========================================================================

#include "kernels/attention.cuh"
#include "kernels/attention_cpu.hpp"
#include "utils/cuda_check.hpp"
#include "utils/tensor.hpp"
#include "test_utils.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

static int run_case(int B, int H, int S, int D, bool causal,
                    bool check_correctness) {
    std::printf("=== B=%d H=%d S=%d D=%d causal=%d ===\n",
                B, H, S, D, (int)causal);

    size_t numel = static_cast<size_t>(B) * H * S * D;
    std::vector<float> hQ(numel), hK(numel), hV(numel);
    std::vector<float> hO_gpu(numel), hO_cpu(numel);
    fill_random(hQ, 1);
    fill_random(hK, 2);
    fill_random(hV, 3);

    Tensor<float> dQ({B, H, S, D});
    Tensor<float> dK({B, H, S, D});
    Tensor<float> dV({B, H, S, D});
    Tensor<float> dO({B, H, S, D});
    dQ.copy_from_host(hQ.data());
    dK.copy_from_host(hK.data());
    dV.copy_from_host(hV.data());

    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    auto launch = [&]() {
        launch_flash_attention_forward(dQ.data(), dK.data(), dV.data(),
                                       dO.data(), B, H, S, D, scale, causal);
    };

    constexpr int N_ITER = 20;
    float ms = time_kernel(launch, N_ITER);

    // Attention flops (forward): per (b, h): 2 * 2 * S^2 * D
    //   QK^T   :  2 * S^2 * D  (matmul, ignoring scale + mask + softmax)
    //   P V    :  2 * S^2 * D
    // Ignoring the softmax's flops (negligible).
    double flops = 4.0 * B * H * static_cast<double>(S) * S * D;
    double gflops = flops / (ms * 1e-3) / 1e9;
    std::printf("  GPU: %7.3f ms/iter, %7.1f GFLOPS (avg of %d)\n",
                ms, gflops, N_ITER);
    std::printf("PERF: kernel=attention B=%d H=%d S=%d D=%d causal=%d ms=%.4f gflops=%.1f\n",
                B, H, S, D, (int)causal, ms, gflops);

    if (check_correctness) {
        dO.copy_to_host(hO_gpu.data());
        attention_cpu(hQ.data(), hK.data(), hV.data(), hO_cpu.data(),
                      B, H, S, D, scale, causal);

        float abs_err, rel_err;
        compare(hO_cpu, hO_gpu, abs_err, rel_err);
        std::printf("  max abs err: %.3e, max rel err: %.3e\n",
                    abs_err, rel_err);

        // Looser threshold than GEMM because of accumulated expf rounding.
        // Still much tighter than the 1e-2 BF16 budget from M2 §3.
        if (rel_err > 5e-3f) {
            std::printf("  FAIL\n");
            return 1;
        }
        std::printf("  PASS\n");
    }
    return 0;
}

int main() {
    int fails = 0;

    // Small correctness cases. Both causal and non-causal.
    fails += run_case(1, 2, 16, 32, false, true);
    fails += run_case(1, 2, 16, 32, true,  true);
    fails += run_case(2, 4, 64, 64, true,  true);

    // M2 config: D = 96 (hidden 768 / 8 heads).
    fails += run_case(1, 8, 128, 96, true, true);

    // Non-multiple-of-Br lengths to exercise boundary handling.
    fails += run_case(1, 2, 33,  64, true, true);
    fails += run_case(1, 2, 100, 64, true, true);

    // Timing-only cases at realistic sizes.
    run_case(8, 8, 512,  96, true, false);   // M2 default
    run_case(2, 8, 1024, 96, true, false);
    run_case(1, 8, 2048, 96, true, false);

    if (fails) {
        std::printf("\n%d case(s) FAILED\n", fails);
        return 1;
    }
    std::printf("\nAll correctness cases passed.\n");
    return 0;
}
