// ===========================================================================
// test_layernorm.cu
//
// Correctness + performance harness for the fused LayerNorm forward kernel.
//
// Shapes are picked to match what the actual transformer will run with:
//   (N = batch*seq, H = hidden_dim).
// Our M2 config is hidden=768, sequence=512, batch=8 -> N=4096, H=768.
// We also sweep a few smaller cases to make sure the boundary handling for
// non-multiple-of-BLOCK row widths is right.
// ===========================================================================

#include "kernels/layernorm.cuh"
#include "kernels/layernorm_cpu.hpp"
#include "utils/cuda_check.hpp"
#include "utils/tensor.hpp"
#include "test_utils.hpp"

#include <cstdio>
#include <vector>

static int run_case(int N, int H, bool check_correctness) {
    std::printf("=== N=%d H=%d ===\n", N, H);

    std::vector<float> hx(static_cast<size_t>(N) * H);
    std::vector<float> hgamma(H);
    std::vector<float> hbeta(H);
    std::vector<float> hy_gpu(static_cast<size_t>(N) * H);
    std::vector<float> hy_cpu(static_cast<size_t>(N) * H);

    fill_random(hx, 1);
    fill_random(hgamma, 2);
    fill_random(hbeta, 3);

    Tensor<float> dx({N, H});
    Tensor<float> dgamma({H});
    Tensor<float> dbeta({H});
    Tensor<float> dy({N, H});

    dx.copy_from_host(hx.data());
    dgamma.copy_from_host(hgamma.data());
    dbeta.copy_from_host(hbeta.data());

    const float eps = 1e-5f;
    auto launch = [&]() {
        launch_layernorm_forward(dx.data(), dgamma.data(), dbeta.data(),
                                 dy.data(), N, H, eps);
    };

    constexpr int N_ITER = 20;
    float ms = time_kernel(launch, N_ITER);

    // LayerNorm is memory-bound: read x, gamma, beta; write y.
    // Bytes moved per iteration: (N*H + H + H + N*H) * 4 = ~8*N*H bytes
    // for large N. We use 8*N*H as the bandwidth estimate, treating gamma
    // and beta as cache-resident after the first few rows.
    double bytes = 8.0 * static_cast<double>(N) * H;
    double gbps  = bytes / (ms * 1e-3) / 1e9;
    std::printf("  GPU: %7.3f ms/iter, %7.1f GB/s (avg of %d)\n",
                ms, gbps, N_ITER);
    std::printf("PERF: kernel=layernorm N=%d H=%d ms=%.4f bandwidth_gbs=%.1f\n",
                N, H, ms, gbps);

    if (check_correctness) {
        dy.copy_to_host(hy_gpu.data());
        layernorm_cpu(hx.data(), hgamma.data(), hbeta.data(),
                      hy_cpu.data(), N, H, eps);

        float abs_err, rel_err;
        compare(hy_cpu, hy_gpu, abs_err, rel_err);
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

    // Small cases with full correctness check.
    fails += run_case(4,   64,  true);
    fails += run_case(32,  256, true);
    fails += run_case(128, 768, true);   // realistic hidden dim
    fails += run_case(100, 555, true);   // awkward non-multiple sizes

    // Realistic transformer shapes (N = batch*seq, H = hidden).
    // M2 config: hidden = 768, sequence = 512, batch = 8 -> N = 4096.
    run_case(4096, 768,  false);
    run_case(8192, 768,  false);
    run_case(4096, 3072, false);  // FFN intermediate width = 4*hidden

    if (fails) {
        std::printf("\n%d case(s) FAILED\n", fails);
        return 1;
    }
    std::printf("\nAll correctness cases passed.\n");
    return 0;
}
