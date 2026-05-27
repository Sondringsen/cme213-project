// ===========================================================================
// test_backward_cross_entropy.cu
//
// Correctness test for launch_cross_entropy_backward.
//
// The gradient of per-token cross-entropy loss w.r.t. logits is:
//   dlogits[n, j] = dlosses[n] * (softmax(logits[n])[j] - 1{j == targets[n]})
//
// We test with dlosses = all-ones (uniform upstream gradient) and also with
// random dlosses to cover the general case. The GPU result is compared to a
// CPU reference using the same LSE (log-sum-exp) trick for numerical stability.
// ===========================================================================

#include "kernels/cross_entropy.cuh"
#include "utils/cuda_check.hpp"
#include "utils/tensor.hpp"
#include "test_utils.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

// ---------------------------------------------------------------------------
// CPU reference: dlogits[n,j] = dlosses[n] * (softmax(logits[n])[j] - 1{j==t})
// ---------------------------------------------------------------------------
static void cross_entropy_backward_cpu(
        const float* logits, const int* targets, const float* dlosses,
        float* dlogits, int N, int V) {

    for (int n = 0; n < N; ++n) {
        const float* logits_n  = logits  + n * V;
        float*       dlogits_n = dlogits + n * V;
        float dloss = dlosses[n];
        int   tgt   = targets[n];

        // Find max for numerical stability
        float row_max = -INFINITY;
        for (int j = 0; j < V; ++j) row_max = std::fmax(row_max, logits_n[j]);

        // Sum of exp(logit - max)
        float Z = 0.0f;
        for (int j = 0; j < V; ++j) Z += std::expf(logits_n[j] - row_max);

        // Write dlogits
        for (int j = 0; j < V; ++j) {
            float softmax_j = std::expf(logits_n[j] - row_max) / Z;
            float indicator = (j == tgt) ? 1.0f : 0.0f;
            dlogits_n[j] = dloss * (softmax_j - indicator);
        }
    }
}

// ---------------------------------------------------------------------------
// One test case.
// ---------------------------------------------------------------------------
static int run_case(int N, int V, bool uniform_dloss) {
    std::vector<float> h_logits(N * V);
    std::vector<int>   h_targets(N);
    std::vector<float> h_dlosses(N);

    fill_random(h_logits, 1);
    fill_random_int(h_targets, V, 2);
    if (uniform_dloss) {
        std::fill(h_dlosses.begin(), h_dlosses.end(), 1.0f / N);
    } else {
        fill_random(h_dlosses, 3);
        for (auto& v : h_dlosses) v = std::fabs(v);  // keep positive
    }

    Tensor<float> logits({N, V}), dlogits({N, V}), dlosses({N});
    Tensor<int>   targets({N});

    logits.copy_from_host(h_logits.data());
    targets.copy_from_host(h_targets.data());
    dlosses.copy_from_host(h_dlosses.data());
    dlogits.zero();

    launch_cross_entropy_backward(logits.data(), targets.data(),
                                  dlosses.data(), dlogits.data(),
                                  N, V);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gpu_dlogits(N * V);
    dlogits.copy_to_host(gpu_dlogits.data());

    std::vector<float> cpu_dlogits(N * V);
    cross_entropy_backward_cpu(h_logits.data(), h_targets.data(),
                               h_dlosses.data(), cpu_dlogits.data(),
                               N, V);

    float abs_err, rel_err;
    compare(cpu_dlogits, gpu_dlogits, abs_err, rel_err);

    const char* tag = uniform_dloss ? "uniform-dloss" : "random-dloss";
    std::printf("  N=%4d V=%5d [%s] | abs=%.2e rel=%.2e",
                N, V, tag, abs_err, rel_err);

    constexpr float TOL = 1e-3f;
    if (rel_err > TOL) {
        std::printf(" FAIL\n");
        return 1;
    }
    std::printf(" PASS\n");
    return 0;
}

int main() {
    int fails = 0;
    std::printf("=== Cross-Entropy Backward ===\n");
    fails += run_case(4,   128, true);
    fails += run_case(32,  512, true);
    fails += run_case(32,  512, false);
    fails += run_case(128, 1024, false);

    if (fails) {
        std::printf("\n%d case(s) FAILED\n", fails);
        return 1;
    }
    std::printf("\nAll cases PASSED\n");
    return 0;
}
