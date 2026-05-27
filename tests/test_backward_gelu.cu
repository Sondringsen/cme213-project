// ===========================================================================
// test_backward_gelu.cu
//
// Correctness test for launch_gelu_backward.
//
// GELU forward:  y = x * 0.5 * (1 + erf(x / sqrt(2)))
// GELU backward: dx = dy * gelu'(x)
//   gelu'(x) = 0.5*(1+erf(x/sqrt(2))) + x*(1/sqrt(2*pi))*exp(-x^2/2)
//
// The CPU reference uses the same erff / expf calls as the GPU kernel, so
// numerical differences should be at the rounding-error level (< 1e-5 rel).
// ===========================================================================

#include "kernels/gelu.cuh"
#include "utils/cuda_check.hpp"
#include "utils/tensor.hpp"
#include "test_utils.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

// ---------------------------------------------------------------------------
// CPU reference for GELU backward.
// ---------------------------------------------------------------------------
static void gelu_backward_cpu(const float* dy, const float* x,
                               float* dx_out, int n) {
    constexpr float kInvSqrt2   = 0.7071067811865476f;   // 1/sqrt(2)
    constexpr float kInvSqrt2Pi = 0.3989422804014327f;   // 1/sqrt(2*pi)
    for (int i = 0; i < n; ++i) {
        float xi   = x[i];
        float phi  = 0.5f * (1.0f + std::erff(xi * kInvSqrt2));
        float dphi = kInvSqrt2Pi * std::expf(-0.5f * xi * xi);
        dx_out[i]  = dy[i] * (phi + xi * dphi);
    }
}

static int run_case(int n) {
    std::vector<float> hx(n), hdy(n);
    fill_random(hx,  1);
    fill_random(hdy, 2);

    Tensor<float> dx({n}), x({n}), dy({n});
    x.copy_from_host(hx.data());
    dy.copy_from_host(hdy.data());
    dx.zero();

    launch_gelu_backward(dy.data(), x.data(), dx.data(), n);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gpu_dx(n);
    dx.copy_to_host(gpu_dx.data());

    std::vector<float> cpu_dx(n);
    gelu_backward_cpu(hdy.data(), hx.data(), cpu_dx.data(), n);

    float abs_err, rel_err;
    compare(cpu_dx, gpu_dx, abs_err, rel_err);
    std::printf("  n=%7d | abs=%.2e rel=%.2e", n, abs_err, rel_err);

    constexpr float TOL = 1e-4f;
    if (rel_err > TOL) {
        std::printf(" FAIL\n");
        return 1;
    }
    std::printf(" PASS\n");
    return 0;
}

int main() {
    int fails = 0;
    std::printf("=== GELU Backward ===\n");
    fails += run_case(64);
    fails += run_case(1024);
    fails += run_case(4096);
    fails += run_case(100003);  // odd size, tests boundary handling

    if (fails) {
        std::printf("\n%d case(s) FAILED\n", fails);
        return 1;
    }
    std::printf("\nAll cases PASSED\n");
    return 0;
}
