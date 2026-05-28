// ===========================================================================
// test_backward_layernorm.cu
//
// Correctness test for launch_layernorm_backward.
//
// Strategy:
//   1. Run launch_layernorm_forward with mean_out/rstd_out to cache statistics.
//   2. Run launch_layernorm_backward with a random upstream gradient dy.
//   3. Compare dx, dgamma, dbeta against a CPU reference that implements the
//      same chain rule.
//
// CPU reference formula (see layernorm.cu for derivation):
//   dnorm_i = gamma_i * dy_i
//   dx_i    = rstd * (dnorm_i - mean_j(dnorm_j) - xhat_i * mean_j(dnorm_j*xhat_j))
//   dgamma_i = sum_n dy[n,i] * xhat[n,i]
//   dbeta_i  = sum_n dy[n,i]
// ===========================================================================

#include "kernels/layernorm.cuh"
#include "utils/cuda_check.hpp"
#include "utils/tensor.hpp"
#include "test_utils.hpp"

#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

// ---------------------------------------------------------------------------
// CPU reference for layernorm backward.
// Caller must zero dgamma and dbeta before the first call (they accumulate).
// ---------------------------------------------------------------------------
static void layernorm_backward_cpu(
        const float* dy, const float* x, const float* gamma,
        float* dx, float* dgamma, float* dbeta,
        int N, int H, float eps) {

    std::fill(dgamma, dgamma + H, 0.0f);
    std::fill(dbeta,  dbeta  + H, 0.0f);

    for (int n = 0; n < N; ++n) {
        const float* xn  = x  + n * H;
        const float* dyn = dy + n * H;
        float*       dxn = dx + n * H;

        // Recompute mean and rstd from x (same as forward)
        float mean = 0.0f;
        for (int i = 0; i < H; ++i) mean += xn[i];
        mean /= H;

        float var = 0.0f;
        for (int i = 0; i < H; ++i) {
            float d = xn[i] - mean;
            var += d * d;
        }
        var /= H;
        float rstd = 1.0f / sqrtf(var + eps);

        // Two reductions (with gamma weights — see layernorm.cu comment)
        float sum_dy     = 0.0f;
        float sum_dyxhat = 0.0f;
        for (int i = 0; i < H; ++i) {
            float xhat   = (xn[i] - mean) * rstd;
            float dnorm  = dyn[i] * gamma[i];
            sum_dy     += dnorm;
            sum_dyxhat += dnorm * xhat;
        }
        float inv_H = 1.0f / H;

        // dx and parameter gradient accumulation
        for (int i = 0; i < H; ++i) {
            float xhat  = (xn[i] - mean) * rstd;
            float dnorm = dyn[i] * gamma[i];
            dxn[i]     = rstd * (dnorm - inv_H * sum_dy - xhat * inv_H * sum_dyxhat);
            dgamma[i] += dyn[i] * xhat;
            dbeta[i]  += dyn[i];
        }
    }
}

// ---------------------------------------------------------------------------
// One test case: returns 0 on PASS, 1 on FAIL.
// ---------------------------------------------------------------------------
static int run_case(int N, int H) {
    const float eps = 1e-5f;

    // Host buffers
    std::vector<float> hx(N * H), hgamma(H), hbeta(H), hdy(N * H);
    fill_random(hx,     1);
    fill_random(hgamma, 2);
    fill_random(hbeta,  3);
    fill_random(hdy,    4);

    // GPU tensors
    Tensor<float> dx({N, H}), dgamma({H}), dbeta({H});
    Tensor<float> x({N, H}), gamma({H}), beta({H});
    Tensor<float> y({N, H}), dy({N, H});
    Tensor<float> mean_buf({N}), rstd_buf({N});

    x.copy_from_host(hx.data());
    gamma.copy_from_host(hgamma.data());
    beta.copy_from_host(hbeta.data());
    dy.copy_from_host(hdy.data());
    dx.zero(); dgamma.zero(); dbeta.zero();

    // GPU: forward (to get cached mean/rstd) + backward
    launch_layernorm_forward(x.data(), gamma.data(), beta.data(),
                             y.data(), N, H, eps,
                             mean_buf.data(), rstd_buf.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    launch_layernorm_backward(dy.data(), x.data(), gamma.data(),
                              mean_buf.data(), rstd_buf.data(),
                              dx.data(), dgamma.data(), dbeta.data(),
                              N, H);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy GPU results to host
    std::vector<float> gpu_dx(N * H), gpu_dgamma(H), gpu_dbeta(H);
    dx.copy_to_host(gpu_dx.data());
    dgamma.copy_to_host(gpu_dgamma.data());
    dbeta.copy_to_host(gpu_dbeta.data());

    // CPU reference
    std::vector<float> cpu_dx(N * H), cpu_dgamma(H), cpu_dbeta(H);
    layernorm_backward_cpu(hdy.data(), hx.data(), hgamma.data(),
                           cpu_dx.data(), cpu_dgamma.data(), cpu_dbeta.data(),
                           N, H, eps);

    // Compare
    float dx_abs, dx_rel, dg_abs, dg_rel, db_abs, db_rel;
    compare(cpu_dx,     gpu_dx,     dx_abs, dx_rel);
    compare(cpu_dgamma, gpu_dgamma, dg_abs, dg_rel);
    compare(cpu_dbeta,  gpu_dbeta,  db_abs, db_rel);

    std::printf("  N=%4d H=%4d | dx  abs=%.2e rel=%.2e | dgamma abs=%.2e rel=%.2e | dbeta abs=%.2e rel=%.2e\n",
                N, H, dx_abs, dx_rel, dg_abs, dg_rel, db_abs, db_rel);

    constexpr float TOL = 1e-3f;
    if (dx_rel > TOL || dg_rel > TOL || db_rel > TOL) {
        std::printf("  FAIL\n");
        return 1;
    }
    std::printf("  PASS\n");
    return 0;
}

int main() {
    int fails = 0;
    std::printf("=== LayerNorm Backward ===\n");
    fails += run_case(4,   64);
    fails += run_case(32,  256);
    fails += run_case(128, 768);
    fails += run_case(100, 555);   // non-power-of-two
    fails += run_case(4096, 768);  // realistic transformer shape

    if (fails) {
        std::printf("\n%d case(s) FAILED\n", fails);
        return 1;
    }
    std::printf("\nAll cases PASSED\n");
    return 0;
}
