#pragma once

// ---------------------------------------------------------------------------
// LayerNorm layer: wraps launch_layernorm_forward / launch_layernorm_backward.
//
// Owns gamma and beta parameters and their gradients.
// Caches x, mean, rstd from the forward pass for use in backward.
// ---------------------------------------------------------------------------

#include "kernels/layernorm.cuh"
#include "utils/tensor.hpp"
#include <vector>

struct LayerNormLayer {
    int     H;
    float   eps;

    Tensor<float> gamma;    // (H,) scale
    Tensor<float> beta;     // (H,) shift
    Tensor<float> d_gamma;  // gradient w.r.t. gamma
    Tensor<float> d_beta;   // gradient w.r.t. beta

    // Cached from forward (device pointers into caller-managed buffers)
    const float* x_cache    = nullptr;  // input x (N, H), not owned
    int          N_cache     = 0;

    // These ARE owned: allocated once to maximum expected N*H size
    Tensor<float> mean_buf;  // (N,) — needs to be large enough
    Tensor<float> rstd_buf;  // (N,)

    // max_N: upper bound on the batch×sequence dimension
    LayerNormLayer(int hidden, int max_N, float epsilon = 1e-5f)
        : H(hidden), eps(epsilon),
          gamma({hidden}), beta({hidden}),
          d_gamma({hidden}), d_beta({hidden}),
          mean_buf({max_N}), rstd_buf({max_N}) {
        std::vector<float> ones(hidden, 1.0f);
        gamma.copy_from_host(ones.data());
        beta.zero();
        d_gamma.zero();
        d_beta.zero();
    }

    void forward(const float* x, float* out, int N, cudaStream_t stream = 0) {
        x_cache = x;
        N_cache = N;
        launch_layernorm_forward(x, gamma.data(), beta.data(), out,
                                 N, H, eps,
                                 mean_buf.data(), rstd_buf.data(),
                                 stream);
    }

    // backward: compute d_x (written to d_x); accumulate d_gamma, d_beta.
    // Caller must zero d_gamma and d_beta before the first backward call
    // within a training step.
    void backward(const float* d_out, float* d_x, cudaStream_t stream = 0) {
        launch_layernorm_backward(d_out, x_cache,
                                  gamma.data(),
                                  mean_buf.data(), rstd_buf.data(),
                                  d_x, d_gamma.data(), d_beta.data(),
                                  N_cache, H, stream);
    }
};
