#pragma once

// ---------------------------------------------------------------------------
// Linear layer: out = x * W^T  (or equivalently, W x^T column-wise)
//
// For a transformer, the typical convention is:
//   x    : (N, in_features)   — input batch
//   W    : (out_features, in_features)  — weight matrix
//   out  : (N, out_features)
//
// This is equivalent to out = x W^T, computed as:
//   GEMM(x, W^T, out, N, out_features, in_features)
//   = launch_gemm_nt(x, W, out, N, out_features, in_features)
//
// Backward (given d_out upstream gradient, shape N × out_features):
//   d_x = d_out * W          (N × in_features)
//       = launch_gemm_tiled(d_out, W, d_x, N, in_features, out_features)
//   d_W = d_out^T * x        (out_features × in_features)
//       = launch_gemm_tn(d_out, x, d_W, out_features, in_features, N)
//
// The struct caches the input x from forward so that backward can compute d_W.
// d_W is accumulated (+=) across time steps; caller zeros it before each step.
// ---------------------------------------------------------------------------

#include "kernels/gemm.cuh"
#include "utils/tensor.hpp"
#include <cmath>
#include <random>
#include <vector>

struct Linear {
    int in_features;
    int out_features;

    Tensor<float> W;      // (out_features, in_features)
    Tensor<float> d_W;    // gradient w.r.t. W (caller zeros before backward)

    // Cached from forward pass — needed in backward to compute d_W
    const float* x_cache = nullptr;  // raw device pointer (not owned)
    int          N_cache  = 0;

    Linear(int in_f, int out_f)
        : in_features(in_f), out_features(out_f),
          W({out_f, in_f}), d_W({out_f, in_f}) {
        d_W.zero();
    }

    // Initialize weights
    void init_xavier(unsigned seed = 0) {
        float lim = std::sqrt(6.0f / static_cast<float>(in_features + out_features));
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-lim, lim);
        std::vector<float> h_W(W.numel());
        for (auto& v : h_W) v = dist(gen);
        W.copy_from_host(h_W.data());
    }

    // forward: out = x * W^T
    //   x   : device pointer to (N, in_features)
    //   out : device pointer to (N, out_features)
    void forward(const float* x, float* out, int N,
                 cudaStream_t stream = 0) {
        x_cache = x;
        N_cache  = N;
        // out = x W^T  →  gemm_nt(x, W, out, N, out_features, in_features)
        launch_gemm_nt(x, W.data(), out, N, out_features, in_features, stream);
    }

    // backward: compute d_x and accumulate d_W
    //   d_out : device pointer to (N, out_features) — upstream gradient
    //   d_x   : device pointer to (N, in_features)  — output gradient (overwritten)
    void backward(const float* d_out, float* d_x,
                  cudaStream_t stream = 0) {
        // d_x = d_out * W  →  gemm_tiled(d_out, W, d_x, N, in_features, out_features)
        launch_gemm_tiled(d_out, W.data(), d_x,
                          N_cache, in_features, out_features, stream);

        // d_W += d_out^T * x  →  gemm_tn(d_out, x_cache, d_W, out_features, in_features, N)
        // Note: this overwrites d_W with d_out^T * x (not accumulates).
        // For multi-step accumulation, zero d_W externally or use atomicAdd.
        // During a single forward-backward step, d_W is computed fresh here.
        launch_gemm_tn(d_out, x_cache, d_W.data(),
                       out_features, in_features, N_cache, stream);
    }
};
