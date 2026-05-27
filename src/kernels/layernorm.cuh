#pragma once

// ---------------------------------------------------------------------------
// Public API: fused LayerNorm forward + backward.
//
// Forward:
//   Computes, per row of an (N, H) input:
//       y = gamma * (x - mean) / sqrt(var + eps) + beta
//   Optionally writes mean[n] and rstd[n] (reciprocal std) per row.
//   These must be saved and passed to the backward pass for training.
//   Pass nullptr for mean_out / rstd_out if you don't need them.
//
// Backward:
//   Given upstream gradient dy (N, H) and the cached mean and rstd from
//   the forward, computes:
//       dx     (N, H): gradient w.r.t. x
//       dgamma (H,):   gradient w.r.t. gamma (caller must zero before call)
//       dbeta  (H,):   gradient w.r.t. beta  (caller must zero before call)
//
// All buffers are FP32 device pointers, row-major.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

void launch_layernorm_forward(const float* x,
                              const float* gamma,
                              const float* beta,
                              float* y,
                              int N, int H, float eps,
                              float* mean_out = nullptr,
                              float* rstd_out = nullptr,
                              cudaStream_t stream = 0);

void launch_layernorm_backward(const float* dy,
                               const float* x,
                               const float* gamma,
                               const float* mean,
                               const float* rstd,
                               float* dx,
                               float* dgamma,
                               float* dbeta,
                               int N, int H,
                               cudaStream_t stream = 0);
