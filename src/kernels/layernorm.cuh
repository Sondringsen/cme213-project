#pragma once

// ---------------------------------------------------------------------------
// Public API: fused LayerNorm forward.
//
// Computes, per row of an (N, H) input,
//     y = gamma * (x - mean) / sqrt(var + eps) + beta
// where mean and var are taken along the H dimension.
//
// All buffers are FP32 device pointers.
//   x     : (N, H)
//   gamma : (H,)
//   beta  : (H,)
//   y     : (N, H)
// eps is the usual numerical guard (typically 1e-5).
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

void launch_layernorm_forward(const float* dx,
                              const float* dgamma,
                              const float* dbeta,
                              float* dy,
                              int N, int H, float eps,
                              cudaStream_t stream = 0);
