#pragma once

// ---------------------------------------------------------------------------
// Public API: numerically-stable softmax forward, applied along the last
// dimension of an (N, V) input.
//
//   x : (N, V)
//   y : (N, V)
//
// For each row,
//   m       = max(x)
//   y[i]    = exp(x[i] - m) / sum_j exp(x[j] - m)
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

void launch_softmax_forward(const float* dx, float* dy,
                            int N, int V, cudaStream_t stream = 0);
