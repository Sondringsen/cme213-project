#pragma once

// ---------------------------------------------------------------------------
// Public API: per-token cross-entropy loss, forward and backward.
//
// Forward:
//   Inputs:  logits  (N, V) FP32 row-major
//            targets (N,)   int32, each in [0, V)
//   Output:  losses  (N,)   FP32, one loss value per token
//   Computes loss[n] = -log(softmax(logits[n])[targets[n]]) via the LSE trick.
//
// Backward:
//   Given upstream gradient dlosses (N,) — one scalar per token loss —
//   writes the gradient w.r.t. logits:
//       dlogits[n, j] = dlosses[n] * (softmax(logits[n])[j] - 1{j == targets[n]})
//   This is a 3-pass kernel: max, sum-exp, then write N*V dlogits values.
//   The gradient is NOT divided by N; callers set dlosses[n] = 1/N to average.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

void launch_cross_entropy_forward(const float* dlogits,
                                  const int*   dtargets,
                                  float*       dlosses,
                                  int N, int V,
                                  cudaStream_t stream = 0);

void launch_cross_entropy_backward(const float* logits,
                                   const int*   targets,
                                   const float* dlosses,
                                   float*       dlogits,
                                   int N, int V,
                                   cudaStream_t stream = 0);
