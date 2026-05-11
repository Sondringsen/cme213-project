#pragma once

// ---------------------------------------------------------------------------
// Public API: per-token cross-entropy loss for language modelling.
//
// Inputs:
//   logits  : (N, V) FP32 row-major
//   targets : (N,)   int32, each in [0, V)
// Output:
//   losses  : (N,)   FP32, one loss value per token
//
// We compute    loss[n] = -log(softmax(logits[n])[targets[n]])
// without ever materializing the softmax probabilities, using the
// log-sum-exp identity:
//     loss[n] = log(sum_j exp(logits[n,j] - m)) + m - logits[n, targets[n]]
// where m = max_j logits[n,j].
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

void launch_cross_entropy_forward(const float* dlogits,
                                  const int* dtargets,
                                  float* dlosses,
                                  int N, int V,
                                  cudaStream_t stream = 0);
