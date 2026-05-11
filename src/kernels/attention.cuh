#pragma once

// ---------------------------------------------------------------------------
// Public API: Flash Attention forward (FP32, single-GPU).
//
// Inputs (all device pointers, row-major, FP32):
//   Q, K, V : shape (B, H, S, D)
// Output:
//   O       : shape (B, H, S, D)
//
//   B : batch size
//   H : number of attention heads
//   S : sequence length
//   D : head dimension (must be one of the values supported by the
//       dispatcher in attention.cu -- currently 32, 64, 96, 128)
//
// scale : multiplicative factor applied to QK^T before the softmax.
//         Standard attention uses 1/sqrt(D); pass it explicitly so callers
//         can experiment with other choices.
// causal: if true, each query position only attends to keys at positions
//         <= its own (autoregressive language modelling).
//
// The output is written in-place to O; Q, K, V are not modified.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

void launch_flash_attention_forward(const float* dQ,
                                    const float* dK,
                                    const float* dV,
                                    float* dO,
                                    int B, int H, int S, int D,
                                    float scale, bool causal,
                                    cudaStream_t stream = 0);
