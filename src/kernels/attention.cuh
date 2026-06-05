#pragma once

// ---------------------------------------------------------------------------
// Public API: Flash Attention forward + naive attention backward (FP32).
//
// Forward (Flash Attention — O(S*D) memory):
//   Inputs  Q, K, V : (B, H, S, D)
//   Output  O       : (B, H, S, D)
//   scale: pre-scale factor applied to QK^T (typically 1/sqrt(D))
//   causal: if true, token i only attends to positions 0..i
//
// Backward (naive — O(S^2) memory):
//   Inputs : Q, K, V from forward; dO upstream gradient; scale, causal flag
//   Outputs: dQ, dK, dV  each (B, H, S, D)
//   The backward materializes the full S×S attention weight matrix P per
//   (batch, head). This is correct but memory-heavy for long sequences.
//   A memory-efficient Flash Attention backward is planned for Milestone 5.
//
//   Temp buffers P and dP (each B*H*S*S floats) are allocated and freed
//   internally by launch_attention_backward.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

void launch_flash_attention_forward(const float* dQ,
                                    const float* dK,
                                    const float* dV,
                                    float* dO,
                                    int B, int H, int S, int D,
                                    float scale, bool causal,
                                    cudaStream_t stream = 0);

// P_buf and dP_buf must each point to B*H*S*S pre-allocated device floats.
// The caller owns these buffers; this function does not allocate or free them.
void launch_attention_backward(const float* Q,
                               const float* K,
                               const float* V,
                               const float* dO,
                               float* dQ,
                               float* dK,
                               float* dV,
                               int B, int H, int S, int D,
                               float scale, bool causal,
                               float* P_buf, float* dP_buf,
                               cudaStream_t stream = 0);
