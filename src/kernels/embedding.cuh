#pragma once

// ---------------------------------------------------------------------------
// Token embedding forward + backward (FP32, row-major).
//
// Forward: lookup — for each token in the sequence, copy the corresponding
//   embedding row from the weight table.
//     input_ids : (N,) int32, values in [0, V)
//     weight    : (V, D) — embedding table
//     out       : (N, D) — output embeddings
//
// Backward: scatter-add — for each token, add its upstream gradient row
//   into the corresponding weight gradient row. Because multiple tokens may
//   map to the same vocabulary index, we use atomicAdd.
//     d_out     : (N, D) — upstream gradient
//     d_weight  : (V, D) — gradient w.r.t. weight (caller must zero first)
//     input_ids : (N,)   — same token indices used in forward
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

void launch_embedding_forward(const int*   input_ids,
                              const float* weight,
                              float*       out,
                              int N, int V, int D,
                              cudaStream_t stream = 0);

void launch_embedding_backward(const float* d_out,
                               const int*   input_ids,
                               float*       d_weight,
                               int N, int V, int D,
                               cudaStream_t stream = 0);

// Positional embedding: adds pos_embed[n % S, :] to out[n, :] in-place.
void launch_pos_embed_forward(const float* pos_embed, float* out,
                              int B, int S, int C,
                              cudaStream_t stream = 0);

// Positional embedding backward: d_pos_embed[s, :] += sum_b d_out[b*S+s, :]
void launch_pos_embed_backward(const float* d_out, float* d_pos_embed,
                               int B, int S, int C,
                               cudaStream_t stream = 0);
