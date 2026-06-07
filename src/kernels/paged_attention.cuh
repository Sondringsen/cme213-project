#pragma once

// ---------------------------------------------------------------------------
// PagedAttention decode kernel (inference, FP32).
//
// Reference: Kwon et al., "Efficient Memory Management for Large Language Model
// Serving with PagedAttention", SOSP 2023 (the vLLM paper).
//
// What this is for
// ----------------
// During autoregressive decoding, each new token attends to the KV cache for
// every previous token in its sequence. With B concurrent requests of variable
// length, naively allocating B * S_max contiguous KV buffers wastes ~50% of
// memory on padding. PagedAttention splits the cache into fixed-size blocks
// managed from a global pool, and each sequence holds a small block table
// mapping its logical positions to physical block ids.
//
// This is NOT the training Flash Attention kernel. It is the decode-step
// kernel: one new query per sequence, attending to the full historical KV.
//
// Layout
// ------
//   Q            : float [B, H, D]                          one new query per seq
//   K_blocks     : float [N_blocks, H, BLOCK_SIZE, D]       head-contiguous within block
//   V_blocks     : float [N_blocks, H, BLOCK_SIZE, D]
//   block_table  : int   [B, max_blocks_per_seq]            -1 for unused entries
//   seq_lens     : int   [B]                                current length per seq
//   O            : float [B, H, D]                          output
//
// Grid: (B, H). Block: 128 threads. Supported (D, BLOCK_SIZE):
//   D ∈ {16, 32, 64, 128},  BLOCK_SIZE ∈ {16}
//
// The dense_attention_decode launcher is a paired reference used by the
// benchmark to isolate the cost of the block-table indirection from the rest
// of the decode kernel. Same arithmetic, contiguous K/V layout.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

void launch_paged_attention_decode(const float* dQ,
                                   const float* dK_blocks,
                                   const float* dV_blocks,
                                   const int*   d_block_table,
                                   const int*   d_seq_lens,
                                   float*       dO,
                                   int B, int H, int D,
                                   int block_size, int max_blocks_per_seq,
                                   float scale,
                                   cudaStream_t stream = 0);

void launch_dense_attention_decode(const float* dQ,
                                   const float* dK,
                                   const float* dV,
                                   const int*   d_seq_lens,
                                   float*       dO,
                                   int B, int H, int D, int seq_max,
                                   float scale,
                                   cudaStream_t stream = 0);
