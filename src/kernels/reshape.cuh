#pragma once

// ---------------------------------------------------------------------------
// Layout permutation helpers for the attention layer.
//
// The model uses (B*S, C) flat layout for activations (C = H*D).
// The attention kernel uses (B, H, S, D) layout.
// These two kernels convert between them.
//
// launch_flat_to_bhsd : (B*S, C) → (B, H, S, D)
//   flat[b*S+s][h*D+d] → bhsd[b][h][s][d]
//
// launch_bhsd_to_flat : (B, H, S, D) → (B*S, C)
//   bhsd[b][h][s][d] → flat[b*S+s][h*D+d]
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

void launch_flat_to_bhsd(const float* flat, float* bhsd,
                         int B, int H, int S, int D,
                         cudaStream_t stream = 0);

void launch_bhsd_to_flat(const float* bhsd, float* flat,
                         int B, int H, int S, int D,
                         cudaStream_t stream = 0);
