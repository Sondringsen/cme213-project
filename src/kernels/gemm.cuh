#pragma once

// ---------------------------------------------------------------------------
// Public API for our hand-written GEMM kernels (FP32, row-major, device).
//
// Forward:
//   launch_gemm_tiled(A, B, C, M, N, K)  — C = A * B
//     A: M×K,  B: K×N,  C: M×N
//
// Backward helpers (used in backward passes of linear layers and attention):
//   launch_gemm_tn(A, B, C, M, N, K)  — C = A^T * B
//     A is stored as K×M,  B: K×N,  C: M×N
//
//   launch_gemm_nt(A, B, C, M, N, K)  — C = A * B^T
//     A: M×K,  B is stored as N×K,  C: M×N
//
// All launchers are asynchronous; synchronize with cudaStreamSynchronize.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

void launch_gemm_tiled(const float* dA, const float* dB, float* dC,
                       int M, int N, int K, cudaStream_t stream = 0);

// C = A^T * B  (A stored as K×M)
void launch_gemm_tn(const float* dA, const float* dB, float* dC,
                    int M, int N, int K, cudaStream_t stream = 0);

// C = A * B^T  (B stored as N×K)
void launch_gemm_nt(const float* dA, const float* dB, float* dC,
                    int M, int N, int K, cudaStream_t stream = 0);
