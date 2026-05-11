#pragma once

// ---------------------------------------------------------------------------
// Public API for our hand-written GEMM kernels.
//
// All matrices are FP32, row-major, and live on the device. We compute
//     C = A * B
// with A: M x K, B: K x N, C: M x N. There is intentionally no alpha/beta
// (cuBLAS-style C = alpha*A*B + beta*C); we add that only when we need it.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

// Launch the tiled GEMM kernel.
//   dA, dB, dC : device pointers
//   M, N, K    : matrix dimensions (see header comment)
//   stream     : CUDA stream (default 0 = the default stream)
//
// Returns immediately; the kernel is asynchronous. Call cudaStreamSynchronize
// (or rely on a subsequent blocking call) to wait for completion.
void launch_gemm_tiled(const float* dA, const float* dB, float* dC,
                       int M, int N, int K, cudaStream_t stream = 0);
