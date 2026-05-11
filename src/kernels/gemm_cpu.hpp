#pragma once

// ---------------------------------------------------------------------------
// CPU reference implementation of single-precision GEMM.
//
// This exists *only* to validate the GPU kernels: we run the GPU output and
// the CPU output on the same inputs and compare. We deliberately do not
// optimize it (no BLAS, no SIMD, no parallelism) -- it should be obvious by
// inspection that the CPU answer is correct.
// ---------------------------------------------------------------------------

// Computes C = A * B with A: M x K, B: K x N, C: M x N, all row-major FP32.
// O(M*N*K). Not fast; not meant to be.
void gemm_cpu(const float* A, const float* B, float* C,
              int M, int N, int K);
