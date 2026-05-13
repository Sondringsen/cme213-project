#pragma once

// ---------------------------------------------------------------------------
// Public API: GELU (Gaussian Error Linear Unit) forward, element-wise.
//
// Computes the exact formulation used by PyTorch (approximate='none'):
//     gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
//
// The input is treated as a flat 1-D array of n_elements FP32 values;
// callers can pass any shape as long as the pointer is contiguous.
//
// This is a purely pointwise kernel -- no communication between elements,
// no shared memory. Performance is limited by HBM bandwidth (reads x,
// writes y) and by the throughput of erff() on the target architecture.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

void launch_gelu_forward(const float* dx, float* dy,
                         int n_elements, cudaStream_t stream = 0);
