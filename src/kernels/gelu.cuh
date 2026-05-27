#pragma once

// ---------------------------------------------------------------------------
// Public API: GELU forward and backward (exact erf formulation, FP32).
//
// Forward:
//   gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))   [F.gelu(x, approximate='none')]
//
// Backward:
//   Given upstream gradient dy, computes dx = dy * gelu'(x) where:
//   gelu'(x) = 0.5*(1+erf(x/sqrt(2))) + x*(1/sqrt(2*pi))*exp(-x^2/2)
//            = gelu(x)/x + x * phi(x)   (phi = standard normal PDF)
//
// Both kernels operate element-wise on a flat 1-D array of n_elements floats.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

void launch_gelu_forward(const float* dx, float* dy,
                         int n_elements, cudaStream_t stream = 0);

void launch_gelu_backward(const float* dy, const float* x,
                          float* dx_out, int n_elements,
                          cudaStream_t stream = 0);
