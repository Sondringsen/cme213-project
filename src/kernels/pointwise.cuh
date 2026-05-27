#pragma once

// ---------------------------------------------------------------------------
// Simple pointwise element-wise operations used by layer backward passes.
//
// launch_add_inplace:  a[i] += b[i]
// launch_scale_inplace: a[i] *= s
// launch_fill:         a[i]  = val
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

void launch_add_inplace(float* a, const float* b, int n, cudaStream_t stream = 0);
void launch_scale_inplace(float* a, float s, int n, cudaStream_t stream = 0);
void launch_fill(float* a, float val, int n, cudaStream_t stream = 0);
