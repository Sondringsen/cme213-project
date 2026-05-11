#pragma once

// Reference CPU implementation of LayerNorm forward, for correctness checks.
//   x     : (N, H) row-major
//   gamma : (H,)
//   beta  : (H,)
//   y     : (N, H) row-major output
void layernorm_cpu(const float* x, const float* gamma, const float* beta,
                   float* y, int N, int H, float eps);
