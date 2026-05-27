#pragma once

// ---------------------------------------------------------------------------
// Fused Adam optimizer kernel (FP32 parameters, FP32 moments).
//
// Applies the Adam update rule in-place:
//     m  = beta1 * m + (1 - beta1) * g
//     v  = beta2 * v + (1 - beta2) * g^2
//     m_hat = m / (1 - beta1^t)
//     v_hat = v / (1 - beta2^t)
//     param -= lr * m_hat / (sqrt(v_hat) + eps)
//
// All arrays (param, grad, m, v) are flat FP32 device buffers of length n.
// The caller maintains m and v across steps (they are updated in-place).
// t is the current step number (1-indexed), used for bias correction.
//
// Typical usage:
//   For each parameter tensor:
//       launch_adam(param.data(), grad.data(), m.data(), v.data(),
//                   lr, beta1, beta2, eps, step, param.numel());
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

void launch_adam(float* param,
                 const float* grad,
                 float* m,
                 float* v,
                 float lr,
                 float beta1, float beta2, float eps,
                 int t,          // current step (1-indexed)
                 int n,          // number of elements
                 cudaStream_t stream = 0);
