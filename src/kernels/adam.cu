#include "kernels/adam.cuh"
#include "utils/cuda_check.hpp"

// ===========================================================================
// Fused Adam optimizer update kernel.
//
// One thread per parameter. All five arrays (param, grad, m, v, and the
// output param) are accessed in a single pass — no extra global memory
// round-trips compared to separate kernels for m-update, v-update, and
// param-update.
//
// Why bias correction?
//   At step t=1, m is initialized to 0 so m = (1-beta1)*g underestimates
//   the true first moment by a factor of (1-beta1^t). Dividing by
//   (1-beta1^t) corrects for this warm-up bias. Without it, the first few
//   steps would use a very small effective learning rate.
//
// Hardware note:
//   rsqrtf() uses the SFU (Special Function Unit) on NVIDIA GPUs — roughly
//   the same throughput as a multiply. It's slightly less accurate than
//   1/sqrtf() but sufficient for optimization.
// ===========================================================================

constexpr int ADAM_BLOCK = 256;

__global__ void adam_kernel(float*       __restrict__ param,
                            const float* __restrict__ grad,
                            float*       __restrict__ m,
                            float*       __restrict__ v,
                            float lr,
                            float beta1, float beta2, float eps,
                            float bc1,   // bias correction 1: 1 / (1 - beta1^t)
                            float bc2,   // bias correction 2: 1 / (1 - beta2^t)
                            int n) {
    int i = blockIdx.x * ADAM_BLOCK + threadIdx.x;
    if (i >= n) return;

    float g  = grad[i];
    float mi = beta1 * m[i] + (1.0f - beta1) * g;
    float vi = beta2 * v[i] + (1.0f - beta2) * g * g;

    m[i] = mi;
    v[i] = vi;

    float m_hat = mi * bc1;
    float v_hat = vi * bc2;

    // param -= lr * m_hat / sqrt(v_hat + eps)
    param[i] -= lr * m_hat * rsqrtf(v_hat + eps);
}

void launch_adam(float* param, const float* grad, float* m, float* v,
                 float lr, float beta1, float beta2, float eps,
                 int t, int n, cudaStream_t stream) {
    // Precompute bias corrections on the host to avoid per-thread powf() calls.
    float bc1 = 1.0f / (1.0f - powf(beta1, static_cast<float>(t)));
    float bc2 = 1.0f / (1.0f - powf(beta2, static_cast<float>(t)));

    dim3 block(ADAM_BLOCK);
    dim3 grid((n + ADAM_BLOCK - 1) / ADAM_BLOCK);
    adam_kernel<<<grid, block, 0, stream>>>(
        param, grad, m, v, lr, beta1, beta2, eps, bc1, bc2, n);
    CUDA_CHECK_LAST();
}
