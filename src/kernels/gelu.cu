#include "kernels/gelu.cuh"
#include "utils/cuda_check.hpp"

// ===========================================================================
// GELU forward + backward kernels.
//
// Both are fully pointwise: one thread per element, no shared memory, no
// inter-thread communication. Performance is limited by HBM bandwidth for
// the reads/writes and by SFU throughput for erff() and expf().
//
// Backward formula:
//   gelu'(x) = 0.5 * (1 + erf(x/sqrt(2))) + x * phi(x)
// where phi(x) = (1/sqrt(2*pi)) * exp(-x^2/2) is the standard normal PDF.
// The two terms are the derivative of (x * Phi(x)) via product rule:
//   d/dx [x * Phi(x)] = Phi(x) + x * phi(x)
// with Phi(x) = 0.5*(1+erf(x/sqrt(2))).
// ===========================================================================

constexpr int GELU_BLOCK = 256;

static constexpr float kInvSqrt2     = 0.7071067811865476f;   // 1/sqrt(2)
static constexpr float kInvSqrt2Pi   = 0.3989422804014327f;   // 1/sqrt(2*pi)

__global__ void gelu_forward_kernel(const float* __restrict__ x,
                                    float*       __restrict__ y,
                                    int n) {
    int i = blockIdx.x * GELU_BLOCK + threadIdx.x;
    if (i >= n) return;
    float xi = x[i];
    y[i] = xi * 0.5f * (1.0f + erff(xi * kInvSqrt2));
}

// ---------------------------------------------------------------------------
// gelu_backward_kernel
//
// dx_out[i] = dy[i] * gelu'(x[i])
//
// We recompute gelu'(x) from x rather than caching the forward output, to
// avoid the extra memory needed to save intermediate activations. The cost is
// one extra erff() + expf() call per element, which is acceptable given that
// GELU backward is SFU-bound regardless.
// ---------------------------------------------------------------------------
__global__ void gelu_backward_kernel(const float* __restrict__ dy,
                                     const float* __restrict__ x,
                                     float*       __restrict__ dx_out,
                                     int n) {
    int i = blockIdx.x * GELU_BLOCK + threadIdx.x;
    if (i >= n) return;

    float xi = x[i];

    // Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
    float phi_x = 0.5f * (1.0f + erff(xi * kInvSqrt2));

    // phi_standard(x) = (1/sqrt(2*pi)) * exp(-x^2 / 2)
    float phi_std = kInvSqrt2Pi * expf(-0.5f * xi * xi);

    float gelu_prime = phi_x + xi * phi_std;
    dx_out[i] = dy[i] * gelu_prime;
}

void launch_gelu_forward(const float* dx, float* dy,
                         int n_elements, cudaStream_t stream) {
    dim3 block(GELU_BLOCK);
    dim3 grid((n_elements + GELU_BLOCK - 1) / GELU_BLOCK);
    gelu_forward_kernel<<<grid, block, 0, stream>>>(dx, dy, n_elements);
    CUDA_CHECK_LAST();
}

void launch_gelu_backward(const float* dy, const float* x,
                          float* dx_out, int n_elements,
                          cudaStream_t stream) {
    dim3 block(GELU_BLOCK);
    dim3 grid((n_elements + GELU_BLOCK - 1) / GELU_BLOCK);
    gelu_backward_kernel<<<grid, block, 0, stream>>>(dy, x, dx_out, n_elements);
    CUDA_CHECK_LAST();
}
