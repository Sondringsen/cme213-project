#include "kernels/gelu.cuh"
#include "utils/cuda_check.hpp"

// ===========================================================================
// GELU forward kernel.
//
// What is GELU?
// -------------
// GELU (Hendrycks & Gimpel, 2016) is the activation function used in every
// GPT-2 FFN layer. It is defined as:
//
//     gelu(x) = x * Phi(x)
//
// where Phi is the standard normal CDF. Using the identity
// Phi(x) = 0.5 * (1 + erf(x / sqrt(2))), we get the exact form:
//
//     gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
//
// This is what PyTorch computes with F.gelu(x, approximate='none'), which
// is the version we implement here for exact numerical agreement.
//
// Why not the tanh approximation?
// --------------------------------
// GPT-2's original code used the tanh approximation:
//     gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))
// This was a workaround for GPUs that lacked fast erff(). Modern GPUs have
// hardware erff() through CUDA's special function unit (SFU), so the exact
// form is both accurate and fast.
//
// Performance characteristics
// ---------------------------
// GELU is purely pointwise: one input read, one output write, ~6 flops.
// Arithmetic intensity ≈ 6 / 8 = 0.75 flop/byte -- well below the
// compute-to-bandwidth ratio of any modern GPU. The kernel is firmly
// memory-bandwidth bound. The limiting factor in practice is erff() SFU
// throughput, not DRAM bandwidth.
//
// Optimization opportunities (not done yet)
// ------------------------------------------
// - float4 vector loads: read 4 floats per thread in a single 128-bit
//   transaction, reducing the number of memory instructions by 4x.
// - The fused GELU (not a separate kernel but part of a wider GEMM+GELU
//   fused op) would eliminate the extra round-trip to HBM; that's a
//   follow-up for the MLP layer.
// ===========================================================================

constexpr int GELU_BLOCK = 256;

// 1/sqrt(2), precomputed to avoid repeated division in the inner loop.
static constexpr float kInvSqrt2 = 0.7071067811865476f;

/*
 * gelu_forward_kernel
 *
 * Thread/block layout:
 *   blockDim = (GELU_BLOCK,)
 *   gridDim  = (ceil(n / GELU_BLOCK),)
 *
 * Each thread handles exactly one element. Threads beyond the end of the
 * array are guarded with an early return.
 *
 * erff() is the single-precision variant of erf(). On current NVIDIA GPUs
 * it maps to the SFU and has ~19 cycles of throughput latency.
 */
__global__ void gelu_forward_kernel(const float* __restrict__ x,
                                    float*       __restrict__ y,
                                    int n) {
    int i = blockIdx.x * GELU_BLOCK + threadIdx.x;
    if (i >= n) return;

    float xi = x[i];
    // Multiply 0.5f last to keep the intermediate value scaled near 1.0,
    // reducing relative rounding error compared to computing (0.5*xi) first.
    y[i] = xi * 0.5f * (1.0f + erff(xi * kInvSqrt2));
}

void launch_gelu_forward(const float* dx, float* dy,
                         int n_elements, cudaStream_t stream) {
    dim3 block(GELU_BLOCK);
    dim3 grid((n_elements + GELU_BLOCK - 1) / GELU_BLOCK);
    gelu_forward_kernel<<<grid, block, 0, stream>>>(dx, dy, n_elements);
    CUDA_CHECK_LAST();
}
