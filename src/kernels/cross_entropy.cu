#include "kernels/cross_entropy.cuh"
#include "utils/cuda_check.hpp"

// ===========================================================================
// Cross-entropy loss forward, with the log-sum-exp (LSE) trick.
//
// Naive: probs = softmax(logits); loss = -log(probs[target]).
// This materializes the full V-wide probability vector per token, then
// reads one entry. Wasteful.
//
// LSE identity, used here:
//     softmax(x)[t] = exp(x[t] - m) / sum_j exp(x[j] - m)
//     log softmax(x)[t] = (x[t] - m) - log sum_j exp(x[j] - m)
//     -log softmax(x)[t] = log sum_j exp(x[j] - m) + m - x[t]
//                       = LSE(x) - x[t]
//
// So we only need two reductions over V (max, then sum-exp) and one
// scalar write per token. No V-wide intermediate.
//
// Memory: read V logits + read 1 target -> write 1 loss. Bandwidth-bound.
// At N >> V/peak-bw-per-block it should hit near-peak HBM bandwidth.
// ===========================================================================

constexpr int CE_BLOCK = 256;

/*
 * cross_entropy_forward_kernel
 *
 * Thread/block layout:
 *   gridDim  = (N,)         one block per token
 *   blockDim = (CE_BLOCK,)  256 threads per token
 *
 * Each thread strides over the vocab axis in steps of CE_BLOCK, doing
 * coalesced loads. Two shared-memory tree reductions give us m and Z.
 * Thread 0 alone writes the scalar loss.
 *
 * The kernel also handles the "fused gradient" path naturally -- once we
 * have m and Z, ∂L/∂logits[n,j] = exp(logits[n,j]-m)/Z - 1[j==target] is
 * cheap to write out. We'll add that output buffer when we wire up the
 * backward pass in Milestone 4.
 */
__global__ void cross_entropy_forward_kernel(const float* __restrict__ logits,
                                             const int*   __restrict__ targets,
                                             float*       __restrict__ losses,
                                             int N, int V) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float* row_logits = logits + row * V;
    int   target            = targets[row];
    int   tid               = threadIdx.x;

    __shared__ float s_data[CE_BLOCK];

    // ---- Pass 1: per-thread max, then block-wide max ----
    float local_max = -INFINITY;
    for (int i = tid; i < V; i += CE_BLOCK) {
        local_max = fmaxf(local_max, row_logits[i]);
    }
    s_data[tid] = local_max;
    __syncthreads();

    #pragma unroll
    for (int s = CE_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) s_data[tid] = fmaxf(s_data[tid], s_data[tid + s]);
        __syncthreads();
    }
    float row_max = s_data[0];
    __syncthreads();  // protect s_data before pass 2 reuse

    // ---- Pass 2: per-thread sum of exp(x - max), then block-wide sum ----
    float local_sum = 0.0f;
    for (int i = tid; i < V; i += CE_BLOCK) {
        local_sum += __expf(row_logits[i] - row_max);
    }
    s_data[tid] = local_sum;
    __syncthreads();

    #pragma unroll
    for (int s = CE_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) s_data[tid] += s_data[tid + s];
        __syncthreads();
    }
    float row_sum = s_data[0];

    // ---- Scalar write of the per-token loss ----
    if (tid == 0) {
        float lse = logf(row_sum) + row_max;
        losses[row] = lse - row_logits[target];
    }
}

void launch_cross_entropy_forward(const float* dlogits,
                                  const int*   dtargets,
                                  float*       dlosses,
                                  int N, int V,
                                  cudaStream_t stream) {
    dim3 grid(N);
    dim3 block(CE_BLOCK);
    cross_entropy_forward_kernel<<<grid, block, 0, stream>>>(
        dlogits, dtargets, dlosses, N, V);
    CUDA_CHECK_LAST();
}
