#include "kernels/cross_entropy.cuh"
#include "utils/cuda_check.hpp"

// ===========================================================================
// Cross-entropy loss forward + backward.
//
// Forward uses the log-sum-exp (LSE) trick: we need only max and sum-exp per
// row, not the full softmax vector. See cross_entropy.cuh for the math.
//
// Backward: given dlosses[n] (the upstream gradient for each token's scalar
// loss), we need to write:
//     dlogits[n, j] = dlosses[n] * (softmax(logits[n])[j] - 1{j == targets[n]})
//
// This requires knowing the softmax values, which means re-doing max + sum-exp.
// We then stream over the V logits a third time to write dlogits — three passes
// total (max, sum-exp, write). Each pass is coalesced and cache-friendly.
// ===========================================================================

constexpr int CE_BLOCK = 256;

// ---- Forward kernel (unchanged from Milestone 3) ----
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

    // Pass 1: per-block max
    float local_max = -INFINITY;
    for (int i = tid; i < V; i += CE_BLOCK) local_max = fmaxf(local_max, row_logits[i]);
    s_data[tid] = local_max;
    __syncthreads();

    #pragma unroll
    for (int s = CE_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) s_data[tid] = fmaxf(s_data[tid], s_data[tid + s]);
        __syncthreads();
    }
    float row_max = s_data[0];
    __syncthreads();

    // Pass 2: sum of exp(x - max)
    float local_sum = 0.0f;
    for (int i = tid; i < V; i += CE_BLOCK) local_sum += __expf(row_logits[i] - row_max);
    s_data[tid] = local_sum;
    __syncthreads();

    #pragma unroll
    for (int s = CE_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) s_data[tid] += s_data[tid + s];
        __syncthreads();
    }
    float row_sum = s_data[0];

    if (tid == 0) {
        float lse = logf(row_sum) + row_max;
        losses[row] = lse - row_logits[target];
    }
}

// ---- Backward kernel ----
// One block per token (row). Three passes: max, sum-exp, write dlogits.
// This kernel writes N*V values, which is large for typical vocabs (V≈50k).
// We accept this cost because there's no way to avoid it: the gradient
// equals softmax(logits) elementwise minus the one-hot target, and softmax
// is defined over all V positions.
__global__ void cross_entropy_backward_kernel(const float* __restrict__ logits,
                                              const int*   __restrict__ targets,
                                              const float* __restrict__ dlosses,
                                              float*       __restrict__ dlogits,
                                              int N, int V) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float* row_logits  = logits  + row * V;
    float*       row_dlogits = dlogits + row * V;
    int target  = targets[row];
    float dloss = dlosses[row];  // upstream scalar gradient for this token
    int tid = threadIdx.x;

    __shared__ float s_data[CE_BLOCK];

    // Pass 1: find per-row max (for numerical stability in exp)
    float local_max = -INFINITY;
    for (int i = tid; i < V; i += CE_BLOCK) local_max = fmaxf(local_max, row_logits[i]);
    s_data[tid] = local_max;
    __syncthreads();

    #pragma unroll
    for (int s = CE_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) s_data[tid] = fmaxf(s_data[tid], s_data[tid + s]);
        __syncthreads();
    }
    float row_max = s_data[0];
    __syncthreads();

    // Pass 2: compute Z = sum_j exp(logit_j - max)
    float local_sum = 0.0f;
    for (int i = tid; i < V; i += CE_BLOCK) local_sum += __expf(row_logits[i] - row_max);
    s_data[tid] = local_sum;
    __syncthreads();

    #pragma unroll
    for (int s = CE_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) s_data[tid] += s_data[tid + s];
        __syncthreads();
    }
    float row_Z = s_data[0];

    // Pass 3: write dlogits[n, j] = dloss * (softmax_j - indicator)
    // softmax_j = exp(logit_j - max) / Z
    for (int i = tid; i < V; i += CE_BLOCK) {
        float softmax_i = __expf(row_logits[i] - row_max) / row_Z;
        float indicator = (i == target) ? 1.0f : 0.0f;
        row_dlogits[i] = dloss * (softmax_i - indicator);
    }
}

void launch_cross_entropy_forward(const float* dlogits, const int* dtargets,
                                  float* dlosses, int N, int V,
                                  cudaStream_t stream) {
    dim3 grid(N);
    dim3 block(CE_BLOCK);
    cross_entropy_forward_kernel<<<grid, block, 0, stream>>>(
        dlogits, dtargets, dlosses, N, V);
    CUDA_CHECK_LAST();
}

void launch_cross_entropy_backward(const float* logits, const int* targets,
                                   const float* dlosses, float* dlogits,
                                   int N, int V, cudaStream_t stream) {
    dim3 grid(N);
    dim3 block(CE_BLOCK);
    cross_entropy_backward_kernel<<<grid, block, 0, stream>>>(
        logits, targets, dlosses, dlogits, N, V);
    CUDA_CHECK_LAST();
}
