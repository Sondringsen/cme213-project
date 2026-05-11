#include "kernels/softmax.cuh"
#include "utils/cuda_check.hpp"

// ===========================================================================
// Row-wise numerically stable softmax.
//
// y[i] = exp(x[i] - m) / Z,   m = max_j x[j],   Z = sum_j exp(x[j] - m)
//
// The "- m" shift is mandatory: with no shift, exp(x[i]) overflows for any
// x[i] > ~88 in FP32. The shift leaves the result mathematically identical
// (the m cancels in the ratio) but keeps every exp argument <= 0, so the
// result is in [0, 1] and well-behaved.
//
// We do *three* sweeps over each row:
//   1. max-reduce to find m
//   2. sum-reduce of exp(x - m) to find Z
//   3. write y[i] = exp(x[i] - m) / Z
// We can't fold passes 2 and 3 (the divide needs the full sum), and we
// can't cache x in shared memory because V can be huge (vocab = 30k =>
// 120 KB per row, well over the per-block shared-mem budget). So we eat
// the extra exp call in pass 3.
// ===========================================================================

constexpr int SM_BLOCK = 256;

// Helper: block-wide reduction of `val` using shared-memory tree.
// `op` is "max" or "sum" via a template parameter. After this returns,
// every thread sees the same reduced value (written to s_data[0]).
template <typename Op>
__device__ inline float block_reduce(float val, float* s_data, Op op) {
    int tid = threadIdx.x;
    s_data[tid] = val;
    __syncthreads();

    #pragma unroll
    for (int s = SM_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) s_data[tid] = op(s_data[tid], s_data[tid + s]);
        __syncthreads();
    }
    return s_data[0];
}

/*
 * softmax_forward_kernel
 *
 * Thread/block layout:
 *   gridDim  = (N,)        one block per row
 *   blockDim = (SM_BLOCK,) 256 threads per row
 *
 * Each thread strides over its row in steps of SM_BLOCK, so the global
 * loads are coalesced (consecutive threads -> consecutive addresses).
 *
 * Performance shape: 3 reads of x + 1 write of y. ~16 bytes/element moved.
 * At V = 30k this kernel is firmly memory-bound and we expect it to hit
 * near-peak HBM bandwidth on modern GPUs.
 */
__global__ void softmax_forward_kernel(const float* __restrict__ x,
                                       float* __restrict__ y,
                                       int N, int V) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float* row_x = x + row * V;
    float*       row_y = y + row * V;
    int tid = threadIdx.x;

    __shared__ float s_data[SM_BLOCK];

    // ---- Pass 1: per-thread max, then block-wide max ----
    float local_max = -INFINITY;
    for (int i = tid; i < V; i += SM_BLOCK) {
        local_max = fmaxf(local_max, row_x[i]);
    }
    float row_max = block_reduce(local_max, s_data,
                                 [] __device__ (float a, float b) {
                                     return fmaxf(a, b);
                                 });
    // Note: block_reduce's final __syncthreads is inside the loop, so
    // s_data[0] is stable by the time every thread reads it.

    // ---- Pass 2: per-thread sum of exp(x-max), then block-wide sum ----
    float local_sum = 0.0f;
    for (int i = tid; i < V; i += SM_BLOCK) {
        local_sum += __expf(row_x[i] - row_max);
    }
    float row_sum = block_reduce(local_sum, s_data,
                                 [] __device__ (float a, float b) {
                                     return a + b;
                                 });

    // ---- Pass 3: write normalized output ----
    float inv_sum = 1.0f / row_sum;
    for (int i = tid; i < V; i += SM_BLOCK) {
        row_y[i] = __expf(row_x[i] - row_max) * inv_sum;
    }
}

void launch_softmax_forward(const float* dx, float* dy,
                            int N, int V, cudaStream_t stream) {
    dim3 grid(N);
    dim3 block(SM_BLOCK);
    softmax_forward_kernel<<<grid, block, 0, stream>>>(dx, dy, N, V);
    CUDA_CHECK_LAST();
}
