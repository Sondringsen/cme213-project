#include "kernels/layernorm.cuh"
#include "utils/cuda_check.hpp"

// ===========================================================================
// Fused LayerNorm forward kernel.
//
// LayerNorm is intrinsically memory-bound: it does O(H) flops per element
// but moves the entire input tensor through global memory. The win from
// "fusing" is reducing the *number of global memory passes*, not the
// flop count.
//
// Naive (3-pass) implementation:
//     pass 1: compute mean per row -> global write
//     pass 2: compute variance per row -> global write
//     pass 3: normalize using mean, variance -> global read + global write
// Total: 3 reads + 3 writes of the input/output tensors per element.
//
// Our fused (single-kernel) version:
//     pass 1 (within the kernel): read x, accumulate sum and sum-of-squares,
//                                 block-reduce, derive mean and rstd
//     pass 2 (within the kernel): re-read x, write normalized output
// Total: 2 reads + 1 write. Same flops, ~half the memory traffic.
//
// We could go further (cache x in shared memory between the two passes,
// reaching 1 read + 1 write) but for H = 768 the saving is small and the
// extra shared-memory pressure can hurt occupancy. Future optimization.
// ===========================================================================

// 256 threads/block: enough warps (8) to keep the SM busy and to amortize
// the block-level reduction, and small enough that we get good occupancy.
constexpr int LN_BLOCK = 256;

/*
 * layernorm_forward_kernel
 *
 * Thread/block layout:
 *   gridDim  = (N,)             one block per row of the input
 *   blockDim = (LN_BLOCK,)      256 threads per row
 *
 * Each thread strides across the H elements of its row with step LN_BLOCK,
 * so for H = 768 each thread touches 3 elements. The strided pattern keeps
 * consecutive threads on consecutive addresses, so the global loads are
 * coalesced.
 *
 * We accumulate the per-thread partial sums in registers, then collapse
 * them to a single scalar via a classic tree reduction in shared memory:
 *   for s = BLOCK/2, BLOCK/4, ... 1:
 *       if tid < s: s_data[tid] += s_data[tid + s]
 *       __syncthreads()
 * This runs in log2(BLOCK) = 8 steps.
 *
 * Numerical note: we compute var = E[x^2] - (E[x])^2. This is fine in FP32
 * for the layer sizes we care about (H up to a few thousand), but it can
 * lose precision badly in BF16. When we switch to BF16 we should swap to
 * Welford's online variance for that reason.
 */
__global__ void layernorm_forward_kernel(const float* __restrict__ x,
                                         const float* __restrict__ gamma,
                                         const float* __restrict__ beta,
                                         float* __restrict__ y,
                                         int N, int H, float eps) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float* row_x = x + row * H;
    float*       row_y = y + row * H;

    int tid = threadIdx.x;

    // ---- Pass 1: per-thread partial sum and sum-of-squares ----
    float sum    = 0.0f;
    float sum_sq = 0.0f;
    for (int i = tid; i < H; i += LN_BLOCK) {
        float v = row_x[i];
        sum    += v;
        sum_sq += v * v;
    }

    // ---- Block-level reduction in shared memory ----
    // Two parallel reductions (sum and sum-of-squares). We share a single
    // 2*LN_BLOCK array rather than declaring two separate ones for clarity
    // of memory layout.
    __shared__ float s_sum[LN_BLOCK];
    __shared__ float s_sq[LN_BLOCK];
    s_sum[tid] = sum;
    s_sq[tid]  = sum_sq;
    __syncthreads();

    // Tree reduction: in each step, the lower half of active threads pulls
    // in a value from the upper half. After log2(LN_BLOCK) steps, s_sum[0]
    // holds the total sum, s_sq[0] the total sum-of-squares.
    #pragma unroll
    for (int s = LN_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sq[tid]  += s_sq[tid + s];
        }
        __syncthreads();
    }

    // Every thread now reads the reduced values (broadcast from shared mem).
    float mean = s_sum[0] / H;
    float var  = s_sq[0]  / H - mean * mean;
    // rsqrtf is a hardware intrinsic; faster than 1.0f / sqrtf().
    float rstd = rsqrtf(var + eps);

    // ---- Pass 2: write normalized + affine output ----
    // Same strided pattern, same coalescing. gamma and beta are reused
    // across all rows, so they'll mostly stay in L2 cache after the first
    // few blocks.
    for (int i = tid; i < H; i += LN_BLOCK) {
        float v = row_x[i];
        row_y[i] = gamma[i] * (v - mean) * rstd + beta[i];
    }
}

// Host launcher.
void launch_layernorm_forward(const float* dx,
                              const float* dgamma,
                              const float* dbeta,
                              float* dy,
                              int N, int H, float eps,
                              cudaStream_t stream) {
    dim3 grid(N);
    dim3 block(LN_BLOCK);
    layernorm_forward_kernel<<<grid, block, 0, stream>>>(
        dx, dgamma, dbeta, dy, N, H, eps);
    CUDA_CHECK_LAST();
}
