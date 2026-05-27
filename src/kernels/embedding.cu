#include "kernels/embedding.cuh"
#include "utils/cuda_check.hpp"

// ===========================================================================
// Token embedding: forward (gather) and backward (scatter-add).
//
// Both kernels assign one thread block per token. Each thread within the
// block handles stride D/BLOCK elements of the D-dimensional embedding row.
// This gives coalesced access along the D dimension for both read and write.
//
// Backward uses atomicAdd because multiple tokens in a batch can share the
// same vocabulary index. AtomicAdd on float is non-deterministic in order
// but correct in magnitude (each update is applied exactly once).
// ===========================================================================

constexpr int EMB_BLOCK = 256;

// Forward: out[n, :] = weight[input_ids[n], :]
__global__ void embedding_forward_kernel(const int*   __restrict__ input_ids,
                                         const float* __restrict__ weight,
                                         float*       __restrict__ out,
                                         int N, int D) {
    int n   = blockIdx.x;
    int tid = threadIdx.x;
    if (n >= N) return;

    int vid = input_ids[n];
    const float* w_row  = weight + vid * D;
    float*       o_row  = out    + n   * D;

    for (int d = tid; d < D; d += EMB_BLOCK) o_row[d] = w_row[d];
}

// Backward: d_weight[input_ids[n], :] += d_out[n, :]
__global__ void embedding_backward_kernel(const float* __restrict__ d_out,
                                          const int*   __restrict__ input_ids,
                                          float*       __restrict__ d_weight,
                                          int N, int D) {
    int n   = blockIdx.x;
    int tid = threadIdx.x;
    if (n >= N) return;

    int vid = input_ids[n];
    const float* g_row  = d_out    + n   * D;
    float*       dw_row = d_weight + vid * D;

    for (int d = tid; d < D; d += EMB_BLOCK) atomicAdd(&dw_row[d], g_row[d]);
}

void launch_embedding_forward(const int* input_ids, const float* weight,
                              float* out, int N, int V, int D,
                              cudaStream_t stream) {
    (void)V;  // not needed at launch time (bounds checking omitted for speed)
    dim3 grid(N);
    dim3 block(EMB_BLOCK);
    embedding_forward_kernel<<<grid, block, 0, stream>>>(input_ids, weight, out, N, D);
    CUDA_CHECK_LAST();
}

void launch_embedding_backward(const float* d_out, const int* input_ids,
                               float* d_weight, int N, int V, int D,
                               cudaStream_t stream) {
    (void)V;
    dim3 grid(N);
    dim3 block(EMB_BLOCK);
    embedding_backward_kernel<<<grid, block, 0, stream>>>(d_out, input_ids, d_weight, N, D);
    CUDA_CHECK_LAST();
}
