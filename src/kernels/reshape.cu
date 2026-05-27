#include "kernels/reshape.cuh"
#include "utils/cuda_check.hpp"

// ===========================================================================
// Reshape / permutation kernels for the attention layer.
//
// Both kernels assign one thread per element and decode the linear index into
// (b, h, s, d) coordinates using integer division — no shared memory needed.
// The workload is perfectly load-balanced; performance is limited by HBM BW.
// ===========================================================================

constexpr int RESHAPE_BLOCK = 256;

// (B*S, C) → (B, H, S, D) where C = H*D
__global__ void flat_to_bhsd_kernel(const float* __restrict__ flat,
                                    float*       __restrict__ bhsd,
                                    int B, int H, int S, int D) {
    int idx = blockIdx.x * RESHAPE_BLOCK + threadIdx.x;
    int total = B * H * S * D;
    if (idx >= total) return;

    // Decode from linearized (b, h, s, d) index
    int tmp = idx;
    int d   = tmp % D; tmp /= D;
    int s   = tmp % S; tmp /= S;
    int h   = tmp % H;
    int b   = tmp / H;

    int C = H * D;
    // flat layout: [b*S+s][h*D+d]  →  flat[(b*S+s)*C + h*D + d]
    bhsd[idx] = flat[(b * S + s) * C + h * D + d];
}

// (B, H, S, D) → (B*S, C) where C = H*D
__global__ void bhsd_to_flat_kernel(const float* __restrict__ bhsd,
                                    float*       __restrict__ flat,
                                    int B, int H, int S, int D) {
    int idx = blockIdx.x * RESHAPE_BLOCK + threadIdx.x;
    int total = B * H * S * D;
    if (idx >= total) return;

    int tmp = idx;
    int d   = tmp % D; tmp /= D;
    int s   = tmp % S; tmp /= S;
    int h   = tmp % H;
    int b   = tmp / H;

    int C = H * D;
    flat[(b * S + s) * C + h * D + d] = bhsd[idx];
}

void launch_flat_to_bhsd(const float* flat, float* bhsd,
                         int B, int H, int S, int D, cudaStream_t stream) {
    int total = B * H * S * D;
    dim3 block(RESHAPE_BLOCK);
    dim3 grid((total + RESHAPE_BLOCK - 1) / RESHAPE_BLOCK);
    flat_to_bhsd_kernel<<<grid, block, 0, stream>>>(flat, bhsd, B, H, S, D);
    CUDA_CHECK_LAST();
}

void launch_bhsd_to_flat(const float* bhsd, float* flat,
                         int B, int H, int S, int D, cudaStream_t stream) {
    int total = B * H * S * D;
    dim3 block(RESHAPE_BLOCK);
    dim3 grid((total + RESHAPE_BLOCK - 1) / RESHAPE_BLOCK);
    bhsd_to_flat_kernel<<<grid, block, 0, stream>>>(bhsd, flat, B, H, S, D);
    CUDA_CHECK_LAST();
}
