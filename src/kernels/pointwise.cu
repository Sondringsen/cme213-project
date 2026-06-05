#include "kernels/pointwise.cuh"
#include "utils/cuda_check.hpp"

constexpr int PW_BLOCK = 256;

__global__ void add_inplace_kernel(float* a, const float* b, int n) {
    int base = (blockIdx.x * PW_BLOCK + threadIdx.x) * 4;
    if (base >= n) return;

    if (base + 3 < n) {
        float4 a4 = reinterpret_cast<const float4*>(a)[base / 4];
        float4 b4 = reinterpret_cast<const float4*>(b)[base / 4];
        a4.x += b4.x; a4.y += b4.y; a4.z += b4.z; a4.w += b4.w;
        reinterpret_cast<float4*>(a)[base / 4] = a4;
    } else {
        for (int i = base; i < n; ++i) a[i] += b[i];
    }
}

__global__ void scale_inplace_kernel(float* a, float s, int n) {
    int i = blockIdx.x * PW_BLOCK + threadIdx.x;
    if (i < n) a[i] *= s;
}

void launch_add_inplace(float* a, const float* b, int n, cudaStream_t stream) {
    dim3 block(PW_BLOCK);
    dim3 grid((n + PW_BLOCK * 4 - 1) / (PW_BLOCK * 4));
    add_inplace_kernel<<<grid, block, 0, stream>>>(a, b, n);
    CUDA_CHECK_LAST();
}

void launch_scale_inplace(float* a, float s, int n, cudaStream_t stream) {
    dim3 block(PW_BLOCK);
    dim3 grid((n + PW_BLOCK - 1) / PW_BLOCK);
    scale_inplace_kernel<<<grid, block, 0, stream>>>(a, s, n);
    CUDA_CHECK_LAST();
}

__global__ void fill_kernel(float* a, float val, int n) {
    int i = blockIdx.x * PW_BLOCK + threadIdx.x;
    if (i < n) a[i] = val;
}

void launch_fill(float* a, float val, int n, cudaStream_t stream) {
    dim3 block(PW_BLOCK);
    dim3 grid((n + PW_BLOCK - 1) / PW_BLOCK);
    fill_kernel<<<grid, block, 0, stream>>>(a, val, n);
    CUDA_CHECK_LAST();
}
