#include "kernels/pointwise.cuh"
#include "utils/cuda_check.hpp"

constexpr int PW_BLOCK = 256;

__global__ void add_inplace_kernel(float* a, const float* b, int n) {
    int i = blockIdx.x * PW_BLOCK + threadIdx.x;
    if (i < n) a[i] += b[i];
}

__global__ void scale_inplace_kernel(float* a, float s, int n) {
    int i = blockIdx.x * PW_BLOCK + threadIdx.x;
    if (i < n) a[i] *= s;
}

void launch_add_inplace(float* a, const float* b, int n, cudaStream_t stream) {
    dim3 block(PW_BLOCK);
    dim3 grid((n + PW_BLOCK - 1) / PW_BLOCK);
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
