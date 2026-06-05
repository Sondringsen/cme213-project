#include "kernels/layernorm.cuh"
#include "utils/cuda_check.hpp"

// ===========================================================================
// Fused LayerNorm forward + backward kernels.
//
// Forward: y = gamma * (x - mean) / sqrt(var + eps) + beta  (per row of N×H)
//
// Why save mean and rstd?
// The backward pass needs (x - mean) * rstd (the "xhat" value) for each
// input element. Rather than re-reading x and recomputing mean/rstd, we save
// the two scalars per row (tiny: 2*N floats) in the forward pass. This halves
// the backward's global memory traffic compared to a full recomputation.
//
// Backward formula derivation:
//   Let xhat_i = (x_i - mean) * rstd,  y_i = gamma_i * xhat_i + beta_i
//   Define dnorm_i = gamma_i * dy_i   (gradient w.r.t. xhat_i via chain rule)
//
//   Chain rule through beta: dbeta_i = sum_n dy[n,i]
//   Chain rule through gamma: dgamma_i = sum_n dy[n,i] * xhat[n,i]
//   Chain rule through x (per row):
//     dx_i = rstd * (dnorm_i - mean_j(dnorm_j) - xhat_i * mean_j(dnorm_j * xhat_j))
//
//   The gamma weights MUST be included inside the mean terms — factoring
//   gamma_i outside is only valid when gamma is a scalar constant.
// ===========================================================================

constexpr int LN_BLOCK = 256;

// ---------------------------------------------------------------------------
// Forward kernel — same as before, extended to optionally write mean and rstd.
// ---------------------------------------------------------------------------
__global__ void layernorm_forward_kernel(const float* __restrict__ x,
                                         const float* __restrict__ gamma,
                                         const float* __restrict__ beta,
                                         float* __restrict__ y,
                                         float* __restrict__ mean_out,
                                         float* __restrict__ rstd_out,
                                         int N, int H, float eps) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float* row_x = x + row * H;
    float*       row_y = y + row * H;
    int tid = threadIdx.x;

    // ---- Pass 1: compute mean ----
    float sum = 0.0f;
    for (int i = tid; i < H; i += LN_BLOCK) sum += row_x[i];

    __shared__ float s_sum[LN_BLOCK];
    s_sum[tid] = sum;
    __syncthreads();

    #pragma unroll
    for (int s = LN_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) s_sum[tid] += s_sum[tid + s];
        __syncthreads();
    }
    float mean = s_sum[0] / H;

    // Optionally save mean for the backward pass.
    if (tid == 0 && mean_out) mean_out[row] = mean;

    // ---- Pass 2a: compute variance as E[(x - mean)^2] ----
    float var_sum = 0.0f;
    for (int i = tid; i < H; i += LN_BLOCK) {
        float d = row_x[i] - mean;
        var_sum += d * d;
    }
    s_sum[tid] = var_sum;
    __syncthreads();

    #pragma unroll
    for (int s = LN_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) s_sum[tid] += s_sum[tid + s];
        __syncthreads();
    }
    float rstd = rsqrtf(s_sum[0] / H + eps);

    // Optionally save rstd for the backward pass.
    if (tid == 0 && rstd_out) rstd_out[row] = rstd;

    // ---- Pass 2b: write normalized + affine output (float4 vectorized) ----
    // H is assumed to be a multiple of 4 (true for all realistic hidden sizes).
    // Each thread processes 4 consecutive elements per iteration for wider
    // memory transactions: 3 reads (x, gamma, beta) + 1 write (y) per element.
    for (int i = tid * 4; i + 3 < H; i += LN_BLOCK * 4) {
        float4 xi = reinterpret_cast<const float4*>(row_x)[i / 4];
        float4 gi = reinterpret_cast<const float4*>(gamma)[i / 4];
        float4 bi = reinterpret_cast<const float4*>(beta)[i / 4];
        float4 yi;
        yi.x = gi.x * (xi.x - mean) * rstd + bi.x;
        yi.y = gi.y * (xi.y - mean) * rstd + bi.y;
        yi.z = gi.z * (xi.z - mean) * rstd + bi.z;
        yi.w = gi.w * (xi.w - mean) * rstd + bi.w;
        reinterpret_cast<float4*>(row_y)[i / 4] = yi;
    }
    // Scalar tail for the rare case H % 4 != 0
    for (int i = (H / 4) * 4 + tid; i < H; i += LN_BLOCK) {
        row_y[i] = gamma[i] * (row_x[i] - mean) * rstd + beta[i];
    }
}

void launch_layernorm_forward(const float* x, const float* gamma, const float* beta,
                              float* y, int N, int H, float eps,
                              float* mean_out, float* rstd_out,
                              cudaStream_t stream) {
    dim3 grid(N);
    dim3 block(LN_BLOCK);
    layernorm_forward_kernel<<<grid, block, 0, stream>>>(
        x, gamma, beta, y, mean_out, rstd_out, N, H, eps);
    CUDA_CHECK_LAST();
}

// ---------------------------------------------------------------------------
// Backward kernel.
//
// Thread/block layout: same as forward — one block per row, LN_BLOCK threads.
// Each block:
//   1. Computes two reductions: sum_dy and sum_dy_xhat
//   2. Writes dx row by row using those two scalars
//   3. Accumulates dgamma and dbeta via atomicAdd across all rows
//
// The caller must zero dgamma and dbeta before calling this kernel.
// ---------------------------------------------------------------------------
__global__ void layernorm_backward_kernel(const float* __restrict__ dy,
                                          const float* __restrict__ x,
                                          const float* __restrict__ gamma,
                                          const float* __restrict__ mean,
                                          const float* __restrict__ rstd,
                                          float* __restrict__ dx,
                                          float* __restrict__ dgamma,
                                          float* __restrict__ dbeta,
                                          int N, int H) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float* row_dy = dy + row * H;
    const float* row_x  = x  + row * H;
    float*       row_dx = dx + row * H;
    int tid = threadIdx.x;

    float m = mean[row];
    float r = rstd[row];

    // ---- Pass 1: two simultaneous reductions ----
    // We need dnorm_i = gamma_i * dy_i (the gradient w.r.t. xhat_i).
    // sum_dy     = sum_j gamma_j * dy_j          (for the mean subtraction term)
    // sum_dyxhat = sum_j gamma_j * dy_j * xhat_j (for the xhat-scaled term)
    // Both must include gamma — factoring it outside would be wrong when
    // gamma is non-constant (i.e., always in practice).
    float local_sum_dy     = 0.0f;
    float local_sum_dyxhat = 0.0f;

    for (int i = tid; i < H; i += LN_BLOCK) {
        float xi   = row_x[i];
        float dyi  = row_dy[i];
        float gi   = gamma[i];
        float xhat = (xi - m) * r;
        local_sum_dy     += dyi * gi;
        local_sum_dyxhat += dyi * gi * xhat;
    }

    __shared__ float s_sum_dy[LN_BLOCK];
    __shared__ float s_sum_dyxhat[LN_BLOCK];
    s_sum_dy[tid]     = local_sum_dy;
    s_sum_dyxhat[tid] = local_sum_dyxhat;
    __syncthreads();

    #pragma unroll
    for (int s = LN_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum_dy[tid]     += s_sum_dy[tid + s];
            s_sum_dyxhat[tid] += s_sum_dyxhat[tid + s];
        }
        __syncthreads();
    }

    float sum_dy     = s_sum_dy[0];
    float sum_dyxhat = s_sum_dyxhat[0];
    float inv_H      = 1.0f / static_cast<float>(H);

    // ---- Pass 2: write dx, accumulate dgamma and dbeta ----
    // dx_i = rstd * (gamma_i*dy_i - sum_dy/H - xhat_i * sum_dyxhat/H)
    // Note: gamma is baked into sum_dy and sum_dyxhat (Pass 1), so the
    // subtracted mean terms already account for gamma across the row.
    for (int i = tid; i < H; i += LN_BLOCK) {
        float xi     = row_x[i];
        float dyi    = row_dy[i];
        float gi     = gamma[i];
        float xhat   = (xi - m) * r;
        float dnorm  = dyi * gi;           // gradient w.r.t. xhat_i

        row_dx[i] = r * (dnorm - inv_H * sum_dy - xhat * inv_H * sum_dyxhat);

        atomicAdd(&dgamma[i], dyi * xhat);
        atomicAdd(&dbeta[i],  dyi);
    }
}

void launch_layernorm_backward(const float* dy, const float* x,
                               const float* gamma,
                               const float* mean, const float* rstd,
                               float* dx, float* dgamma, float* dbeta,
                               int N, int H, cudaStream_t stream) {
    dim3 grid(N);
    dim3 block(LN_BLOCK);
    layernorm_backward_kernel<<<grid, block, 0, stream>>>(
        dy, x, gamma, mean, rstd, dx, dgamma, dbeta, N, H);
    CUDA_CHECK_LAST();
}
