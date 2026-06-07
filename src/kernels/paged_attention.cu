#include "kernels/paged_attention.cuh"
#include "utils/cuda_check.hpp"

#include <cstdio>
#include <cstdlib>

// ===========================================================================
// PagedAttention decode kernel.
//
// Per (b, h) CUDA block, computes
//     o = softmax(q · K_seq[b]^T * scale) · V_seq[b]
// where K_seq[b], V_seq[b] are the gathered KV cache for sequence b,
// scattered across physical blocks indexed by block_table[b, *].
//
// Online softmax (same recurrence as Flash Attention):
//   For each logical block, compute scores against the new K tile, get the
//   block max m_block, set m_new = max(m_old, m_block), rescale the running
//   output by alpha = exp(m_old - m_new), and add the block's contribution.
//   Final O = o_acc / l_acc.
//
// Threading
// ---------
//   blockDim = 128 (fixed). The first BLOCK_SIZE threads compute one score
//   each per K tile; the first D threads each own one element of the output
//   accumulator and one slot of the per-iteration V-weighted sum. The
//   remaining threads help cooperatively load K/V tiles.
//
// Indirection cost
// ----------------
//   The only difference from a contiguous-KV decode is the K_base/V_base
//   pointer computation: paged reads `block_table[b * max_blocks + blk]` per
//   logical block. That's one extra global load per BLOCK_SIZE tokens,
//   amortized against the BLOCK_SIZE * D-element tile load that follows.
// ===========================================================================

template <int D, int BLOCK_SIZE, int BLOCK_DIM>
__global__ void paged_attention_decode_kernel(
        const float* __restrict__ Q,            // (B, H, D)
        const float* __restrict__ K_blocks,     // (N_blocks, H, BLOCK_SIZE, D)
        const float* __restrict__ V_blocks,     // (N_blocks, H, BLOCK_SIZE, D)
        const int*   __restrict__ block_table,  // (B, max_blocks)
        const int*   __restrict__ seq_lens,     // (B,)
        float*       __restrict__ O,            // (B, H, D)
        int H, int max_blocks, float scale) {
    static_assert(BLOCK_DIM >= D, "BLOCK_DIM must be >= D");
    static_assert(BLOCK_DIM >= BLOCK_SIZE, "BLOCK_DIM must be >= BLOCK_SIZE");

    int b   = blockIdx.x;
    int h   = blockIdx.y;
    int tid = threadIdx.x;
    int len = seq_lens[b];

    __shared__ float q_smem[D];
    __shared__ float k_tile[BLOCK_SIZE][D];
    __shared__ float v_tile[BLOCK_SIZE][D];
    __shared__ float scores[BLOCK_SIZE];
    __shared__ float weights[BLOCK_SIZE];

    // Empty-sequence early-out (new request, no context yet).
    if (len <= 0) {
        if (tid < D) O[(b * H + h) * D + tid] = 0.0f;
        return;
    }

    if (tid < D) q_smem[tid] = Q[(b * H + h) * D + tid];

    // Online-softmax running state, kept in per-thread registers. Every thread
    // computes identical scalar values from shared inputs; cheaper than writing
    // them through shared memory.
    float m_state = -INFINITY;
    float l_state = 0.0f;
    float o_acc   = 0.0f;  // meaningful only for tid < D
    __syncthreads();

    int n_blocks = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int blk = 0; blk < n_blocks; ++blk) {
        int phys             = block_table[b * max_blocks + blk];
        int tokens_in_block  = min(BLOCK_SIZE, len - blk * BLOCK_SIZE);
        const float* K_base  = K_blocks
                             + static_cast<size_t>(phys * H + h) * BLOCK_SIZE * D;
        const float* V_base  = V_blocks
                             + static_cast<size_t>(phys * H + h) * BLOCK_SIZE * D;

        // Cooperative tile load. Out-of-range rows in the last block get zeros;
        // their scores are explicitly set to -inf below so they contribute 0.
        for (int i = tid; i < BLOCK_SIZE * D; i += BLOCK_DIM) {
            int t = i / D;
            int d = i % D;
            bool valid = (t < tokens_in_block);
            k_tile[t][d] = valid ? K_base[t * D + d] : 0.0f;
            v_tile[t][d] = valid ? V_base[t * D + d] : 0.0f;
        }
        __syncthreads();

        // Scores: first BLOCK_SIZE threads each compute one dot product.
        if (tid < BLOCK_SIZE) {
            if (tid < tokens_in_block) {
                float s = 0.0f;
                #pragma unroll
                for (int d = 0; d < D; ++d) s += q_smem[d] * k_tile[tid][d];
                scores[tid] = s * scale;
            } else {
                scores[tid] = -INFINITY;
            }
        }
        __syncthreads();

        // Block max, m_new, alpha — identical scalar work in every thread.
        float block_max = -INFINITY;
        #pragma unroll
        for (int t = 0; t < BLOCK_SIZE; ++t) block_max = fmaxf(block_max, scores[t]);
        float m_new  = fmaxf(m_state, block_max);
        float alpha  = (m_state == -INFINITY) ? 0.0f : __expf(m_state - m_new);

        // Per-token softmax weights (BLOCK_SIZE-wide expf), once into shared.
        if (tid < BLOCK_SIZE) {
            float s = scores[tid];
            weights[tid] = (s == -INFINITY) ? 0.0f : __expf(s - m_new);
        }
        __syncthreads();

        // l_block in registers — every thread reads the same weights and
        // computes the same sum.
        float l_block = 0.0f;
        #pragma unroll
        for (int t = 0; t < BLOCK_SIZE; ++t) l_block += weights[t];

        // Output accumulate: each thread owns one output dimension.
        if (tid < D) {
            float o_new = alpha * o_acc;
            #pragma unroll
            for (int t = 0; t < BLOCK_SIZE; ++t) o_new += weights[t] * v_tile[t][tid];
            o_acc = o_new;
        }

        l_state = alpha * l_state + l_block;
        m_state = m_new;
        __syncthreads();  // before next iteration overwrites shared tiles
    }

    if (tid < D) O[(b * H + h) * D + tid] = o_acc / l_state;
}

// ===========================================================================
// Dense decode reference. Identical computation, contiguous KV layout.
// Used by the benchmark to isolate the block-table indirection cost.
// ===========================================================================
template <int D, int BLOCK_SIZE, int BLOCK_DIM>
__global__ void dense_attention_decode_kernel(
        const float* __restrict__ Q,         // (B, H, D)
        const float* __restrict__ K,         // (B, H, seq_max, D)
        const float* __restrict__ V,         // (B, H, seq_max, D)
        const int*   __restrict__ seq_lens,  // (B,)
        float*       __restrict__ O,         // (B, H, D)
        int H, int seq_max, float scale) {
    static_assert(BLOCK_DIM >= D, "BLOCK_DIM must be >= D");
    static_assert(BLOCK_DIM >= BLOCK_SIZE, "BLOCK_DIM must be >= BLOCK_SIZE");

    int b   = blockIdx.x;
    int h   = blockIdx.y;
    int tid = threadIdx.x;
    int len = seq_lens[b];

    __shared__ float q_smem[D];
    __shared__ float k_tile[BLOCK_SIZE][D];
    __shared__ float v_tile[BLOCK_SIZE][D];
    __shared__ float scores[BLOCK_SIZE];
    __shared__ float weights[BLOCK_SIZE];

    if (len <= 0) {
        if (tid < D) O[(b * H + h) * D + tid] = 0.0f;
        return;
    }

    if (tid < D) q_smem[tid] = Q[(b * H + h) * D + tid];

    float m_state = -INFINITY;
    float l_state = 0.0f;
    float o_acc   = 0.0f;
    __syncthreads();

    const float* K_base = K + static_cast<size_t>(b * H + h) * seq_max * D;
    const float* V_base = V + static_cast<size_t>(b * H + h) * seq_max * D;

    int n_blocks = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int blk = 0; blk < n_blocks; ++blk) {
        int tokens_in_block = min(BLOCK_SIZE, len - blk * BLOCK_SIZE);
        int row_offset      = blk * BLOCK_SIZE;

        for (int i = tid; i < BLOCK_SIZE * D; i += BLOCK_DIM) {
            int t = i / D;
            int d = i % D;
            bool valid = (t < tokens_in_block);
            k_tile[t][d] = valid ? K_base[(row_offset + t) * D + d] : 0.0f;
            v_tile[t][d] = valid ? V_base[(row_offset + t) * D + d] : 0.0f;
        }
        __syncthreads();

        if (tid < BLOCK_SIZE) {
            if (tid < tokens_in_block) {
                float s = 0.0f;
                #pragma unroll
                for (int d = 0; d < D; ++d) s += q_smem[d] * k_tile[tid][d];
                scores[tid] = s * scale;
            } else {
                scores[tid] = -INFINITY;
            }
        }
        __syncthreads();

        float block_max = -INFINITY;
        #pragma unroll
        for (int t = 0; t < BLOCK_SIZE; ++t) block_max = fmaxf(block_max, scores[t]);
        float m_new  = fmaxf(m_state, block_max);
        float alpha  = (m_state == -INFINITY) ? 0.0f : __expf(m_state - m_new);

        if (tid < BLOCK_SIZE) {
            float s = scores[tid];
            weights[tid] = (s == -INFINITY) ? 0.0f : __expf(s - m_new);
        }
        __syncthreads();

        float l_block = 0.0f;
        #pragma unroll
        for (int t = 0; t < BLOCK_SIZE; ++t) l_block += weights[t];

        if (tid < D) {
            float o_new = alpha * o_acc;
            #pragma unroll
            for (int t = 0; t < BLOCK_SIZE; ++t) o_new += weights[t] * v_tile[t][tid];
            o_acc = o_new;
        }

        l_state = alpha * l_state + l_block;
        m_state = m_new;
        __syncthreads();
    }

    if (tid < D) O[(b * H + h) * D + tid] = o_acc / l_state;
}

// ===========================================================================
// Dispatchers. D is a template parameter so per-thread arrays live in
// registers (the same reason Flash Attention is templated on D).
// ===========================================================================

constexpr int PA_BLOCK_DIM = 128;

#define DISPATCH_PAGED(D_VAL)                                                 \
    case D_VAL: {                                                             \
        paged_attention_decode_kernel<D_VAL, 16, PA_BLOCK_DIM>                \
            <<<grid, block, 0, stream>>>(dQ, dK_blocks, dV_blocks,            \
                                         d_block_table, d_seq_lens, dO,      \
                                         H, max_blocks_per_seq, scale);      \
        break;                                                                \
    }

void launch_paged_attention_decode(const float* dQ,
                                   const float* dK_blocks,
                                   const float* dV_blocks,
                                   const int*   d_block_table,
                                   const int*   d_seq_lens,
                                   float*       dO,
                                   int B, int H, int D,
                                   int block_size, int max_blocks_per_seq,
                                   float scale,
                                   cudaStream_t stream) {
    if (block_size != 16) {
        std::fprintf(stderr,
            "paged_attention_decode: only block_size=16 is wired up "
            "(got %d). Add a template instantiation in paged_attention.cu.\n",
            block_size);
        std::abort();
    }
    dim3 grid(B, H);
    dim3 block(PA_BLOCK_DIM);
    switch (D) {
        DISPATCH_PAGED(16)
        DISPATCH_PAGED(32)
        DISPATCH_PAGED(64)
        DISPATCH_PAGED(128)
        default:
            std::fprintf(stderr,
                "paged_attention_decode: unsupported D=%d. "
                "Add a case in launch_paged_attention_decode.\n", D);
            std::abort();
    }
    CUDA_CHECK_LAST();
}

#define DISPATCH_DENSE(D_VAL)                                                 \
    case D_VAL: {                                                             \
        dense_attention_decode_kernel<D_VAL, 16, PA_BLOCK_DIM>                \
            <<<grid, block, 0, stream>>>(dQ, dK, dV, d_seq_lens, dO,          \
                                         H, seq_max, scale);                 \
        break;                                                                \
    }

void launch_dense_attention_decode(const float* dQ,
                                   const float* dK,
                                   const float* dV,
                                   const int*   d_seq_lens,
                                   float*       dO,
                                   int B, int H, int D, int seq_max,
                                   float scale,
                                   cudaStream_t stream) {
    dim3 grid(B, H);
    dim3 block(PA_BLOCK_DIM);
    switch (D) {
        DISPATCH_DENSE(16)
        DISPATCH_DENSE(32)
        DISPATCH_DENSE(64)
        DISPATCH_DENSE(128)
        default:
            std::fprintf(stderr,
                "dense_attention_decode: unsupported D=%d.\n", D);
            std::abort();
    }
    CUDA_CHECK_LAST();
}
