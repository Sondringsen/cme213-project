#include "kernels/attention.cuh"
#include "kernels/gemm.cuh"
#include "utils/cuda_check.hpp"
#include <cstdlib>   // malloc / free
#include <cstdio>

// ===========================================================================
// Flash Attention forward (FP32).
//
// What's the problem?
// -------------------
// Standard attention computes
//     S = Q K^T / sqrt(D)        # shape (S, S)
//     P = softmax(S, dim=-1)     # shape (S, S)
//     O = P V                    # shape (S, D)
// where Q, K, V, O have shape (S, D) (per batch and per head; the batch
// and head axes are independent and parallelizable).
//
// The (S, S) attention matrix is enormous: for our M2 config (S = 512,
// 12 layers, 8 heads, B = 8) one layer's attention is 8*8*512*512 = 16 M
// entries = 64 MB. At S = 2048 it's a gigabyte. Materializing it forces
// global-memory writes and reads we can avoid.
//
// What does Flash Attention do?
// -----------------------------
// It computes O *without ever materializing S or P*, by tiling the K, V
// matrices into SRAM (shared memory) and using an *online* softmax. The
// recurrence is:
//
// After processing the first k key/value blocks, per query row we keep
//     m_k    = running max of attention scores so far
//     l_k    = running sum of exp(s - m_k) so far
//     O_k    = running un-normalized output  (sum exp(s - m_k) * V)
//
// When the next block arrives:
//     m_new = max(m_k, max of new scores)
//     alpha = exp(m_k - m_new)
//     l_new = alpha * l_k + sum_{new j} exp(s_j - m_new)
//     O_new = alpha * O_k + sum_{new j} exp(s_j - m_new) * V_j
//     m_k   <- m_new, l_k <- l_new, O_k <- O_new
//
// At the end, the actual output is O = O_k / l_k. Because the recurrence
// rescales by alpha <= 1, it stays numerically stable even when later
// blocks dominate the softmax.
//
// Why is this faster?
// -------------------
// Total global memory traffic per layer becomes O(B*H*S*D) instead of
// O(B*H*S^2 + B*H*S*D). For long sequences this is a huge win, both in
// bytes moved and in HBM round-trips.
//
// Parallelization
// ---------------
// One block per (batch * head, query block) tuple, where a "query block"
// is Br = 32 consecutive queries.
// blockDim is 128 threads. The first Br = 32 threads do the per-query
// online-softmax / accumulation work (each thread owns one query row).
// All 128 threads cooperate on the K, V tile loads -- this is why we use
// 128 instead of 32: loads finish 4x faster and the extra threads cost
// nothing during compute (they just sit idle behind the `is_compute`
// branch). A later optimization (FA-2 style) replaces those idle threads
// with cooperative work across queries within a warp.
// ===========================================================================

constexpr int FA_BLOCK_DIM = 128;   // threads per block
constexpr int FA_Br        = 32;    // queries per block
constexpr int FA_Bc        = 32;    // keys per K/V tile

/*
 * flash_attention_forward_kernel<Br, Bc, D>
 *
 * Template parameters:
 *   Br : queries per block
 *   Bc : keys per K/V tile
 *   D  : head dimension (compile-time so that per-thread arrays sized D
 *        can live in registers rather than spilling to local memory)
 *
 * Grid:
 *   blockIdx.x : query block index (0 .. ceil(S/Br) - 1)
 *   blockIdx.y : flattened (batch * head) index (0 .. B*H - 1)
 *
 * Shared memory:
 *   K_tile, V_tile : (Bc, D) each -- holds one K/V tile in fast SRAM
 *
 * Per-thread registers (compute threads only):
 *   q_reg[D]    : this thread's query row
 *   o[D]        : running un-normalized output for this row
 *   s_local[Bc] : scores for the current K tile
 *   m_state, l_state : online-softmax running max and normalizer
 */
template <int Br, int Bc, int D>
__global__ void flash_attention_forward_kernel(const float* __restrict__ Q,
                                               const float* __restrict__ K,
                                               const float* __restrict__ V,
                                               float* __restrict__ O,
                                               int S, float scale,
                                               bool causal) {
    int bh       = blockIdx.y;          // batch * head index
    int q_block  = blockIdx.x;          // which query tile this block owns
    int q_start  = q_block * Br;        // global index of the first query
    int tid      = threadIdx.x;
    int q_idx    = q_start + tid;       // this thread's global query index
                                        //  (meaningful only for tid < Br)
    bool is_compute = (tid < Br);

    // Offsets into the (B*H)-flattened Q, K, V, O tensors.
    const float* Qbh = Q + bh * S * D;
    const float* Kbh = K + bh * S * D;
    const float* Vbh = V + bh * S * D;
    float*       Obh = O + bh * S * D;

    __shared__ float K_tile[Bc][D];
    __shared__ float V_tile[Bc][D];

    // ---- Per-thread compute state (only used if is_compute) ----
    float q_reg[D];
    float o[D];
    float m_state = -INFINITY;
    float l_state = 0.0f;

    if (is_compute) {
        // Load this thread's query row. Out-of-range queries (q_idx >= S)
        // get zeros; we'll skip the final write for them anyway.
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            q_reg[d] = (q_idx < S) ? Qbh[q_idx * D + d] : 0.0f;
            o[d]     = 0.0f;
        }
    }

    int n_kv_blocks = (S + Bc - 1) / Bc;
    for (int kv = 0; kv < n_kv_blocks; ++kv) {
        int kv_start = kv * Bc;

        // ---- Cooperative load of K and V tiles into shared memory ----
        // All FA_BLOCK_DIM threads (including the non-compute ones) load.
        // Each load is one element; we stride by FA_BLOCK_DIM. For
        // Bc = D = 32 we get 1024 elements / 128 threads = 8 loads/thread.
        // Out-of-range rows pad with zeros so the dot product contributes
        // nothing for them.
        for (int i = tid; i < Bc * D; i += FA_BLOCK_DIM) {
            int row    = i / D;
            int col    = i % D;
            int g_row  = kv_start + row;
            float kv_q = (g_row < S) ? Kbh[g_row * D + col] : 0.0f;
            float vv_q = (g_row < S) ? Vbh[g_row * D + col] : 0.0f;
            K_tile[row][col] = kv_q;
            V_tile[row][col] = vv_q;
        }
        __syncthreads();  // Tiles must be fully loaded before compute reads.

        if (is_compute) {
            // ---- Compute attention scores S = q_reg @ K_tile^T * scale ----
            // Plus the causal / out-of-range mask. Each thread computes one
            // row of S (Bc scalars).
            float s_local[Bc];
            #pragma unroll
            for (int j = 0; j < Bc; ++j) {
                int kv_idx = kv_start + j;
                bool oob   = (kv_idx >= S);
                bool fut   = (causal && kv_idx > q_idx);
                if (oob || fut) {
                    s_local[j] = -INFINITY;
                } else {
                    float dot = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < D; ++d) {
                        dot += q_reg[d] * K_tile[j][d];
                    }
                    s_local[j] = dot * scale;
                }
            }

            // ---- Online softmax: new running max ----
            float m_block = -INFINITY;
            #pragma unroll
            for (int j = 0; j < Bc; ++j) {
                m_block = fmaxf(m_block, s_local[j]);
            }
            float m_new = fmaxf(m_state, m_block);

            // ---- Rescale previous output and l ----
            // alpha = exp(m_old - m_new), in [0, 1].
            // Special case: if m_state is -INFINITY this is the first block
            // contributing anything. We must NOT call expf(-inf - finite)
            // here because that's well-defined (= 0) but expf(-inf - (-inf))
            // is NaN, and m_new could legitimately still be -inf if every
            // key in this block is masked. Guarding with an explicit check
            // is simpler than reasoning about every corner.
            float alpha = (m_state == -INFINITY) ? 0.0f
                                                 : __expf(m_state - m_new);
            #pragma unroll
            for (int d = 0; d < D; ++d) o[d] *= alpha;
            l_state *= alpha;

            // ---- Add contributions from this K/V tile ----
            #pragma unroll
            for (int j = 0; j < Bc; ++j) {
                // Masked entries: exp(-inf - finite) = 0 anyway, but writing
                // it explicitly avoids any NaN if m_new also ended up -inf.
                float p = (s_local[j] == -INFINITY) ? 0.0f
                                                    : __expf(s_local[j] - m_new);
                l_state += p;
                #pragma unroll
                for (int d = 0; d < D; ++d) {
                    o[d] += p * V_tile[j][d];
                }
            }

            m_state = m_new;
        }

        // Sync before the next iteration so non-compute threads don't race
        // ahead and overwrite the tile while compute threads are still
        // reading.
        __syncthreads();
    }

    // ---- Final normalization and write to global O ----
    if (is_compute && q_idx < S) {
        float inv_l = 1.0f / l_state;
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            Obh[q_idx * D + d] = o[d] * inv_l;
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatcher: pick the right template instantiation based on D.
// D is a kernel template parameter (not runtime) so that per-thread arrays
// of size D can live in registers; if D were dynamic the compiler would
// spill them to "local memory", which is actually global memory and ruins
// performance.
// ---------------------------------------------------------------------------
void launch_flash_attention_forward(const float* dQ, const float* dK,
                                    const float* dV, float* dO,
                                    int B, int H, int S, int D,
                                    float scale, bool causal,
                                    cudaStream_t stream) {
    dim3 grid((S + FA_Br - 1) / FA_Br, B * H);
    dim3 block(FA_BLOCK_DIM);

    switch (D) {
        case 32:
            flash_attention_forward_kernel<FA_Br, FA_Bc, 32>
                <<<grid, block, 0, stream>>>(dQ, dK, dV, dO, S, scale, causal);
            break;
        case 64:
            flash_attention_forward_kernel<FA_Br, FA_Bc, 64>
                <<<grid, block, 0, stream>>>(dQ, dK, dV, dO, S, scale, causal);
            break;
        case 96:
            flash_attention_forward_kernel<FA_Br, FA_Bc, 96>
                <<<grid, block, 0, stream>>>(dQ, dK, dV, dO, S, scale, causal);
            break;
        case 128:
            flash_attention_forward_kernel<FA_Br, FA_Bc, 128>
                <<<grid, block, 0, stream>>>(dQ, dK, dV, dO, S, scale, causal);
            break;
        default:
            std::fprintf(stderr,
                "Flash Attention: unsupported head dim D=%d. "
                "Add a case in launch_flash_attention_forward.\n", D);
            std::abort();
    }
    CUDA_CHECK_LAST();
}

// ===========================================================================
// Naive attention backward — materializes the S×S score matrix P.
//
// Strategy (per (batch, head) pair):
//   Forward re-computation:
//     A_raw[i,j] = (Q[i] · K[j]) * scale
//     A[i,j]     = softmax_i(A_raw)                  → P buffer
//   Backward steps:
//     dV    = P^T  × dO                              (gemm_tn)
//     dP    = dO   × V^T                             (gemm_nt)
//     dS    = softmax_backward(P, dP) * scale        (row-parallel kernel)
//     dQ    = dS   × K                               (gemm_tiled)
//     dK    = dS^T × Q                               (gemm_tn)
//
// Memory: two temp buffers of B*H*S*S floats are allocated and freed here.
// For B=8, H=8, S=512: 2 × 8×8×512²×4 = 128 MB — large but manageable.
// ===========================================================================

// ---- Kernel 1: re-compute attention weights P for one (b,h) slice ----
// Grid: (S,) one block per query row i; Block: (256,) threads
// Each block: dot(Q[i], K[j]) * scale for all j, causal mask, then softmax
template <int D>
__global__ void attention_weights_kernel(const float* __restrict__ Q,  // (S, D) one bh
                                         const float* __restrict__ K,  // (S, D) one bh
                                         float*       __restrict__ P,  // (S, S) one bh
                                         int S, float scale, bool causal) {
    constexpr int SMEM_BLOCK = 256;
    int i   = blockIdx.x;
    int tid = threadIdx.x;
    if (i >= S) return;

    // Cache query row in shared memory (D ≤ 128 → always fits)
    __shared__ float q_smem[D];
    if (tid < D) q_smem[tid] = Q[i * D + tid];
    __syncthreads();

    float* p_row = P + i * S;

    // Compute raw scores for all j
    for (int j = tid; j < S; j += SMEM_BLOCK) {
        bool masked = causal && (j > i);
        if (masked) {
            p_row[j] = -INFINITY;
        } else {
            float dot = 0.0f;
            #pragma unroll
            for (int d = 0; d < D; ++d) dot += q_smem[d] * K[j * D + d];
            p_row[j] = dot * scale;
        }
    }
    __syncthreads();

    // Row-wise softmax
    __shared__ float s_buf[SMEM_BLOCK];

    float lmax = -INFINITY;
    for (int j = tid; j < S; j += SMEM_BLOCK) lmax = fmaxf(lmax, p_row[j]);
    s_buf[tid] = lmax;
    __syncthreads();
    for (int s = SMEM_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) s_buf[tid] = fmaxf(s_buf[tid], s_buf[tid + s]);
        __syncthreads();
    }
    float row_max = s_buf[0];
    __syncthreads();

    float lsum = 0.0f;
    for (int j = tid; j < S; j += SMEM_BLOCK) {
        float v = (p_row[j] == -INFINITY) ? 0.0f : __expf(p_row[j] - row_max);
        p_row[j] = v;
        lsum += v;
    }
    s_buf[tid] = lsum;
    __syncthreads();
    for (int s = SMEM_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) s_buf[tid] += s_buf[tid + s];
        __syncthreads();
    }
    float row_sum = s_buf[0];

    for (int j = tid; j < S; j += SMEM_BLOCK) p_row[j] /= row_sum;
}

// ---- Kernel 2: softmax backward + scale ----
// dS_ij = scale * P_ij * (dP_ij - sum_k P_ik * dP_ik)
// Writes result into dS (may be the same buffer as dP)
// Grid: (S,) one block per row; Block: (256,)
__global__ void softmax_backward_kernel(const float* __restrict__ P,   // (S, S)
                                        const float* __restrict__ dP,  // (S, S)
                                        float*       __restrict__ dS,  // (S, S)
                                        int S, float scale) {
    constexpr int SMEM_BLOCK = 256;
    int i   = blockIdx.x;
    int tid = threadIdx.x;
    if (i >= S) return;

    const float* p_row  = P  + i * S;
    const float* dp_row = dP + i * S;
    float*       ds_row = dS + i * S;

    // Reduce: row_dot = sum_j P_ij * dP_ij
    __shared__ float s_buf[SMEM_BLOCK];
    float local_dot = 0.0f;
    for (int j = tid; j < S; j += SMEM_BLOCK) local_dot += p_row[j] * dp_row[j];
    s_buf[tid] = local_dot;
    __syncthreads();
    for (int s = SMEM_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) s_buf[tid] += s_buf[tid + s];
        __syncthreads();
    }
    float row_dot = s_buf[0];

    // Write dS (scale already folded in)
    for (int j = tid; j < S; j += SMEM_BLOCK) {
        ds_row[j] = scale * p_row[j] * (dp_row[j] - row_dot);
    }
}

// ---- Dispatcher for attention_weights_kernel ----
static void launch_attention_weights(const float* Q, const float* K,
                                     float* P, int S, int D,
                                     float scale, bool causal,
                                     cudaStream_t stream) {
    dim3 grid(S);
    dim3 block(256);
    switch (D) {
        case 32:  attention_weights_kernel< 32><<<grid, block, 0, stream>>>(Q, K, P, S, scale, causal); break;
        case 64:  attention_weights_kernel< 64><<<grid, block, 0, stream>>>(Q, K, P, S, scale, causal); break;
        case 96:  attention_weights_kernel< 96><<<grid, block, 0, stream>>>(Q, K, P, S, scale, causal); break;
        case 128: attention_weights_kernel<128><<<grid, block, 0, stream>>>(Q, K, P, S, scale, causal); break;
        default:
            std::fprintf(stderr, "Attention backward: unsupported D=%d\n", D);
            std::abort();
    }
    CUDA_CHECK_LAST();
}

void launch_attention_backward(const float* Q, const float* K, const float* V,
                               const float* dO,
                               float* dQ, float* dK, float* dV,
                               int B, int H, int S, int D,
                               float scale, bool causal,
                               cudaStream_t stream) {
    int BH  = B * H;
    int SD  = S * D;
    int SS  = S * S;

    // Allocate temp buffers for P and dP (each BH × S × S floats)
    float* P  = nullptr;
    float* dP = nullptr;
    CUDA_CHECK(cudaMalloc(&P,  static_cast<size_t>(BH) * SS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dP, static_cast<size_t>(BH) * SS * sizeof(float)));

    // Step 1: compute attention weights P for all (b,h) pairs.
    // Grid: (S, BH) — one block per (query row, batch-head)
    // We run attention_weights_kernel per bh in a loop to keep code simple.
    for (int bh = 0; bh < BH; ++bh) {
        launch_attention_weights(Q + bh * SD, K + bh * SD,
                                 P + bh * SS,
                                 S, D, scale, causal, stream);
    }

    // Step 2: dV = P^T × dO   →  dV[bh] = P[bh]^T × dO[bh]
    // P stored as (S, S), P^T conceptually. launch_gemm_tn: C = A^T B, A stored K×M
    // Here: A = P (stored S×S), B = dO (S×D), C = dV (S×D)
    // M=S, N=D, K=S
    for (int bh = 0; bh < BH; ++bh) {
        launch_gemm_tn(P + bh * SS, dO + bh * SD, dV + bh * SD, S, D, S, stream);
    }

    // Step 3: dP = dO × V^T   →  dP[bh] = dO[bh] × V[bh]^T
    // launch_gemm_nt: C = A × B^T, B stored N×K
    // A = dO (S×D), B = V (stored S×D = N×K), C = dP (S×S)
    // M=S, N=S, K=D
    for (int bh = 0; bh < BH; ++bh) {
        launch_gemm_nt(dO + bh * SD, V + bh * SD, dP + bh * SS, S, S, D, stream);
    }

    // Step 4: dS = softmax_backward(P, dP) * scale
    // We write back into dP (safe: P is still unmodified, needed only for this step)
    // After this, dP buffer holds dS = dL/d(A_raw) = dL/dA * scale
    for (int bh = 0; bh < BH; ++bh) {
        dim3 grid(S);
        dim3 block(256);
        softmax_backward_kernel<<<grid, block, 0, stream>>>(
            P + bh * SS, dP + bh * SS, dP + bh * SS, S, scale);
        CUDA_CHECK_LAST();
    }

    // Step 5: dQ = dS × K
    // dS (= dP buffer) is (S, S); K is (S, D)
    // M=S, N=D, K=S
    for (int bh = 0; bh < BH; ++bh) {
        launch_gemm_tiled(dP + bh * SS, K + bh * SD, dQ + bh * SD, S, D, S, stream);
    }

    // Step 6: dK = dS^T × Q
    // launch_gemm_tn: C = A^T B, A stored K×M
    // A = dS (stored S×S), B = Q (S×D), C = dK (S×D)
    // M=S, N=D, K=S
    for (int bh = 0; bh < BH; ++bh) {
        launch_gemm_tn(dP + bh * SS, Q + bh * SD, dK + bh * SD, S, D, S, stream);
    }

    CUDA_CHECK(cudaFree(P));
    CUDA_CHECK(cudaFree(dP));
}
