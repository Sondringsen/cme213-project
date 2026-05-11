#include "kernels/attention.cuh"
#include "utils/cuda_check.hpp"

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
