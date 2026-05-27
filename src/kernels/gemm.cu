#include "kernels/gemm.cuh"
#include "utils/cuda_check.hpp"

// ===========================================================================
// Register-tiled FP32 GEMM kernel.
//
// Computes C = A * B  (A: M×K, B: K×N, C: M×N, all row-major FP32 on device).
//
// Why register tiling?
// --------------------
// The "textbook" tiled GEMM (TILE=32) has every thread compute one output
// element using a K-long shared-memory dot product. At TILE=32 that gives
// arithmetic intensity ~32 FLOP/byte from global memory, which is
// compute-bound — but each thread still loads 2×K floats from shared memory,
// and the 1024-thread block has very limited register reuse.
//
// Register tiling goes further: each thread accumulates a TM×TN (8×8) sub-tile
// in registers across the entire K dimension. The shared memory tile is still
// there (BM×BK for A, BK×BN for B), but each thread reads TM elements of A
// and TN elements of B from shared memory and computes TM×TN=64 FMA ops
// before touching shared memory again. This:
//   - Reduces shared-memory read traffic by ~TM×TN / (TM+TN) ≈ 4.6×
//   - Increases arithmetic intensity by the same factor
//   - Amortizes __syncthreads() overhead over 64 instead of 1 output
//
// Block/thread layout:
//   - Block handles a BM×BN = 128×128 output tile
//   - Thread block size: (BN/TN, BM/TM) = (16, 16) = 256 threads
//   - Thread (tx, ty) accumulates the TN×TM sub-tile starting at
//       C[blockIdx.y*BM + ty*TM][blockIdx.x*BN + tx*TN]
//
// Shared memory tiles:
//   - Asub[BM][BK] = 128×8 = 4 KB
//   - Bsub[BK][BN] = 8×128 = 4 KB
//   - Total 8 KB/block — well within the 64 KB limit on sm_75
//
// For each K step (BK=8 at a time):
//   1. 256 threads cooperatively load Asub and Bsub (4 floats each, coalesced)
//   2. Each thread loads TM A-values and TN B-values into registers and
//      accumulates the outer product into acc[TM][TN]
//   3. Repeat until K is exhausted, then write acc to global C
// ===========================================================================

// Block tile dimensions
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;

// Per-thread register sub-tile dimensions
constexpr int TM = 8;
constexpr int TN = 8;

// Derived: (BM/TM) * (BN/TN) = 256 threads per block
constexpr int NUM_THREADS = (BM / TM) * (BN / TN);

// Elements each thread loads per cooperative tile fill
// BM*BK / NUM_THREADS = 1024/256 = 4, same for B
constexpr int A_LOADS = (BM * BK) / NUM_THREADS;
constexpr int B_LOADS = (BK * BN) / NUM_THREADS;

// ---------------------------------------------------------------------------
// gemm_register_tiled_kernel
// ---------------------------------------------------------------------------
__global__ void gemm_register_tiled_kernel(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float* __restrict__ C,
                                           int M, int N, int K) {
    // Shared-memory staging tiles. One pair per block, filled cooperatively
    // each K-step, then consumed by all threads via register loads.
    __shared__ float Asub[BM][BK];
    __shared__ float Bsub[BK][BN];

    // threadIdx.x: column-tile index in [0, BN/TN) = [0, 16)
    // threadIdx.y: row-tile index    in [0, BM/TM) = [0, 16)
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Flat ID used for cooperative tile loads (0..255)
    int tid = ty * (BN / TN) + tx;

    // Base corner of this block's output tile in global C
    int row_base = blockIdx.y * BM;
    int col_base = blockIdx.x * BN;

    // Per-thread accumulator — lives in registers for the entire K loop.
    // The compiler keeps these as register variables (not spilled to local
    // memory) because TM, TN are compile-time constants and the loop is unrolled.
    float acc[TM][TN] = {};

    // --- Precompute load indices (constant across K tiles) ---
    //
    // For A[BM][BK]: map tid so that consecutive tids read consecutive K
    // positions (coalesced along K). Group of NUM_THREADS/BK = 32 threads
    // share one A-row; each group reads one BK-wide segment of that row.
    //   a_inner_col = tid % BK  → K position within the tile
    //   a_inner_row = tid / BK  → M position within the tile (first of A_LOADS strides)
    int a_inner_col = tid % BK;
    int a_inner_row = tid / BK;   // ∈ [0, NUM_THREADS/BK) = [0, 32)

    // For B[BK][BN]: map tid so consecutive tids read consecutive N positions
    // (128 threads per row → perfectly coalesced 512-byte loads).
    //   b_inner_col = tid % BN  → N position within the tile
    //   b_inner_row = tid / BN  → K position within the tile
    int b_inner_col = tid % BN;
    int b_inner_row = tid / BN;   // ∈ [0, NUM_THREADS/BN) = [0, 2)

    int num_k_tiles = (K + BK - 1) / BK;

    for (int t = 0; t < num_k_tiles; ++t) {
        int k_offset = t * BK;

        // ---- Cooperative load of Asub[BM][BK] ----
        // Each thread loads A_LOADS=4 elements, striding by NUM_THREADS/BK=32
        // rows in the M direction. Consecutive tids read the same M row but
        // consecutive K positions (coalesced within each BK-wide row group).
        #pragma unroll
        for (int i = 0; i < A_LOADS; ++i) {
            int r = a_inner_row + i * (NUM_THREADS / BK);  // row in Asub, 0..BM-1
            int global_r = row_base + r;
            int global_c = k_offset + a_inner_col;
            Asub[r][a_inner_col] =
                (global_r < M && global_c < K) ? A[global_r * K + global_c] : 0.0f;
        }

        // ---- Cooperative load of Bsub[BK][BN] ----
        // Each thread loads B_LOADS=4 elements; consecutive tids read consecutive
        // N positions within the same K row → 128-element coalesced transaction.
        #pragma unroll
        for (int i = 0; i < B_LOADS; ++i) {
            int r = b_inner_row + i * (NUM_THREADS / BN);  // row in Bsub, 0..BK-1
            int global_r = k_offset + r;
            int global_c = col_base + b_inner_col;
            Bsub[r][b_inner_col] =
                (global_r < K && global_c < N) ? B[global_r * N + global_c] : 0.0f;
        }

        // All 256 threads must finish filling both tiles before any thread
        // reads from them in the accumulate step.
        __syncthreads();

        // ---- Accumulate: TM×TN×BK FMA ops ----
        //
        // For each k in the current tile:
        //   1. Load this thread's TM A-values into a_reg[] (one column of Asub)
        //   2. Load this thread's TN B-values into b_reg[] (one row of Bsub)
        //   3. Outer-product accumulate: acc[tm][tn] += a_reg[tm] * b_reg[tn]
        //
        // Storing in a_reg/b_reg first avoids reading the same shared-memory
        // element once per (tm, tn) pair — shared memory has finite bandwidth
        // and this halves the number of smem reads relative to the naive order.
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // Load TM elements from A's k-th column (this thread's M range)
            float a_reg[TM];
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm) {
                a_reg[tm] = Asub[ty * TM + tm][k];
            }

            // Load TN elements from B's k-th row (this thread's N range)
            float b_reg[TN];
            #pragma unroll
            for (int tn = 0; tn < TN; ++tn) {
                b_reg[tn] = Bsub[k][tx * TN + tn];
            }

            // Outer product: 64 FMA ops, all in registers
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm) {
                #pragma unroll
                for (int tn = 0; tn < TN; ++tn) {
                    acc[tm][tn] += a_reg[tm] * b_reg[tn];
                }
            }
        }

        // Must finish consuming the tile before the next iteration overwrites it.
        __syncthreads();
    }

    // ---- Write TM×TN results to global C ----
    // Each thread writes 64 elements. Boundary guards handle non-multiples of BM/BN.
    #pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
        int r = row_base + ty * TM + tm;
        if (r >= M) break;
        #pragma unroll
        for (int tn = 0; tn < TN; ++tn) {
            int c = col_base + tx * TN + tn;
            if (c < N) {
                C[r * N + c] = acc[tm][tn];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// launch_gemm_tiled — host-side launcher (public API, unchanged from before)
// ---------------------------------------------------------------------------
void launch_gemm_tiled(const float* dA, const float* dB, float* dC,
                       int M, int N, int K, cudaStream_t stream) {
    // Block: (BN/TN, BM/TM) = (16, 16) = 256 threads
    // Grid:  one block per BM×BN output tile
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_register_tiled_kernel<<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK_LAST();
}

// ===========================================================================
// Transpose-aware GEMM kernels for backward passes.
//
// These use a simpler TILE=32 tiled kernel (not register-tiled) because:
//   1. They are used in backward passes where throughput is less critical.
//   2. Transposed access patterns are harder to optimize with register tiling.
//
// launch_gemm_tn: C = A^T * B
//   A is stored as (K, M), i.e. A^T is (M, K).
//   B is (K, N), C is (M, N).
//
// launch_gemm_nt: C = A * B^T
//   A is (M, K), B is stored as (N, K), i.e. B^T is (K, N).
//   C is (M, N).
// ===========================================================================

constexpr int TILE = 32;

// ---- C = A^T * B, A stored as K×M ----
__global__ void gemm_tn_kernel(const float* __restrict__ A,  // K×M
                               const float* __restrict__ B,  // K×N
                               float* __restrict__ C,         // M×N
                               int M, int N, int K) {
    __shared__ float Asub[TILE][TILE];
    __shared__ float Bsub[TILE][TILE];

    // Output: C[row, col]
    int row = blockIdx.y * TILE + threadIdx.y;  // M dimension
    int col = blockIdx.x * TILE + threadIdx.x;  // N dimension

    float acc = 0.0f;
    int num_k_tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < num_k_tiles; ++t) {
        // A^T[row, t*TILE + tx] = A[(t*TILE + tx), row] = A[(t*TILE+tx)*M + row]
        int k_A = t * TILE + threadIdx.x;
        Asub[threadIdx.y][threadIdx.x] =
            (row < M && k_A < K) ? A[k_A * M + row] : 0.0f;

        // B[t*TILE + ty, col] = B[(t*TILE+ty)*N + col]
        int k_B = t * TILE + threadIdx.y;
        Bsub[threadIdx.y][threadIdx.x] =
            (k_B < K && col < N) ? B[k_B * N + col] : 0.0f;

        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k) acc += Asub[threadIdx.y][k] * Bsub[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) C[row * N + col] = acc;
}

// ---- C = A * B^T, B stored as N×K ----
__global__ void gemm_nt_kernel(const float* __restrict__ A,  // M×K
                               const float* __restrict__ B,  // N×K
                               float* __restrict__ C,         // M×N
                               int M, int N, int K) {
    __shared__ float Asub[TILE][TILE];
    __shared__ float Bsub[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    int num_k_tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < num_k_tiles; ++t) {
        // A[row, t*TILE + tx] = A[row*K + t*TILE + tx]
        int k_A = t * TILE + threadIdx.x;
        Asub[threadIdx.y][threadIdx.x] =
            (row < M && k_A < K) ? A[row * K + k_A] : 0.0f;

        // B^T[t*TILE + ty, col] = B[col, t*TILE+ty] = B[col*K + t*TILE+ty]
        int k_B = t * TILE + threadIdx.y;
        Bsub[threadIdx.y][threadIdx.x] =
            (col < N && k_B < K) ? B[col * K + k_B] : 0.0f;

        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k) acc += Asub[threadIdx.y][k] * Bsub[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) C[row * N + col] = acc;
}

void launch_gemm_tn(const float* dA, const float* dB, float* dC,
                    int M, int N, int K, cudaStream_t stream) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    gemm_tn_kernel<<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK_LAST();
}

void launch_gemm_nt(const float* dA, const float* dB, float* dC,
                    int M, int N, int K, cudaStream_t stream) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    gemm_nt_kernel<<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK_LAST();
}
