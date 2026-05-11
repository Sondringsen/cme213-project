#include "kernels/gemm.cuh"
#include "utils/cuda_check.hpp"

// ===========================================================================
// Tiled FP32 GEMM kernel.
//
// This is the "textbook" GPU GEMM, the same algorithm covered in CME 213 HW4
// but written from scratch and packaged as the first reference kernel for
// the project. It is intentionally NOT the fastest possible GEMM -- it is
// the simplest one that gets meaningful speedup over the naive
// O(MNK)-global-loads version. A faster variant with register tiling, vector
// loads, and software pipelining is a follow-up.
//
// Why tiling?
// -----------
// The naive GEMM has each thread compute one element of C with a K-long
// global-memory dot product. For each output element it issues 2K global
// loads and does 2K flops -> arithmetic intensity ~1 flop/byte, far below
// the GPU's compute-to-bandwidth ratio. The kernel is memory-bound and
// nowhere near peak.
//
// Tiling fixes this by loading TILE x TILE sub-matrices of A and B into
// shared memory and *reusing* each loaded element TILE times. Effective
// arithmetic intensity goes up by a factor of TILE, which is enough to make
// the kernel compute-bound at reasonable problem sizes.
//
// CUDA primitives used:
//   - Shared memory: a small (~48-96 KB / SM), low-latency scratchpad shared
//     by all threads in a block. Roughly 100x lower latency than global
//     memory.
//   - __syncthreads(): a barrier across the threads of a block. Required so
//     that we don't read a tile before it's fully loaded, or overwrite a
//     tile before all readers are done.
//   - Coalesced loads: when consecutive threads in a warp read consecutive
//     addresses in global memory, the hardware combines those into a single
//     transaction. Our load pattern (threadIdx.x is the contiguous
//     dimension) is coalesced.
// ===========================================================================

// TILE = 32 means each block has 32*32 = 1024 threads, the maximum allowed
// per block on every architecture we care about. Each thread computes one
// element of the output tile.
//
// Changing TILE here doesn't change correctness, only performance. The next
// optimization step is to keep TILE = 32 but have each thread compute a
// small TM x TN sub-tile (register tiling); that's done in a follow-up.
constexpr int TILE = 32;

// ---------------------------------------------------------------------------
// gemm_tiled_kernel
//
// Computes C = A * B where A is M x K, B is K x N, C is M x N, all FP32,
// all row-major.
//
// Thread/block layout:
//   blockDim = (TILE, TILE)
//   gridDim  = (ceil_div(N, TILE), ceil_div(M, TILE))
//
// Mapping:
//   threadIdx.(x, y) <-> (col-within-tile, row-within-tile)
//   blockIdx.(x, y)  <-> (col-tile-index, row-tile-index)
// So thread (tx, ty) of block (bx, by) is responsible for the C element at
//   row = by*TILE + ty, col = bx*TILE + tx.
// ---------------------------------------------------------------------------
__global__ void gemm_tiled_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K) {
    // Shared-memory staging tiles. These are allocated once per block.
    __shared__ float Asub[TILE][TILE];
    __shared__ float Bsub[TILE][TILE];

    // Global indices of *this thread's* output element.
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    // Accumulator lives in a register. This is where compute happens.
    float acc = 0.0f;

    // We march across the K dimension TILE columns at a time. Each iteration
    // we load one A sub-tile (TILE rows starting at `row`, TILE cols starting
    // at `t*TILE`) and one B sub-tile (TILE rows starting at `t*TILE`, TILE
    // cols starting at `col`).
    int num_k_tiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < num_k_tiles; ++t) {
        // ---- Cooperative load of one tile of A and one tile of B ----
        // Each thread loads exactly one element of each. Because consecutive
        // threads in a warp differ only in threadIdx.x, and threadIdx.x maps
        // to the column index, the loads are coalesced into a small number
        // of memory transactions.
        int a_col = t * TILE + threadIdx.x;   // column of A we read
        int b_row = t * TILE + threadIdx.y;   // row of B we read

        // Out-of-range reads pad with zero so they contribute nothing to the
        // dot product. This is the cheapest way to handle non-multiple-of-
        // TILE matrix sizes.
        Asub[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bsub[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        // Barrier: don't start consuming the tile until every thread in the
        // block has finished filling its slot.
        __syncthreads();

        // ---- Inner product over the tile ----
        // All reads here are from shared memory, ~100x faster than the
        // global reads we'd otherwise issue 2*TILE more times.
        //
        // #pragma unroll lets nvcc hoist the loop and keep `acc` in a
        // register; this is a free speedup since TILE is a compile-time
        // constant.
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += Asub[threadIdx.y][k] * Bsub[k][threadIdx.x];
        }

        // Barrier: don't overwrite the tile on the next iteration until
        // every thread has finished reading the current one.
        __syncthreads();
    }

    // Boundary guard: threads that fall outside the output matrix skip the
    // store. Their accumulator is garbage anyway because the loads padded
    // with zeros, but we must not write past the end of C.
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// ---------------------------------------------------------------------------
// Host-side launcher: picks the grid/block dimensions and fires the kernel.
// Kept separate from the kernel so callers don't have to think about the
// launch geometry.
// ---------------------------------------------------------------------------
void launch_gemm_tiled(const float* dA, const float* dB, float* dC,
                       int M, int N, int K, cudaStream_t stream) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    gemm_tiled_kernel<<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK_LAST();
}
