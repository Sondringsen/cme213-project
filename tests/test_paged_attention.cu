// ===========================================================================
// test_paged_attention.cu
//
// Correctness for the PagedAttention decode kernel.
//
// The test pre-fills K and V into a global block pool with random data, then
// assigns each sequence a shuffled subset of physical blocks (this exercises
// the gather indirection — if we used identity mapping, the test would pass
// even if the block_table were ignored). A logical dense view is built by
// gathering blocks in their LOGICAL order, and the CPU reference is computed
// from that dense view. Comparison is between paged-GPU and dense-CPU outputs.
//
// We also run the dense-decode reference kernel on the same dense view; it
// should match both within FP32 expf tolerance. This isolates "did paging
// break anything" from "did our decode-style online softmax break anything".
// ===========================================================================

#include "kernels/paged_attention.cuh"
#include "utils/cuda_check.hpp"
#include "utils/tensor.hpp"
#include "test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

constexpr int BLOCK_SIZE = 16;

// CPU reference: one decode step. Inputs are the *logical dense* K and V
// (already gathered in logical order), shapes (B, H, S_max, D). Output is
// (B, H, D). seq_lens[b] = number of valid history tokens.
static void paged_attention_decode_cpu(const std::vector<float>& Q,
                                       const std::vector<float>& K_dense,
                                       const std::vector<float>& V_dense,
                                       const std::vector<int>&   seq_lens,
                                       std::vector<float>&       O,
                                       int B, int H, int S_max, int D,
                                       float scale) {
    O.assign(static_cast<size_t>(B) * H * D, 0.0f);
    std::vector<float> scores(S_max);
    for (int b = 0; b < B; ++b) {
        int len = seq_lens[b];
        for (int h = 0; h < H; ++h) {
            if (len <= 0) continue;

            // Scores
            float m = -INFINITY;
            for (int t = 0; t < len; ++t) {
                float s = 0.0f;
                for (int d = 0; d < D; ++d) {
                    float q = Q[(b * H + h) * D + d];
                    float k = K_dense[((b * H + h) * S_max + t) * D + d];
                    s += q * k;
                }
                scores[t] = s * scale;
                if (scores[t] > m) m = scores[t];
            }

            // Softmax weights
            float l = 0.0f;
            for (int t = 0; t < len; ++t) {
                scores[t] = std::exp(scores[t] - m);
                l += scores[t];
            }
            float inv_l = 1.0f / l;

            // Output
            for (int d = 0; d < D; ++d) {
                float acc = 0.0f;
                for (int t = 0; t < len; ++t) {
                    float v = V_dense[((b * H + h) * S_max + t) * D + d];
                    acc += scores[t] * v;
                }
                O[(b * H + h) * D + d] = acc * inv_l;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Build a paged layout from a logical dense KV.
//
//   - Allocates `total_blocks` physical blocks in the global pool.
//   - For each sequence b, assigns ceil(seq_lens[b]/BLOCK_SIZE) random unused
//     blocks (so gather indirection is non-trivial).
//   - Scatters K_dense / V_dense values into those physical blocks.
//
// Output:
//   K_blocks, V_blocks : (total_blocks * H * BLOCK_SIZE * D) flat floats
//   block_table        : (B * max_blocks) ints (unused entries = -1)
// ---------------------------------------------------------------------------
static void build_paged_from_dense(const std::vector<float>& K_dense,
                                   const std::vector<float>& V_dense,
                                   const std::vector<int>&   seq_lens,
                                   int B, int H, int S_max, int D,
                                   int total_blocks, int max_blocks,
                                   std::vector<float>& K_blocks,
                                   std::vector<float>& V_blocks,
                                   std::vector<int>&   block_table,
                                   unsigned shuffle_seed) {
    size_t block_floats = static_cast<size_t>(H) * BLOCK_SIZE * D;
    K_blocks.assign(static_cast<size_t>(total_blocks) * block_floats, 0.0f);
    V_blocks.assign(static_cast<size_t>(total_blocks) * block_floats, 0.0f);
    block_table.assign(static_cast<size_t>(B) * max_blocks, -1);

    // Shuffled free list of physical block ids — guarantees scrambled mapping.
    std::vector<int> free_list(total_blocks);
    std::iota(free_list.begin(), free_list.end(), 0);
    std::mt19937 gen(shuffle_seed);
    std::shuffle(free_list.begin(), free_list.end(), gen);
    int free_cursor = 0;

    for (int b = 0; b < B; ++b) {
        int len = seq_lens[b];
        int nblk = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (nblk > max_blocks) {
            std::fprintf(stderr, "[test] seq %d needs %d blocks > max %d\n",
                         b, nblk, max_blocks);
            std::abort();
        }
        for (int blk = 0; blk < nblk; ++blk) {
            int phys = free_list.at(free_cursor++);
            block_table[b * max_blocks + blk] = phys;

            int tokens_in_block = std::min(BLOCK_SIZE, len - blk * BLOCK_SIZE);
            for (int h = 0; h < H; ++h) {
                for (int t = 0; t < tokens_in_block; ++t) {
                    int logical_t = blk * BLOCK_SIZE + t;
                    for (int d = 0; d < D; ++d) {
                        size_t src = ((b * H + h) * S_max + logical_t) * D + d;
                        size_t dst = (static_cast<size_t>(phys) * H + h) * BLOCK_SIZE * D
                                   + static_cast<size_t>(t) * D + d;
                        K_blocks[dst] = K_dense[src];
                        V_blocks[dst] = V_dense[src];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Single test case. Returns 0 on pass, 1 on fail.
// ---------------------------------------------------------------------------
static int run_case(int B, int H, int D,
                    const std::vector<int>& seq_lens,
                    unsigned seed) {
    int S_max = *std::max_element(seq_lens.begin(), seq_lens.end());
    int max_blocks = (S_max + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int needed_blocks = 0;
    for (int len : seq_lens) needed_blocks += (len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int total_blocks = needed_blocks + 8;  // a little slack for shuffle

    std::printf("=== B=%d H=%d D=%d S_max=%d blocks_used/total=%d/%d ===\n",
                B, H, D, S_max, needed_blocks, total_blocks);

    // Allocate host buffers.
    std::vector<float> hQ(static_cast<size_t>(B) * H * D);
    std::vector<float> hK_dense(static_cast<size_t>(B) * H * S_max * D, 0.0f);
    std::vector<float> hV_dense(static_cast<size_t>(B) * H * S_max * D, 0.0f);
    fill_random(hQ, seed);
    // Fill only valid positions; rest stays zero (won't be read anyway).
    {
        std::mt19937 gen(seed + 1);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int b = 0; b < B; ++b) {
            int len = seq_lens[b];
            for (int h = 0; h < H; ++h) {
                for (int t = 0; t < len; ++t) {
                    for (int d = 0; d < D; ++d) {
                        size_t idx = ((b * H + h) * S_max + t) * D + d;
                        hK_dense[idx] = dist(gen);
                        hV_dense[idx] = dist(gen);
                    }
                }
            }
        }
    }

    // Build paged layout from dense.
    std::vector<float> hK_blocks, hV_blocks;
    std::vector<int>   h_block_table;
    build_paged_from_dense(hK_dense, hV_dense, seq_lens,
                           B, H, S_max, D, total_blocks, max_blocks,
                           hK_blocks, hV_blocks, h_block_table, seed + 2);

    // CPU reference.
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    std::vector<float> hO_cpu(static_cast<size_t>(B) * H * D, 0.0f);
    paged_attention_decode_cpu(hQ, hK_dense, hV_dense, seq_lens, hO_cpu,
                               B, H, S_max, D, scale);

    // GPU: paged.
    Tensor<float> dQ({B, H, D});
    Tensor<float> dK_blocks({total_blocks, H, BLOCK_SIZE, D});
    Tensor<float> dV_blocks({total_blocks, H, BLOCK_SIZE, D});
    Tensor<int>   d_block_table({B, max_blocks});
    Tensor<int>   d_seq_lens({B});
    Tensor<float> dO_paged({B, H, D});
    dQ.copy_from_host(hQ.data());
    dK_blocks.copy_from_host(hK_blocks.data());
    dV_blocks.copy_from_host(hV_blocks.data());
    d_block_table.copy_from_host(h_block_table.data());
    d_seq_lens.copy_from_host(seq_lens.data());
    launch_paged_attention_decode(dQ.data(), dK_blocks.data(), dV_blocks.data(),
                                  d_block_table.data(), d_seq_lens.data(),
                                  dO_paged.data(),
                                  B, H, D, BLOCK_SIZE, max_blocks, scale);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> hO_paged(hO_cpu.size());
    dO_paged.copy_to_host(hO_paged.data());

    // GPU: dense reference kernel on the SAME logical dense KV.
    Tensor<float> dK_dense({B, H, S_max, D});
    Tensor<float> dV_dense({B, H, S_max, D});
    Tensor<float> dO_dense({B, H, D});
    dK_dense.copy_from_host(hK_dense.data());
    dV_dense.copy_from_host(hV_dense.data());
    launch_dense_attention_decode(dQ.data(), dK_dense.data(), dV_dense.data(),
                                  d_seq_lens.data(), dO_dense.data(),
                                  B, H, D, S_max, scale);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> hO_dense(hO_cpu.size());
    dO_dense.copy_to_host(hO_dense.data());

    float abs_paged, rel_paged, abs_dense, rel_dense, abs_pd, rel_pd;
    compare(hO_cpu,   hO_paged, abs_paged, rel_paged);
    compare(hO_cpu,   hO_dense, abs_dense, rel_dense);
    compare(hO_dense, hO_paged, abs_pd,    rel_pd);
    std::printf("  paged vs CPU : abs=%.3e rel=%.3e\n", abs_paged, rel_paged);
    std::printf("  dense vs CPU : abs=%.3e rel=%.3e\n", abs_dense, rel_dense);
    std::printf("  paged vs dense GPU: abs=%.3e rel=%.3e\n", abs_pd, rel_pd);

    int fail = 0;
    if (rel_paged > 5e-3f && abs_paged > 1e-5f) { std::printf("  FAIL: paged\n"); fail = 1; }
    if (rel_dense > 5e-3f && abs_dense > 1e-5f) { std::printf("  FAIL: dense\n"); fail = 1; }
    if (!fail) std::printf("  PASS\n");
    return fail;
}

int main() {
    int fails = 0;

    // One block, simplest case.
    fails += run_case(1, 1, 32, {16}, 1);

    // Tail handling: non-multiple-of-BLOCK_SIZE lengths.
    fails += run_case(1, 2, 64, {33}, 2);
    fails += run_case(1, 2, 64, {17}, 3);  // just past one block

    // Mixed lengths across requests.
    fails += run_case(4, 4, 64, {16, 32, 48, 64}, 4);

    // Larger: training-ish sizes.
    fails += run_case(8, 8, 64,  {128, 128, 96, 80, 64, 48, 32, 17}, 5);
    fails += run_case(4, 4, 128, {128, 100, 64, 33}, 6);

    // All requests at max length.
    fails += run_case(4, 8, 64, {128, 128, 128, 128}, 7);

    if (fails) {
        std::printf("\n%d case(s) FAILED\n", fails);
        return 1;
    }
    std::printf("\nAll PagedAttention correctness cases passed.\n");
    return 0;
}
