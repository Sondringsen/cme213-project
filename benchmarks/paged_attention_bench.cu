// ===========================================================================
// paged_attention_bench.cu
//
// Two metrics on the same workload: memory utilization (paged vs naive-padded
// KV layout) and decode throughput (tokens/sec, same kernels).
//
// Workload model
// --------------
// B concurrent requests with lengths drawn from a truncated exponential
// distribution (mean=64, capped at S_max=128). This mimics LLM-serving
// workloads: most requests are short, a long tail of larger ones.
//
// Memory utilization (analytical, no GPU)
// ---------------------------------------
//   useful_bytes = sum_b len[b] * H * D * 4 * 2          (K and V)
//   naive_alloc  = B * S_max * H * D * 4 * 2
//   paged_alloc  = (sum_b ceil(len[b]/BLOCK_SIZE)) * BLOCK_SIZE * H * D * 4 * 2
// Paged converts external (cross-request) and internal (per-request padding)
// fragmentation into one fixed cost: at most BLOCK_SIZE-1 padded tokens at
// the tail of each sequence.
//
// Throughput (timed kernel)
// -------------------------
//   Time N iterations of one decode step. Each iteration produces B tokens.
//   tokens_per_sec = N * B / elapsed_s.
//
// The dense-vs-paged kernel comparison isolates the indirection cost; the
// memory comparison shows what paging buys at the workload level. Both go
// into the report — they tell different parts of the same story.
// ===========================================================================

#include "kernels/paged_attention.cuh"
#include "utils/cuda_check.hpp"
#include "utils/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

constexpr int BLOCK_SIZE = 16;

// ---------------------------------------------------------------------------
// Truncated exponential: rate = 1/mean, capped at S_max. The +1 keeps len >= 1
// so every request has at least one historical token to attend to.
// ---------------------------------------------------------------------------
static std::vector<int> sample_lengths(int B, float mean, int S_max,
                                       unsigned seed) {
    std::mt19937 gen(seed);
    std::exponential_distribution<float> dist(1.0f / mean);
    std::vector<int> lens(B);
    for (int b = 0; b < B; ++b) {
        float x = dist(gen);
        int   l = std::max(1, std::min(S_max, static_cast<int>(x) + 1));
        lens[b] = l;
    }
    return lens;
}

// ---------------------------------------------------------------------------
// Convert lengths to (paged block ids per sequence, total blocks consumed).
// Physical block ids are sequential here (no shuffle) — the benchmark only
// needs to time the kernel, not stress the indirection pattern. The
// correctness test already verifies the gather under a scrambled mapping.
// ---------------------------------------------------------------------------
static void layout_blocks(const std::vector<int>& lens, int max_blocks,
                          std::vector<int>& block_table,
                          int& blocks_used) {
    int B = static_cast<int>(lens.size());
    block_table.assign(static_cast<size_t>(B) * max_blocks, -1);
    int cursor = 0;
    for (int b = 0; b < B; ++b) {
        int nblk = (lens[b] + BLOCK_SIZE - 1) / BLOCK_SIZE;
        for (int blk = 0; blk < nblk; ++blk) {
            block_table[b * max_blocks + blk] = cursor++;
        }
    }
    blocks_used = cursor;
}

// ---------------------------------------------------------------------------
// One sweep point. H, D, S_max fixed for the run; B varies across the sweep.
// Emits two PERF: lines (paged, dense) plus the memory accounting.
// ---------------------------------------------------------------------------
static void run_case(int B, int H, int D, int S_max, float mean_len,
                     int n_iter, unsigned seed) {
    auto lens = sample_lengths(B, mean_len, S_max, seed);

    long long sum_len = std::accumulate(lens.begin(), lens.end(), 0LL);
    int  max_blocks   = (S_max + BLOCK_SIZE - 1) / BLOCK_SIZE;

    std::vector<int> block_table;
    int blocks_used = 0;
    layout_blocks(lens, max_blocks, block_table, blocks_used);

    // ---- Memory accounting (workload-level, no GPU work) ----
    long long useful_bytes      = sum_len * H * D * 4LL * 2LL;
    long long naive_alloc_bytes = static_cast<long long>(B) * S_max * H * D * 4LL * 2LL;
    long long paged_alloc_bytes = static_cast<long long>(blocks_used) * BLOCK_SIZE
                                                       * H * D * 4LL * 2LL;
    double naive_util = static_cast<double>(useful_bytes) / naive_alloc_bytes;
    double paged_util = static_cast<double>(useful_bytes) / paged_alloc_bytes;

    // ---- Device buffers ----
    Tensor<float> dQ({B, H, D});
    Tensor<float> dO({B, H, D});
    Tensor<int>   d_seq_lens({B});
    Tensor<int>   d_block_table({B, max_blocks});

    // Paged pool — exactly blocks_used physical blocks of size H*BLOCK_SIZE*D.
    Tensor<float> dK_blocks({blocks_used, H, BLOCK_SIZE, D});
    Tensor<float> dV_blocks({blocks_used, H, BLOCK_SIZE, D});

    // Dense pool — sized for the naive layout (B * S_max).
    Tensor<float> dK_dense({B, H, S_max, D});
    Tensor<float> dV_dense({B, H, S_max, D});

    // Initialize all device buffers with random data — exact values don't
    // matter for timing, but we want them non-zero / non-nan.
    {
        std::vector<float> tmp;
        std::mt19937 g(seed + 11);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        auto randfill = [&](Tensor<float>& t) {
            tmp.resize(t.numel());
            for (auto& v : tmp) v = dist(g);
            t.copy_from_host(tmp.data());
        };
        randfill(dQ);
        randfill(dK_blocks);
        randfill(dV_blocks);
        randfill(dK_dense);
        randfill(dV_dense);
    }
    d_seq_lens.copy_from_host(lens.data());
    d_block_table.copy_from_host(block_table.data());

    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // ---- Time paged ----
    auto paged_launch = [&]() {
        launch_paged_attention_decode(dQ.data(), dK_blocks.data(), dV_blocks.data(),
                                      d_block_table.data(), d_seq_lens.data(),
                                      dO.data(),
                                      B, H, D, BLOCK_SIZE, max_blocks, scale);
    };
    paged_launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < n_iter; ++i) paged_launch();
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float paged_ms_total = 0.0f;
    cudaEventElapsedTime(&paged_ms_total, e0, e1);
    double paged_ms = paged_ms_total / n_iter;

    // ---- Time dense ----
    auto dense_launch = [&]() {
        launch_dense_attention_decode(dQ.data(), dK_dense.data(), dV_dense.data(),
                                      d_seq_lens.data(), dO.data(),
                                      B, H, D, S_max, scale);
    };
    dense_launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(e0);
    for (int i = 0; i < n_iter; ++i) dense_launch();
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float dense_ms_total = 0.0f;
    cudaEventElapsedTime(&dense_ms_total, e0, e1);
    double dense_ms = dense_ms_total / n_iter;
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);

    double paged_tps = (paged_ms > 0.0) ? (1000.0 * B / paged_ms) : 0.0;
    double dense_tps = (dense_ms > 0.0) ? (1000.0 * B / dense_ms) : 0.0;
    double mean_len_actual = static_cast<double>(sum_len) / B;

    std::printf("B=%4d  meanlen=%5.1f  blocks=%4d  "
                "naive_util=%.2f paged_util=%.2f  "
                "paged %.3f ms (%.0f tok/s)  dense %.3f ms (%.0f tok/s)\n",
                B, mean_len_actual, blocks_used,
                naive_util, paged_util,
                paged_ms, paged_tps, dense_ms, dense_tps);

    std::printf("PERF: kernel=paged B=%d H=%d D=%d S_max=%d mean_len=%.1f "
                "useful_bytes=%lld alloc_bytes=%lld util=%.4f step_ms=%.4f tps=%.1f\n",
                B, H, D, S_max, mean_len_actual,
                useful_bytes, paged_alloc_bytes, paged_util, paged_ms, paged_tps);

    std::printf("PERF: kernel=dense B=%d H=%d D=%d S_max=%d mean_len=%.1f "
                "useful_bytes=%lld alloc_bytes=%lld util=%.4f step_ms=%.4f tps=%.1f\n",
                B, H, D, S_max, mean_len_actual,
                useful_bytes, naive_alloc_bytes, naive_util, dense_ms, dense_tps);
}

int main(int argc, char** argv) {
    // Default sweep. Override per arg if you want a single point.
    int   H        = (argc > 1) ? std::atoi(argv[1]) : 8;
    int   D        = (argc > 2) ? std::atoi(argv[2]) : 64;
    int   S_max    = (argc > 3) ? std::atoi(argv[3]) : 128;
    float mean_len = (argc > 4) ? std::atof(argv[4]) : 64.0f;
    int   n_iter   = (argc > 5) ? std::atoi(argv[5]) : 200;

    std::printf("PagedAttention decode benchmark\n");
    std::printf("  H=%d D=%d S_max=%d mean_len=%.1f n_iter=%d block_size=%d\n\n",
                H, D, S_max, mean_len, n_iter, BLOCK_SIZE);

    for (int B : {1, 4, 16, 64, 256}) {
        run_case(B, H, D, S_max, mean_len, n_iter,
                 static_cast<unsigned>(0xC0FFEE) + B);
    }
    return 0;
}
