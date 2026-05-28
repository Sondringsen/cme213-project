// ===========================================================================
// main_distributed.cu — Distributed GPT-2 training with MPI data parallelism.
//
// Each MPI rank:
//   1. Initializes on a separate GPU (rank % n_gpus_per_node).
//   2. Builds an identical GPT-2 model with local batch size B/n_ranks.
//   3. Receives weight initialization from rank 0 via broadcast_weights().
//
// Training loop (each step):
//   a. Slice the global synthetic batch for this rank.
//   b. Copy local batch tokens to GPU.
//   c. forward_backward() — local forward + backward pass.
//   d. allreduce_gradients() — average gradients across all ranks.
//   e. optimizer_step() — identical Adam update on all ranks.
//
// Since every rank sees the same averaged gradient and runs the same Adam
// step, weights stay synchronized without re-broadcasting after each step.
//
// Usage (on the cluster):
//   mpirun -np 4 ./train_distributed [n_steps] [n_layers] [C] [S] [total_B]
// ===========================================================================

#include "model/gpt2.hpp"
#include "mpi/data_parallel.hpp"
#include "training/trainer.hpp"
#include "utils/cuda_check.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <chrono>

// NVTX range markers so Nsight Systems shows named regions in the timeline.
// Compiled out when nvToolsExt is not available (no overhead in that case).
#ifdef WITH_NVTX
#  include <nvToolsExt.h>
#  define NVTX_PUSH(name) nvtxRangePushA(name)
#  define NVTX_POP()      nvtxRangePop()
#else
#  define NVTX_PUSH(name) ((void)0)
#  define NVTX_POP()      ((void)0)
#endif

// ---------------------------------------------------------------------------
// Synthetic batch generator.  Every call produces a new random batch so that
// the model actually has something to train on (loss should decrease over time
// as it learns to predict the random sequences it sees repeatedly).
// Deterministic given (step, rank) so multi-rank runs are reproducible.
// ---------------------------------------------------------------------------
static void gen_batch(int* ids, int* targets, int total_tokens, int V,
                      unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, V - 1);
    for (int i = 0; i < total_tokens; ++i) {
        ids[i]     = dist(gen);
        targets[i] = dist(gen);
    }
}

int main(int argc, char** argv) {
    // Parse optional CLI arguments
    int n_steps   = (argc > 1) ? std::atoi(argv[1]) : 20;
    int n_layers  = (argc > 2) ? std::atoi(argv[2]) : 4;
    int C         = (argc > 3) ? std::atoi(argv[3]) : 128;
    int S         = (argc > 4) ? std::atoi(argv[4]) : 64;
    int total_B   = (argc > 5) ? std::atoi(argv[5]) : 8;  // global batch

    // ---- MPI init ----
    mpi_init(argc, argv);
    int rank    = mpi_rank();
    int n_ranks = mpi_size();

    if (total_B % n_ranks != 0) {
        if (rank == 0)
            std::fprintf(stderr,
                "[Error] total_B=%d must be divisible by n_ranks=%d\n",
                total_B, n_ranks);
        mpi_finalize();
        return 1;
    }

    int local_B = total_B / n_ranks;

    if (rank == 0) {
        std::printf("=== Distributed GPT-2 training ===\n");
        std::printf("  ranks=%d  local_B=%d  S=%d  C=%d  layers=%d  steps=%d\n",
                    n_ranks, local_B, S, C, n_layers, n_steps);
    }

    // ---- Model construction ----
    GPT2Config cfg;
    cfg.n_layers = n_layers;
    cfg.n_heads  = (C >= 128) ? 8 : 4;   // keep head_dim ≥ 16
    cfg.C        = C;
    cfg.V        = 512;    // small vocab for the synthetic demo
    cfg.S        = S;
    cfg.B        = local_B;

    GPT2 model(cfg);

    // ---- Broadcast rank-0's random init to all ranks ----
    broadcast_weights(model);

    // ---- Trainer (owns Adam moment buffers) ----
    TrainerConfig tcfg;
    tcfg.lr = 1e-3f;
    Trainer trainer(model, tcfg);

    // ---- Host batch buffers ----
    int global_tokens = total_B  * S;
    int local_tokens  = local_B  * S;

    std::vector<int> h_full_ids    (global_tokens);
    std::vector<int> h_full_targets(global_tokens);
    std::vector<int> h_local_ids   (local_tokens);
    std::vector<int> h_local_tgts  (local_tokens);

    // ---- Device batch buffers ----
    int *d_ids = nullptr, *d_targets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ids,     local_tokens * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_targets, local_tokens * sizeof(int)));

    // ---- Training loop ----
    double total_step_time = 0.0;
    double total_comm_time = 0.0;

    for (int step = 1; step <= n_steps; ++step) {
        char step_label[32];
        std::snprintf(step_label, sizeof(step_label), "step_%d", step);
        NVTX_PUSH(step_label);

        // Generate the same global batch on every rank (same seed), then
        // slice the local portion.  Using the step as the seed means each
        // step has a fresh batch.
        gen_batch(h_full_ids.data(), h_full_targets.data(),
                  global_tokens, cfg.V, static_cast<unsigned>(step));

        scatter_batch(h_full_ids.data(), h_full_targets.data(),
                      h_local_ids.data(), h_local_tgts.data(),
                      local_B, S, rank);

        CUDA_CHECK(cudaMemcpy(d_ids, h_local_ids.data(),
                              local_tokens * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_targets, h_local_tgts.data(),
                              local_tokens * sizeof(int), cudaMemcpyHostToDevice));

        // Time the full step
        auto t0 = std::chrono::steady_clock::now();

        // a. Local forward + backward
        NVTX_PUSH("forward_backward");
        float loss = trainer.forward_backward(d_ids, d_targets);
        NVTX_POP();

        // b. Gradient all-reduce (timed separately for the scaling study)
        auto t_comm0 = std::chrono::steady_clock::now();
        NVTX_PUSH("allreduce");
        allreduce_gradients(model);
        NVTX_POP();
        auto t_comm1 = std::chrono::steady_clock::now();

        // c. Adam step (identical on all ranks)
        NVTX_PUSH("optimizer");
        trainer.optimizer_step();
        NVTX_POP();

        auto t1 = std::chrono::steady_clock::now();

        double step_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double comm_ms = std::chrono::duration<double, std::milli>(t_comm1 - t_comm0).count();
        total_step_time += step_ms;
        total_comm_time += comm_ms;

        NVTX_POP(); // step_N

        if (rank == 0) {
            std::printf("step %3d | loss %.4f | step %.1f ms | comm %.1f ms (%.1f%%)\n",
                        step, loss, step_ms, comm_ms,
                        100.0 * comm_ms / step_ms);
            std::fflush(stdout);
        }
    }

    // ---- Summary (rank 0 only) ----
    if (rank == 0 && n_steps > 0) {
        double avg_step = total_step_time / n_steps;
        double avg_comm = total_comm_time / n_steps;
        std::printf("\n=== Summary ===\n");
        std::printf("  avg step time : %.2f ms\n", avg_step);
        std::printf("  avg comm time : %.2f ms (%.1f%% of step)\n",
                    avg_comm, 100.0 * avg_comm / avg_step);
        std::printf("PERF: ranks=%d local_B=%d S=%d C=%d layers=%d "
                    "avg_step_ms=%.2f comm_fraction=%.3f\n",
                    n_ranks, local_B, S, C, n_layers,
                    avg_step, avg_comm / avg_step);
    }

    CUDA_CHECK(cudaFree(d_ids));
    CUDA_CHECK(cudaFree(d_targets));
    mpi_finalize();
    return 0;
}
