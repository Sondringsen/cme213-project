#pragma once

// ---------------------------------------------------------------------------
// MPI data-parallelism helpers for distributed GPT-2 training.
//
// Design (data parallelism):
//   Each MPI rank owns one GPU and processes a local slice of the global
//   mini-batch.  After the backward pass, gradients are averaged across all
//   ranks with MPI_Allreduce.  Every rank then runs an identical Adam step,
//   so weights stay in sync without an explicit broadcast after each step.
//
// Gradient transfer strategy:
//   Device → host copy, MPI_Allreduce (sum), scale by 1/n_ranks, host →
//   device copy.  This avoids requiring CUDA-aware MPI.  A future version
//   can add CUDA-aware fast paths when the cluster supports them.
//
// Usage:
//   mpi_init(argc, argv);               // call first; sets device
//   broadcast_weights(model);           // once at startup
//   for each step:
//     float loss = trainer.forward_backward(local_ids, local_targets);
//     allreduce_gradients(model);        // average gradients across ranks
//     trainer.optimizer_step();         // identical Adam on all ranks
//   mpi_finalize();
// ---------------------------------------------------------------------------

#ifdef MPI_ENABLED
#include <mpi.h>
#endif

#include "model/gpt2.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <vector>

// ---------------------------------------------------------------------------
// Global state: rank and size are set by mpi_init and read by the helpers.
// ---------------------------------------------------------------------------
namespace mpi_state {
    inline int rank = 0;
    inline int size = 1;
}

// ---------------------------------------------------------------------------
// mpi_init — initialize MPI, assign GPU (one GPU per rank in round-robin),
//            and record rank/size in global state.
// ---------------------------------------------------------------------------
inline void mpi_init(int argc, char** argv) {
#ifdef MPI_ENABLED
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_state::rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_state::size);

    // Assign one GPU per rank. The Turing nodes have 4 GPUs each; for
    // single-node runs this cycles through devices 0–3.
    int n_devices = 0;
    cudaGetDeviceCount(&n_devices);
    int dev = mpi_state::rank % (n_devices > 0 ? n_devices : 1);
    cudaSetDevice(dev);

    if (mpi_state::rank == 0) {
        std::printf("[MPI] %d rank(s), rank 0 on GPU %d\n",
                    mpi_state::size, dev);
    }
#else
    (void)argc; (void)argv;
#endif
}

// ---------------------------------------------------------------------------
// mpi_finalize — call at program exit.
// ---------------------------------------------------------------------------
inline void mpi_finalize() {
#ifdef MPI_ENABLED
    MPI_Finalize();
#endif
}

inline int mpi_rank() { return mpi_state::rank; }
inline int mpi_size() { return mpi_state::size; }

// ---------------------------------------------------------------------------
// broadcast_weights — copy all model parameters from rank 0 to every rank.
// Call once after model construction so all ranks start from the same weights.
// ---------------------------------------------------------------------------
inline void broadcast_weights(GPT2& model) {
#ifdef MPI_ENABLED
    if (mpi_state::size == 1) return;

    auto pg = model.param_grads();
    for (auto& p : pg) {
        // Copy weights from device to a host buffer on every rank.
        // MPI_Bcast then overwrites non-root ranks with rank-0's values.
        std::vector<float> h_param(static_cast<size_t>(p.n));
        cudaMemcpy(h_param.data(), p.param,
                   static_cast<size_t>(p.n) * sizeof(float),
                   cudaMemcpyDeviceToHost);

        MPI_Bcast(h_param.data(), p.n, MPI_FLOAT, 0, MPI_COMM_WORLD);

        cudaMemcpy(p.param, h_param.data(),
                   static_cast<size_t>(p.n) * sizeof(float),
                   cudaMemcpyHostToDevice);
    }
#else
    (void)model;
#endif
}

// ---------------------------------------------------------------------------
// allreduce_gradients — sum gradients across all ranks then divide by n_ranks.
// After this call every rank holds the same averaged gradient, so identical
// Adam steps will keep weights synchronized.
//
// All gradients are packed into one pinned host buffer (lazy-allocated on
// first call), reducing ~50 per-tensor MPI_Allreduce calls to one.
// Pinned memory enables full PCIe DMA bandwidth on the D↔H transfers.
// ---------------------------------------------------------------------------
inline void allreduce_gradients(GPT2& model) {
#ifdef MPI_ENABLED
    if (mpi_state::size == 1) return;

    auto pg = model.param_grads();

    // Lazy-allocate a pinned host buffer sized to hold all gradients.
    static float* s_buf   = nullptr;
    static int    s_total = 0;

    int total = 0;
    for (auto& p : pg) total += p.n;

    if (s_buf == nullptr || total != s_total) {
        if (s_buf) cudaFreeHost(s_buf);
        cudaMallocHost(&s_buf, static_cast<size_t>(total) * sizeof(float));
        s_total = total;
    }

    // Pack all gradients from GPU into the contiguous host buffer.
    float* ptr = s_buf;
    for (auto& p : pg) {
        cudaMemcpy(ptr, p.grad,
                   static_cast<size_t>(p.n) * sizeof(float),
                   cudaMemcpyDeviceToHost);
        ptr += p.n;
    }

    // One MPI_Allreduce on the full buffer instead of ~50 per-tensor calls.
    MPI_Allreduce(MPI_IN_PLACE, s_buf, s_total, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    // Scale from sum → average and scatter back to device.
    float inv_n = 1.0f / static_cast<float>(mpi_state::size);
    ptr = s_buf;
    for (auto& p : pg) {
        for (int i = 0; i < p.n; ++i) ptr[i] *= inv_n;
        cudaMemcpy(const_cast<float*>(p.grad), ptr,
                   static_cast<size_t>(p.n) * sizeof(float),
                   cudaMemcpyHostToDevice);
        ptr += p.n;
    }
#else
    (void)model;
#endif
}

// ---------------------------------------------------------------------------
// scatter_batch — extract rank's slice from a global host batch.
//
// The caller pre-generates the full global batch (total_B * S tokens) on
// host, then calls this to get each rank's contiguous sub-batch.  Each rank
// receives tokens [rank*local_B*S, (rank+1)*local_B*S).
//
// Parameters:
//   full_ids     — host array (total_B * S,)
//   full_targets — host array (total_B * S,)
//   local_ids    — output host array (local_B * S,) pre-allocated by caller
//   local_targets— output host array (local_B * S,) pre-allocated by caller
//   local_B      — local batch size (total_B / n_ranks)
//   S            — sequence length
//   rank         — this rank's index
// ---------------------------------------------------------------------------
inline void scatter_batch(const int* full_ids, const int* full_targets,
                          int* local_ids, int* local_targets,
                          int local_B, int S, int rank) {
    int offset = rank * local_B * S;
    int count  = local_B * S;
    std::copy(full_ids     + offset, full_ids     + offset + count, local_ids);
    std::copy(full_targets + offset, full_targets + offset + count, local_targets);
}
