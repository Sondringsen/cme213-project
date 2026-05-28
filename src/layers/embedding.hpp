#pragma once

// ---------------------------------------------------------------------------
// Token embedding layer: maps integer token IDs to dense vectors.
//
// Forward:  out[n] = weight[ids[n]]       (gather)
// Backward: d_weight[ids[n]] += d_out[n]  (scatter-add, using atomicAdd)
//
// Caller must zero d_weight before accumulating across tokens.
// ---------------------------------------------------------------------------

#include "kernels/embedding.cuh"
#include "utils/tensor.hpp"
#include <random>
#include <vector>

struct EmbeddingLayer {
    int V;  // vocabulary size
    int D;  // embedding dimension

    Tensor<float> weight;    // (V, D)
    Tensor<float> d_weight;  // gradient w.r.t. weight (caller zeros)

    // Cached input ids from forward (device pointer, not owned)
    const int* ids_cache = nullptr;
    int        N_cache   = 0;

    EmbeddingLayer(int vocab_size, int embed_dim, unsigned seed = 1)
        : V(vocab_size), D(embed_dim),
          weight({vocab_size, embed_dim}),
          d_weight({vocab_size, embed_dim}) {
        // GPT-2 uses N(0, 0.02) for embedding init.
        std::mt19937 gen(seed);
        std::normal_distribution<float> dist(0.0f, 0.02f);
        std::vector<float> h_w(static_cast<size_t>(vocab_size) * embed_dim);
        for (auto& v : h_w) v = dist(gen);
        weight.copy_from_host(h_w.data());
        d_weight.zero();
    }

    // forward: ids (N,) → out (N, D)
    void forward(const int* ids, float* out, int N, cudaStream_t stream = 0) {
        ids_cache = ids;
        N_cache   = N;
        launch_embedding_forward(ids, weight.data(), out, N, V, D, stream);
    }

    // backward: d_out (N, D) → d_weight (V, D) accumulated
    void backward(const float* d_out, cudaStream_t stream = 0) {
        launch_embedding_backward(d_out, ids_cache, d_weight.data(),
                                  N_cache, V, D, stream);
    }
};
