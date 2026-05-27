#pragma once

// ---------------------------------------------------------------------------
// Single-GPU trainer for GPT2.
//
// Each call to step() runs:
//   1. Forward pass  → logits
//   2. Cross-entropy loss  (per-token, shape B*S)
//   3. Cross-entropy backward with upstream gradient = 1/(B*S) per token
//   4. Model backward  → gradients for all parameters
//   5. Adam update for every parameter tensor
//
// Returns the mean loss over the B*S tokens as a float (copied to host).
// Adam moment buffers (m, v) are allocated on first call and reused.
// ---------------------------------------------------------------------------

#include "kernels/kernels.cuh"
#include "model/gpt2.hpp"
#include "utils/tensor.hpp"
#include <memory>
#include <numeric>
#include <vector>
#include <cuda_runtime.h>

struct TrainerConfig {
    float lr     = 3e-4f;
    float beta1  = 0.9f;
    float beta2  = 0.999f;
    float eps    = 1e-8f;
};

struct Trainer {
    GPT2&         model;
    TrainerConfig cfg;
    int           step_count = 0;

    // Owned device buffers for loss computation
    Tensor<float> losses;    // (N,) per-token losses
    Tensor<float> dlosses;   // (N,) upstream gradient = 1/N per token
    Tensor<float> d_logits;  // (N, V) gradient w.r.t. logits

    // Adam moment buffers — allocated once on first step
    std::vector<std::unique_ptr<Tensor<float>>> m_bufs;
    std::vector<std::unique_ptr<Tensor<float>>> v_bufs;
    bool moments_initialized = false;

    Trainer(GPT2& model_, const TrainerConfig& cfg_ = {})
        : model(model_), cfg(cfg_),
          losses({model_.cfg.B * model_.cfg.S}),
          dlosses({model_.cfg.B * model_.cfg.S}),
          d_logits({model_.cfg.B * model_.cfg.S, model_.cfg.V}) {}

    // Run forward + backward pass. Returns mean cross-entropy loss (copies
    // losses to host; adds one device sync). Call optimizer_step() after
    // this — and after gradient all-reduce in a distributed setting.
    float forward_backward(const int* ids, const int* targets,
                           cudaStream_t stream = 0) {
        ++step_count;
        int N = model.cfg.B * model.cfg.S;
        int V = model.cfg.V;
        float inv_N = 1.0f / static_cast<float>(N);

        // 1. Forward
        model.forward(ids, stream);

        // 2. Loss (per-token)
        launch_cross_entropy_forward(model.logits.data(), targets,
                                     losses.data(), N, V, stream);

        // 3. Backward through CE loss
        //    dlosses[i] = 1/N so the gradient is mean-normalized
        launch_fill(dlosses.data(), inv_N, N, stream);
        launch_cross_entropy_backward(model.logits.data(), targets,
                                      dlosses.data(), d_logits.data(),
                                      N, V, stream);

        // 4. Model backward
        model.zero_grad(stream);
        model.backward(d_logits.data(), stream);

        // Return mean loss (host copy — one sync)
        cudaStreamSynchronize(stream);
        std::vector<float> h_losses(static_cast<size_t>(N));
        losses.copy_to_host(h_losses.data());
        return std::accumulate(h_losses.begin(), h_losses.end(), 0.0f) / N;
    }

    // Apply Adam optimizer to all parameters. Call after forward_backward()
    // (and, for multi-GPU, after gradient all-reduce).
    void optimizer_step(cudaStream_t stream = 0) {
        auto pg = model.param_grads();
        if (!moments_initialized) {
            for (auto& p : pg) {
                m_bufs.push_back(std::make_unique<Tensor<float>>(
                    std::vector<int>{p.n}));
                v_bufs.push_back(std::make_unique<Tensor<float>>(
                    std::vector<int>{p.n}));
                m_bufs.back()->zero();
                v_bufs.back()->zero();
            }
            moments_initialized = true;
        }

        for (size_t i = 0; i < pg.size(); ++i) {
            launch_adam(pg[i].param, pg[i].grad,
                        m_bufs[i]->data(), v_bufs[i]->data(),
                        cfg.lr, cfg.beta1, cfg.beta2, cfg.eps,
                        step_count, pg[i].n, stream);
        }
    }

    // Convenience: forward_backward + optimizer_step in a single call.
    // Use this for single-GPU training. For multi-GPU, call the two
    // functions separately so gradient all-reduce can happen between them.
    float step(const int* ids, const int* targets, cudaStream_t stream = 0) {
        float loss = forward_backward(ids, targets, stream);
        optimizer_step(stream);
        return loss;
    }
};
