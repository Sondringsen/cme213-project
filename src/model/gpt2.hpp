#pragma once

// ---------------------------------------------------------------------------
// Minimal GPT-2-style transformer language model.
//
// Architecture:
//   Token embedding  (V → C)
//   N_layers × TransformerBlock
//   Final LayerNorm
//   LM head: Linear (C → V)  [weight tied to embedding for memory efficiency]
//
// All parameters live on GPU as FP32 Tensors. The model owns all parameters
// and their gradients. The trainer calls forward(), backward(), and then
// uses params()/grads() to iterate over all (param, grad) pairs for Adam.
//
// Config: follows GPT-2 small (12 layers, 12 heads, C=768, V=50257) but
// caller can pass any values.
// ---------------------------------------------------------------------------

#include "kernels/kernels.cuh"
#include "layers/embedding.hpp"
#include "layers/transformer_block.hpp"
#include "layers/layernorm_layer.hpp"
#include "layers/linear.hpp"
#include "utils/tensor.hpp"
#include <vector>
#include <cstdio>

struct GPT2Config {
    int V       = 50257;  // vocab size
    int C       = 768;    // model dim (must be divisible by n_heads)
    int n_heads = 12;
    int n_layers= 12;
    int S       = 512;    // sequence length
    int B       = 8;      // batch size (local, per MPI rank)
};

struct GPT2 {
    GPT2Config cfg;

    EmbeddingLayer              embed;
    std::vector<TransformerBlock*> blocks;  // owning pointers
    LayerNormLayer              final_ln;
    Linear                      lm_head;   // (V, C) — tied with embed.weight

    // Intermediate activations (device, owned)
    Tensor<float> embed_out;    // (B*S, C)
    Tensor<float> block_in;     // (B*S, C) — alternating buffer
    Tensor<float> block_out;    // (B*S, C)
    Tensor<float> ln_out;       // (B*S, C)
    Tensor<float> logits;       // (B*S, V)

    // For backward
    Tensor<float> d_logits;     // (B*S, V) — gradient from cross-entropy
    Tensor<float> d_ln_out;     // (B*S, C)
    Tensor<float> d_block;      // (B*S, C) — gradient flowing through blocks

    explicit GPT2(const GPT2Config& config)
        : cfg(config),
          embed(config.V, config.C),
          final_ln(config.C, config.B * config.S),
          lm_head(config.C, config.V),
          embed_out({config.B * config.S, config.C}),
          block_in({config.B * config.S, config.C}),
          block_out({config.B * config.S, config.C}),
          ln_out({config.B * config.S, config.C}),
          logits({config.B * config.S, config.V}),
          d_logits({config.B * config.S, config.V}),
          d_ln_out({config.B * config.S, config.C}),
          d_block({config.B * config.S, config.C}) {
        blocks.reserve(config.n_layers);
        for (int i = 0; i < config.n_layers; ++i) {
            blocks.push_back(
                new TransformerBlock(config.C, config.n_heads,
                                     config.S, config.B, /*causal=*/true));
        }
    }

    ~GPT2() {
        for (auto* b : blocks) delete b;
    }

    GPT2(const GPT2&)            = delete;
    GPT2& operator=(const GPT2&) = delete;

    // forward: ids (B*S,) → logits (B*S, V)
    // After calling, logits.data() contains the output logits.
    void forward(const int* ids, cudaStream_t stream = 0) {
        int N = cfg.B * cfg.S;

        embed.forward(ids, embed_out.data(), N, stream);

        // Copy embed_out into block_in for the first block
        cudaMemcpyAsync(block_in.data(), embed_out.data(),
                        static_cast<size_t>(N) * cfg.C * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);

        for (int i = 0; i < cfg.n_layers; ++i) {
            blocks[i]->forward(block_in.data(), block_out.data(), stream);
            // Swap buffers for next block
            cudaMemcpyAsync(block_in.data(), block_out.data(),
                            static_cast<size_t>(N) * cfg.C * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
        }

        final_ln.forward(block_in.data(), ln_out.data(), N, stream);
        lm_head.forward(ln_out.data(), logits.data(), N, stream);
    }

    // backward: given d_logits (B*S, V), propagate gradients to all parameters.
    // The caller typically fills d_logits via launch_cross_entropy_backward.
    void backward(const float* d_logits_in, cudaStream_t stream = 0) {
        int N = cfg.B * cfg.S;

        // LM head backward
        lm_head.backward(d_logits_in, d_ln_out.data(), stream);

        // Final LayerNorm backward
        final_ln.backward(d_ln_out.data(), d_block.data(), stream);

        // Transformer blocks — backward in reverse order
        for (int i = cfg.n_layers - 1; i >= 0; --i) {
            Tensor<float> d_in({N, cfg.C});
            blocks[i]->backward(d_block.data(), d_in.data(), stream);
            cudaMemcpyAsync(d_block.data(), d_in.data(),
                            static_cast<size_t>(N) * cfg.C * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
        }

        // Embedding backward (scatter-add into embed.d_weight)
        embed.backward(d_block.data(), stream);
    }

    // Collect all (param, grad, n_elements) tuples for the Adam step.
    // Caller iterates over this to apply the optimizer.
    struct ParamGrad {
        float*       param;
        const float* grad;
        int          n;
    };

    std::vector<ParamGrad> param_grads() {
        std::vector<ParamGrad> pg;
        auto add = [&](float* p, const float* g, size_t n) {
            pg.push_back({p, g, static_cast<int>(n)});
        };

        add(embed.weight.data(), embed.d_weight.data(), embed.weight.numel());
        for (auto* b : blocks) {
            // LayerNorm 1
            add(b->ln1.gamma.data(), b->ln1.d_gamma.data(), b->ln1.gamma.numel());
            add(b->ln1.beta.data(),  b->ln1.d_beta.data(),  b->ln1.beta.numel());
            // MHA projections (W_q, W_k, W_v, W_out)
            auto& mha = b->mha;
            add(mha.W_q.W.data(), mha.W_q.d_W.data(), mha.W_q.W.numel());
            add(mha.W_k.W.data(), mha.W_k.d_W.data(), mha.W_k.W.numel());
            add(mha.W_v.W.data(), mha.W_v.d_W.data(), mha.W_v.W.numel());
            add(mha.W_out.W.data(), mha.W_out.d_W.data(), mha.W_out.W.numel());
            // LayerNorm 2
            add(b->ln2.gamma.data(), b->ln2.d_gamma.data(), b->ln2.gamma.numel());
            add(b->ln2.beta.data(),  b->ln2.d_beta.data(),  b->ln2.beta.numel());
            // FFN
            add(b->fc1.W.data(), b->fc1.d_W.data(), b->fc1.W.numel());
            add(b->fc2.W.data(), b->fc2.d_W.data(), b->fc2.W.numel());
        }
        // Final LN
        add(final_ln.gamma.data(), final_ln.d_gamma.data(), final_ln.gamma.numel());
        add(final_ln.beta.data(),  final_ln.d_beta.data(),  final_ln.beta.numel());
        // LM head
        add(lm_head.W.data(), lm_head.d_W.data(), lm_head.W.numel());

        return pg;
    }

    // Zero all parameter gradients. Call before each backward pass.
    void zero_grad(cudaStream_t stream = 0) {
        embed.d_weight.zero();
        for (auto* b : blocks) {
            b->ln1.d_gamma.zero(); b->ln1.d_beta.zero();
            b->mha.W_q.d_W.zero();
            b->mha.W_k.d_W.zero();
            b->mha.W_v.d_W.zero();
            b->mha.W_out.d_W.zero();
            b->ln2.d_gamma.zero(); b->ln2.d_beta.zero();
            b->fc1.d_W.zero();
            b->fc2.d_W.zero();
        }
        final_ln.d_gamma.zero(); final_ln.d_beta.zero();
        lm_head.d_W.zero();
        (void)stream;
    }
};
