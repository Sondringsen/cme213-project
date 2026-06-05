#pragma once

// ---------------------------------------------------------------------------
// Multi-Head Self-Attention layer.
//
// Model shape: x is (B*S, C) where C = H * D.
//
// Forward:
//   1. Project x → Q, K, V separately: each (B*S, C)
//   2. Reshape each from (B*S, C) to (B, H, S, D)
//   3. Flash Attention forward → attn_out (B, H, S, D)
//   4. Reshape attn_out → (B*S, C)
//   5. Output projection: out = attn_flat * W_out^T
//
// Backward is the mirror sequence in reverse.
// ---------------------------------------------------------------------------

#include "kernels/attention.cuh"
#include "kernels/reshape.cuh"
#include "kernels/pointwise.cuh"
#include "layers/linear.hpp"
#include "utils/tensor.hpp"
#include <cmath>

struct MultiHeadAttention {
    int B, H, S, D, C;
    float scale;
    bool  causal;

    Linear W_q;    // (C, C)
    Linear W_k;    // (C, C)
    Linear W_v;    // (C, C)
    Linear W_out;  // (C, C)

    // Forward intermediate buffers
    Tensor<float> Q_flat;   // (B*S, C)
    Tensor<float> K_flat;   // (B*S, C)
    Tensor<float> V_flat;   // (B*S, C)
    Tensor<float> Q_bhsd;   // (B, H, S, D)
    Tensor<float> K_bhsd;   // (B, H, S, D)
    Tensor<float> V_bhsd;   // (B, H, S, D)
    Tensor<float> O_bhsd;   // (B, H, S, D)
    Tensor<float> O_flat;   // (B*S, C)

    // Backward scratch buffers — pre-allocated once to avoid per-step cudaMalloc.
    // The attention backward materializes two S×S matrices (P and dP) per
    // batch*head; allocating them here amortizes the ~30 ms/step allocation cost.
    Tensor<float> d_O_flat_buf;   // (B*S, C)
    Tensor<float> d_O_bhsd_buf;   // (B, H, S, D)
    Tensor<float> d_Q_bhsd_buf;   // (B, H, S, D)
    Tensor<float> d_K_bhsd_buf;   // (B, H, S, D)
    Tensor<float> d_V_bhsd_buf;   // (B, H, S, D)
    Tensor<float> d_Q_flat_buf;   // (B*S, C)
    Tensor<float> d_K_flat_buf;   // (B*S, C)
    Tensor<float> d_V_flat_buf;   // (B*S, C)
    Tensor<float> tmp_buf;        // (B*S, C) — W_q/k/v backward scratch
    Tensor<float> attn_P_buf;     // (B*H, S*S) — attention weights
    Tensor<float> attn_dP_buf;    // (B*H, S*S) — attention weight gradients

    MultiHeadAttention(int B_, int H_, int S_, int D_, bool causal_ = true)
        : B(B_), H(H_), S(S_), D(D_), C(H_ * D_),
          scale(1.0f / std::sqrt(static_cast<float>(D_))),
          causal(causal_),
          W_q(C, C), W_k(C, C), W_v(C, C), W_out(C, C),
          Q_flat({B_* S_, C}), K_flat({B_* S_, C}), V_flat({B_* S_, C}),
          Q_bhsd({B_, H_, S_, D_}), K_bhsd({B_, H_, S_, D_}),
          V_bhsd({B_, H_, S_, D_}),
          O_bhsd({B_, H_, S_, D_}),
          O_flat({B_* S_, C}),
          d_O_flat_buf({B_* S_, C}),
          d_O_bhsd_buf({B_, H_, S_, D_}),
          d_Q_bhsd_buf({B_, H_, S_, D_}),
          d_K_bhsd_buf({B_, H_, S_, D_}),
          d_V_bhsd_buf({B_, H_, S_, D_}),
          d_Q_flat_buf({B_* S_, C}),
          d_K_flat_buf({B_* S_, C}),
          d_V_flat_buf({B_* S_, C}),
          tmp_buf({B_* S_, C}),
          attn_P_buf({B_* H_, S_* S_}),
          attn_dP_buf({B_* H_, S_* S_}) {}

    // forward: x (B*S, C) → out (B*S, C)
    void forward(const float* x, float* out, cudaStream_t stream = 0) {
        int N = B * S;

        // Project x → Q, K, V  (each N × C)
        W_q.forward(x, Q_flat.data(), N, stream);
        W_k.forward(x, K_flat.data(), N, stream);
        W_v.forward(x, V_flat.data(), N, stream);

        // Reshape (N, C) → (B, H, S, D)
        launch_flat_to_bhsd(Q_flat.data(), Q_bhsd.data(), B, H, S, D, stream);
        launch_flat_to_bhsd(K_flat.data(), K_bhsd.data(), B, H, S, D, stream);
        launch_flat_to_bhsd(V_flat.data(), V_bhsd.data(), B, H, S, D, stream);

        // Flash Attention
        launch_flash_attention_forward(Q_bhsd.data(), K_bhsd.data(), V_bhsd.data(),
                                       O_bhsd.data(), B, H, S, D, scale, causal, stream);

        // Reshape (B, H, S, D) → (N, C)
        launch_bhsd_to_flat(O_bhsd.data(), O_flat.data(), B, H, S, D, stream);

        // Output projection
        W_out.forward(O_flat.data(), out, N, stream);
    }

    // backward: d_out (B*S, C) → d_x (B*S, C)
    void backward(const float* d_out, float* d_x, cudaStream_t stream = 0) {
        int N = B * S;

        // 5b. Backward through output projection
        W_out.backward(d_out, d_O_flat_buf.data(), stream);

        // 4b. Reshape d_O_flat (N, C) → d_O_bhsd (B, H, S, D)
        launch_flat_to_bhsd(d_O_flat_buf.data(), d_O_bhsd_buf.data(), B, H, S, D, stream);

        // 3b. Attention backward (uses pre-allocated P and dP scratch buffers)
        launch_attention_backward(Q_bhsd.data(), K_bhsd.data(), V_bhsd.data(),
                                  d_O_bhsd_buf.data(),
                                  d_Q_bhsd_buf.data(), d_K_bhsd_buf.data(), d_V_bhsd_buf.data(),
                                  B, H, S, D, scale, causal,
                                  attn_P_buf.data(), attn_dP_buf.data(), stream);

        // 2b. Reshape (B, H, S, D) → (N, C)
        launch_bhsd_to_flat(d_Q_bhsd_buf.data(), d_Q_flat_buf.data(), B, H, S, D, stream);
        launch_bhsd_to_flat(d_K_bhsd_buf.data(), d_K_flat_buf.data(), B, H, S, D, stream);
        launch_bhsd_to_flat(d_V_bhsd_buf.data(), d_V_flat_buf.data(), B, H, S, D, stream);

        // 1b. Backward through Q, K, V projections
        //     d_x accumulates contributions from all three; must zero first
        cudaMemsetAsync(d_x, 0, static_cast<size_t>(N) * C * sizeof(float), stream);

        W_q.backward(d_Q_flat_buf.data(), tmp_buf.data(), stream);
        launch_add_inplace(d_x, tmp_buf.data(), N * C, stream);

        W_k.backward(d_K_flat_buf.data(), tmp_buf.data(), stream);
        launch_add_inplace(d_x, tmp_buf.data(), N * C, stream);

        W_v.backward(d_V_flat_buf.data(), tmp_buf.data(), stream);
        launch_add_inplace(d_x, tmp_buf.data(), N * C, stream);
    }
};
