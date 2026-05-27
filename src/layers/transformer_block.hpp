#pragma once

// ---------------------------------------------------------------------------
// Transformer block: Pre-LN variant (LayerNorm before each sub-layer).
//
// Forward:
//   h  = x + MHA(LN1(x))          — self-attention with residual
//   out = h + FFN(LN2(h))          — feed-forward with residual
//
// where FFN(z) = GELU(z * W1^T) * W2^T (two-layer MLP, hidden = 4*C).
//
// Backward is the mirror in reverse, with gradients flowing through both
// the main path and the residual connection at each stage.
// ---------------------------------------------------------------------------

#include "kernels/gelu.cuh"
#include "kernels/pointwise.cuh"
#include "layers/layernorm_layer.hpp"
#include "layers/attention_layer.hpp"
#include "layers/linear.hpp"
#include "utils/tensor.hpp"
#include <cuda_runtime.h>

struct TransformerBlock {
    int C;    // model dimension
    int FC;   // FFN hidden dimension (4*C)

    LayerNormLayer   ln1;
    MultiHeadAttention mha;
    LayerNormLayer   ln2;
    Linear           fc1;   // (FC, C)
    Linear           fc2;   // (C, FC)

    // Cached forward activations for backward
    // All are device pointers into owned Tensors
    Tensor<float> ln1_out;    // LN1(x)
    Tensor<float> attn_out;   // output of MHA (before residual)
    Tensor<float> h;          // x + attn_out  (residual after MHA)
    Tensor<float> ln2_out;    // LN2(h)
    Tensor<float> fc1_out;    // FC1 pre-GELU
    Tensor<float> gelu_out;   // GELU(fc1_out)

    // N = B*S is fixed at construction time (simplifies buffer sizes)
    TransformerBlock(int C_, int H_attn, int S, int B, bool causal = true)
        : C(C_), FC(4 * C_),
          ln1(C_, B * S),
          mha(B, H_attn, S, C_ / H_attn, causal),
          ln2(C_, B * S),
          fc1(C_, 4 * C_),
          fc2(4 * C_, C_),
          ln1_out({B * S, C_}),
          attn_out({B * S, C_}),
          h({B * S, C_}),
          ln2_out({B * S, C_}),
          fc1_out({B * S, 4 * C_}),
          gelu_out({B * S, 4 * C_}) {}

    // forward: x (B*S, C) → out (B*S, C)
    void forward(const float* x, float* out, cudaStream_t stream = 0) {
        int N = static_cast<int>(ln1_out.numel() / C);

        // ---- Attention sub-layer ----
        ln1.forward(x, ln1_out.data(), N, stream);
        mha.forward(ln1_out.data(), attn_out.data(), stream);
        // h = x + attn_out  (residual)
        cudaMemcpyAsync(h.data(), x, static_cast<size_t>(N) * C * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
        launch_add_inplace(h.data(), attn_out.data(), N * C, stream);

        // ---- FFN sub-layer ----
        ln2.forward(h.data(), ln2_out.data(), N, stream);
        fc1.forward(ln2_out.data(), fc1_out.data(), N, stream);
        launch_gelu_forward(fc1_out.data(), gelu_out.data(), N * FC, stream);
        fc2.forward(gelu_out.data(), out, N, stream);
        // out = h + fc2_out  (residual)
        launch_add_inplace(out, h.data(), N * C, stream);
    }

    // backward: d_out (B*S, C) → d_x (B*S, C)
    void backward(const float* d_out, float* d_x, cudaStream_t stream = 0) {
        int N = static_cast<int>(ln1_out.numel() / C);

        // ---- Backward through FFN sub-layer ----
        // Residual: gradient passes through directly to d_h, and also
        // flows through the FFN path.
        Tensor<float> d_h({N, C});
        cudaMemcpyAsync(d_h.data(), d_out,
                        static_cast<size_t>(N) * C * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);

        // fc2 backward: d_gelu_out = d_out * W_fc2
        Tensor<float> d_gelu_out({N, FC});
        fc2.backward(d_out, d_gelu_out.data(), stream);

        // GELU backward: d_fc1_out = d_gelu_out * gelu'(fc1_out)
        Tensor<float> d_fc1_out({N, FC});
        launch_gelu_backward(d_gelu_out.data(), fc1_out.data(),
                             d_fc1_out.data(), N * FC, stream);

        // fc1 backward: d_ln2_out = d_fc1_out * W_fc1
        Tensor<float> d_ln2_out({N, C});
        fc1.backward(d_fc1_out.data(), d_ln2_out.data(), stream);

        // ln2 backward: d_h_ffn (contribution to d_h from FFN path)
        Tensor<float> d_h_ffn({N, C});
        ln2.backward(d_ln2_out.data(), d_h_ffn.data(), stream);

        // Accumulate FFN gradient into d_h
        launch_add_inplace(d_h.data(), d_h_ffn.data(), N * C, stream);

        // ---- Backward through attention sub-layer ----
        // Residual: d_x gets a copy of d_h
        cudaMemcpyAsync(d_x, d_h.data(),
                        static_cast<size_t>(N) * C * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);

        // mha backward: d_ln1_out
        Tensor<float> d_ln1_out({N, C});
        mha.backward(d_h.data(), d_ln1_out.data(), stream);

        // ln1 backward: d_x_attn (contribution from attention path)
        Tensor<float> d_x_attn({N, C});
        ln1.backward(d_ln1_out.data(), d_x_attn.data(), stream);

        // Accumulate attention gradient into d_x
        launch_add_inplace(d_x, d_x_attn.data(), N * C, stream);
    }
};
