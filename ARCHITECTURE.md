# CME213 GPT-2 — Architecture Overview

This document explains how all the pieces of the codebase connect and how data flows through the system during training.

---

## The Four Layers of Abstraction

```
src/
├── kernels/      ← Layer 1: raw CUDA kernels (compute primitives)
├── layers/       ← Layer 2: stateful modules (weights + forward/backward logic)
├── model/        ← Layer 3: full GPT-2 assembled from layers
└── training/     ← Layer 4: training loop (loss, optimizer, step)
        +
src/mpi/          ← Layer 5: multi-GPU gradient synchronization
```

Each layer only talks to the one below it. Tests can target any layer independently.

---

## Layer 1 — Kernels (`src/kernels/`)

These are pure compute functions. They take raw device pointers and do math. No state, no memory ownership.

| File | What it computes | Key functions |
|------|-----------------|---------------|
| `gemm.cu` | Matrix multiply C = A × B | `launch_gemm_tiled` (forward), `launch_gemm_tn` (A^T×B), `launch_gemm_nt` (A×B^T) |
| `layernorm.cu` | Per-row normalize + affine | `launch_layernorm_forward` (saves mean/rstd), `launch_layernorm_backward` |
| `gelu.cu` | GELU activation | `launch_gelu_forward`, `launch_gelu_backward` |
| `softmax.cu` | Row-wise softmax | `launch_softmax_forward` |
| `cross_entropy.cu` | Per-token CE loss | `launch_cross_entropy_forward`, `launch_cross_entropy_backward` |
| `attention.cu` | Self-attention | `launch_flash_attention_forward` (online softmax, O(S·D) memory), `launch_attention_backward` (naive, O(S²) memory) |
| `embedding.cu` | Token lookup/scatter | `launch_embedding_forward` (gather), `launch_embedding_backward` (atomicAdd scatter) |
| `adam.cu` | Parameter update | `launch_adam` (fused bias-corrected Adam) |
| `reshape.cu` | Layout conversion | `launch_flat_to_bhsd`, `launch_bhsd_to_flat` |
| `pointwise.cu` | Element-wise helpers | `launch_add_inplace`, `launch_scale_inplace`, `launch_fill` |

All kernel headers are re-exported through `kernels/kernels.cuh` — you only need one include to get everything.

### Why separate forward and backward?
Each kernel file contains both forward and backward functions. For example, `layernorm.cu` has both `launch_layernorm_forward` and `launch_layernorm_backward`. The backward function takes the cached `mean` and `rstd` that the forward optionally wrote — this avoids recomputing them.

GEMM has no separate backward kernel: `dA = dC × B^T` and `dB = A^T × dC` are just the transposed GEMM variants (`launch_gemm_tiled` and `launch_gemm_tn`).

---

## Layer 2 — Layers (`src/layers/`)

Each layer is a C++ struct that:
- **Owns** its weight `Tensor<float>` members
- **Caches** device pointers from the forward pass (needed for backward)
- Has a `forward()` and `backward()` method that call kernel launchers

| Struct | File | Owns |
|--------|------|------|
| `Linear` | `linear.hpp` | `W`, `d_W`; caches `x_cache` pointer |
| `LayerNormLayer` | `layernorm_layer.hpp` | `gamma`, `beta`, `d_gamma`, `d_beta`, `mean_buf`, `rstd_buf` |
| `EmbeddingLayer` | `embedding.hpp` | `weight`, `d_weight` |
| `MultiHeadAttention` | `attention_layer.hpp` | 4 × `Linear` (W_q, W_k, W_v, W_out); Q/K/V/O intermediate buffers |
| `TransformerBlock` | `transformer_block.hpp` | 2 × `LayerNormLayer`, 1 × `MultiHeadAttention`, 2 × `Linear` (fc1, fc2); activation buffers |

### How `Linear` works

```
Forward:  out = x @ W.T        (GEMM: N×in × in×out = N×out)
Backward: d_x = d_out @ W      (GEMM: N×out × out×in = N×in)
          d_W += d_out.T @ x   (GEMM: out×N × N×in = out×in)
```

`W` is stored as `(out_features, in_features)` so that `out = x @ W.T` maps naturally to `launch_gemm_nt(x, W, out, ...)`.

### How `MultiHeadAttention` handles layout

The model works in flat `(B*S, C)` layout, but the attention kernel needs `(B, H, S, D)` layout (so each head is contiguous). The layer inserts reshape kernels:

```
x (B*S, C)
    │ W_q / W_k / W_v projections (Linear.forward)
    ▼
Q_flat, K_flat, V_flat  (B*S, C)
    │ launch_flat_to_bhsd
    ▼
Q_bhsd, K_bhsd, V_bhsd  (B, H, S, D)
    │ launch_flash_attention_forward
    ▼
O_bhsd  (B, H, S, D)
    │ launch_bhsd_to_flat
    ▼
O_flat  (B*S, C)
    │ W_out projection (Linear.forward)
    ▼
out  (B*S, C)
```

### How `TransformerBlock` wires everything (Pre-LN style)

```
x  (B*S, C)
│
├──────────────────────┐  residual path (copy of x)
│                      │
▼  LN1.forward(x)      │
ln1_out                │
│                      │
▼  MHA.forward         │
attn_out               │
│                      │
▼  add_inplace(x)  ◄───┘
h = x + attn_out
│
├──────────────────────┐  residual path (copy of h)
│                      │
▼  LN2.forward(h)      │
ln2_out                │
│                      │
▼  fc1 (Linear C→4C)   │
fc1_out                │
│                      │
▼  GELU                │
gelu_out               │
│                      │
▼  fc2 (Linear 4C→C)   │
ffn_out                │
│                      │
▼  add_inplace(h)  ◄───┘
out = h + ffn_out
```

The backward is exactly the reverse: it peels off each operation and propagates the gradient back through it, accumulating the residual branch contributions.

---

## Layer 3 — Model (`src/model/gpt2.hpp`)

`GPT2` assembles all layers into a complete language model:

```
token_ids  (B*S,)
    │ EmbeddingLayer.forward
    ▼
embed_out  (B*S, C)
    │
    │  × n_layers:
    │    TransformerBlock[i].forward
    ▼
block_out  (B*S, C)
    │ LayerNormLayer (final LN).forward
    ▼
ln_out  (B*S, C)
    │ Linear (lm_head, C→V).forward
    ▼
logits  (B*S, V)
```

The backward is the mirror:

```
d_logits  (B*S, V)
    │ lm_head.backward
    ▼
d_ln_out  (B*S, C)
    │ final_ln.backward
    ▼
d_block  (B*S, C)
    │
    │  × n_layers (reverse order):
    │    TransformerBlock[i].backward  →  allocates d_in temporarily
    ▼
d_block  (B*S, C)  ← gradient flowing into the embedding
    │ EmbeddingLayer.backward  (atomicAdd into d_weight)
    ▼
d_weight  (V, C)
```

`GPT2::param_grads()` returns a flat list of `{float* param, const float* grad, int n}` structs covering every parameter in the model, in a consistent order. The trainer and MPI code use this list to iterate over all parameters without knowing the model's internal structure.

`GPT2::zero_grad()` zeros all gradient tensors before the backward pass.

---

## Layer 4 — Trainer (`src/training/trainer.hpp`)

`Trainer` owns the loss buffers and Adam moment buffers, and orchestrates one training step:

```
┌─────────────────────────────────────────────────────────────────────┐
│  forward_backward(ids, targets)                                     │
│                                                                     │
│  1. model.forward(ids)                → logits (B*S, V)            │
│  2. launch_cross_entropy_forward      → losses (B*S,)              │
│  3. launch_fill(dlosses, 1/N)         → dlosses = 1/N per token    │
│  4. launch_cross_entropy_backward     → d_logits (B*S, V)          │
│  5. model.zero_grad()                 zero all d_W / d_gamma etc.  │
│  6. model.backward(d_logits)          → all parameter gradients    │
│  7. sync + copy losses → host                                       │
│  8. return mean(losses)                                             │
└─────────────────────────────────────────────────────────────────────┘
                    ↕  (MPI all-reduce happens here in distributed mode)
┌─────────────────────────────────────────────────────────────────────┐
│  optimizer_step()                                                   │
│                                                                     │
│  for each (param, grad, n) in model.param_grads():                 │
│      launch_adam(param, grad, m, v, lr, β1, β2, ε, t, n)          │
│                                                                     │
│  Adam moment buffers (m, v) are allocated lazily on first call     │
│  and reused across steps.                                           │
└─────────────────────────────────────────────────────────────────────┘
```

For single-GPU training, the convenience `step()` method calls both in sequence.

### Why is the loss gradient `1/N`?

`launch_cross_entropy_forward` produces one loss scalar per token. The mathematical mean loss is `sum(losses) / N`. Its gradient w.r.t. each token's loss is `1/N`. Setting `dlosses[i] = 1/N` and passing that into `launch_cross_entropy_backward` makes the parameter gradients automatically normalized by the batch size — so the Adam learning rate is scale-invariant to batch size.

---

## Layer 5 — MPI (`src/mpi/data_parallel.hpp`)

Multi-GPU training uses **data parallelism**: each rank processes a different slice of the global mini-batch, then gradients are averaged.

```
Global batch (total_B × S tokens, on host)
        │
        │  scatter_batch(rank, n_ranks)
        ▼
Local batch (local_B × S tokens, one per rank)
        │
        │  copy to GPU
        ▼
  trainer.forward_backward(local_ids, local_targets)
        │  (each rank computes gradients on its own GPU)
        ▼
  allreduce_gradients(model)
        │  MPI_Allreduce: sum all gradients across ranks, divide by n_ranks
        │  Implementation: device→host copy → MPI_Allreduce → host→device copy
        ▼
  trainer.optimizer_step()
        │  Identical Adam step on all ranks (same averaged gradient → same update)
        │  → Weights stay in sync without re-broadcasting
        ▼
  (repeat)
```

`broadcast_weights(model)` is called once at startup so every rank begins with the same weights (rank 0's initialization is the canonical one).

### Why do weights stay in sync after step 1?
After the first `broadcast_weights`, every rank has identical weights. After each step:
- `allreduce_gradients` ensures every rank has the **same** averaged gradient
- Since Adam is deterministic and all ranks have the same `(weights, gradient, m, v, t)`, every rank computes the **same** parameter update
- Therefore weights remain identical across ranks indefinitely, without any further synchronization

---

## Tensor Memory Layout

Everything is row-major FP32. The key shape conventions:

| Tensor | Shape | Meaning |
|--------|-------|---------|
| Input tokens | `(B*S,)` int32 | Flat batch of token IDs |
| Hidden states | `(B*S, C)` | B sequences × S positions, each C-dimensional |
| Attention (projected) | `(B, H, S, D)` | Batch × Heads × Sequence × Head-dim |
| Logits | `(B*S, V)` | One vocabulary distribution per token |
| Embedding weight | `(V, C)` | One row per vocabulary entry |
| Linear weight | `(out, in)` | Stored transposed-friendly for `x @ W.T` |
| LN parameters | `(C,)` | Per-channel scale (gamma) and shift (beta) |

`B = batch size`, `S = sequence length`, `C = model dimension`, `H = num heads`, `D = C/H = head dimension`, `V = vocab size`.

---

## What Happens During One Training Step

Here is a concrete trace for `B=2, S=4, C=8, H=2, D=4, V=16` (tiny model):

```
ids     = [[3,1,9,2], [7,0,4,6]]    # shape (2,4) = (B,S), flattened to (8,)

─── Forward ──────────────────────────────────────────────────
embed   (8, 8)    ← embed.weight[ids[i]]  for each token i
block0  (8, 8)    ← LN → MHA → residual → LN → FFN → residual
block1  (8, 8)    ← same
final_ln(8, 8)    ← normalize
logits  (8, 16)   ← lm_head: (8,8) @ (8,16) = (8,16)

─── Loss ─────────────────────────────────────────────────────
losses  (8,)      ← -log(softmax(logits)[i, targets[i]])
mean_loss = sum(losses) / 8

─── Backward ─────────────────────────────────────────────────
d_logits (8,16)   ← (softmax[i,j] - 1{j==target[i]}) / 8
d_ln_out (8, 8)   ← lm_head.backward
d_block  (8, 8)   ← final_ln.backward
                  ← block1.backward (reverse sub-layers)
                  ← block0.backward
                  ← embed.backward  (atomicAdd into d_weight)

─── Optimizer ────────────────────────────────────────────────
For every (param, grad):
    m = 0.9*m + 0.1*grad
    v = 0.999*v + 0.001*grad²
    param -= lr * (m/(1-0.9^t)) / (sqrt(v/(1-0.999^t)) + 1e-8)
```

---

## File Dependency Map

```
kernels.cuh          ← umbrella include for all kernel headers
    ├── gemm.cuh
    ├── layernorm.cuh
    ├── gelu.cuh
    ├── softmax.cuh
    ├── cross_entropy.cuh
    ├── attention.cuh
    ├── embedding.cuh
    ├── adam.cuh
    ├── reshape.cuh
    └── pointwise.cuh

layers/linear.hpp         uses: gemm.cuh, tensor.hpp
layers/layernorm_layer.hpp uses: layernorm.cuh, tensor.hpp
layers/embedding.hpp       uses: embedding.cuh, tensor.hpp
layers/attention_layer.hpp uses: attention.cuh, reshape.cuh, pointwise.cuh, linear.hpp, tensor.hpp
layers/transformer_block.hpp uses: gelu.cuh, pointwise.cuh, layernorm_layer.hpp,
                                   attention_layer.hpp, linear.hpp, tensor.hpp

model/gpt2.hpp       uses: kernels.cuh, embedding.hpp, transformer_block.hpp,
                           layernorm_layer.hpp, linear.hpp, tensor.hpp

training/trainer.hpp  uses: kernels.cuh, gpt2.hpp, tensor.hpp

mpi/data_parallel.hpp uses: gpt2.hpp (via param_grads()), cuda_runtime.h, mpi.h

main_distributed.cu   uses: gpt2.hpp, data_parallel.hpp, trainer.hpp
```

---

## Tests

| Test binary | What it checks |
|-------------|----------------|
| `test_gemm` | Forward GEMM vs CPU |
| `test_layernorm` | Forward LN vs CPU reference |
| `test_softmax` | Forward softmax vs CPU |
| `test_gelu` | Forward GELU vs CPU |
| `test_cross_entropy` | Forward CE loss vs CPU |
| `test_attention` | Flash Attention forward vs CPU |
| `test_backward_layernorm` | LN backward: dx, dgamma, dbeta |
| `test_backward_gelu` | GELU backward: dx |
| `test_backward_cross_entropy` | CE backward: dlogits |
| `test_backward_attention` | Attention backward: dQ, dK, dV |
| `test_vs_pytorch` | All forward kernels vs PyTorch ref `.bin` files |

All tests use the same pattern: fill with deterministic random data → run GPU kernel → compare to CPU reference with ~1e-3 relative tolerance.

---

## Building and Running

```bash
# On the cluster (sm_75 = Turing RTX 6000)
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=75
make -j4

# Run all forward + backward tests
ctest --output-on-failure

# Single-GPU training demo
./train_distributed 20

# Multi-GPU (4 ranks)
mpirun -np 4 ./train_distributed 20 4 128 64 32
#                                steps layers C  S  total_B
```
