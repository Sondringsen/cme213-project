# CLAUDE.md — Project Context for CME 213 Final Project

## Project Overview

We are building a small GPT-2-style transformer language model from scratch in C++/CUDA, distributed across multiple GPUs using MPI. The goal is to learn and demonstrate the parallel computing primitives behind modern LLM training: custom CUDA kernels, Flash Attention, mixed-precision arithmetic, and Ring All-Reduce gradient synchronization.

**Note:** The project proposal (`Milestone1.tex`) is a starting point, not a contract. We can and will deviate from it as the project evolves — scope, design choices, and features are all negotiable.

---

## Coding Guidelines

- **Always explain the code.** We are both new to C++ and CUDA. When writing or modifying code, include clear inline comments explaining what each block does and *why*. Do not assume familiarity with CUDA concepts like warps, shared memory, or memory coalescing — explain these where they appear.
- **Document every function.** Each function should have a short comment header describing its purpose, inputs, and outputs.
- **Prefer clarity over cleverness.** We would rather understand a slightly slower implementation than be confused by a heavily optimized one. Optimizations can come later, and should always be explained when introduced.
- **Validate before optimizing.** Every kernel or module should be tested for correctness against a reference (PyTorch, cuBLAS, or a simple CPU implementation) before we move on to optimization.

---

## Step-by-Step Implementation Plan

This is a rough chronological roadmap. Each step is a self-contained module we can implement, test, and check off. If you need a detailed plan for any step, just ask.

### Phase 1 — Infrastructure & Scaffolding
1. **Project build system** — Set up `CMakeLists.txt` with CUDA and MPI support. Make sure a "hello world" CUDA program compiles and runs.
2. **Data pipeline** — Load and tokenize a small text corpus (WikiText-2 or similar). Implement a simple vocabulary builder and a batch sampler that produces `(input_ids, target_ids)` pairs.
3. **Tensor abstraction** — A minimal `Tensor` struct/class that holds a flat GPU buffer, shape, and stride. Supports allocation, deallocation, and host↔device copy.

### Phase 2 — Reference Forward Pass (CPU / cuBLAS)
4. **Embedding layer** — Word embedding lookup (and later POS embedding concatenation). Just an index-into-matrix operation.
5. **Linear layer (GEMM)** — Forward pass using cuBLAS first; this is the reference we will later replace with a custom kernel.
6. **LayerNorm** — Forward pass (reference CPU or cuBLAS-free implementation).
7. **Softmax & Cross-Entropy Loss** — Numerically stable softmax; cross-entropy loss over vocabulary logits.
8. **Attention (naive)** — Standard scaled dot-product attention without any memory tricks, for correctness reference.
9. **Full transformer block** — Wire together: LayerNorm → Attention → residual → LayerNorm → FFN → residual.
10. **Training loop skeleton** — Forward pass → loss → (placeholder) backward → (placeholder) optimizer step. Confirm loss decreases with gradient descent on a tiny dataset.

### Phase 3 — Custom CUDA Kernels
11. **Tiled GEMM kernel** — Replace cuBLAS with a hand-written tiled matrix multiplication using shared memory blocking and register tiling.
12. **Fused pointwise kernels** — Fused GELU activation, fused LayerNorm (forward), fused residual add. Minimize global memory round-trips.
13. **Flash Attention forward** — Online softmax + SRAM tiling to compute attention without materializing the full N×N matrix.
14. **Backward passes** — Backprop through GEMM, LayerNorm, GELU, and Flash Attention. This is the hardest phase.
15. **Adam optimizer kernel** — Fused parameter update using Adam with BF16 weights and FP32 master weights.

### Phase 4 — Mixed Precision
16. **BF16 storage** — Add BF16 tensor type; convert forward/backward kernels to operate in BF16 with FP32 accumulation where needed.
17. **FP32 master weights** — Store a FP32 copy of weights for the optimizer step; load/store in BF16 for compute.

### Phase 5 — Multi-GPU with MPI
18. **MPI setup** — Initialize MPI ranks, assign each rank a GPU, shard the dataset across ranks.
19. **Ring All-Reduce** — Implement gradient synchronization via Ring All-Reduce using CUDA-aware MPI.
20. **Overlap communication and computation** — Use CUDA streams to overlap the All-Reduce of layer ℓ's gradients with the backward pass of earlier layers.

### Phase 6 — POS Extension & Experiments
21. **POS embedding** — Add a part-of-speech tagger (or pre-tagged data) and concatenate POS embeddings to word embeddings at the input stage.
22. **Task-parallel sweep** — Run baseline and POS-augmented variants concurrently on disjoint GPU groups; compare throughput and perplexity.

### Phase 7 — Performance Analysis & Report
23. **Roofline analysis** — Profile main kernels (GEMM, Flash Attention) and plot against the hardware roofline.
24. **Scaling study** — Strong and weak scaling curves; isoefficiency analysis; α+βn communication breakdown.
25. **Final report** — 6-page writeup covering design, implementation, and performance results.

---

## Directory Structure

```
cme213-project/
├── src/
│   ├── kernels/        # Custom CUDA kernels (gemm, attention, layernorm, etc.)
│   ├── layers/         # High-level layer modules (embedding, linear, attention block, etc.)
│   ├── model/          # Full transformer model assembly and config
│   ├── training/       # Training loop, optimizer, loss
│   ├── data/           # Tokenizer, dataset loader, batch sampler
│   ├── mpi/            # MPI utilities, Ring All-Reduce
│   └── utils/          # Tensor class, memory helpers, logging, timing
├── tests/              # Per-module correctness tests (vs. PyTorch / cuBLAS reference)
├── benchmarks/         # Roofline, scaling, and throughput benchmarks
├── scripts/            # Python scripts for data prep, reference outputs, plotting
├── reports/            # Milestone write-ups and final report (LaTeX)
│   └── Milestone1.tex
└── CMakeLists.txt
