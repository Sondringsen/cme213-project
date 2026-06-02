# CLAUDE.md — Project Context for CME 213 Final Project

## Project Overview

A GPT-2-style transformer training pipeline in C++/CUDA, distributed across multiple GPUs with MPI. The system is complete: all kernels, backward passes, the full training loop, and MPI data parallelism are implemented. The remaining work is performance optimization and the final report.

**Evaluation note:** The staff grades on analysis quality, not code quality (AI-assisted code is allowed). Every effort should go toward Nsight profiling, kernel optimization, and producing quantitative analysis for the report.

**Plan:** Read `plan/PLAN.md` for the step-by-step completion plan. Individual steps have detailed files in `plan/stepN_*.md`.

---

## Current State

### Implemented
- Forward and backward CUDA kernels: GEMM (tiled, shared memory), LayerNorm, GELU, Softmax, Cross-Entropy, Flash Attention (forward + naive O(S²) backward), Embedding, Adam
- Full layer hierarchy: Linear, LayerNormLayer, EmbeddingLayer, MultiHeadAttention, TransformerBlock (Pre-LN style)
- GPT2 model + Trainer (forward → CE loss → backward → Adam)
- MPI data parallelism: allreduce gradients (host-staged, ~50 per-tensor calls), broadcast weights, scatter batch
- NVTX annotations, SLURM profiling and scaling scripts

### Key bottlenecks (Nsight-identified)
1. ~50 per-tensor MPI_Allreduce calls: ~14 ms/step, latency-bound
2. Per-step cudaMalloc in backward: ~30 ms/step
3. GEMM at 13.5% of FP32 peak (no register tiling yet)
4. Bandwidth-bound kernels (LayerNorm, GELU, Softmax, CE) without float4 loads

### Not implemented (discuss in report)
- CUDA-aware MPI, BF16, communication/computation overlap, full 12-layer GPT-2

---

## Coding Guidelines

- **Prefer performance.** We know CUDA and C++ well enough to optimize. Comments are only needed to explain non-obvious choices or tricky invariants.
- **No unnecessary comments.** Don't explain what the code does if the variable names already do. A kernel called `gemm_reg_tiled` doesn't need a comment saying "this is the register-tiled GEMM kernel."
- **Validate correctness first.** All existing tests must pass after any change. Run `./build/test_<kernel>` before and after each modification.
- **Benchmark every change.** Run `run_distributed.sh` or the relevant `ncu` command before and after any optimization. Numbers go in the report.

---

## Directory Structure

```
cme213-project/
├── src/
│   ├── kernels/        # CUDA kernels (gemm, attention, layernorm, gelu, softmax, cross_entropy, embedding, adam, reshape, pointwise)
│   ├── layers/         # Layer modules (Linear, LayerNormLayer, EmbeddingLayer, MultiHeadAttention, TransformerBlock)
│   ├── model/          # GPT2 model assembly (gpt2.hpp)
│   ├── training/       # Trainer (trainer.hpp)
│   ├── data/           # (empty — data pipeline not yet implemented)
│   ├── mpi/            # MPI data parallelism (data_parallel.hpp)
│   └── utils/          # Tensor, cuda_check, utils
├── tests/              # Per-kernel correctness tests
├── benchmarks/         # (allreduce_bench.cu to be added)
├── scripts/            # Python and shell scripts for profiling, plotting, data prep
├── reports/            # Milestone write-ups + FinalReport.tex
├── plan/               # Step-by-step completion plan
├── plots/              # Generated figures for the report
├── profiles/           # Nsight .ncu-rep and .nsys-rep files
├── logs/               # SLURM job output
└── Makefile / CMakeLists.txt
```

### Building
```bash
make CUDA_ARCH=75          # single GPU tests
mpirun -np 4 ./build/train_distributed 20 4 256 128 16
```

### Cluster constraint
All SLURM jobs must be under 15 minutes. Use small model configs (LAYERS=4, C=256, S=128) for benchmarks. Avoid STEPS > 20 for timing runs.
