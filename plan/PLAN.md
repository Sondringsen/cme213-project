# CME 213 Final Project — Completion Plan

**Deadline:** June 8, 2026  
**Today:** June 1, 2026 (7 days remaining)  
**Authors:** Nils Astrup Toft & Sondre Rogde

---

## Evaluation Philosophy

The staff evaluates **analysis** more than code quality (code was written with AI assistance, so its assessment is limited). The goal is to produce a 6-page report with deep, credible performance analysis backed by Nsight screenshots, roofline plots, scaling numbers, and a communication breakdown.

---

## State of the Project (as of June 1)

### Done
- All 6 forward CUDA kernels: tiled GEMM, fused LayerNorm, online Softmax, GELU, Cross-Entropy, Flash Attention
- All backward kernels: LayerNorm backward, GELU backward, Cross-Entropy backward, naive O(S²) Attention backward
- Embedding forward (gather) + backward (atomicAdd scatter)
- Fused Adam optimizer kernel
- Full layer hierarchy: Linear, LayerNormLayer, EmbeddingLayer, MultiHeadAttention, TransformerBlock
- GPT2 model + Trainer orchestrating forward→loss→backward→Adam
- MPI data parallelism with MPI_Allreduce (non-CUDA-aware, ~50 per-tensor calls, host-staged)
- Strong and weak scaling data: 1/2/4 GPUs, numbers in Milestone 4 Table 1
- NVTX annotations in the training loop
- Nsight profiling SLURM script (`run_profile.sh`)
- SLURM scripts for tests (`run.sh`) and distributed training (`run_distributed.sh`)
- Existing roofline plot from Milestone 3

### Known Bottlenecks (from Milestone 4 + Nsight)
1. **~50 per-tensor MPI_Allreduce calls** — latency-bound, ~14 ms/step overhead
2. **~30 ms/step of synchronous cudaMalloc/cudaFree** — per-step backward Tensor allocations (Nsight identified)
3. **GEMM at only ~13.5% of FP32 peak** at 4096×4096 — register tiling not yet implemented
4. **Bandwidth-bound kernels** (LayerNorm, GELU, Softmax, Cross-Entropy) — no vectorized loads yet
5. **No real data pipeline** — training on synthetic random tokens only
6. **No BF16 implementation** — all tensors are FP32

### Not Implemented (and will be discussed in report only)
- CUDA-aware MPI (RTX 6000 node uses PCIe; analyze the PCIe staging cost)
- BF16 mixed precision (Turing has no native BF16 tensor cores; analyze theoretically)
- Communication/computation overlap (CUDA streams)
- Full GPT-2 (12L, C=768) benchmarks (may be too slow; use medium config)
- POS-augmented variant (dropped in favor of deeper performance analysis)

---

## Step-by-Step Plan

### Step 1 — Run Baseline Profiling (cluster, ~2 hours)
**File:** `plan/step1_nsight_profiling.md`  
Run Nsight Compute and Nsight Systems on the current codebase before any optimizations. Save the `.ncu-rep` and `.nsys-rep` files as the **before** baseline. This gives before/after screenshots for the report.

Key outputs:
- `profiles/ncu_baseline.ncu-rep` — per-kernel roofline metrics
- `profiles/nsys_baseline.nsys-rep` — training timeline with NVTX regions

### Step 2 — GEMM Register Tiling (code, ~3 hours)
**File:** `plan/step2_kernel_optimization.md`  
Implement 4×4 or 8×8 per-thread register tiling in `src/kernels/gemm.cu` to increase arithmetic intensity per global memory access and push GEMM from ~13.5% to ≥50% of FP32 peak. Add vectorized `float4` loads to the four bandwidth-bound kernels (LayerNorm, GELU, Softmax, Cross-Entropy). Re-run `ncu` after each change to capture **after** screenshots.

Key outputs:
- Updated `src/kernels/gemm.cu`
- Updated `src/kernels/layernorm.cu` (float4 loads)
- `profiles/ncu_after_gemm.ncu-rep`
- Before/after GFLOPS numbers for the report

### Step 3 — Fused AllReduce + Pre-allocated Scratch (code, ~3 hours)
**File:** `plan/step3_mpi_optimization.md`  
Two MPI/memory optimizations to close the two biggest non-kernel bottlenecks:
1. **Fused allreduce**: pack all ~3.4M gradient parameters into a single flat buffer, one `MPI_Allreduce(MPI_SUM)`, scatter back. Expected: collapse ~14 ms latency-bound overhead to ~1-2 ms bandwidth-bound.
2. **Pre-allocated backward scratch**: identify all `Tensor<float>` objects created inside backward() each step and move their allocation to model/layer construction. Expected: eliminate the ~30 ms/step cudaMalloc overhead Nsight identified.

Run `run_distributed.sh` before and after each change to measure improvement. Re-run `run_profile.sh` for updated Nsight Systems timeline.

Key outputs:
- Updated `src/mpi/data_parallel.hpp`
- Updated backward() methods in `src/layers/`
- Updated scaling table (strong/weak, 1/2/4 GPUs) with new numbers

### Step 4 — Communication Analysis Benchmark (code + analysis, ~2 hours)
**File:** `plan/step4_communication_analysis.md`  
Write a standalone MPI benchmark (`benchmarks/allreduce_bench.cu`) that measures `MPI_Allreduce` latency for varying message sizes n ∈ {1K, 10K, 100K, 1M, 10M, 100M} floats, with P ∈ {1, 2, 4} ranks. Fit the α+βn model to the data. Compare with the theoretical Ring All-Reduce lower bound:

```
T_ring = 2 * (P-1)/P * (α + βn)
```

Plot latency vs message size on a log-log scale. This gives the full communication analysis for the report.

Key outputs:
- `benchmarks/allreduce_bench.cu`
- `scripts/run_allreduce_bench.sh`
- α and β values for P=2 and P=4
- `plots/comm_alpha_beta.png`

### Step 5 — Extended Scaling Experiments (cluster, ~1 hour)
**File:** `plan/step5_scaling_experiments.md`  
Run the updated (post-optimization) `run_distributed.sh` with a larger model config for the final report scaling numbers. Also verify that the fused allreduce actually improves scaling efficiency. Produce the final strong/weak scaling table and plot.

Config for final report: LAYERS=4, C=256, S=128 (same as MS4). Use STEPS=10.
**Cluster constraint: all SLURM jobs must finish within 15 minutes.**

Key outputs:
- Updated `plots/scaling.png` (before vs after optimization)
- Final scaling table (strong + weak, 1/2/4 GPUs, compute/comm breakdown)

### Step 6 — WikiText-2 Training (optional, ~3 hours if time allows)
**File:** `plan/step6_wikitext_training.md`  
Implement a minimal data pipeline to train on real text. If this takes longer than 3 hours, skip and discuss as future work in the report. Priority is low compared to Steps 1-5.

If implemented:
- Python script: download WikiText-2, word-level tokenize, write binary file of int32 token IDs
- C++ data loader: read the binary file, produce (input_ids, target_ids) pairs for each batch
- Run 100-step training, show loss curve

Key outputs:
- `scripts/prepare_data.py`
- `src/data/data_loader.hpp`
- `plots/loss_curve.png`

### Step 7 — BF16 Analysis (report-only, ~30 min)
No code needed. In the report, explain:
- **Why BF16 was not implemented on Turing (sm_75)**: Turing has FP16 tensor cores but no native BF16 tensor cores. BF16 on Turing executes on regular CUDA cores at the same throughput as FP32 — so BF16 gives 2× memory reduction but zero compute speedup on our hardware.
- **Where BF16 would help**: on Ampere (A100) and Hopper (H100), native BF16 tensor cores give 2× compute throughput vs FP32 tensor cores.
- **FP32 master weights**: Adam's update requires high precision to capture small gradient updates; BF16 accumulation loses this. Our design stores FP32 weights and accumulates in FP32.

### Step 8 — Write the Final Report (LaTeX, ~6 hours)
**File:** `plan/step7_final_report.md` (see detailed per-section writing guide there)

The skeleton is in `reports/FinalReport.tex`. The structure follows the 10 required sections from `FinalReport_desc.pdf`:
1. Project description
2. Algorithms and state of the art (cuBLAS, Flash Attention, Horovod)
3. Parallelization strategy (kernel design + MPI decomposition)
4. Performance model (roofline predictions, α+βn, Amdahl, isoefficiency)
5. Benchmarking and instrumentation (ncu, nsys, NVTX, allreduce_bench)
6. Bottleneck analysis (identify each bottleneck and tie to hardware)
7. Algorithmic variants (register tiling, float4, fused allreduce, pre-alloc scratch)
8. Scalability analysis (strong + weak scaling, Amdahl f, isoefficiency)
9. Correctness and verification (test suite, bit-identical multi-rank checks)
10. Discussion, limitations, future work

References, figures, and appendices do NOT count against the 6-page limit.
1-inch margins, 11pt, single-spaced.

Appendix A: Nsight Compute screenshots (GEMM before/after)
Appendix B: Nsight Systems timeline (before/after optimization)
Appendix C: Full kernel performance table + roofline + α+βn plot

---

## Timeline

| Date | Task |
|------|------|
| Jun 1 | Create plan files + FinalReport.tex outline (this session) |
| Jun 2 | Step 1: baseline Nsight profiling on cluster |
| Jun 2-3 | Step 2: GEMM register tiling + float4 loads + re-profile |
| Jun 3-4 | Step 3: fused allreduce + pre-allocated scratch |
| Jun 4 | Step 4: communication analysis benchmark |
| Jun 4-5 | Step 5: extended scaling experiments |
| Jun 5 (optional) | Step 6: WikiText-2 data pipeline |
| Jun 5-7 | Step 8: write report |
| Jun 8 | Final review + submit |

---

## Shell Scripts to Create

| Script | Purpose |
|--------|---------|
| `scripts/run_ncu_baseline.sh` | Run ncu before optimizations |
| `scripts/run_ncu_gemm.sh` | Profile GEMM specifically with full metrics |
| `scripts/run_nsys_training.sh` | Nsight Systems timeline (single + multi-GPU) |
| `scripts/run_allreduce_bench.sh` | α+βn communication benchmark |
| `scripts/run_scaling_final.sh` | Final scaling study with optimized code |
| `scripts/plot_scaling.py` | Generate scaling plot from PERF: lines |
| `scripts/plot_comm.py` | Fit and plot α+βn from benchmark output |

---

## Report Figures to Generate

| Figure | Source | Section |
|--------|--------|---------|
| Roofline plot (ncu) | `profiles/ncu_after_gemm.ncu-rep` | §4 Kernels |
| GEMM before/after bar chart | Before/after ncu numbers | §4 Kernels |
| Strong scaling efficiency | `run_scaling_final.sh` output | §6 Analysis |
| Weak scaling efficiency | same | §6 Analysis |
| Communication α+βn log-log plot | `allreduce_bench` output | §6 Analysis |
| Nsight Compute screenshot (GEMM before) | ncu GUI | Appendix A |
| Nsight Compute screenshot (GEMM after) | ncu GUI | Appendix A |
| Nsight Systems timeline | nsys GUI | Appendix B |

---

## Important Context for Sub-agents

### Cluster Details
- Partition: `gpu-turing`, nodes have 4 × Quadro RTX 6000 (sm_75)
- Peak FP32: 16.31 TFLOPS, Peak bandwidth: 672 GB/s, Ridge point: ~24 FLOP/byte
- Build: `make CUDA_ARCH=75`
- MPI: OpenMPI, non-CUDA-aware

### Key File Locations
- Main training entry point: `src/main_distributed.cu`
- GEMM kernel: `src/kernels/gemm.cu` + `src/kernels/gemm.cuh`
- MPI allreduce: `src/mpi/data_parallel.hpp` function `allreduce_gradients()`
- Backward scratch allocations: `src/layers/transformer_block.hpp`, `src/layers/attention_layer.hpp`
- Existing profiling script: `run_profile.sh`
- Existing scaling script: `run_distributed.sh`

### Model Parameters (Milestone 4 baseline)
- 4 layers, C=256, S=128, V=512 → ~3.4M parameters
- Step time: 75.97 ms (1 GPU), 54.26 ms (4 GPUs, strong scaling)
- Comm time: 14 ms/step (4 GPUs), latency-bound (50 separate MPI calls)
- cudaMalloc overhead: ~30 ms/step (Nsight-identified, in "compute" segment)

### Critical Numbers for the Report
- Strong scaling efficiency at 4 GPUs: 35% (poor due to bottlenecks)
- Weak scaling efficiency at 4 GPUs: 41%
- GEMM at 4096×4096: 13.5% of FP32 peak (before register tiling)
- Target after optimization: ≥50% GEMM utilization, ≥70% scaling efficiency
