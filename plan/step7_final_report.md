# Step 7 — Final Report Writing Guide

**File:** `reports/FinalReport.tex`
**Deadline:** June 8, 2026
**Format:** 11pt, 1-inch margins, single-spaced. 6 pages MAX for text; references, figures, and appendices do NOT count toward the limit.

The skeleton is in `reports/FinalReport.tex`. The 10-section structure follows `FinalReport_desc.pdf` exactly.

---

## The 10 Required Sections

| # | Section | Focus | Pages (text) |
|---|---------|-------|------|
| 1 | Project description | Inputs, outputs, assumptions, why parallelism is needed | 0.4 |
| 2 | Algorithms and state of the art | Core algorithms, SOTA survey, your choices vs. alternatives | 0.5 |
| 3 | Parallelization strategy | Kernel design (threads, shared memory, coalescing), MPI decomposition | 0.9 |
| 4 | Performance model | Roofline (predicted AI), α+βn, Amdahl, isoefficiency | 0.5 |
| 5 | Benchmarking and instrumentation | Tools (ncu, nsys), what you measured, compare to model | 0.4 |
| 6 | Bottleneck analysis | Compute/BW/latency/cudaMalloc; tie to hardware and lecture | 0.4 |
| 7 | Algorithmic variants | Register tiling, float4, fused allreduce, pre-alloc scratch; quantify each | 0.6 |
| 8 | Scalability analysis | Strong + weak scaling, efficiency, Amdahl serial fraction, isoefficiency | 0.4 |
| 9 | Correctness and verification | Per-kernel tests, multi-rank checks, bit-identical loss at step 1 | 0.3 |
| 10 | Discussion, limitations, future work | Trade-offs, what you'd do differently, biggest payoffs | 0.4 |
| Abstract | | | 0.2 |
| **Total** | | | **~5.0** |

Figures (roofline, scaling, comm), the kernel table, and references are extra pages.

---

## Grading Weights (from description)

| Category | Weight |
|----------|--------|
| Parallelization (CUDA + MPI design) | 20% |
| Performance model, benchmarking, bottleneck analysis | 20% |
| Scalability study and algorithmic variants | 15% |
| Correctness, testing, verification | 10% |
| Writing quality, organization, clarity | 15% |
| Visual quality of plots and figures | 10% |
| Depth of analysis and connection to lecture | 10% |

**Implication:** §4 + §5 + §6 + §7 + §8 together account for 55% of the grade. These are where effort matters most.

---

## Section-by-Section Writing Notes

### §1 Project Description
- State inputs (token ID sequences, shape (B, S)), outputs (logits (B, S, V), CE loss), and what happens in between (transformer forward/backward/Adam)
- Explain why multi-GPU: large batch sizes, training time — reference GPT-3 scale if needed
- Keep short. This is the framing paragraph, not the main content.

### §2 Algorithms and SOTA
- GEMM: one sentence — register-tiled kernel achieves X% of cuBLAS. No analysis; this was covered in hw4. cuBLAS uses tensor cores + software pipelining as the practical ceiling.
- Flash Attention: reference Dao et al. 2022. Note that the forward kernel was implemented in hw7; this project adds the naïve O(S²) backward and integrates both into the training loop. Mention FA-2 (better parallelism across heads) as an extension not implemented.
- Softmax: online one-pass reduction (also a hw7 building block) — one sentence.
- Data parallelism: Horovod and PyTorch DDP use bucketed/fused allreduce. We implement the same from scratch to understand and quantify the gain.
- llm.c (Karpathy) as a related from-scratch reference implementation.

### §3 Parallelization Strategy
Most important section technically. Cover:
1. Five-layer abstraction (brief — Table 1 in the main text handles this)
2. GEMM: one sentence on tile layout + register tiling. No deep analysis.
3. Flash Attention: one sentence on SRAM tiling and memory saving. No deep analysis — done in hw7.
4. LayerNorm: this is the kernel to explain in depth.
   - Why two-pass: the one-pass formula `var = E[x²] - E[x]²` catastrophically cancels when inputs are large relative to variance. We encountered this as NaN/incorrect variance during development. Two-pass computes mean first, then `E[(x-mean)²]` — numerically stable but reads x twice.
   - Bandwidth floor: 16H bytes/row forward (x read twice, gamma+beta once, y once). State this as the roofline prediction.
   - Backward: three separate per-row reductions needed (sum of dy, sum of dy×xhat, then dx). Each requires a shared-memory tree reduction, leaving half the warps idle at each step. Explain the occupancy trade-off.
   - Float4: reduces the normalization pass from 3×H scalar loads to 3×H/4 float4 loads.
5. MPI: one GPU per rank, scatter_batch (host-side slice), broadcast_weights (once), allreduce_gradients (fused, pinned buffer), Adam (identical on all ranks)
6. Why we DON'T use CUDA-aware MPI (not enabled on cluster) and how that affects the gap from Ring bound

### §4 Performance Model
Must be quantitative with falsifiable predictions:
- LayerNorm: bandwidth AI = 2 FLOP / (16 bytes) = 0.125 FLOP/byte — well below the ridge at 24. Predict bandwidth-bound. Theoretical peak GB/s = achieved FLOP/s ÷ AI. With float4: same AI, but fewer instructions → predict bandwidth utilization increases. State the expected GB/s ceiling = 672 GB/s.
- GEMM: one number — AI = (2MNK) / (4(MK+KN+MN)) FLOP/byte. For square matrices: AI = K/2 FLOP/byte. At K=128: 64 FLOP/byte, above ridge → predict compute-bound. Measured GFLOPS/peak gives efficiency.
- Communication: α+βn model. At 50 calls: 50α dominates. After fusing: α + β × 13.6 MB ~ 1-2 ms. These are predictions; Section 7 verifies them.
- Amdahl: E(P) = T1/(P*T_P). Predict residual serial fraction = PCIe staging.
- Isoefficiency: derive minimum batch size for 90% efficiency from measured C_comm.

### §5 Benchmarking and Instrumentation
Key point: explain WHAT each tool measures and HOW you used it.
- ncu --set full: collects hardware counters → achieved TFLOPS/GB/s on the roofline
- nsys --trace=cuda,mpi,nvtx,osrt: full timeline → found the 30 ms cudaMalloc overhead
- NVTX markers: forward_backward, allreduce, optimizer → structured timeline
- std::chrono: per-step wall time; allreduce timed separately
- allreduce_bench: 100 trials per (P, n) combination for α+βn fitting

Compare measurements to model predictions at end of this section.

### §6 Bottleneck Analysis
Four bottlenecks, each tied to hardware:
1. LayerNorm below bandwidth ceiling → 32-bit scalar loads leave 75% of the 128-bit memory bus unused; two-pass reads x twice, doubling the already-high bandwidth demand
2. BW-bound kernel group (GELU, Softmax, CE, Adam) → same 32-bit load issue; scalar loads issue 4× more memory transactions than float4
3. 50-call allreduce → latency term 50α >> bandwidth term β × 13.6 MB
4. Per-step cudaMalloc → host-device synchronization stalls the GPU pipeline for ~30 ms/step (found by nsys, invisible to step-time chrono)

Connect each to lecture: GPU memory hierarchy and load width (1, 2), α+β model (3), CPU-GPU synchronization (4).

### §7 Algorithmic Variants
For each variant: state what changed (one sentence), predict the effect (from §4 model), report the measured result, explain how the bottleneck shifts.

1. **float4 loads (LayerNorm)**: replaces 3H scalar loads per row with 3H/4 float4 loads in the normalization pass. Prediction: reduces instruction count 4×; if instruction overhead was the bottleneck, GB/s increases toward ceiling. Report: X GB/s → Y GB/s.
2. **float4 loads (GELU)**: same pattern on a pure elementwise kernel — cleaner case, easier to reason about. Report: X GB/s → Y GB/s.
3. **Fused allreduce**: single flat buffer replaces 50 per-tensor calls. Prediction: 50α → α; bandwidth term unchanged. Report: X ms/step → Y ms/step at 4 GPUs.
4. **Pre-allocated backward scratch**: eliminates per-step cudaMalloc/Free. Prediction: removes 30 ms/step overhead seen in nsys. Report: step time before/after.

### §8 Scalability Analysis
- Report Table (scaling_final.tex data) with MS4 before and final after
- Compute Amdahl f from E(P=4) after optimization
- Isoefficiency: C_comm (measured fused allreduce time) / 0.1 = minimum compute per rank
- Figure: two-panel (stacked bar + efficiency curves)
- Connect weak scaling drop to bandwidth cost: even fused, D→H+H→D adds overhead

### §9 Correctness and Verification
Already written in the skeleton (uses Milestone 4 data). Just fill in:
- Max absolute/relative error for each kernel from the test suite output
- The bit-identical step-1 loss (6.6030 across P=1,2,4) from MS4
- Convergence toward log(V) = log(512) ≈ 6.24

### §10 Discussion
Three key takeaways (already in skeleton):
1. Profiler-guided opt: nsys found the cudaMalloc problem that chrono hid
2. Latency vs. bandwidth in MPI: at small model size, 50*alpha >> beta*n
3. BF16 hardware dependency: explain Turing vs. Ampere/Hopper

---

## On the Homework Overlap

Acknowledge the overlap briefly in §2. The differentiators to emphasize throughout the report:
- **Full backward passes** for all operations — homework was forward-only
- **LayerNorm** as the novel kernel to analyze — not covered in any homework; encountered real numerical issues (one-pass instability) during development that are worth discussing
- **End-to-end integration** into a training pipeline — correctness is validated against bit-identical loss across ranks, not just per-kernel tests
- **Quantitative before/after analysis** using Nsight across all kernels — no homework does this at this scope

Suggested framing in §2: "This work builds on the tiled GEMM and Flash Attention kernels from the course homework; those kernels are briefly characterized in §4 but not analyzed in depth. The novel contributions are the LayerNorm forward/backward implementation (§3), the fused MPI allreduce (§7), and the end-to-end quantitative performance analysis (§5–§8)."

---

## Style Rules

- No EM dashes (---). Use commas, semicolons, or separate sentences.
- No filler phrases: "rarely dissected", "state-of-the-art", "from first principles" (use sparingly).
- Define all notation on first use: B, S, C, H, D, V are defined in §1.
- Use \textbf{} for key terms at their first introduction.
- Every figure must be referenced in the text before it appears.
- Every table must be self-contained (no reading the text required to understand it).
- Report real numbers. If you don't have them yet, write [FILL] and get them before June 8.

---

## Figures Needed (all except placeholders should be complete before writing the report)

| Figure | Script/Source | Location |
|--------|---------------|---------|
| Roofline (all kernels) | ncu GUI — Roofline Analysis view | `plots/roofline_all_kernels.png` |
| LayerNorm roofline before float4 | ncu GUI screenshot | `plots/ncu_layernorm_before.png` |
| LayerNorm roofline after float4 | ncu GUI screenshot | `plots/ncu_layernorm_after.png` |
| Scaling (stacked bar + efficiency) | `scripts/plot_scaling.py` | `plots/scaling_final.png` |
| α+βn comm | `scripts/plot_comm.py` | `plots/comm_alpha_beta.png` |
| nsys before (full step + allreduce zoom) | nsys GUI screenshot | `plots/nsys_before.png` |
| nsys after (full step + allreduce zoom) | nsys GUI screenshot | `plots/nsys_after.png` |

Make all plots with clean fonts, axis labels, and legends. No default matplotlib styling.

---

## Cluster Time Limit

All SLURM jobs must finish within **15 minutes**. Use:
- STEPS=10 for scaling runs (enough for stable timing)
- LAYERS=4, C=256, S=128 (small model) for all benchmarks
- `--launch-count 200` for ncu (limits to first 200 kernel launches)
