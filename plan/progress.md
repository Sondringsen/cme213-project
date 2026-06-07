# Project Progress

## Step 1 — Baseline Nsight Profiling
- [x] Run `scripts/run_ncu_baseline.sh` on cluster
- [x] Save `profiles/ncu_baseline.ncu-rep` and `profiles/nsys_baseline.nsys-rep`

## Step 2 — Kernel Profiling and Optimization
- [x] Forward GEMM: register-tiled (BM=128, BN=128, BK=8, TM=8, TN=8) — 2.20 → 9.56 TFLOPS
- [x] Backward GEMM: register-tiled `gemm_tn_reg`, `gemm_nt_reg` with coalesced loads + bank-conflict-free padding
- [x] `test_gemm`: CPU correctness + cuBLAS performance comparison
- [x] Baseline profile → `profiles/ncu_kernels_before.ncu-rep`
- [x] float4 loads added to `gelu.cu`, `softmax.cu`, `layernorm.cu`, `cross_entropy.cu`, `pointwise.cu` (add_inplace)
- [x] Post-optimization profile → `profiles/ncu_kernels_after.ncu-rep`
- [x] Attention backward BH-loop fusion (single launch covers all batch-heads)
- [x] Per-kernel GB/s before/after recorded for report Table 2

## Step 3 — MPI / Memory Optimizations
- [x] Fused allreduce (single pinned host buffer, one MPI call)
- [x] Pre-allocated backward scratch in `MultiHeadAttention` and `TransformerBlock`
- [x] 4-rank step time: 74.5 ms → 59.8 ms

## Step 4 — Communication Benchmark
- [x] `benchmarks/allreduce_bench.cu` (via `scripts/run_allreduce_bench.sh`)
- [x] α+βn fit and `plots/comm_alpha_beta.png`

## Step 5 — Scaling Experiments
- [x] Strong/weak scaling with optimized code; final numbers in Table 1 of FinalReport.tex

## Step 6 — WikiText-2 (was optional, completed)
- [x] `scripts/prepare_data.py` data pipeline
- [x] `src/data/data_loader.hpp`
- [x] `main_distributed.cu` accepts data path argument
- [x] `main_inference.cu` for text generation
- [x] Model checkpoint save/load (`src/utils/model_io.hpp`)
- [x] Loss curve `plots/loss_curve.png` (9.27 → < 6.0 over 300 steps with 4 ranks)

## Step 7 — BF16 Analysis (report-only)
- [x] Discussion paragraph drafted in FinalReport.tex (Limitations section)

## Step 7b — PagedAttention Decode Kernel (NEW)
**File:** `plan/step8_paged_attention.md`

### Phase A — Kernel (~3–4 h)
- [ ] `src/kernels/paged_attention.cuh` declaration
- [ ] `src/kernels/paged_attention.cu` kernel + dispatcher (template on D ∈ {16,32,64}, block_size=16)

### Phase B — Block manager + correctness test (~2 h)
- [ ] `src/inference/paged_kv_cache.hpp` block manager (free-list allocator)
- [ ] `tests/test_paged_attention.cu` — compare paged vs dense Flash Attention reference

### Phase C — Benchmark (~2 h)
- [ ] `benchmarks/paged_attention_bench.cu` — paged vs naive padded
- [ ] `scripts/run_paged_bench.sh` (< 5 min SLURM)
- [ ] `scripts/plot_paged.py` — two-panel figure (mem util + throughput)
- [ ] `plots/paged_attention.png`

### Phase D — Report (~1 h, folded into Step 8)
- [ ] Bibliography entry for Kwon et al. 2023
- [ ] §7 paragraph "Variant 5: PagedAttention for inference KV-cache management"
- [ ] Appendix D with paged_attention.png
- [ ] Abstract sentence on inference extension

## Step 8 — Final Report
- [x] Skeleton written through all 10 sections
- [ ] Fix block-dimension typo at line 109 (16×16 = 256 threads, not 8×8)
- [ ] Confirm H value in abstract/intro matches actual benchmark runs (default is H=8 at C=256, report says H=16)
- [ ] Add Adam row to Table 2 (kernel performance) — flagged as the bandwidth reference in Step 1 notes
- [ ] State the n_layers=6 config explicitly anywhere "before/after" comparisons appear (Milestone 4 used n_layers=4 — make sure baselines don't look silently improved)
- [ ] Tighten abstract narrative: currently a list of numbers; should frame as "built, profiled, found 4 bottlenecks, fixed, then extended to inference via PagedAttention"
- [ ] Cross-reference Step 7b numbers in §7 and Appendix D
