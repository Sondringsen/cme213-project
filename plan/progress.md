# Project Progress

## Step 1 — Baseline Nsight Profiling
- [ ] Run `scripts/run_ncu_baseline.sh` on cluster
- [ ] Save `profiles/ncu_baseline.ncu-rep` and `profiles/nsys_baseline.nsys-rep`

## Step 2 — Kernel Profiling and Optimization
- [x] Forward GEMM: register-tiled (BM=128, BN=128, BK=8, TM=8, TN=8) — **no deep analysis in report, just one number**
- [x] Backward GEMM: TILE=32 shared-memory tiled (`launch_gemm_tn`, `launch_gemm_nt`)
- [x] `test_gemm`: CPU correctness (small sizes) + cuBLAS performance comparison (large sizes)
- [ ] Run `test_gemm` on cluster — record GFLOPS vs cuBLAS ratio (one sentence in report)
- [ ] Write `scripts/run_ncu_kernels.sh` (single script, all kernels)
- [ ] Run baseline profile → `profiles/ncu_kernels_before.ncu-rep`
- [ ] Add float4 loads to `gelu.cu`, `softmax.cu`, `layernorm.cu`; verify tests pass
- [ ] Run post-optimization profile → `profiles/ncu_kernels_after.ncu-rep`
- [ ] Extract Flash Attention roofline position (forward vs backward) for report
- [ ] Record per-kernel GB/s before/after float4 for report table

## Step 3 — MPI / Memory Optimizations
- [ ] Fused allreduce (single flat buffer)
- [ ] Pre-allocated backward scratch

## Step 4 — Communication Benchmark
- [ ] `benchmarks/allreduce_bench.cu`
- [ ] α+βn fit and plot

## Step 5 — Scaling Experiments
- [ ] Re-run strong/weak scaling with optimized code

## Step 6 — WikiText-2 (optional)
- [ ] Data pipeline if time allows
- [ ] Paged attention for inference

## Step 7 — Final Report
- [ ] Write all sections in `reports/FinalReport.tex`
