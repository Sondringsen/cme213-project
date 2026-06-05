# Step 2 — Kernel Profiling and Optimization

**Goal:** Profile all training kernels on the roofline, implement float4 vectorized loads for bandwidth-bound kernels (centered on LayerNorm), and capture before/after ncu data for the report.

**Estimated time:** 2 hours

---

## Kernel Analysis Plan

### GEMM — one sentence in the report

Already register-tiled (BM=128, BN=128, BK=8, TM=8, TN=8). Run `./build/test_gemm` for the GFLOPS vs cuBLAS ratio. That number goes in §2 ("our register-tiled kernel achieves X% of cuBLAS") and nothing more — this is homework material.

### Flash Attention — one paragraph in the report

Also covered in hw7. Worth one paragraph noting where it sits on the roofline and the memory-saving tradeoff (no S×S materialization). The forward vs backward contrast (O(S·D) vs O(S²) memory traffic) is visible in the ncu data and worth one sentence.

### Softmax — one sentence

Part of the hw7 Flash Attention building blocks. Just report the achieved bandwidth as part of the bandwidth-bound kernel group.

### LayerNorm — the analysis centerpiece

This is the kernel to examine in depth. It was not covered in any homework and has real design choices worth explaining.

**Why it is interesting:**

1. **Two-pass vs one-pass numerical stability.** The one-pass formula `var = E[x²] - E[x]²` suffers catastrophic cancellation when input values are large relative to the variance (common after embedding lookup without pre-normalization). We hit this during development — the kernel produced NaN or wildly incorrect variance for certain input ranges. The fix is the two-pass approach: first reduce to compute the mean, then reduce again for `var = E[(x - mean)²]`. This costs one extra global read of x per row, which is directly visible in the bandwidth measurement.

2. **Bandwidth floor from two-pass.** With H hidden units per row, the forward reads x twice (mean pass + variance pass), plus gamma and beta once, and writes y once. Theoretical minimum: `(3H + H) × 4 = 16H bytes` per row. Comparing the achieved GB/s against this theoretical floor tells you whether the kernel is fully bandwidth-bound or has overhead.

3. **Backward is more complex.** The backward needs three independent per-row reductions: `sum(dy)`, `sum(dy × xhat)`, and `dx` itself. Each requires a shared memory tree reduction, which leaves half the warps idle for log2(blockDim) steps. The occupancy cost of 256 threads × 3 reduction arrays in shared memory is worth noting.

4. **float4 improvement.** The normalization pass (read x, gamma, beta → write y) is a clean float4 target. Before/after GB/s gives a concrete number for §7.

**From ncu, answer these questions:**
- Achieved GB/s vs the 16H bytes/row theoretical floor — how close?
- Occupancy — how many warps active vs max? Is shared memory or register count the limiter?
- Warp State Statistics — what fraction of cycles are stalled on memory?
- After float4: does the achieved GB/s increase, or were we already memory-latency bound?

---

## Optimization: float4 Vectorized Loads

### LayerNorm (`src/kernels/layernorm.cu`)

In the normalization pass, load x, gamma, and beta as `float4` (4 floats per instruction instead of 1). The two reduction passes (which accumulate into a scalar) stay unchanged.

```cuda
// Normalization pass — replace scalar loads with float4
for (int i = tid * 4; i + 3 < H; i += LN_BLOCK * 4) {
    float4 xi    = reinterpret_cast<const float4*>(row_x)[i / 4];
    float4 gi    = reinterpret_cast<const float4*>(gamma)[i / 4];
    float4 bi    = reinterpret_cast<const float4*>(beta)[i / 4];
    // apply normalization to xi.x, xi.y, xi.z, xi.w ...
    float4 yi; /* ... */
    reinterpret_cast<float4*>(row_y)[i / 4] = yi;
}
// scalar tail for H % 4 != 0
```

Requirement: H must be a multiple of 4 (true for all realistic hidden sizes).

### GELU (`src/kernels/gelu.cu`) — simplest

Pure elementwise. Each thread processes 4 elements instead of 1.

```cuda
int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
if (i + 3 < N) {
    float4 x4 = reinterpret_cast<const float4*>(x)[i / 4];
    float4 y4 = {gelu(x4.x), gelu(x4.y), gelu(x4.z), gelu(x4.w)};
    reinterpret_cast<float4*>(out)[i / 4] = y4;
}
// scalar tail
```

Same pattern for GELU backward.

### Softmax and Cross-Entropy — if time allows

Both process one row per block. Float4 the data load in the max/sum pass. Lower priority — the GELU and LayerNorm numbers are enough for §7.

**After each change:** run the corresponding test binary to verify correctness.

---

## Profiling

Use the existing `run_profile.sh` (do not create a new script). Run it once before float4 changes, rename outputs to `*_before.*`, then run again after.

```bash
# Baseline (before float4)
sbatch run_profile.sh
mv profiles/ncu_kernels.ncu-rep  profiles/ncu_kernels_before.ncu-rep
mv profiles/nsys_rank0.nsys-rep  profiles/nsys_before_rank0.nsys-rep

# After float4 changes
sbatch run_profile.sh
# outputs go to profiles/ncu_kernels.ncu-rep (the "after" file)
```

---

## Report Content from This Step

### §2 Algorithms (brief mentions)
- GEMM: "register-tiled kernel achieves X% of cuBLAS; see HW4 for tiling analysis."
- Flash Attention: "extends the hw7 forward kernel with a naïve O(S²) backward; forward avoids materializing the S×S attention matrix."
- Softmax: "uses online one-pass max-then-sum reduction from hw7."

### §3 Parallelization (LayerNorm section)
- Explain two-pass design and why one-pass is numerically unstable
- State the bandwidth floor: 16H bytes/row forward, ~20H bytes/row backward
- Mention float4 and alignment requirement

### §6 Bottleneck Analysis
- LayerNorm: bandwidth-bound, X% of 672 GB/s peak; two-pass doubles read traffic vs one-pass

### §7 Algorithmic Variants
- float4 for LayerNorm: X GB/s → Y GB/s (Z% improvement)
- float4 for GELU: same pattern, different numbers

### Appendix A
- Roofline screenshot: LayerNorm before and after float4 (two dots, one moves toward bandwidth ceiling)
- Roofline screenshot: GELU, Softmax, CE, Adam as a group (all near bandwidth ceiling)

---

## Checklist

- [ ] Run `./build/test_gemm` — record GFLOPS vs cuBLAS ratio for §2 sentence
- [ ] Run baseline `sbatch run_profile.sh` — rename to `*_before.*`
- [ ] Add float4 to `gelu.cu` (forward + backward); verify `test_gelu` passes
- [ ] Add float4 to `layernorm.cu` normalization pass; verify `test_layernorm` passes
- [ ] Add float4 to `softmax.cu` if time allows; verify `test_softmax` passes
- [ ] Run post-optimization `sbatch run_profile.sh`
- [ ] Screenshot LayerNorm roofline before/after from ncu GUI → `plots/ncu_layernorm_before.png`, `plots/ncu_layernorm_after.png`
- [ ] Record per-kernel GB/s for the report table (LayerNorm, GELU, Softmax, CE, Adam)
