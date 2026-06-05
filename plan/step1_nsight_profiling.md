# Step 1 — Baseline Nsight Profiling

**Goal:** Capture "before" profiling data before any optimizations. Run `run_profile.sh` unchanged — it already produces both the ncu roofline file and the nsys timeline in one job.

**Estimated time:** 30 min work + cluster queue wait

---

## What the Script Does

`run_profile.sh` already handles everything:

| Output | Tool | Config |
|--------|------|--------|
| `profiles/ncu_kernels.ncu-rep` | Nsight Compute | 1 rank, 2 layers, C=128, S=32, 1 step, --launch-count 200 |
| `profiles/nsys_rank{0-3}.nsys-rep` | Nsight Systems | 4 ranks, 4 layers, C=256, S=128, 5 steps |

Submit as-is:

```bash
sbatch run_profile.sh
```

Run it **before** any float4 changes. After float4 optimization (Step 2), rename the outputs and run again:

```bash
# After Step 2 float4 changes, before running run_profile.sh again:
mv profiles/ncu_kernels.ncu-rep profiles/ncu_kernels_before.ncu-rep
mv profiles/nsys_rank0.nsys-rep profiles/nsys_before_rank0.nsys-rep
# ... then sbatch run_profile.sh  →  produces ncu_kernels.ncu-rep as "after"
```

Download to local machine:

```bash
scp <cluster>:<project_dir>/profiles/*.ncu-rep  ./profiles/
scp <cluster>:<project_dir>/profiles/*.nsys-rep ./profiles/
```

---

## What to Look for in Nsight Compute

Open `ncu_kernels.ncu-rep` in the Nsight Compute GUI. Go to **Roofline Analysis**.

### LayerNorm — primary analysis target

This is the kernel to examine most carefully.

- **Memory throughput**: LayerNorm reads x, gamma, beta twice (two-pass: mean then variance), so theoretical minimum bandwidth is `2 × 3 × H × 4 bytes` per row. Check how close the kernel gets to the 672 GB/s ceiling.
- **Occupancy**: the block size is 256 threads, one block per row. Check whether shared memory for the reduction (256 floats = 1 KB) limits occupancy.
- **Warp efficiency**: the tree reduction at the end of each pass leaves half the warps idle. Look for this in the Warp State Statistics view.
- **Two-pass vs one-pass**: the code uses two passes for numerical stability (catastrophic cancellation in the one-pass `var = E[x²] - E[x]²` when inputs are large). This costs one extra read of x per row — visible as higher-than-expected bytes/row ratio.

### Adam — bandwidth reference

Should sit on the bandwidth ceiling. Reads 2 arrays (param, grad) and writes 3 (param, m, v) → 5 × 4 bytes per element. Check achieved GB/s vs `5 × 4 × n_params × freq`.

### GELU, Softmax, Cross-Entropy

These should all cluster near the bandwidth ceiling. If any are significantly below it, float4 loads will help. Note their positions as a group.

### GEMM, Flash Attention — brief notes only

Both were covered in homework; no deep analysis needed. Just note their roofline positions (GEMM: near compute ridge; Flash Attention: somewhere between compute and bandwidth bounds depending on sequence length) for the one-sentence report mentions.

---

## What to Look for in Nsight Systems

Open `nsys_before_rank0.nsys-rep` in Nsight Systems.

**Screenshots to capture for the report (Appendix B):**

1. **One full training step** — shows the three NVTX phases: `forward_backward` (green), `allreduce` (yellow), `optimizer` (small). The allreduce gap should be visually large (~14 ms vs ~75 ms total).

2. **Zoom into the allreduce region** — shows the D→H memcpy, ~50 MPI_Allreduce calls, then H→D memcpy. This is the "before fused allreduce" picture.

3. **Zoom into the forward_backward region** — look for `cudaMalloc`/`cudaFree` spikes (the ~30 ms overhead). They appear as orange/red marks in the CUDA API row.

---

## Hardware Reference (Quadro RTX 6000, sm_75)

| Property | Value |
|----------|-------|
| FP32 peak | 16.31 TFLOPS |
| Memory bandwidth | 672 GB/s |
| Ridge point | ~24.3 FLOP/byte |
| L2 cache | 4 MB |
| Shared mem / SM | 64 KB |
| Max threads / SM | 2048 |
| Tensor cores | FP16 only |

---

## Checklist

- [ ] `sbatch run_profile.sh` — wait for completion
- [ ] Rename outputs to `*_before.*` before running again after optimizations
- [ ] Download `.ncu-rep` and `.nsys-rep` to local machine
- [ ] Open ncu GUI — screenshot LayerNorm roofline position, note GB/s
- [ ] Open nsys GUI — screenshot full step timeline and allreduce zoom
- [ ] Note Adam, GELU, Softmax, Cross-Entropy GB/s from ncu for Table 2 in report
