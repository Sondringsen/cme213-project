# Step 1 — Baseline Nsight Profiling

**Goal:** Capture "before" profiling data before any optimizations. These screenshots and numbers go into the report as the baseline.

**Estimated time:** 2 hours (mostly cluster queue wait)

---

## What We Are Profiling

1. **Nsight Compute (ncu)** — per-kernel hardware counters:
   - Achieved TFLOPS and GB/s for every kernel
   - SM occupancy
   - Roofline position (compute-bound vs memory-bound)
   - Register usage, shared memory utilization

2. **Nsight Systems (nsys)** — end-to-end CUDA + MPI timeline:
   - NVTX regions: `step_N`, `forward_backward`, `allreduce`, `optimizer`
   - cudaMalloc / cudaFree calls (the 30 ms/step overhead)
   - cudaMemcpy D→H and H→D for staging allreduce
   - MPI_Allreduce gaps between CUDA work

---

## Steps

### 1. Build (on cluster)

```bash
make CUDA_ARCH=75
```

Verify the binary exists: `ls -lh build/train_distributed`

### 2. Run Nsight Compute baseline

The existing `run_profile.sh` already does this. Submit it as-is for the baseline:

```bash
sbatch run_profile.sh
```

This produces `profiles/ncu_kernels.ncu-rep` (single rank, 2 layers, C=128, S=32, 1 step).

**Important:** The `--launch-count 200` flag limits ncu to the first 200 kernel launches so the job doesn't time out. With 2 layers and 1 step, all major kernels appear at least once.

For a deeper GEMM-only profile (needed for before/after comparison), also run:

```bash
sbatch scripts/run_ncu_gemm.sh
```

(Create this script — see below.)

### 3. Run Nsight Systems baseline

Also produced by `run_profile.sh`:
- `profiles/nsys_rank0.nsys-rep` through `profiles/nsys_rank3.nsys-rep`

Open `nsys_rank0.nsys-rep` in Nsight Systems on your local machine. The NVTX timeline should show:
- Green `forward_backward` regions
- Yellow `allreduce` regions (these are the ~14 ms/step gaps)
- Small `optimizer` regions

**Screenshot to capture for the report:**
- One full step showing all three NVTX phases + the MPI gaps
- Zoom into the allreduce region to show the D→H + MPI + H→D pattern

### 4. Download profiles to local machine

```bash
# Run from your local machine:
scp <cluster>:<project_dir>/profiles/ncu_kernels.ncu-rep ./profiles/
scp <cluster>:<project_dir>/profiles/nsys_rank0.nsys-rep ./profiles/
```

---

## New Script to Create: `scripts/run_ncu_gemm.sh`

This is a focused ncu run targeting only the GEMM kernel at a large matrix size (4096×4096) for maximum precision on the roofline:

```bash
#!/bin/bash
#SBATCH --job-name=ncu_gemm
#SBATCH --output=logs/ncu_gemm_%j.out
#SBATCH --partition=gpu-turing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00

cd "$SLURM_SUBMIT_DIR"
mkdir -p profiles logs

make CUDA_ARCH=75

# Profile only the GEMM test binary, full metrics
ncu \
    --output=profiles/ncu_gemm_before \
    --set full \
    --kernel-name-base function \
    --kernel-name gemm_tiled_kernel \
    --force-overwrite \
    ./build/test_gemm

echo "Done. File: profiles/ncu_gemm_before.ncu-rep"
```

**Note:** The kernel name `gemm_tiled_kernel` must match the actual `__global__` function name in `src/kernels/gemm.cu`. Check with:
```bash
strings build/test_gemm | grep "kernel"
```

---

## What to Look for in ncu

Open `ncu_kernels.ncu-rep` or `ncu_gemm_before.ncu-rep` in Nsight Compute on your local machine.

### For GEMM
- **Roofline chart**: should show GEMM near the compute-bound ridge line, but below it. Note the exact FLOP/byte (arithmetic intensity) and achieved GFLOPS/s.
- **SM occupancy**: typically low (<50%) without register tiling because the thread block is large
- **L2 cache hit rate**: should be reasonable for tiled GEMM
- **Warp efficiency**: look for predicated-off lanes

### For Flash Attention
- **Memory throughput**: should be much lower than peak (attention is compute-heavy but uses SRAM tricks)
- **Compute throughput**: should be higher than naive attention

### For LayerNorm / GELU / Softmax / Cross-Entropy
- **Memory throughput**: these should be close to the 672 GB/s bandwidth ceiling
- **Compute throughput**: very low (barely any FLOPS per byte loaded)
- **If significantly below bandwidth ceiling**: float4 vectorized loads will help

---

## What to Capture for the Report

| Item | From | Report section |
|------|------|---------------|
| ncu roofline chart (all kernels) | ncu GUI | §4, Figure 1 |
| GEMM: achieved GFLOPS, % of peak | ncu metrics | §4, Table 2 |
| Flash Attention: achieved GFLOPS, GB/s | ncu metrics | §4, Table 2 |
| LayerNorm achieved bandwidth, % of peak | ncu metrics | §4, Table 2 |
| nsys timeline screenshot (one full step) | nsys GUI | Appendix B |
| nsys zoom into allreduce gap | nsys GUI | Appendix B |

---

## Context for Interpreting Results

- **Quadro RTX 6000 (Turing sm_75):**
  - FP32 peak: 16.31 TFLOPS
  - Memory bandwidth: 672 GB/s
  - Ridge point: ~24.3 FLOP/byte
  - L2 cache: 4 MB
  - Shared memory per SM: 64 KB (configurable)
  - Max registers per SM: 65536
  - Max threads per SM: 2048
  - Tensor cores: FP16 only (not BF16)

- **Arithmetic intensity for GEMM** at tile size T=32:
  - Each thread loads 2T floats and does 2T² operations per (T×T) output tile
  - AI = 2T² / (2T × 4 bytes) = T/4 = 8 FLOP/byte for T=32
  - This is **below the ridge point** (~24 FLOP/byte), meaning without register tiling, GEMM is memory-bound even though it should be compute-bound at this matrix size
  - With 4×4 register tiling, AI improves to 4× higher → 32 FLOP/byte, crossing the ridge point

- **Arithmetic intensity for Flash Attention** (forward):
  - Loads Q, K, V tiles; computes QK^T + softmax + V; writes O
  - Avoids materializing full S×S matrix
  - AI ≈ O(S·D) / O(S·D) = O(1) per sequence step but proportional to block size
  - Practically: Flash Attention should be higher AI than memory-bound threshold
