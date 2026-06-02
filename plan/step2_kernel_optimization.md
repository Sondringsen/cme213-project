# Step 2 — Kernel Optimization

**Goal:** Improve GEMM utilization from ~13.5% to ≥50% of FP32 peak via register tiling, and improve bandwidth-bound kernels via float4 vectorized loads. Capture before/after ncu screenshots for the report.

**Estimated time:** 3 hours

---

## Optimization 1: GEMM Register Tiling

### What and Why

The current tiled GEMM (`src/kernels/gemm.cu`) uses a 32×32 thread block where each thread computes exactly 1 output element. This means:
- Per thread: loads 32 A values + 32 B values from shared memory → 64 loads → computes 32 FMAs (64 FLOP)
- Arithmetic intensity: 64 FLOP / (64 × 4 bytes) = 0.25 FLOP/byte per thread (very low in registers)
- Most time is spent on shared memory reads, not compute

With **4×4 register tiling**, each thread computes a 4×4 output sub-tile:
- Per thread: loads 4 A values + 4 B values per K-step → 8 loads → computes 32 FMAs (64 FLOP)
- Arithmetic intensity per thread: 64 FLOP / (8 × 4 bytes) = 2 FLOP/byte (8× better than 1×1)
- Register file holds the 4×4 accumulators across the K-loop → reuse without shared memory traffic

### Implementation Plan

File: `src/kernels/gemm.cu`

**Thread block configuration:**
- Instead of 32×32 threads computing 32×32 output: use 8×8 threads computing 32×32 output (each thread handles 4×4)
- Thread block size: BM=BN=32 (output tile), BK=8 (K-dimension per step)
- Threads per block: 8×8 = 64
- Each thread: TM=4, TN=4 output registers

**Pseudocode for new kernel:**
```cuda
__global__ void gemm_reg_tiled(const float* A, const float* B, float* C,
                                 int M, int N, int K) {
    // BM=32, BN=32, BK=8, TM=4, TN=4
    // One thread block computes a 32×32 output tile
    // Thread (ty, tx) where ty,tx in [0,8) computes a 4×4 sub-tile

    __shared__ float smA[BM][BK];   // 32×8 = 256 floats = 1 KB
    __shared__ float smB[BK][BN];   // 8×32 = 256 floats = 1 KB

    float regA[TM] = {0};  // 4 values from A
    float regB[TN] = {0};  // 4 values from B
    float regC[TM][TN] = {0};  // 4×4 accumulators (16 registers)

    int row = blockIdx.y * BM + ty * TM;
    int col = blockIdx.x * BN + tx * TN;

    for (int k = 0; k < K; k += BK) {
        // Collaborative load: all 64 threads load a 32×8 tile of A and 8×32 of B
        // (requires careful indexing so all threads participate in loading)
        
        __syncthreads();
        
        // Each thread loads TM values of A and TN values of B from shared mem
        for (int kk = 0; kk < BK; kk++) {
            for (int m = 0; m < TM; m++) regA[m] = smA[ty*TM+m][kk];
            for (int n = 0; n < TN; n++) regB[n] = smB[kk][tx*TN+n];
            for (int m = 0; m < TM; m++)
                for (int n = 0; n < TN; n++)
                    regC[m][n] += regA[m] * regB[n];
        }
        
        __syncthreads();
    }
    
    // Write 4×4 output tile to C
    for (int m = 0; m < TM; m++)
        for (int n = 0; n < TN; n++)
            if (row+m < M && col+n < N)
                C[(row+m)*N + (col+n)] = regC[m][n];
}
```

**Key implementation details:**
1. The 64-thread block must cooperatively load the 32×8 and 8×32 shared memory tiles. With 64 threads and 256 elements to load per tile, each thread loads 4 elements.
2. Loading pattern: thread `(ty, tx)` loads `smA[ty*4+i][tx/2]` for i in [0,4) (approximate — need exact bounds checking).
3. Use `#pragma unroll` on the TM and TN inner loops to let the compiler fully unroll the 4×4 accumulation into 16 independent FMAs.
4. Consider double-buffering: prefetch the next K-chunk into a second shared memory buffer while computing with the current one (requires `__pipeline_commit()` on sm_80+, but on sm_75 use manual double-buffering with `cp.async` not available — use two smA/smB pairs and alternate).

**Correctness check:**
Run `./build/test_gemm` after modification. All existing tests must still pass (M=N=K=32, 128, 256, 512, 1024, 4096).

**Performance check after optimization:**
```bash
# Run ncu on the updated binary
ncu --output=profiles/ncu_gemm_after --set full \
    --kernel-name gemm_reg_tiled_kernel \
    --force-overwrite \
    ./build/test_gemm
```

Expected: ≥50% of 16.31 TFLOPS = ≥8.15 TFLOPS at M=N=K=4096.

### Also: Transpose variants

The GEMM backward uses `launch_gemm_tn` (A^T×B) and `launch_gemm_nt` (A×B^T). These can also benefit from register tiling, but apply to the forward GEMM first since it's the hottest path. The transposed variants can use similar 4×4 tiling.

---

## Optimization 2: float4 Vectorized Loads (Bandwidth-Bound Kernels)

### What and Why

LayerNorm, GELU, Softmax, and Cross-Entropy are all memory-bandwidth bound. Each thread currently loads one `float` at a time (32-bit loads). Replacing with `float4` loads (128-bit) reduces:
- Number of load instructions by 4×
- Instruction overhead (fewer memory transactions issued)
- L1 cache pressure (wider loads have better cache efficiency)

This is especially effective on Turing where the L1/shared memory path is wide.

### Implementation Pattern

For any elementwise kernel that processes N floats:

```cuda
// Before: each thread loads 1 float
__global__ void kernel(float* x, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = some_op(x[i]);
}

// After: each thread loads 4 floats
__global__ void kernel_vec4(float* x, float* out, int N) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < N) {
        float4 val = reinterpret_cast<float4*>(x)[i/4];
        float4 res;
        res.x = some_op(val.x);
        res.y = some_op(val.y);
        res.z = some_op(val.z);
        res.w = some_op(val.w);
        reinterpret_cast<float4*>(out)[i/4] = res;
    }
    // Handle tail: if N % 4 != 0, process remaining elements scalar
}
```

**Requirement:** The input pointer must be 16-byte aligned (guaranteed by cudaMalloc) and N must be a multiple of 4 (enforce in the layer with padding if needed, or handle the tail).

### Kernels to Vectorize

1. **`src/kernels/gelu.cu`** — `launch_gelu_forward` and `launch_gelu_backward`:
   - Each element: `x → x * 0.5 * (1 + erf(x/sqrt(2)))`
   - Pure elementwise → trivial to vectorize
   
2. **`src/kernels/layernorm.cu`** — `launch_layernorm_forward`:
   - The main loop over hidden dimension reads x, gamma, beta
   - Can load x and gamma as float4 in the reduction passes
   - More complex due to two-pass reduction; vectorize the data load
   
3. **`src/kernels/softmax.cu`** — `launch_softmax_forward`:
   - Each block handles one row; the reduction is over C elements
   - Load x as float4 in the max-finding and sum passes
   
4. **`src/kernels/cross_entropy.cu`** — `launch_cross_entropy_forward`:
   - Reads logits row; loads as float4 for the softmax reduction

### Correctness Check

Run `./build/test_gelu`, `./build/test_layernorm`, `./build/test_softmax`, `./build/test_cross_entropy` after each change.

---

## Profiling After Optimizations

### Script: `scripts/run_ncu_after.sh`

```bash
#!/bin/bash
#SBATCH --job-name=ncu_after
#SBATCH --output=logs/ncu_after_%j.out
#SBATCH --partition=gpu-turing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00

cd "$SLURM_SUBMIT_DIR"
mkdir -p profiles logs
make CUDA_ARCH=75

# Full training step profile (all kernels, post-optimization)
mpirun -np 1 ncu \
    --output=profiles/ncu_after_full \
    --set full \
    --force-overwrite \
    --launch-count 300 \
    ./build/train_distributed 1 2 128 32 4

echo "Done. profiles/ncu_after_full.ncu-rep"
```

### Expected Results

| Kernel | Before (% peak) | After (% peak) |
|--------|----------------|----------------|
| GEMM (4096×4096) | 13.5% FP32 | ≥50% FP32 |
| Flash Attention | ~40% compute | ~50% compute |
| GELU | ~60% BW | ~80% BW |
| LayerNorm | ~55% BW | ~75% BW |
| Softmax | ~50% BW | ~70% BW |
| Cross-Entropy | ~45% BW | ~65% BW |

(Exact numbers will come from ncu; these are estimates for planning.)

---

## Report Content from This Step

### Section 4 (CUDA Kernels) content
- Before/after GEMM roofline positions
- Explain register tiling: "each thread now accumulates a 4×4 output tile in registers, reducing shared memory traffic by 4× and achieving X% of peak"
- Explain float4: "replacing scalar loads with 128-bit vectorized loads reduces instruction count 4× and improves bandwidth utilization from X% to Y%"
- Table: per-kernel achieved GFLOPS/GB/s before and after

### Appendix A content
- Screenshot 1: ncu roofline view, GEMM before (dot far below compute ceiling)
- Screenshot 2: ncu roofline view, GEMM after (dot near compute ceiling)
- Screenshot 3: ncu Warp State Statistics showing improved issue efficiency

---

## Summary Checklist

- [ ] Implement 4×4 register tiling in `gemm_tiled_kernel` in `src/kernels/gemm.cu`
- [ ] Verify all `test_gemm` tests pass
- [ ] Run `ncu` before and after, save to `profiles/ncu_gemm_before.ncu-rep` and `profiles/ncu_gemm_after.ncu-rep`
- [ ] Record GFLOPS numbers for the report table
- [ ] Add float4 loads to `gelu.cu`, `layernorm.cu`, `softmax.cu`, `cross_entropy.cu`
- [ ] Verify all corresponding tests pass
- [ ] Run `ncu_after_full.sh` for the complete training step profile
- [ ] Screenshot the roofline charts from ncu GUI
- [ ] Save screenshots to `plots/` for the report
