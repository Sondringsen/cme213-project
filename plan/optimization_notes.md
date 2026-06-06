# Kernel Optimization Notes — CME 213 Final Report Reference

This file is the single source of truth for all kernel changes made during Step 2.
Each entry answers: what was changed, what was wrong before, why the fix works,
and what numbers to fill in from the post-optimization profile.

---

## 1. float4 vectorized loads — GELU forward + backward
**File:** `src/kernels/gelu.cu`

**Problem:** Each thread loaded 1 float per instruction. With 1 read (x) + 1 write (y)
per element in the forward pass, the kernel was issuing 2× as many memory instructions
as needed, limiting memory throughput to ~17% (from ncu CSV: ID 14).

**Fix:** Each thread now loads and stores 4 floats per instruction via `float4`
reinterpret casts. Grid size reduced to `ceil(n / (BLOCK × 4))`. Scalar tail handles
`n % 4 != 0`. The backward uses a shared `gelu_prime()` device helper.

**Expected gain:** ~2× reduction in load/store instruction count; bandwidth-bound
kernels typically see 10–30% wall-clock improvement from this.

**Report numbers to fill in:** Before GB/s → After GB/s (from ncu before/after).

---

## 2. float4 vectorized loads — LayerNorm normalization pass
**File:** `src/kernels/layernorm.cu`

**Problem:** The normalization output pass (Pass 2b) reads x, gamma, beta and writes y
— 3 reads + 1 write = 16 bytes/element — in scalar 1-float-per-instruction loops.
The two reduction passes (mean, variance) stay scalar because they reduce into a
single accumulator and don't have the same vectorization opportunity.

**Fix:** Pass 2b loops in strides of `LN_BLOCK × 4`, loading x/gamma/beta as `float4`
and writing y as `float4`. The scalar tail loop handles `H % 4 != 0` (never true for
power-of-2 hidden sizes). Grid and block are unchanged — still one block per row.

**Theoretical bandwidth floor:** Forward reads x twice (pass 1 + pass 2), gamma, beta
once each, writes y once → `(2+1+1+1) × H × 4 = 20H bytes/row`. float4 lowers the
instruction count for the bandwidth-bound pass without changing the byte count.

**Report numbers to fill in:** Achieved GB/s before/after from ncu; compare against
672 GB/s peak to place on roofline.

---

## 3. float4 vectorized loads — add_inplace
**File:** `src/kernels/pointwise.cu`

**Problem:** `add_inplace_kernel` was scalar (1 float/thread). It runs 3× per step in
the residual connections and showed ~10% memory bandwidth (ncu CSV IDs 11, 16, 27, 32,
47). 1 read (a) + 1 read (b) + 1 write (a) = 12 bytes/element.

**Fix:** Same float4 pattern as GELU. Grid shrinks to `ceil(n / (BLOCK × 4))`.

---

## 4. Attention backward BH loop fusion
**File:** `src/kernels/attention.cu`

**Problem:** `attention_weights_kernel` and `softmax_backward_kernel` were each
launched in a C++ loop `for (int bh = 0; bh < B*H; ++bh)`, one kernel call per
batch-head pair. With B=8, H=8 this produced 32 separate launches each with a grid
of only S=32 blocks (at NCU config) or S=128 blocks (training config). ncu reported
23.1% achieved occupancy vs 100% theoretical (CSV IDs 51–82), and 76% estimated
speedup.

**Root cause:** The S-block grid (32 or 128) is smaller than the GPU's 72 SMs, so
most SMs are idle during each launch. With 32 serialized launches, the total idle
time dominates.

**Fix:** Added `blockIdx.y = bh` as the batch-head dimension. One launch with grid
`(S, BH)` now covers all S×BH = 4096 blocks (training config) in a single dispatch,
fully saturating all 72 SMs.

The GEMM steps (dV, dP, dQ, dK) inside the same function still loop over bh because
fixing those requires batched GEMM — a larger change noted as future work.

**Report numbers to fill in:** Before = 23.1% occupancy, 32 launches × ~4.8µs = ~154µs.
After = single launch, occupancy should approach theoretical.

---

## 5. Coalesced memory loads — gemm_nt and gemm_tn
**File:** `src/kernels/gemm.cu`

**Problem:** Both transposed GEMM variants had one uncoalesced global memory load per
tile load phase.

*Why coalescing matters:* Within a warp, all 32 threads execute the same instruction
simultaneously. If their memory addresses are contiguous (4 bytes apart), the L2 cache
serves the entire warp from one 128-byte cache line. If the addresses stride by K or M
floats, the hardware needs up to 32 separate cache line fetches — 32× more memory
transactions.

In a (TILE=32, TILE=32) block, a warp is one full row: fixed `threadIdx.y`,
`threadIdx.x` varying 0→31.

**`gemm_tn_kernel` (C = A^T × B, A stored K×M):**
Original load: `A[k_A * M + row]` where `k_A = t*TILE + threadIdx.x` varies,
`row = blockIdx.y*TILE + threadIdx.y` fixed. Adjacent threads access
`A[k_A*M + row]` with stride M → uncoalesced.

Fix: swap so `threadIdx.x` indexes the contiguous M dimension:
- `row_A = blockIdx.y*TILE + threadIdx.x` (varies across warp → stride 1 ✓)
- `k_A = t*TILE + threadIdx.y` (fixed within warp)
- Store transposed: `Asub[threadIdx.x][threadIdx.y]`

`Asub` declared as `float Asub[TILE][TILE+1]` — the +1 padding makes each write hit
a different shared memory bank (bank = `(tx*33 + ty) % 32` cycles through all 32
banks as tx varies). The accumulate reads `Asub[ty][k]` — identical index to before,
correct because `Asub[tx][ty]` stores the transposed element.

**`gemm_nt_kernel` (C = A × B^T, B stored N×K):**
Original load: `B[col * K + k_B]` where `col = col_base + threadIdx.x` varies,
`k_B = t*TILE + threadIdx.y` fixed. Adjacent threads access B at stride K → uncoalesced.

Fix: swap so `threadIdx.x` indexes the contiguous K dimension:
- `k_B = t*TILE + threadIdx.x` (varies → stride 1 ✓)
- `col_B = blockIdx.x*TILE + threadIdx.y` (fixed within warp)
- Store transposed: `Bsub[threadIdx.y][threadIdx.x]`

`Bsub` declared as `float Bsub[TILE][TILE+1]`. Accumulate changes from
`Bsub[k][threadIdx.x]` to `Bsub[threadIdx.x][k]` to read the same logical element.

**Report numbers to fill in:** Before memory throughput ~12–38% (ncu CSV). After
should move significantly higher (expect 2–4× improvement in bandwidth utilization
for the matrix sizes in training).

---

## 5b. Register tiling — gemm_nt and gemm_tn
**File:** `src/kernels/gemm.cu`

**What changed:** The TILE=32 kernels (one output element per thread) were replaced
with register-tiled versions using the same BM=128, BN=128, BK=8, TM=8, TN=8 block
structure as the NN kernel. Each thread now accumulates an 8×8 register sub-tile (64
FMAs per smem load) instead of 1.

**`gemm_tn_reg_kernel`:** A (K×M) is loaded as `A[k][m] → Asub[m][k]`, using
`tid % BM` as the fast M-index so adjacent threads read stride-1 addresses.
`Asub[BM][BK+1]` — the +1 stride of 9 is coprime to 32, so all 32 warp writes land
on distinct banks (conflict-free). The accumulation phase is then identical to the NN
kernel since Asub and Bsub have the same logical layout.

**`gemm_nt_reg_kernel`:** B (N×K) is loaded as `B[n][k] → Bsub[n][k]`, using
`tid % BK` as the fast K-index. `Bsub[BN][BK+1]`. The accumulation reads
`b_reg[tn] = Bsub[tx*TN + tn][k]` (N-major) instead of `Bsub[k][tx*TN+tn]`.

**Expected gain:** ~4× more FMAs per smem access; combined with the coalescing fix
from 5a, the backward GEMMs should now be both bandwidth-efficient and
compute-intensive. Run `ncu` on `test_backward_attention` before/after to measure.

**Report numbers to fill in:** Achieved GB/s and FP32 utilization before/after from ncu.

---

## 6. Attention backward scratch buffer pre-allocation
**Files:** `src/layers/attention_layer.hpp`, `src/kernels/attention.cu`,
`src/kernels/attention.cuh`

**Problem:** `MultiHeadAttention::backward()` allocated **11 device buffers** on every
backward pass call:
- 9 `Tensor<float>` temporaries (d_O_flat, d_O_bhsd, d_Q/K/V_bhsd, d_Q/K/V_flat, tmp)
- 2 explicit `cudaMalloc` calls inside `launch_attention_backward` for the S×S
  attention weight matrices P and dP (each B×H×S×S floats = ~4 MB at training config)

These showed as the large cudaMalloc spikes in Nsight Systems immediately before the
`attention_weights_kernel` launch. At B=8, H=8, S=128 each P buffer is
8×8×128×128×4 = 4 MB. CUDA's allocator also triggered additional internal pool
management allocations (the "smaller cudaMallocs" visible in the timeline).

**Why cudaMalloc is slow:** `cudaMalloc` is a synchronizing runtime call — it flushes
the GPU command queue and blocks the CPU until the allocation completes. Doing this
11 times per step adds tens of milliseconds of latency.

**Fix:** All 11 buffers are now member variables of `MultiHeadAttention`, initialized
once in the constructor. `launch_attention_backward` accepts `float* P_buf, float* dP_buf`
parameters instead of allocating internally. The backward() method passes the
pre-allocated members.

**Report numbers to fill in:** Nsight Systems before: large cudaMalloc spike ~30 ms/step.
After: no allocation in the critical path; step time should drop measurably.

---

## 7. TransformerBlock backward scratch buffer pre-allocation
**File:** `src/layers/transformer_block.hpp`

**Problem:** `TransformerBlock::backward()` allocated 7 `Tensor<float>` temporaries on
every backward call (`d_h`, `d_gelu_out`, `d_fc1_out`, `d_ln2_out`, `d_h_ffn`,
`d_ln1_out`, `d_x_attn`). Each triggers a `cudaMalloc`/`cudaFree` pair, and with
N_layers blocks per step the allocations compound. The attention layer's scratch was
already pre-allocated (entry 6), making the TransformerBlock the remaining source of
per-step cudaMalloc noise.

**Fix:** All 7 temporaries are now member variables (`d_h_buf`, `d_gelu_out_buf`,
`d_fc1_out_buf`, `d_ln2_out_buf`, `d_h_ffn_buf`, `d_ln1_out_buf`, `d_x_attn_buf`),
initialized in the constructor alongside the existing forward buffers. `backward()`
writes directly into them with no allocation in the critical path.

**Report numbers to fill in:** Step time (1 rank, no comm) before/after from
`run_distributed.sh`; the difference isolates the allocation overhead.

---

## 8. Fused MPI AllReduce with pinned host buffer
**File:** `src/mpi/data_parallel.hpp`

**Problem:** `allreduce_gradients()` called `MPI_Allreduce` once per parameter tensor
(~50 calls for a 4-layer model). Each call incurs a startup latency α ≈ 100–300 µs,
totalling ~14 ms/step that is pure latency overhead, not bandwidth-limited. The function
also allocated a `std::vector<float>` per tensor per step (heap alloc + dealloc × 50).

**Fix:** All gradients are packed into a single contiguous pinned host buffer, then one
`MPI_Allreduce` is issued on the full buffer, and results are scattered back.
The buffer is lazy-allocated (static local, `cudaMallocHost`) on the first call and
reused every subsequent step. Pinned memory enables full PCIe DMA bandwidth (~16 GB/s)
for the D→H and H→D transfers, versus pageable memory which goes through an extra
intermediate copy.

**Expected gain:** ~50 α-latencies eliminated. Before: ~50 × 300 µs ≈ 14 ms. After:
1 × α + β × total_grad_bytes ≈ 300 µs + 850 µs ≈ 1.2 ms.

**Report numbers to fill in:** `comm ms` at 4 ranks before/after from `run_distributed.sh`.

---

## Profiling script changes
**File:** `run_profile.sh`

**Problem 1:** `--launch-count 200` capped the profile before the full backward pass
completed, so many backward kernels (layernorm backward, GEMM backward, etc.) were
never captured. The resulting CSV contained mostly forward-pass and attention-backward
BH-loop instances.

**Problem 2:** `NCU_C=128, NCU_S=32` made the attention backward GEMMs trivially
small (32×16×32 → 1-block launches), producing artificially high "98% speedup"
estimates that do not reflect training-scale performance.

**Fix:** 
- Removed `--launch-count 200`; added `--nvtx --nvtx-include "forward_backward"` to
  scope the profile to exactly the training compute path (excludes init and optimizer)
- Bumped NCU config to match NSYS config: `C=256, S=128, LAYERS=4`
