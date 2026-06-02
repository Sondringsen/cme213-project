# Step 3 — MPI Optimization: Fused AllReduce + Pre-allocated Scratch

**Goal:** Eliminate the two largest non-kernel bottlenecks identified by Nsight in Milestone 4:
1. ~14 ms/step from 50 latency-bound MPI_Allreduce calls → fuse into one
2. ~30 ms/step from per-step cudaMalloc/cudaFree in backward → pre-allocate

**Estimated time:** 3 hours

---

## Bottleneck 1: Per-tensor MPI_Allreduce

### Current state (`src/mpi/data_parallel.hpp`)

```cpp
void allreduce_gradients(GPT2& model) {
    for (auto& pg : model.param_grads()) {
        // D→H copy
        cudaMemcpy(h_buf, pg.grad, pg.n * sizeof(float), cudaMemcpyDeviceToHost);
        // One MPI call per tensor (~50 calls for a 3.4M param model)
        MPI_Allreduce(MPI_IN_PLACE, h_buf, pg.n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        // H→D copy
        cudaMemcpy(pg.param, h_buf, pg.n * sizeof(float), cudaMemcpyDeviceToHost);
    }
    // Scale by 1/n_ranks
}
```

The bottleneck: ~50 MPI_Allreduce calls each with a startup latency α ≈ 100-300 µs (measured by the α+βn benchmark in Step 4). Total: 50 × 300 µs = 15 ms, which matches the 14 ms observed.

### Fix: Fused AllReduce

Pack all gradient tensors into one contiguous host buffer, do one MPI_Allreduce, scatter back.

**Implementation in `src/mpi/data_parallel.hpp`:**

```cpp
// Module-level: pre-allocated host buffer (allocated once in allreduce_gradients_init)
static float* s_grad_buf = nullptr;
static int    s_grad_total = 0;

void allreduce_gradients_init(GPT2& model) {
    int total = 0;
    for (auto& pg : model.param_grads()) total += pg.n;
    s_grad_total = total;
    s_grad_buf = new float[total];  // or use pinned: cudaMallocHost
}

void allreduce_gradients(GPT2& model) {
    auto pgs = model.param_grads();
    
    // 1. Pack all gradients from GPU into one host buffer (one cudaMemcpy per tensor)
    float* ptr = s_grad_buf;
    for (auto& pg : pgs) {
        cudaMemcpy(ptr, pg.grad, pg.n * sizeof(float), cudaMemcpyDeviceToHost);
        ptr += pg.n;
    }
    
    // 2. One MPI_Allreduce on the full buffer
    MPI_Allreduce(MPI_IN_PLACE, s_grad_buf, s_grad_total, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    
    // 3. Scale and scatter back
    float scale = 1.0f / mpi_size();
    ptr = s_grad_buf;
    for (auto& pg : pgs) {
        // scale
        for (int i = 0; i < pg.n; i++) ptr[i] *= scale;
        cudaMemcpy(pg.grad_device_ptr, ptr, pg.n * sizeof(float), cudaMemcpyHostToDevice);
        ptr += pg.n;
    }
}
```

**Important:** `param_grads()` must return `{float* param_device, float* grad_device, int n}` structs. Verify the field names in `src/model/gpt2.hpp`.

**Upgrade: use pinned memory** for the host buffer to maximize PCIe throughput:
```cpp
cudaMallocHost(&s_grad_buf, s_grad_total * sizeof(float));
```
Pinned memory allows DMA transfers at full PCIe bandwidth (~16 GB/s) vs pageable memory which goes through an extra copy.

**Call `allreduce_gradients_init(model)` once** in `main_distributed.cu` after `broadcast_weights(model)`.

### Expected speedup

Current: ~50 MPI calls × α ≈ 14 ms  
After: 1 MPI call × α + β × 13.6 MB  
  = ~300 µs + 13.6e6 bytes × (1/(16 GB/s)) ≈ 300 µs + 850 µs ≈ 1.2 ms  
Expected reduction: 14 ms → ~1-2 ms (6-10× improvement)

### How to verify

1. Run `run_distributed.sh` before and after:
   - Look for `comm X.X ms` lines in output
   - Comm should drop from ~14 ms to ~2 ms

2. Re-run `run_profile.sh` for updated Nsight Systems timeline:
   - The yellow `allreduce` NVTX regions should shrink dramatically

---

## Bottleneck 2: Per-step cudaMalloc in Backward

### What Nsight found

Nsight Systems attributed ~30 ms/step to synchronous `cudaMalloc`/`cudaFree` calls happening inside `backward()`. These come from intermediate `Tensor<float>` objects created during the backward pass.

### Where the allocations happen

From `ARCHITECTURE.md` and the layer files:

1. **`src/layers/transformer_block.hpp` `backward()`**: creates `d_in` tensor temporarily
2. **`src/layers/attention_layer.hpp` `backward()`**: creates `dQ`, `dK`, `dV`, `dO_proj` temporaries
3. **`launch_attention_backward`** in `src/kernels/attention.cu`: may allocate score matrix

These allocations happen N_layers × times per step, so for 4 layers: ~4 × (several) = many cudaMalloc calls.

### Fix: Pre-allocate and cache

**Approach:** Give each layer a "scratch buffer" sized for its largest backward allocation, allocated once at layer construction time and reused every backward pass.

**Step 1: Identify all temporary tensors in backward()**

Read through:
- `src/layers/transformer_block.hpp` — find `Tensor<float> d_in` (or similar)
- `src/layers/attention_layer.hpp` — find `dQ`, `dK`, `dV`, `dO_flat` temporaries

**Step 2: Move them to member variables**

In `MultiHeadAttention` struct, add:
```cpp
struct MultiHeadAttention {
    // ... existing members ...
    
    // Pre-allocated backward scratch (sized at construction)
    Tensor<float> d_q_flat, d_k_flat, d_v_flat;  // (B*S, C) each
    Tensor<float> d_q_bhsd, d_k_bhsd, d_v_bhsd;  // (B, H, S, D) each
    Tensor<float> d_attn_out_flat;                 // (B*S, C)
    
    MultiHeadAttention(int B, int S, int C, int H)
        : /* ... existing init ... */
          d_q_flat(B*S, C), d_k_flat(B*S, C), d_v_flat(B*S, C),
          d_q_bhsd(B*H*S*(C/H)), d_k_bhsd(B*H*S*(C/H)), d_v_bhsd(B*H*S*(C/H)),
          d_attn_out_flat(B*S, C)
    {}
    
    void backward(/* ... */) {
        // Use d_q_flat.data() etc. instead of creating new Tensors
    }
};
```

Similarly in `TransformerBlock`:
```cpp
struct TransformerBlock {
    // ... existing ...
    Tensor<float> d_residual1, d_residual2;  // (B*S, C) scratch
    
    TransformerBlock(int B, int S, int C, int H)
        : /* ... */
          d_residual1(B*S, C), d_residual2(B*S, C)
    {}
};
```

**Step 3: Verify no dangling pointers**

The forward pass may cache raw device pointers (e.g., `float* x_cache` in `Linear`) pointing into tensors that must persist. Ensure the pre-allocated scratch tensors are not overwritten by another use before the backward consumes them.

### How to verify

1. Run `mpirun -np 1 ./build/train_distributed 5 2 128 32 4` before and after.
2. Step time should decrease by ~30 ms.
3. Re-run `run_profile.sh` and look at Nsight Systems: the `cudaMalloc`/`cudaFree` calls in the "OS runtime" track should disappear from within the `forward_backward` NVTX region.

---

## Combined Effect on Scaling

After both optimizations, re-run the full scaling study with `run_distributed.sh`. The Milestone 4 numbers were:

| Sweep | ranks | step (ms) | comm (ms) | efficiency |
|-------|-------|-----------|-----------|------------|
| strong | 4 | 54.26 | 14.00 | 35% |
| weak | 4 | 53.49 | 13.91 | 41% |

After optimization (expected):
- Comm: 14 ms → ~2 ms
- cudaMalloc overhead: 30 ms → 0 ms
- step time at 1 GPU: 75.97 ms → ~45 ms (saved 30 ms cudaMalloc)
- step time at 4 GPUs (strong): 54.26 ms → ~25 ms (saved 30 ms + 12 ms comm) ≈ 12 ms + 2 ms comm
- Expected strong efficiency at 4 GPUs: 45/(4×12) ≈ 94% (much better!)

*Note: These are rough estimates; the actual numbers from the cluster are what go in the report.*

---

## Nsight Systems Re-run After Optimization

After implementing both fixes, re-run `run_profile.sh` to get an updated timeline. The report should show:
1. **Before** (Appendix B, Fig B.1): large yellow allreduce gaps, cudaMalloc calls in green compute region
2. **After** (Appendix B, Fig B.2): tiny allreduce, no cudaMalloc noise

This is a compelling before/after story for the analysis section.

---

## Checklist

- [ ] Implement fused allreduce in `src/mpi/data_parallel.hpp`
- [ ] Add `allreduce_gradients_init()` call to `src/main_distributed.cu`
- [ ] Run distributed tests: verify loss trajectories match before/after
- [ ] Measure comm ms before and after; record numbers
- [ ] Identify backward scratch allocations in transformer_block.hpp + attention_layer.hpp
- [ ] Move them to member variables allocated at construction
- [ ] Run single-GPU training: verify step time decreases by ~30 ms
- [ ] Re-run scaling study; record new numbers
- [ ] Re-run `run_profile.sh`; download updated .nsys-rep
- [ ] Screenshot nsys before and after for Appendix B
