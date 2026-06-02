# Step 4 — Communication Analysis (α+βn Benchmark)

**Goal:** Fit the α+βn latency model to MPI_Allreduce timing data; compare with the theoretical Ring All-Reduce bound; produce a log-log plot for the report.

**Estimated time:** 2 hours

---

## Why This Matters

The report must include a communication cost breakdown. The standard model for a collective is:

```
T(n, P) = α + β·n
```

where:
- α = startup latency (seconds) — fixed cost per collective call
- β = inverse bandwidth (seconds/byte) — cost per byte transferred
- n = message size in bytes

For a Ring All-Reduce specifically, the theoretical lower bound is:
```
T_ring(n, P) = 2·(P−1)/P · (α + β·n/P)
```

which at large n reduces to β·2n (each byte traverses the ring twice), and at small n is dominated by the 2·(P−1)/P·α latency.

Measuring T(n) for various n and P gives us α and β, and lets us compare our implementation against the theoretical bound and identify where our host-staging overhead diverges from theory.

---

## Benchmark Program: `benchmarks/allreduce_bench.cu`

### What it does

For each (P, n) combination:
1. Allocate a GPU buffer of n floats
2. Copy to host buffer
3. Run 10 warm-up MPI_Allreduce calls
4. Time 100 MPI_Allreduce calls
5. Report average latency in microseconds

Output format (one line per config):
```
ALLREDUCE: ranks=2 n_bytes=4096 avg_us=312.5
```

### Implementation

```cpp
// benchmarks/allreduce_bench.cu
#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <chrono>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, n_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
    
    // Message sizes: 256 B, 1 KB, 4 KB, 16 KB, 64 KB, 256 KB, 1 MB, 4 MB, 16 MB
    std::vector<int> sizes_bytes = {
        256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216
    };
    
    const int WARMUP = 10;
    const int TRIALS = 100;
    
    for (int n_bytes : sizes_bytes) {
        int n_floats = n_bytes / sizeof(float);
        std::vector<float> buf(n_floats, 1.0f);
        
        // Warm-up
        for (int i = 0; i < WARMUP; i++) {
            MPI_Allreduce(MPI_IN_PLACE, buf.data(), n_floats,
                          MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Timed trials
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < TRIALS; i++) {
            MPI_Allreduce(MPI_IN_PLACE, buf.data(), n_floats,
                          MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        auto t1 = std::chrono::high_resolution_clock::now();
        
        double avg_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / TRIALS;
        
        if (rank == 0) {
            printf("ALLREDUCE: ranks=%d n_bytes=%d avg_us=%.2f\n",
                   n_ranks, n_bytes, avg_us);
            fflush(stdout);
        }
    }
    
    MPI_Finalize();
    return 0;
}
```

This benchmark runs on CPU buffers only (no GPU involved) to measure pure MPI latency. A second variant with GPU buffers + staging copies measures our actual allreduce cost including PCIe transfers.

### SLURM Script: `scripts/run_allreduce_bench.sh`

```bash
#!/bin/bash
#SBATCH --job-name=allreduce_bench
#SBATCH --output=logs/allreduce_bench_%j.out
#SBATCH --partition=gpu-turing
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
make CUDA_ARCH=75  # also builds the benchmark

echo "=== MPI Allreduce Latency Benchmark ==="
echo "Node: $SLURMD_NODENAME"
echo ""

echo "--- P=1 (baseline, no communication) ---"
mpirun -np 1 ./build/allreduce_bench

echo ""
echo "--- P=2 ---"
mpirun -np 2 ./build/allreduce_bench

echo ""
echo "--- P=4 ---"
mpirun -np 4 ./build/allreduce_bench

echo ""
echo "=== Done ==="
```

Add `allreduce_bench` to `CMakeLists.txt` or `Makefile` as a new executable target.

---

## Analysis: Fitting α+βn

After running the benchmark, extract the ALLREDUCE lines and fit the model.

### Python script: `scripts/plot_comm.py`

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Parse output lines like:
# ALLREDUCE: ranks=2 n_bytes=4096 avg_us=312.5
def parse_allreduce_log(filename):
    data = {}
    with open(filename) as f:
        for line in f:
            if line.startswith('ALLREDUCE:'):
                parts = line.strip().split()
                ranks = int(parts[1].split('=')[1])
                n_bytes = int(parts[2].split('=')[1])
                avg_us = float(parts[3].split('=')[1])
                data.setdefault(ranks, []).append((n_bytes, avg_us))
    return data

def alpha_beta_model(n, alpha, beta):
    return alpha + beta * n   # alpha in us, beta in us/byte

fig, ax = plt.subplots(figsize=(6, 4))

for ranks in [2, 4]:
    pts = sorted(data[ranks])
    n_bytes = np.array([p[0] for p in pts])
    t_us    = np.array([p[1] for p in pts])
    
    popt, _ = curve_fit(alpha_beta_model, n_bytes, t_us)
    alpha, beta = popt
    print(f"P={ranks}: alpha={alpha:.1f} us, beta={beta*1e9:.3f} us/GB = {1/beta/1e9:.2f} GB/s")
    
    # Theoretical Ring: T_ring = 2*(P-1)/P * (alpha + beta*n)
    # where alpha is per-hop latency, beta is inverse bandwidth
    # Here we use the measured (alpha, beta) as the point-to-point parameters
    t_ring = 2 * (ranks-1)/ranks * (alpha + beta * n_bytes)
    
    ax.loglog(n_bytes, t_us, 'o-', label=f'Measured P={ranks}')
    ax.loglog(n_bytes, t_ring, '--', label=f'Theoretical Ring P={ranks}')

ax.set_xlabel('Message size (bytes)')
ax.set_ylabel('Latency (µs)')
ax.set_title('MPI_Allreduce: α+βn fit vs. Ring bound')
ax.legend()
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('plots/comm_alpha_beta.png', dpi=150)
print("Saved plots/comm_alpha_beta.png")
```

---

## What to Put in the Report

### Section 6 (Performance Analysis): Communication subsection

**Key points to make:**

1. **α and β values** for P=2 and P=4: "MPI_Allreduce on our system has a per-call startup latency of α ≈ X µs and an effective bandwidth of 1/β ≈ Y GB/s (PCIe Gen3 theoretical: 16 GB/s)."

2. **Per-tensor vs fused allreduce**: "With ~50 per-tensor calls averaging ~280 µs each, the latency-dominated cost is 50 × α ≈ Z ms. Fusing into one call reduces this to α + β × 13.6 MB ≈ W ms."

3. **Gap from Ring bound**: "Our host-staged implementation adds 2× PCIe overhead (D→H and H→D) on top of the MPI cost. CUDA-aware MPI would eliminate both transfers, reducing allreduce cost by ~2× on a PCIe-connected system."

4. **Figure**: log-log plot with measured vs theoretical Ring, for P=2 and P=4.

### Expected numbers (from MS4 context)

At P=4, fused allreduce on 13.6 MB (3.4M floats):
- PCIe D→H: 13.6 MB / 16 GB/s ≈ 0.85 ms
- MPI_Allreduce: α + β×13.6 MB ≈ 1-2 ms (depending on measured β)
- PCIe H→D: 0.85 ms
- Total: ~3-4 ms (vs 14 ms before fusion)

The gap from the Ring lower bound quantifies the host-staging overhead.

---

## Add to CMakeLists.txt / Makefile

In `Makefile` (existing build system), add:
```makefile
allreduce_bench: benchmarks/allreduce_bench.cu
	$(NVCC) $(NVCC_FLAGS) -I$(SRCDIR) $< $(MPI_LIBS) -o $(BUILD)/allreduce_bench
```

---

## Checklist

- [ ] Write `benchmarks/allreduce_bench.cu`
- [ ] Add build target to Makefile/CMakeLists.txt
- [ ] Write `scripts/run_allreduce_bench.sh`
- [ ] Submit job on cluster; download log
- [ ] Run `scripts/plot_comm.py` to fit α+βn and generate plot
- [ ] Record α and β values for the report
- [ ] Save `plots/comm_alpha_beta.png`
- [ ] Write the communication subsection in the report
