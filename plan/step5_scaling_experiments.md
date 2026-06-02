# Step 5 — Extended Scaling Experiments

**Goal:** Run the final scaling study with the optimized code (fused allreduce + pre-allocated scratch) and produce clean scaling numbers and plots for the report.

**Estimated time:** 1 hour

---

## What to Run

### Configuration for Final Report

Use a single model config matching MS4: `LAYERS=4, C=256, S=128, V=512`.
This is sufficient for the report and runs well within 15 minutes.

**Cluster constraint: all SLURM jobs must complete in 15 minutes.**
With STEPS=10 and small model (LAYERS=4, C=256, S=128), 6 runs (3 strong + 3
weak) take roughly 6 × 30 steps/ms = well under 5 minutes total.

For the strong scaling, keep total_B fixed at 16.
For weak scaling, keep local_B fixed at 4.

Use STEPS=10 (enough for stable timing; first step JIT overhead is small
compared to 10-step average at these model sizes).

### SLURM Script: `scripts/run_scaling_final.sh`

```bash
#!/bin/bash
#SBATCH --job-name=scaling_final
#SBATCH --output=logs/scaling_final_%j.out
#SBATCH --partition=gpu-turing
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:4
#SBATCH --time=00:15:00

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
make CUDA_ARCH=75

STEPS=10
LAYERS=4; C=256; S=128
STRONG_TOTAL_B=16
WEAK_LOCAL_B=4

banner() { echo; echo "=============================="; echo " $*"; echo "=============================="; }

banner "STRONG SCALING (total_B=$STRONG_TOTAL_B)"
for NP in 1 2 4; do
    echo "--- $NP rank(s) ---"
    mpirun -np $NP ./build/train_distributed $STEPS $LAYERS $C $S $STRONG_TOTAL_B
done

banner "WEAK SCALING (local_B=$WEAK_LOCAL_B)"
for NP in 1 2 4; do
    TOTAL=$((WEAK_LOCAL_B * NP))
    echo "--- $NP rank(s) (total_B=$TOTAL) ---"
    mpirun -np $NP ./build/train_distributed $STEPS $LAYERS $C $S $TOTAL
done

banner "Extracting PERF lines"
grep "^PERF:" logs/scaling_final_${SLURM_JOB_ID}.out || grep "^PERF:" $0
```

---

## Extracting Numbers from the Log

After the job finishes:
```bash
grep "PERF:" logs/scaling_final_<jobid>.out
```

Each `PERF:` line has: `ranks, local_B, S, C, layers, avg_step_ms, comm_fraction`

Build a table manually for the report (copy into `results_tables/scaling_final.tex`).

---

## Analysis to Include in the Report

### Strong Scaling

| P | step_ms | comm_ms | speedup | efficiency |
|---|---------|---------|---------|------------|
| 1 | T₁ | 0 | 1.00× | 100% |
| 2 | T₂ | c₂ | T₁/T₂ | T₁/(2T₂) |
| 4 | T₄ | c₄ | T₁/T₄ | T₁/(4T₄) |

Efficiency E(P) = T₁ / (P × T_P). Ideal: 100%. Expected after optimization: ≥70% at P=4.

**Amdahl's law fit:**
The serial fraction f is found from: E(P) ≈ 1 / (f + (1-f)/P)

Solve for f: f = (1/E(P) - 1) / (P - 1/E(P))

Report f as the irreducible serial fraction. If we achieve 70% efficiency at P=4, then f ≈ (1/0.7 - 1)/(4 - 1/0.7) ≈ 0.43/3.57 ≈ 12%. This is the fraction of the step that cannot be parallelized (parameter sync + sequential setup).

### Weak Scaling

| P | local_B | step_ms | efficiency |
|---|---------|---------|------------|
| 1 | 4 | T₁ | 100% |
| 2 | 4 | T₂ | T₁/T₂ |
| 4 | 4 | T₄ | T₁/T₄ |

Weak scaling efficiency: T₁/T_P. Any drop is communication overhead (since compute per rank is identical).

Expected after optimization: ≥80% at P=4 (comm is now ~1-2 ms out of ~20-30 ms total step).

### Isoefficiency Analysis

Isoefficiency measures how much work must grow with P to maintain constant efficiency. For data-parallel training with fixed communication cost C_comm:

```
T_compute ≥ C_comm / (1 - E_target)
```

If C_comm ≈ 2 ms (after fused allreduce) and E_target = 0.9:
```
T_compute ≥ 2 ms / 0.1 = 20 ms
```

This means we need at least 20 ms of compute per rank to maintain 90% efficiency. This translates to a minimum batch size / model size requirement, which you can compute from the per-step timing.

---

## Plotting: `scripts/plot_scaling.py`

```python
import matplotlib.pyplot as plt
import numpy as np

# Fill in from the PERF: lines
# [P, step_ms, comm_ms] for strong scaling (small model, post-optimization)
strong = [
    (1, T1, 0),
    (2, T2, c2),
    (4, T4, c4),
]

# Strong scaling plot
P_vals = [x[0] for x in strong]
step_ms = [x[1] for x in strong]
comm_ms = [x[2] for x in strong]
compute_ms = [s - c for s, c in zip(step_ms, comm_ms)]
T1 = strong[0][1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Left: stacked bar (compute vs comm)
ax1.bar(P_vals, compute_ms, label='Compute', color='steelblue')
ax1.bar(P_vals, comm_ms, bottom=compute_ms, label='Communication', color='orange')
ax1.set_xlabel('Number of GPUs (P)')
ax1.set_ylabel('Step time (ms)')
ax1.set_title('Strong scaling: step time breakdown')
ax1.legend()
ax1.set_xticks(P_vals)

# Efficiency annotations
for p, s in zip(P_vals, step_ms):
    eff = T1 / (p * s) * 100
    ax1.annotate(f'{eff:.0f}%', xy=(p, s), ha='center', va='bottom')

# Right: weak scaling efficiency
# [P, step_ms] for weak scaling
weak = [(1, W1), (2, W2), (4, W4)]
weak_eff = [W1 / ws * 100 for _, ws in weak]
ax2.plot([x[0] for x in weak], weak_eff, 'o-', color='green', label='Weak scaling')
ax2.plot([x[0] for x in strong], [T1/(p*s)*100 for p, s, _ in strong], 
         's--', color='steelblue', label='Strong scaling')
ax2.axhline(100, color='gray', linestyle=':', label='Ideal')
ax2.set_xlabel('Number of GPUs (P)')
ax2.set_ylabel('Parallel efficiency (%)')
ax2.set_title('Scaling efficiency')
ax2.legend()
ax2.set_xticks([1, 2, 4])
ax2.set_ylim(0, 110)

plt.tight_layout()
plt.savefig('plots/scaling_final.png', dpi=150)
print("Saved plots/scaling_final.png")
```

---

## Before vs After Comparison Table

For the report, show both MS4 (before) and final (after) numbers:

| | P=1 | P=2 (eff) | P=4 (eff) |
|--|-----|-----------|-----------|
| MS4 (before) | 75.97 ms | 62% | 35% |
| After fused AR | ? ms | ?% | ?% |
| After + pre-alloc | ? ms | ?% | ?% |

Fill in from cluster runs. This shows the incremental improvement from each optimization and is a strong analysis story.

---

## Checklist

- [ ] Write and submit `scripts/run_scaling_final.sh`
- [ ] Download log file
- [ ] Extract PERF: lines, build scaling table
- [ ] Fill in `scripts/plot_scaling.py` with actual numbers
- [ ] Run plot script, save `plots/scaling_final.png`
- [ ] Compute Amdahl serial fraction f from efficiency numbers
- [ ] Compute isoefficiency minimum batch size
- [ ] Write the scaling subsection in the report
