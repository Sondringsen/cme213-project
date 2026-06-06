# CME 213 Final Project — Small LLM from Scratch

A small GPT-2-style transformer language model built from the ground up in C++/CUDA, distributed across multiple GPUs with MPI. The project implements the core parallel computing primitives behind modern LLM training: hand-written tiled GEMM kernels, Flash Attention (forward and backward), fused LayerNorm/GELU operations, and data-parallel gradient synchronization via MPI Allreduce.

**Course:** CME 213, Spring 2026  
**Authors:** Nils Astrup Toft and Sondre Rogde

---

## Build

```bash
make CUDA_ARCH=75        # Turing (RTX 6000 on teaching cluster)
make CUDA_ARCH=80        # Ampere (A100)
```

---

## Correctness tests

Runs all forward and backward kernel tests plus a single-GPU training sanity check:

```bash
sbatch scripts/run.sh
```

---

## Distributed training (synthetic data)

Strong and weak scaling study with 1, 2, and 4 MPI ranks. Outputs `PERF:` lines for the report tables:

```bash
sbatch scripts/run_distributed.sh
# extract timing numbers:
grep "PERF:" logs/distributed_<jobid>.out
```

---

## WikiText-2 training

**Step 1 — download and tokenize** (run locally, no GPU needed):
```bash
python3 scripts/prepare_data.py
# produces: data/wikitext2_train.bin, data/wikitext2_val.bin, data/vocab.txt
```

**Step 2 — train and save checkpoint** (~2.5 min on cluster for 1000 steps):
```bash
mkdir -p logs checkpoints
mpirun -np 1 ./build/train_distributed \
    1000 6 256 128 8 \
    data/wikitext2_train.bin \
    checkpoints/wikitext_6l.ckpt \
    | tee logs/wikitext_training.log
```

Or submit as a SLURM job (`scripts/run_wikitext.sh`):
```bash
sbatch scripts/run_wikitext.sh
```

**Step 3 — plot loss curve:**
```bash
python3 scripts/plot_loss.py logs/wikitext_training.log plots/loss_curve.png
```

---

## Inference / text generation

Loads a saved checkpoint and generates text autoregressively. No MPI needed.

```bash
# temperature=0 → greedy argmax (may loop on undertrained models)
srun --partition=gpu-turing --gres=gpu:1 ./build/run_inference checkpoints/wikitext_6l.ckpt data/vocab.txt "the president said" 80 0

# temperature=0.8 → sampled (more varied output)
srun --partition=gpu-turing --gres=gpu:1 ./build/run_inference checkpoints/wikitext_6l.ckpt data/vocab.txt "the president said" 80 0.8
```

Usage: `run_inference <checkpoint> <vocab.txt> "<prompt>" [n_tokens=50] [temperature=1.0]`

---

## Profiling (Nsight Systems + Nsight Compute)

Produces a 4-rank Nsight Systems timeline and a single-rank Nsight Compute roofline profile:

```bash
sbatch scripts/run_profile.sh
# download results:
scp <cluster>:<project_dir>/profiles/*.ncu-rep  .
scp <cluster>:<project_dir>/profiles/*.nsys-rep .
```

Open `.nsys-rep` in Nsight Systems and `.ncu-rep` in Nsight Compute.

---

## Allreduce latency benchmark

Measures `MPI_Allreduce` latency vs message size across 1, 2, and 4 ranks. Used to fit the α+βn communication model:

```bash
sbatch scripts/run_allreduce_bench.sh
```

Outputs `ALLREDUCE: ranks=N n_bytes=B avg_us=T` lines for each (rank count, message size) pair.
