#!/bin/bash
# ===========================================================================
# run_distributed.sh -- SLURM job script for the Milestone 4 scaling study.
#
# Requests 4 GPUs on one node, then runs train_distributed with 1, 2, and
# 4 ranks sequentially.  This gives strong-scaling and weak-scaling data
# in a single job, collecting everything needed for the report tables.
#
# Submit with:
#     sbatch run_distributed.sh
#
# Output goes to:
#     logs/distributed_<jobid>.out
#
# After the job finishes, grep the output for lines starting with "PERF:"
# to extract the machine-readable timing numbers:
#     grep PERF logs/distributed_<jobid>.out
# ===========================================================================

#SBATCH --job-name=cme213_distributed
#SBATCH --output=logs/distributed_%j.out
#SBATCH --partition=gpu-turing
#SBATCH --nodes=1
#SBATCH --ntasks=4           # max ranks we will use
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:4         # 4 GPUs on one node
#SBATCH --time=00:15:00

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "=============================================="
echo " CME 213 -- Milestone 4 Scaling Study"
echo "=============================================="
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "GPUs        : $CUDA_VISIBLE_DEVICES"
echo "Starting at : $(date)"
echo

# ---------------------------------------------------------------------------
# Build (in case it hasn't been built yet)
# ---------------------------------------------------------------------------
echo "--- Building (sm_75) ---"
make CUDA_ARCH=75
if [ $? -ne 0 ]; then
    echo "Build FAILED -- aborting."
    exit 1
fi
echo

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------
# Medium model: 6 layers, C=256, S=128 -- fast enough to get clean numbers
# within the 15-minute limit while still exercising the full stack.
#
# For the final report, bump to: LAYERS=12 C=768 S=512 (but allow more time).
#
STEPS=20          # training steps per run (20 is enough for stable timing)
LAYERS=6          # transformer layers
C=256             # model dimension
S=128             # sequence length

# Strong scaling: total batch stays fixed, per-rank batch shrinks with P.
STRONG_TOTAL_B=16  # divisible by 1, 2, and 4

# Weak scaling: per-rank batch stays fixed regardless of P.
WEAK_LOCAL_B=4

# ---------------------------------------------------------------------------
# Helper: print a section banner
# ---------------------------------------------------------------------------
banner() { echo; echo "=============================================="; echo " $*"; echo "=============================================="; }

# ===========================================================================
# STRONG SCALING  (fixed total_B, increasing P)
# Each rank gets total_B / P sequences.
# Ideal: step time halves each time P doubles.
# ===========================================================================
banner "STRONG SCALING  (total_B=${STRONG_TOTAL_B}, S=${S}, C=${C}, layers=${LAYERS})"

echo "--- 1 rank ---"
mpirun -np 1 ./build/train_distributed \
    $STEPS $LAYERS $C $S $STRONG_TOTAL_B
echo

echo "--- 2 ranks ---"
mpirun -np 2 ./build/train_distributed \
    $STEPS $LAYERS $C $S $STRONG_TOTAL_B
echo

echo "--- 4 ranks ---"
mpirun -np 4 ./build/train_distributed \
    $STEPS $LAYERS $C $S $STRONG_TOTAL_B
echo

# ===========================================================================
# WEAK SCALING  (fixed local_B per rank, increasing P)
# Total batch grows with P; per-rank work stays constant.
# Ideal: step time stays flat.  Any increase is communication overhead.
# ===========================================================================
banner "WEAK SCALING  (local_B=${WEAK_LOCAL_B} per rank, S=${S}, C=${C}, layers=${LAYERS})"

echo "--- 1 rank  (total_B=$((WEAK_LOCAL_B * 1))) ---"
mpirun -np 1 ./build/train_distributed \
    $STEPS $LAYERS $C $S $((WEAK_LOCAL_B * 1))
echo

echo "--- 2 ranks (total_B=$((WEAK_LOCAL_B * 2))) ---"
mpirun -np 2 ./build/train_distributed \
    $STEPS $LAYERS $C $S $((WEAK_LOCAL_B * 2))
echo

echo "--- 4 ranks (total_B=$((WEAK_LOCAL_B * 4))) ---"
mpirun -np 4 ./build/train_distributed \
    $STEPS $LAYERS $C $S $((WEAK_LOCAL_B * 4))
echo

# ===========================================================================
# CORRECTNESS CHECK
# Run P=1 and P=4 (4x batch) and compare loss trajectories.
# Because data parallelism with averaged gradients is mathematically
# equivalent to training on the full batch, losses should match.
# ===========================================================================
banner "CORRECTNESS: P=1 vs P=4 loss comparison"

echo "--- P=1 baseline (total_B=4) ---"
mpirun -np 1 ./build/train_distributed \
    10 $LAYERS $C $S 4
echo

echo "--- P=4 equivalent (total_B=16, local_B=4) ---"
mpirun -np 4 ./build/train_distributed \
    10 $LAYERS $C $S 16
echo

# ---------------------------------------------------------------------------
# Summary: extract all PERF lines for easy copy-paste into the report
# ---------------------------------------------------------------------------
banner "PERF SUMMARY (copy these into Milestone4.tex tables)"
echo "(These lines are also in the full log above)"
echo

# Re-run a quick 5-step pass for each config and just print PERF lines.
# (Avoids needing to re-parse the log above.)
for NP in 1 2 4; do
    TOTAL=$((STRONG_TOTAL_B))
    mpirun -np $NP ./build/train_distributed \
        5 $LAYERS $C $S $TOTAL 2>/dev/null | grep "^PERF:"
done

echo
echo "=============================================="
echo " Finished at: $(date)"
echo "=============================================="
