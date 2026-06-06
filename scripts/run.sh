#!/bin/bash
# ===========================================================================
# run.sh -- SLURM job script for CME 213 final project (Milestone 4)
#
# Runs all correctness tests: forward kernels (Milestone 3) + backward
# kernels (Milestone 4).  Uses a single GPU.
#
# Submit with:
#     sbatch run.sh
#
# For the distributed scaling study, use run_distributed.sh instead.
#
# Output goes to:
#     logs/run_<jobid>.out
# ===========================================================================

#SBATCH --job-name=cme213_tests
#SBATCH --output=logs/run_%j.out
#SBATCH --partition=gpu-turing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "=============================================="
echo " CME 213 Final Project -- Milestone 4 Tests"
echo "=============================================="
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "GPU(s)      : $CUDA_VISIBLE_DEVICES"
echo "Starting at : $(date)"
echo

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
echo "----------------------------------------------"
echo " Building (sm_75 = Turing RTX 6000)"
echo "----------------------------------------------"
make CUDA_ARCH=75

if [ $? -ne 0 ]; then
    echo "Build FAILED -- aborting."
    exit 1
fi
echo

# ---------------------------------------------------------------------------
# Forward kernel tests (Milestone 3)
# ---------------------------------------------------------------------------
echo "=============================================="
echo " FORWARD KERNEL TESTS"
echo "=============================================="

echo "--- test_gemm (tiled GEMM, register tiling) ---"
./build/test_gemm
echo

echo "--- test_layernorm ---"
./build/test_layernorm
echo

echo "--- test_softmax ---"
./build/test_softmax
echo

echo "--- test_gelu ---"
./build/test_gelu
echo

echo "--- test_cross_entropy ---"
./build/test_cross_entropy
echo

echo "--- test_attention (Flash Attention) ---"
./build/test_attention
echo

# ---------------------------------------------------------------------------
# Backward kernel tests (Milestone 4)
# ---------------------------------------------------------------------------
echo "=============================================="
echo " BACKWARD KERNEL TESTS"
echo "=============================================="

echo "--- test_backward_layernorm ---"
./build/test_backward_layernorm
echo

echo "--- test_backward_gelu ---"
./build/test_backward_gelu
echo

echo "--- test_backward_cross_entropy ---"
./build/test_backward_cross_entropy
echo

echo "--- test_backward_attention ---"
./build/test_backward_attention
echo

# ---------------------------------------------------------------------------
# PyTorch reference comparison (forward + backward)
# ---------------------------------------------------------------------------
REF_DIR="tests/ref_data"
if ls "$REF_DIR"/*.bin 1>/dev/null 2>&1; then
    echo "--- test_vs_pytorch (GPU vs PyTorch reference) ---"
    ./build/test_vs_pytorch
    echo
else
    echo "(Skipping test_vs_pytorch -- run 'python3 scripts/generate_ref_data.py' first)"
    echo
fi

# ---------------------------------------------------------------------------
# Single-GPU training sanity check
# (verifies loss decreases over 10 steps with a tiny model config)
# ---------------------------------------------------------------------------
echo "=============================================="
echo " SINGLE-GPU TRAINING SANITY CHECK"
echo "=============================================="
echo "--- train_distributed --np 1 (10 steps, small model) ---"
mpirun -np 1 ./build/train_distributed 10 2 128 32 4
echo

echo "=============================================="
echo " Finished at: $(date)"
echo "=============================================="
