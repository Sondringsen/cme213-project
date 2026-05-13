#!/bin/bash
# ===========================================================================
# run.sh -- SLURM job script for CME 213 final project (Milestone 3)
#
# Submit with:
#     sbatch run.sh
#
# Output goes to:
#     logs/run_<jobid>.out   (stdout + stderr interleaved)
#
# Adjust the #SBATCH lines below to match your cluster's setup.
# Common things to change:
#   --partition  : check available partitions with `sinfo`
#   --gres       : GPU type/count (e.g. gpu:1, gpu:titanrtx:1, gpu:a100:1)
#   --time       : keep well under 15 minutes on the shared cluster
# ===========================================================================

#SBATCH --job-name=cme213_kernels
#SBATCH --output=logs/run_%j.out
#SBATCH --partition=gpu-turing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
# Change directory to wherever you submitted from.
cd "$SLURM_SUBMIT_DIR"

# Create log directory if it doesn't exist.
mkdir -p logs

echo "=============================================="
echo " CME 213 Final Project -- Milestone 3 Kernels"
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
echo " Building with CMake"
echo "----------------------------------------------"
# gpu-turing nodes have Quadro RTX 6000 (Turing architecture = sm_75).
make CUDA_ARCH=75

if [ $? -ne 0 ]; then
    echo "Build FAILED -- aborting."
    exit 1
fi
echo

# ---------------------------------------------------------------------------
# Kernel tests
# ---------------------------------------------------------------------------

echo "----------------------------------------------"
echo " test_gemm (tiled GEMM correctness + GFLOPS)"
echo "----------------------------------------------"
./build/test_gemm
echo

echo "----------------------------------------------"
echo " test_layernorm (fused LayerNorm correctness + bandwidth)"
echo "----------------------------------------------"
./build/test_layernorm
echo

echo "----------------------------------------------"
echo " test_softmax (softmax correctness + bandwidth)"
echo "----------------------------------------------"
./build/test_softmax
echo

echo "----------------------------------------------"
echo " test_gelu (GELU correctness + bandwidth)"
echo "----------------------------------------------"
./build/test_gelu
echo

echo "----------------------------------------------"
echo " test_cross_entropy (cross-entropy correctness)"
echo "----------------------------------------------"
./build/test_cross_entropy
echo

echo "----------------------------------------------"
echo " test_attention (Flash Attention correctness + GFLOPS)"
echo "----------------------------------------------"
./build/test_attention
echo

# ---------------------------------------------------------------------------
# PyTorch reference comparison (optional -- only if ref data was generated)
# ---------------------------------------------------------------------------
REF_DIR="tests/ref_data"
if ls "$REF_DIR"/*.bin 1>/dev/null 2>&1; then
    echo "----------------------------------------------"
    echo " test_vs_pytorch (GPU vs PyTorch reference)"
    echo "----------------------------------------------"
    ./build/test_vs_pytorch
    echo
else
    echo "(Skipping test_vs_pytorch -- run 'python3 scripts/generate_ref_data.py' first)"
    echo
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo "=============================================="
echo " Finished at: $(date)"
echo "=============================================="
