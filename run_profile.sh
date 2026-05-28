#!/bin/bash
# ===========================================================================
# run_profile.sh -- SLURM job script for NVIDIA Nsight profiling.
#
# Produces two downloadable profile files:
#
#   profiles/nsys_rank{0,1,2,3}.nsys-rep  -- open in Nsight Systems
#       End-to-end CUDA + MPI + NVTX timeline for each rank.  Shows
#       forward/backward, allreduce, and optimizer phases clearly labeled.
#
#   profiles/ncu_kernels.ncu-rep           -- open in Nsight Compute
#       Per-kernel roofline metrics for a single-rank run: occupancy,
#       memory bandwidth, and compute throughput for every kernel.
#
# Submit with:
#     sbatch run_profile.sh
#
# Download the results (run from your local machine):
#     scp <cluster>:<project_dir>/profiles/*.nsys-rep .
#     scp <cluster>:<project_dir>/profiles/*.ncu-rep  .
#
# Output goes to:
#     logs/profile_<jobid>.out
# ===========================================================================

#SBATCH --job-name=cme213_profile
#SBATCH --output=logs/profile_%j.out
#SBATCH --partition=gpu-turing
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:4
#SBATCH --time=00:15:00

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs profiles

echo "=============================================="
echo " CME 213 -- Nsight Profiling Run"
echo "=============================================="
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "GPUs        : $CUDA_VISIBLE_DEVICES"
echo "Starting at : $(date)"
echo

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
echo "--- Building (sm_75 + NVTX) ---"
make CUDA_ARCH=75
if [ $? -ne 0 ]; then
    echo "Build FAILED -- aborting."
    exit 1
fi
echo

# ---------------------------------------------------------------------------
# Profile configuration
# A small-but-real model: enough to show all phases without hitting the
# 15-minute wall time.  ncu is very slow (serializes kernels) so we use
# a smaller model and fewer steps for that run.
# ---------------------------------------------------------------------------
NSYS_STEPS=5      # steps for the nsys timeline run
NSYS_LAYERS=4
NSYS_C=256
NSYS_S=128
NSYS_TOTAL_B=8    # 2 sequences per rank with 4 ranks

NCU_STEPS=1       # just 1 step — ncu profiles every kernel launch
NCU_LAYERS=2
NCU_C=128
NCU_S=32
NCU_TOTAL_B=4

# ===========================================================================
# Nsight Systems — 4-rank CUDA + MPI + NVTX timeline
#
# Each rank writes its own .nsys-rep file.  The %q{VAR} syntax expands the
# environment variable VAR (set by OpenMPI) into the output filename so all
# four ranks write distinct files without clobbering each other.
#
# --trace=cuda,mpi,nvtx,osrt  captures:
#   cuda  — CUDA API calls and kernel launches
#   mpi   — MPI calls (Allreduce, Bcast, etc.)
#   nvtx  — named regions from our NVTX_PUSH/NVTX_POP annotations
#   osrt  — OS runtime (pthread, memory allocation)
# ===========================================================================
echo "=============================================="
echo " Nsight Systems: 4-rank timeline"
echo " Steps=$NSYS_STEPS  Layers=$NSYS_LAYERS  C=$NSYS_C  S=$NSYS_S"
echo "=============================================="

mpirun -np 4 nsys profile \
    --output=profiles/nsys_rank%q{OMPI_COMM_WORLD_RANK} \
    --trace=cuda,mpi,nvtx,osrt \
    --force-overwrite=true \
    --stats=true \
    ./build/train_distributed \
        $NSYS_STEPS $NSYS_LAYERS $NSYS_C $NSYS_S $NSYS_TOTAL_B

echo
echo "Nsight Systems files:"
ls -lh profiles/nsys_rank*.nsys-rep 2>/dev/null || echo "(no .nsys-rep files found)"
echo

# ===========================================================================
# Nsight Compute — single-rank kernel-level metrics (roofline)
#
# ncu instruments every CUDA kernel and collects hardware performance
# counters.  This serializes execution and is ~100x slower than normal,
# so we use the smallest model config and a single training step.
#
# --set full       collects the full set of metrics (inc. roofline data)
# --launch-count   limits profiling to the first N kernel invocations so
#                  the job does not time out
# --force-overwrite overwrites any existing .ncu-rep file
#
# Open the resulting .ncu-rep in Nsight Compute on your local machine.
# The "Roofline Analysis" section will show each kernel plotted against
# the theoretical memory and compute ceilings of the RTX 6000.
# ===========================================================================
echo "=============================================="
echo " Nsight Compute: single-rank kernel metrics"
echo " Steps=$NCU_STEPS  Layers=$NCU_LAYERS  C=$NCU_C  S=$NCU_S"
echo "=============================================="

mpirun -np 1 ncu \
    --output=profiles/ncu_kernels \
    --set full \
    --force-overwrite \
    --launch-count 200 \
    ./build/train_distributed \
        $NCU_STEPS $NCU_LAYERS $NCU_C $NCU_S $NCU_TOTAL_B

echo
echo "Nsight Compute file:"
ls -lh profiles/ncu_kernels.ncu-rep 2>/dev/null || echo "(no .ncu-rep file found)"
echo

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "=============================================="
echo " Download commands (run from your local machine):"
echo "   scp ${USER}@<cluster>:$(pwd)/profiles/*.nsys-rep ."
echo "   scp ${USER}@<cluster>:$(pwd)/profiles/*.ncu-rep  ."
echo
echo " Open in GUI:"
echo "   nsys_rank*.nsys-rep  ->  Nsight Systems"
echo "   ncu_kernels.ncu-rep  ->  Nsight Compute"
echo "=============================================="
echo "Finished at: $(date)"
