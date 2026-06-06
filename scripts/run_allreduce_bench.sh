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

rm -rf build && cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 . && cmake --build build -j4

echo "=== MPI Allreduce Latency Benchmark ==="
echo "Node: $SLURMD_NODENAME"
echo ""

echo "--- P=1 ---"
mpirun -np 1 ./build/allreduce_bench

echo ""
echo "--- P=2 ---"
mpirun -np 2 ./build/allreduce_bench

echo ""
echo "--- P=4 ---"
mpirun -np 4 ./build/allreduce_bench

echo ""
echo "=== Done ==="
