#!/bin/bash
#SBATCH --job-name=paged_bench
#SBATCH --output=logs/paged_bench_%j.out
#SBATCH --partition=gpu-turing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

rm -rf build && cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 . && cmake --build build -j4

echo "=== PagedAttention decode benchmark ==="
echo "Node: $SLURMD_NODENAME"
echo ""

# Default sweep (B grows internally inside the binary): H=8, D=64, S_max=128,
# mean_len=64, n_iter=200. Run a second config with longer mean to stress
# the long-tail end of the distribution.
echo "--- H=8, D=64, S_max=128, mean_len=64 ---"
./build/paged_attention_bench 8 64 128 64 200

echo ""
echo "--- H=8, D=64, S_max=128, mean_len=96 ---"
./build/paged_attention_bench 8 64 128 96 200

echo ""
echo "--- H=8, D=128, S_max=128, mean_len=64 ---"
./build/paged_attention_bench 8 128 128 64 200

echo ""
echo "=== Correctness gate ==="
./build/test_paged_attention

echo ""
echo "=== Done ==="
