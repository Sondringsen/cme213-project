#!/bin/bash
#SBATCH --job-name=wikitext_train
#SBATCH --output=logs/wikitext_%j.out
#SBATCH --partition=gpu-turing
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --time=00:15:00

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs checkpoints

# 4 ranks, total_B=16 (local_B=4 per GPU)
mpirun -np 4 ./build/train_distributed \
    1000 6 256 128 16 \
    data/wikitext2_train.bin \
    checkpoints/wikitext_6l.ckpt \
    | tee logs/wikitext_training.log