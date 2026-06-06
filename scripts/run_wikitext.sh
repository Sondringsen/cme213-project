#!/bin/bash
#SBATCH --job-name=wikitext_train
#SBATCH --output=logs/wikitext_%j.out
#SBATCH --partition=gpu-turing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs checkpoints

mpirun -np 1 ./build/train_distributed \
    1000 6 256 128 8 \
    data/wikitext2_train.bin \
    checkpoints/wikitext_6l.ckpt \
    | tee logs/wikitext_training.log