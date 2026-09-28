#!/bin/bash
#SBATCH --job-name=grpo
#SBATCH --cpus-per-task=128


export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NUM_GPUS=${SLURM_GPUS_ON_NODE:-4} GRADIENT_CHECKPOINTING=false bash scripts/grpo_run.sh