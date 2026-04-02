#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=DD_Optuna
#SBATCH --output=logs/%x-%j.out

# Activate the environment
source ~/env_sat/bin/activate

# 1. Copy the dataset from the slow network to the fast local RAM-disk
# 1. Create a data folder on the local node
mkdir -p $SLURM_TMPDIR/dataset

# 2. Copy all .pt files from scratch to the local node
# (This happens at hardware speed before Python even starts)
cp /scratch/bc0428/DiscreteDiffusion/dataset/*.pt $SLURM_TMPDIR/dataset/

# 3. Run your script pointing to the LOCAL copy
python hyperparam_opt.py --data_dir=$SLURM_TMPDIR/dataset