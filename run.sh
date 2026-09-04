#!/bin/bash
#SBATCH --job-name=systems
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=24GB
#SBATCH --time=00:30:00
#SBATCH --output=logs/systems_%j.out
#SBATCH --error=logs/systems_%j.err

module load miniconda
conda activate rl-lab

pip install -r requirements.txt

python -m cs336_systems.benchmark
