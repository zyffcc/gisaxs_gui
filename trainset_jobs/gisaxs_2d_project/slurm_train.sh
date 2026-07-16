#!/bin/bash
#SBATCH --job-name=gisaxs_2d_project-train
#SBATCH --partition=allgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
set -euo pipefail
mkdir -p logs results
python train.py
