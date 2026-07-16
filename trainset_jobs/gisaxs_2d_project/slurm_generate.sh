#!/bin/bash
#SBATCH --job-name=gisaxs_2d_project-generate
#SBATCH --partition=allgpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-99
#SBATCH --output=logs/generate_%A_%a.out
#SBATCH --error=logs/generate_%A_%a.err
set -euo pipefail
mkdir -p logs dataset
python generate_dataset.py --config config.yaml --samples 2000 --output dataset
