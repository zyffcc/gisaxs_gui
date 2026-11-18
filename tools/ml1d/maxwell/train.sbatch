#!/bin/bash
# Train the proposal network on one GPU; resumes from NAME.pt when requeued.
#   sbatch train.sbatch CODE_DIR DATA_DIR PRIOR OUT_NAME [extra train_flow.py arguments]
#SBATCH --partition=maxgpu,allgpu
#SBATCH --constraint=A100|H100|L40S|V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --job-name=ml1d_train
#SBATCH --output=logs/%x_%j.log
set -euo pipefail
CODE=$1 DATA=$2 PRIOR=$3 OUT=$4
shift 4
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${ML1D_PYTHON:-python}
cd $CODE
echo "host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) start $(date -Is)"
$PY tools/ml1d/train_flow.py --data $DATA --prior $PRIOR --out $OUT --device cuda --threads 4 --loaders 24 --resume "$@"
echo "done $(date -Is)"
