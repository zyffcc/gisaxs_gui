#!/bin/bash
# One array task = a range of shards of one data set, with all cores of a maxcpu node.
#   sbatch --array=0-N simulate.sbatch CODE_DIR PRIOR OUT COUNT SEED SHARDS_PER_TASK
#SBATCH --partition=maxcpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --time=0-06:00:00
#SBATCH --job-name=ml1d_sim
#SBATCH --output=logs/%x_%A_%a.log
set -euo pipefail
CODE=$1 PRIOR=$2 OUT=$3 COUNT=$4 SEED=$5 PER=$6
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${ML1D_PYTHON:-python}
cd $CODE
FIRST=$((SLURM_ARRAY_TASK_ID * PER))
echo "host $(hostname) cpus ${SLURM_CPUS_PER_TASK} task $SLURM_ARRAY_TASK_ID shards $FIRST+$PER start $(date -Is)"
$PY tools/ml1d/simulate.py --prior $PRIOR --count $COUNT --seed $SEED --out $OUT --shard 10000 \
    --workers ${SLURM_CPUS_PER_TASK} --first-shard $FIRST --shards $PER
echo "done $(date -Is)"
