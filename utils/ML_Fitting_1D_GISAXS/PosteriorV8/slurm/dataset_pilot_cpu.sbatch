#!/bin/bash
#SBATCH --partition=allcpu
#SBATCH --account=hasylab
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

: "${POSTERIOR_V8_SOURCE_ROOT:?set POSTERIOR_V8_SOURCE_ROOT}"
: "${POSTERIOR_V8_OUTPUT_DIR:?set POSTERIOR_V8_OUTPUT_DIR}"

cd "$POSTERIOR_V8_SOURCE_ROOT"
export PYTHONPATH="$POSTERIOR_V8_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

recipe_count="${POSTERIOR_V8_RECIPE_COUNT:-1024}"
shard_index="${POSTERIOR_V8_SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
start_index="${POSTERIOR_V8_START_INDEX:-$((shard_index * recipe_count))}"

conda run --no-capture-output -n tf \
    python -m utils.ML_Fitting_1D_GISAXS.PosteriorV8.build_dataset \
    --output-dir "$POSTERIOR_V8_OUTPUT_DIR" \
    --shard-index "$shard_index" \
    --start-index "$start_index" \
    --recipe-count "$recipe_count" \
    --master-seed "${POSTERIOR_V8_SEED:-20260902}" \
    --topology-schedule "${POSTERIOR_V8_TOPOLOGY_SCHEDULE:-all34}" \
    --views-per-recipe "${POSTERIOR_V8_VIEWS_PER_RECIPE:-3}" \
    --max-raw-points "${POSTERIOR_V8_MAX_RAW_POINTS:-512}"
