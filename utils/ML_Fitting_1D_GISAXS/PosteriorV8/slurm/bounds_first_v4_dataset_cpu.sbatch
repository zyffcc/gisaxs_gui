#!/bin/bash
#SBATCH --partition=allcpu
#SBATCH --account=hasylab
#SBATCH --array=0-0
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/data/dust/user/zhaiyufe/MaxwellRuns/GISAXS_POSTERIOR_V8_20260902/logs/v4-dataset-%A_%a.out
#SBATCH --error=/data/dust/user/zhaiyufe/MaxwellRuns/GISAXS_POSTERIOR_V8_20260902/logs/v4-dataset-%A_%a.err

set -euo pipefail

: "${SLURM_JOB_ID:?submit dataset generation through Slurm; do not run it on max-wgs}"
: "${POSTERIOR_V8_SOURCE_ROOT:?set POSTERIOR_V8_SOURCE_ROOT}"
: "${SLURM_ARRAY_TASK_ID:?submit this wrapper as a Slurm array}"

case "$(hostname -s)" in
    max-wgs*) echo "V4 dataset generation must run on a Slurm worker" >&2; exit 2 ;;
esac

output_dir="${POSTERIOR_V8_V4_OUTPUT_DIR:-/data/dust/user/zhaiyufe/MaxwellRuns/GISAXS_POSTERIOR_V8_20260902/datasets/posterior_v8_bounds_first_v4}"
case "$output_dir" in
    /data/dust/user/zhaiyufe/*) ;;
    *) echo "POSTERIOR_V8_V4_OUTPUT_DIR must be under /data/dust/user/zhaiyufe/" >&2; exit 2 ;;
esac

cd "$POSTERIOR_V8_SOURCE_ROOT"
export PYTHONPATH="$POSTERIOR_V8_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

recipe_count="${POSTERIOR_V8_V4_RECIPES_PER_SHARD:-256}"
dataset_start="${POSTERIOR_V8_V4_DATASET_START:-0}"
shard_index="$SLURM_ARRAY_TASK_ID"
start_recipe_index="$((dataset_start + shard_index * recipe_count))"

conda run --no-capture-output -n tf \
    python -m utils.ML_Fitting_1D_GISAXS.PosteriorV8.build_bounds_first_shards build \
    --output-dir "$output_dir" \
    --shard-index "$shard_index" \
    --start-recipe-index "$start_recipe_index" \
    --recipe-count "$recipe_count" \
    --master-seed "${POSTERIOR_V8_V4_SEED:-20260903}" \
    --topology-schedule "${POSTERIOR_V8_V4_TOPOLOGY_SCHEDULE:-all34}" \
    --views-per-recipe "${POSTERIOR_V8_V4_VIEWS_PER_RECIPE:-3}" \
    --max-raw-points "${POSTERIOR_V8_V4_MAX_RAW_POINTS:-512}"
