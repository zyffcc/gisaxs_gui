#!/bin/bash
#SBATCH --partition=allcpu
#SBATCH --account=hasylab
#SBATCH --array=0-0
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/data/dust/user/zhaiyufe/MaxwellRuns/GISAXS_ONE_CLICK_20260902/logs/v5-sobol-dataset-%A_%a.out
#SBATCH --error=/data/dust/user/zhaiyufe/MaxwellRuns/GISAXS_ONE_CLICK_20260902/logs/v5-sobol-dataset-%A_%a.err

set -euo pipefail

: "${SLURM_JOB_ID:?submit formal dataset generation through Slurm}"
: "${SLURM_ARRAY_TASK_ID:?submit this wrapper as a Slurm array}"
: "${POSTERIOR_V8_SOURCE_ROOT:?set POSTERIOR_V8_SOURCE_ROOT}"
: "${POSTERIOR_V8_V5_SPLIT_PLAN:?set POSTERIOR_V8_V5_SPLIT_PLAN}"
: "${POSTERIOR_V8_V5_SOBOL_DESIGN:?set POSTERIOR_V8_V5_SOBOL_DESIGN}"
: "${POSTERIOR_V8_V5_GROUPED_OUTPUT_DIR:?set POSTERIOR_V8_V5_GROUPED_OUTPUT_DIR}"
: "${POSTERIOR_V8_V5_TARGET_SPLIT:?set POSTERIOR_V8_V5_TARGET_SPLIT}"

case "$(hostname -s)" in
    max-wgs*) echo "formal V5.1 dataset generation cannot run on max-wgs" >&2; exit 2 ;;
esac
case "$POSTERIOR_V8_V5_GROUPED_OUTPUT_DIR" in
    /data/dust/user/zhaiyufe/*) ;;
    *) echo "V5.1 grouped shards must be under /data/dust/user/zhaiyufe/" >&2; exit 2 ;;
esac

mkdir -p "$POSTERIOR_V8_V5_GROUPED_OUTPUT_DIR"
cd "$POSTERIOR_V8_SOURCE_ROOT"
export PYTHONPATH="$POSTERIOR_V8_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# V5SobolDesign deliberately hashes scipy_version.  Freeze this JSON in the
# same Maxwell `tf` environment; a design copied from a different SciPy build
# is rejected before any curve generation or output publication.
recipes_per_shard="${POSTERIOR_V8_V5_RECIPES_PER_SHARD:-256}"
view_indices="${POSTERIOR_V8_V5_VIEW_INDICES:-0,1,2}"
printf -v shard_name '%s-shard-%06d.gvd5' \
    "$POSTERIOR_V8_V5_TARGET_SPLIT" "$SLURM_ARRAY_TASK_ID"
output="$POSTERIOR_V8_V5_GROUPED_OUTPUT_DIR/$shard_name"
arguments=(
    --split-plan "$POSTERIOR_V8_V5_SPLIT_PLAN"
    --sobol-design "$POSTERIOR_V8_V5_SOBOL_DESIGN"
    --target-split "$POSTERIOR_V8_V5_TARGET_SPLIT"
    --shard-index "$SLURM_ARRAY_TASK_ID"
    --count "$recipes_per_shard"
    --view-indices "$view_indices"
    --output "$output"
)
if [[ "${POSTERIOR_V8_V5_INCLUDE_UNVERIFIED_BRANCHES:-0}" == "1" ]]; then
    arguments+=(--include-unverified-branches)
fi
if [[ "${POSTERIOR_V8_V5_DRY_RUN:-0}" == "1" ]]; then
    arguments+=(--dry-run)
fi

conda run --no-capture-output -n tf \
    python -m utils.ML_Fitting_1D_GISAXS.PosteriorV8.build_formal_sobol_grouped_shard_v5 \
    "${arguments[@]}"
