#!/bin/bash
#SBATCH --partition=allcpu
#SBATCH --account=hasylab
#SBATCH --array=0-0
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/data/dust/user/zhaiyufe/MaxwellRuns/v5-unbound-dataset-%A_%a.out
#SBATCH --error=/data/dust/user/zhaiyufe/MaxwellRuns/v5-unbound-dataset-%A_%a.err

set -euo pipefail

: "${SLURM_JOB_ID:?submit formal dataset generation through Slurm}"
: "${SLURM_ARRAY_TASK_ID:?submit this wrapper as a Slurm array}"
: "${POSTERIOR_V8_SOURCE_ROOT:?set POSTERIOR_V8_SOURCE_ROOT}"
: "${POSTERIOR_V8_V5_SPLIT_PLAN:?set POSTERIOR_V8_V5_SPLIT_PLAN}"
: "${POSTERIOR_V8_V5_SOBOL_DESIGN:?set POSTERIOR_V8_V5_SOBOL_DESIGN}"
: "${POSTERIOR_V8_V5_GROUPED_OUTPUT_DIR:?set POSTERIOR_V8_V5_GROUPED_OUTPUT_DIR}"
: "${POSTERIOR_V8_V5_TARGET_SPLIT:?set POSTERIOR_V8_V5_TARGET_SPLIT}"
: "${POSTERIOR_V8_V5_RECIPES_PER_SHARD:?set POSTERIOR_V8_V5_RECIPES_PER_SHARD}"
: "${POSTERIOR_V8_V5_TOTAL_RECIPES:?set POSTERIOR_V8_V5_TOTAL_RECIPES}"

case "$(hostname -s)" in
    max-wgs*) echo "formal V5.1 dataset generation cannot run on max-wgs" >&2; exit 2 ;;
esac
for dust_path in \
    "$POSTERIOR_V8_V5_SPLIT_PLAN" \
    "$POSTERIOR_V8_V5_SOBOL_DESIGN" \
    "$POSTERIOR_V8_V5_GROUPED_OUTPUT_DIR"; do
    case "$dust_path" in
        /data/dust/user/zhaiyufe/*) ;;
        *) echo "V5.1 data and contracts must be under /data/dust/user/zhaiyufe/" >&2; exit 2 ;;
    esac
done
for integer_value in \
    "$SLURM_ARRAY_TASK_ID" \
    "$POSTERIOR_V8_V5_RECIPES_PER_SHARD" \
    "$POSTERIOR_V8_V5_TOTAL_RECIPES"; do
    case "$integer_value" in
        ''|*[!0-9]*) echo "array index, shard size, and total recipes must be integers" >&2; exit 2 ;;
    esac
done
if (( POSTERIOR_V8_V5_RECIPES_PER_SHARD < 1 || POSTERIOR_V8_V5_TOTAL_RECIPES < 1 )); then
    echo "shard size and total recipes must be positive" >&2
    exit 2
fi

shard_start=$((SLURM_ARRAY_TASK_ID * POSTERIOR_V8_V5_RECIPES_PER_SHARD))
if (( shard_start >= POSTERIOR_V8_V5_TOTAL_RECIPES )); then
    echo "array task starts beyond the audited requested recipe total" >&2
    exit 2
fi
remaining=$((POSTERIOR_V8_V5_TOTAL_RECIPES - shard_start))
shard_count=$POSTERIOR_V8_V5_RECIPES_PER_SHARD
if (( remaining < shard_count )); then
    shard_count=$remaining
fi

mkdir -p "$POSTERIOR_V8_V5_GROUPED_OUTPUT_DIR"
cd "$POSTERIOR_V8_SOURCE_ROOT"
export PYTHONPATH="$POSTERIOR_V8_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

printf -v shard_name '%s-shard-%06d.gvd5' \
    "$POSTERIOR_V8_V5_TARGET_SPLIT" "$SLURM_ARRAY_TASK_ID"
output="$POSTERIOR_V8_V5_GROUPED_OUTPUT_DIR/$shard_name"
arguments=(
    --split-plan "$POSTERIOR_V8_V5_SPLIT_PLAN"
    --sobol-design "$POSTERIOR_V8_V5_SOBOL_DESIGN"
    --target-split "$POSTERIOR_V8_V5_TARGET_SPLIT"
    --start "$shard_start"
    --count "$shard_count"
    --view-indices 0,1,2
    --include-unverified-branches
    --output "$output"
)

conda run --no-capture-output -n tf \
    python -m utils.ML_Fitting_1D_GISAXS.PosteriorV8.build_formal_sobol_grouped_shard_v5 \
    "${arguments[@]}"
