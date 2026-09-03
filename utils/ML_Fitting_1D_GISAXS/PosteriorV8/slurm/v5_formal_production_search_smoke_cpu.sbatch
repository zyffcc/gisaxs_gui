#!/bin/bash
#SBATCH --partition=allcpu
#SBATCH --account=hasylab
#SBATCH --array=0-0
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/data/dust/user/zhaiyufe/MaxwellRuns/v5-unbound-formal-smoke-%A_%a.out
#SBATCH --error=/data/dust/user/zhaiyufe/MaxwellRuns/v5-unbound-formal-smoke-%A_%a.err

set -euo pipefail

: "${SLURM_JOB_ID:?formal search must run through Slurm}"
: "${SLURM_ARRAY_TASK_ID:?formal search must be a Slurm array}"
: "${POSTERIOR_V8_SOURCE_ROOT:?set source root}"
: "${POSTERIOR_V8_V5_FORMAL_GLOBAL_PLAN:?set canonical global plan}"
: "${POSTERIOR_V8_V5_EXPECTED_GLOBAL_PLAN_SHA256:?set expected plan SHA-256}"
: "${POSTERIOR_V8_V5_EXPECTED_SHARD_SHA256S:?set ordered shard SHA-256 list}"
: "${POSTERIOR_V8_V5_FORMAL_STAGE_ID:?set formal stage ID}"
: "${POSTERIOR_V8_V5_FORMAL_TARGET_SPLIT:?set target split}"
: "${POSTERIOR_V8_V5_FORMAL_RUN_ROOT:?set formal run root}"
: "${POSTERIOR_V8_V5_SPLIT_PLAN:?set split plan}"
: "${POSTERIOR_V8_V5_SOBOL_DESIGN:?set Sobol design}"
: "${POSTERIOR_V8_V5_LOCAL_SOBOL_SCHEDULE:?set local Sobol schedule}"
: "${POSTERIOR_V8_V5_CALIBRATION_ARTIFACT:?set checked calibration}"

case "$(hostname -s)" in
    max-wgs*) echo "formal exact search cannot run on max-wgs" >&2; exit 2 ;;
esac
case "$SLURM_ARRAY_TASK_ID" in
    ''|*[!0-9]*) echo "array task ID must be unsigned" >&2; exit 2 ;;
esac
case "$POSTERIOR_V8_V5_FORMAL_STAGE_ID" in
    K1|K2|ALL34) ;;
    *) echo "invalid formal stage" >&2; exit 2 ;;
esac
case "$POSTERIOR_V8_V5_FORMAL_TARGET_SPLIT" in
    train|tuning_validation) ;;
    *) echo "invalid formal split" >&2; exit 2 ;;
esac
for dust_path in \
    "$POSTERIOR_V8_SOURCE_ROOT" \
    "$POSTERIOR_V8_V5_FORMAL_GLOBAL_PLAN" \
    "$POSTERIOR_V8_V5_FORMAL_RUN_ROOT" \
    "$POSTERIOR_V8_V5_SPLIT_PLAN" \
    "$POSTERIOR_V8_V5_SOBOL_DESIGN" \
    "$POSTERIOR_V8_V5_LOCAL_SOBOL_SCHEDULE" \
    "$POSTERIOR_V8_V5_CALIBRATION_ARTIFACT"; do
    case "$dust_path" in
        /data/dust/user/zhaiyufe/*) ;;
        *) echo "formal inputs and outputs must be under the Maxwell dust root" >&2; exit 2 ;;
    esac
done

IFS=: read -r -a shard_sha256s <<< "$POSTERIOR_V8_V5_EXPECTED_SHARD_SHA256S"
if (( SLURM_ARRAY_TASK_ID >= ${#shard_sha256s[@]} )); then
    echo "array task has no planned shard SHA-256" >&2
    exit 2
fi
expected_shard_sha256=${shard_sha256s[$SLURM_ARRAY_TASK_ID]}
case "$expected_shard_sha256" in
    *[!0-9a-f]*|'') echo "invalid expected shard SHA-256" >&2; exit 2 ;;
esac
if (( ${#expected_shard_sha256} != 64 )); then
    echo "invalid expected shard SHA-256 length" >&2
    exit 2
fi

cd "$POSTERIOR_V8_SOURCE_ROOT"
export PYTHONPATH="$POSTERIOR_V8_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

conda run --no-capture-output -n tf \
    python -m utils.ML_Fitting_1D_GISAXS.PosteriorV8.run_formal_production_search_smoke_v5 \
    --global-plan "$POSTERIOR_V8_V5_FORMAL_GLOBAL_PLAN" \
    --expected-global-plan-sha256 "$POSTERIOR_V8_V5_EXPECTED_GLOBAL_PLAN_SHA256" \
    --expected-shard-plan-sha256 "$expected_shard_sha256" \
    --stage-id "$POSTERIOR_V8_V5_FORMAL_STAGE_ID" \
    --target-split "$POSTERIOR_V8_V5_FORMAL_TARGET_SPLIT" \
    --array-task-id "$SLURM_ARRAY_TASK_ID" \
    --source-root "$POSTERIOR_V8_SOURCE_ROOT" \
    --run-root "$POSTERIOR_V8_V5_FORMAL_RUN_ROOT" \
    --split-plan "$POSTERIOR_V8_V5_SPLIT_PLAN" \
    --sobol-design "$POSTERIOR_V8_V5_SOBOL_DESIGN" \
    --local-sobol-schedule "$POSTERIOR_V8_V5_LOCAL_SOBOL_SCHEDULE" \
    --calibration-artifact "$POSTERIOR_V8_V5_CALIBRATION_ARTIFACT" \
    --execute
