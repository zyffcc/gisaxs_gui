#!/bin/bash
#SBATCH --partition=allcpu
#SBATCH --account=hasylab
#SBATCH --array=0-59
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/data/dust/user/zhaiyufe/MaxwellRuns/k1-balanced-full-search-v3-unbound-%A_%a.out
#SBATCH --error=/data/dust/user/zhaiyufe/MaxwellRuns/k1-balanced-full-search-v3-unbound-%A_%a.err

set -euo pipefail

: "${SLURM_JOB_ID:?balanced all-K1 full search must run through Slurm}"
: "${SLURM_ARRAY_TASK_ID:?balanced all-K1 full search requires an array task}"
: "${POSTERIOR_V8_SOURCE_ROOT:?set POSTERIOR_V8_SOURCE_ROOT}"
: "${POSTERIOR_V8_SOURCE_ARCHIVE:?set POSTERIOR_V8_SOURCE_ARCHIVE}"
: "${POSTERIOR_V8_K1_FULL_SEARCH_PLAN:?set the immutable full-search plan path}"
: "${POSTERIOR_V8_K1_FULL_SEARCH_PLAN_SHA256:?set the plan self SHA-256}"
: "${POSTERIOR_V8_K1_FULL_SEARCH_PLAN_FILE_SHA256:?set the plan file SHA-256}"
: "${POSTERIOR_V8_K1_FULL_SEARCH_LOCAL_SOBOL_SCHEDULE:?set the local Sobol schedule}"
: "${POSTERIOR_V8_K1_FULL_SEARCH_CALIBRATION:?set the checked calibration}"

case "$(hostname -s)" in
    max-wgs*|max-fs-display*)
        echo "balanced all-K1 full search cannot run on a login host" >&2
        exit 2
        ;;
esac
case "$SLURM_ARRAY_TASK_ID" in
    ''|*[!0-9]*) echo "balanced full-search array task ID must be unsigned" >&2; exit 2 ;;
esac
if (( SLURM_ARRAY_TASK_ID > 59 )); then
    echo "balanced full-search array task ID exceeds the frozen 60-task plan" >&2
    exit 2
fi
for digest_value in \
    "$POSTERIOR_V8_K1_FULL_SEARCH_PLAN_SHA256" \
    "$POSTERIOR_V8_K1_FULL_SEARCH_PLAN_FILE_SHA256"; do
    if [[ ! "$digest_value" =~ ^[0-9a-f]{64}$ ]]; then
        echo "balanced full-search SHA-256 is malformed" >&2
        exit 2
    fi
done
for dust_path in \
    "$POSTERIOR_V8_SOURCE_ROOT" \
    "$POSTERIOR_V8_SOURCE_ARCHIVE" \
    "$POSTERIOR_V8_K1_FULL_SEARCH_PLAN" \
    "$POSTERIOR_V8_K1_FULL_SEARCH_LOCAL_SOBOL_SCHEDULE" \
    "$POSTERIOR_V8_K1_FULL_SEARCH_CALIBRATION"; do
    case "$dust_path" in
        /data/dust/user/zhaiyufe/*) ;;
        *) echo "balanced full-search inputs must remain under the owner dust root" >&2; exit 2 ;;
    esac
done
if [[ ! -d "$POSTERIOR_V8_SOURCE_ROOT" || -L "$POSTERIOR_V8_SOURCE_ROOT" ]]; then
    echo "balanced full-search source root must be a real directory" >&2
    exit 2
fi
for sealed_file in \
    "$POSTERIOR_V8_SOURCE_ARCHIVE" \
    "$POSTERIOR_V8_K1_FULL_SEARCH_PLAN" \
    "$POSTERIOR_V8_K1_FULL_SEARCH_LOCAL_SOBOL_SCHEDULE" \
    "$POSTERIOR_V8_K1_FULL_SEARCH_CALIBRATION"; do
    if [[ ! -f "$sealed_file" \
        || -L "$sealed_file" \
        || "$(stat -c '%a' -- "$sealed_file")" != "400" \
        || "$(stat -c '%h' -- "$sealed_file")" != "1" ]]; then
        echo "balanced full-search file inputs must be 0400/nlink1 regular files" >&2
        exit 2
    fi
done

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

GISAXS_V5_ENV_PYTHON="/data/dust/user/zhaiyufe/conda/envs/gisaxs-v5-r2/bin/python"
if [[ ! -x "$GISAXS_V5_ENV_PYTHON" ]]; then
    echo "the dedicated GISAXS V5 environment is unavailable" >&2
    exit 2
fi

cd "$POSTERIOR_V8_SOURCE_ROOT"
export PYTHONPATH="$POSTERIOR_V8_SOURCE_ROOT"
exec "$GISAXS_V5_ENV_PYTHON" -m \
    utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_balanced_full_search_worker_v5 \
    --plan "$POSTERIOR_V8_K1_FULL_SEARCH_PLAN" \
    --array-task-id "$SLURM_ARRAY_TASK_ID" \
    --expected-plan-sha256 "$POSTERIOR_V8_K1_FULL_SEARCH_PLAN_SHA256" \
    --expected-plan-file-sha256 "$POSTERIOR_V8_K1_FULL_SEARCH_PLAN_FILE_SHA256" \
    --source-root "$POSTERIOR_V8_SOURCE_ROOT" \
    --source-archive "$POSTERIOR_V8_SOURCE_ARCHIVE" \
    --local-sobol-schedule "$POSTERIOR_V8_K1_FULL_SEARCH_LOCAL_SOBOL_SCHEDULE" \
    --calibration "$POSTERIOR_V8_K1_FULL_SEARCH_CALIBRATION"
