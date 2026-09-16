#!/bin/bash
#SBATCH --partition=allcpu
#SBATCH --account=hasylab
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/data/dust/user/zhaiyufe/MaxwellRuns/k1-balanced-all12-v4-collect-unbound-%j.out
#SBATCH --error=/data/dust/user/zhaiyufe/MaxwellRuns/k1-balanced-all12-v4-collect-unbound-%j.err

set -euo pipefail

: "${SLURM_JOB_ID:?balanced K1 dataset collection must run through Slurm}"
: "${POSTERIOR_V8_SOURCE_ROOT:?set POSTERIOR_V8_SOURCE_ROOT}"
: "${POSTERIOR_V8_K1_DATASET_LAUNCH_PLAN:?set the immutable launch plan path}"
: "${POSTERIOR_V8_K1_DATASET_LAUNCH_PLAN_SHA256:?set the launch plan self SHA-256}"

case "$(hostname -s)" in
    max-wgs*|max-fs-display*) echo "balanced K1 dataset collection cannot run on login nodes" >&2; exit 2 ;;
esac
for dust_path in \
    "$POSTERIOR_V8_SOURCE_ROOT" \
    "$POSTERIOR_V8_K1_DATASET_LAUNCH_PLAN"; do
    case "$dust_path" in
        /data/dust/user/zhaiyufe/*) ;;
        *) echo "balanced K1 collection inputs must remain under the owner dust root" >&2; exit 2 ;;
    esac
done
if [[ ! -d "$POSTERIOR_V8_SOURCE_ROOT" || -L "$POSTERIOR_V8_SOURCE_ROOT" ]]; then
    echo "balanced K1 source root must be a real directory" >&2
    exit 2
fi
if [[ ! -f "$POSTERIOR_V8_K1_DATASET_LAUNCH_PLAN" \
    || -L "$POSTERIOR_V8_K1_DATASET_LAUNCH_PLAN" \
    || "$(stat -c '%a' -- "$POSTERIOR_V8_K1_DATASET_LAUNCH_PLAN")" != "400" \
    || "$(stat -c '%h' -- "$POSTERIOR_V8_K1_DATASET_LAUNCH_PLAN")" != "1" ]]; then
    echo "balanced K1 launch plan must be a 0400/nlink1 regular file" >&2
    exit 2
fi
if [[ ! "$POSTERIOR_V8_K1_DATASET_LAUNCH_PLAN_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "balanced K1 launch plan SHA-256 is malformed" >&2
    exit 2
fi

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
    utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_balanced_dataset_collector_v5 \
    --plan "$POSTERIOR_V8_K1_DATASET_LAUNCH_PLAN" \
    --expected-plan-sha256 "$POSTERIOR_V8_K1_DATASET_LAUNCH_PLAN_SHA256"
