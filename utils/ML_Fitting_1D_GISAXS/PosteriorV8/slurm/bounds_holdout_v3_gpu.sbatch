#!/bin/bash
#SBATCH --partition=allgpu
#SBATCH --account=hasylab
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --constraint=GPUx1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/data/dust/user/zhaiyufe/MaxwellRuns/GISAXS_POSTERIOR_V8_20260902/logs/v3-bounds-holdout-%j.out
#SBATCH --error=/data/dust/user/zhaiyufe/MaxwellRuns/GISAXS_POSTERIOR_V8_20260902/logs/v3-bounds-holdout-%j.err

set -euo pipefail

: "${SLURM_JOB_ID:?submit this audit through Slurm; do not run it on max-wgs}"
: "${POSTERIOR_V8_SOURCE_ROOT:?set POSTERIOR_V8_SOURCE_ROOT}"
: "${POSTERIOR_V8_V4_DATASET_DIR:?set POSTERIOR_V8_V4_DATASET_DIR}"
: "${POSTERIOR_V8_V3_TRAINING_RUN:?set POSTERIOR_V8_V3_TRAINING_RUN}"
: "${POSTERIOR_V8_V3_HOLDOUT_OUTPUT:?set POSTERIOR_V8_V3_HOLDOUT_OUTPUT}"

case "$(hostname -s)" in
    max-wgs*) echo "V3 holdout audit must run on a Slurm worker" >&2; exit 2 ;;
esac
for path in \
    "$POSTERIOR_V8_V4_DATASET_DIR" \
    "$POSTERIOR_V8_V3_TRAINING_RUN" \
    "$POSTERIOR_V8_V3_HOLDOUT_OUTPUT"
do
    case "$path" in
        /data/dust/user/zhaiyufe/*) ;;
        *) echo "datasets, models, and audits must be under /data/dust/user/zhaiyufe/" >&2; exit 2 ;;
    esac
done

cd "$POSTERIOR_V8_SOURCE_ROOT"
export PYTHONPATH="$POSTERIOR_V8_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

module purge
source /etc/profile.d/modules.sh
module load maxwell cuda/12.3

shopt -s nullglob
shards=("$POSTERIOR_V8_V4_DATASET_DIR"/*.npz)
if (( ${#shards[@]} == 0 )); then
    echo "no immutable V4 bounds-first shards found" >&2
    exit 2
fi

arguments=(
    --shards "${shards[@]}"
    --training-run "$POSTERIOR_V8_V3_TRAINING_RUN"
    --output "$POSTERIOR_V8_V3_HOLDOUT_OUTPUT"
    --split "${POSTERIOR_V8_HOLDOUT_SPLIT:-test}"
    --range-selection "${POSTERIOR_V8_HOLDOUT_RANGE:-all}"
    --maximum-examples "${POSTERIOR_V8_HOLDOUT_EXAMPLES:-512}"
    --batch-size "${POSTERIOR_V8_HOLDOUT_BATCH_SIZE:-64}"
    --topology-ks "${POSTERIOR_V8_HOLDOUT_TOPOLOGY_KS:-1,3,8}"
    --canonical-joint-ks "${POSTERIOR_V8_HOLDOUT_JOINT_KS:-1,8,32}"
    --oracle-mixture-limit "${POSTERIOR_V8_HOLDOUT_MIXTURES:-12}"
    --oracle-samples-per-mixture "${POSTERIOR_V8_HOLDOUT_SAMPLES_PER_MIXTURE:-2}"
    --oracle-best-of-n "${POSTERIOR_V8_HOLDOUT_BEST_OF_N:-1,4,12,24}"
    --exact-maximum-examples "${POSTERIOR_V8_HOLDOUT_EXACT_EXAMPLES:-8}"
    --target-parameter-mode-count "${POSTERIOR_V8_HOLDOUT_TARGET_MODES:-1}"
    --forward-evaluation-limit "${POSTERIOR_V8_HOLDOUT_FORWARD_BUDGET:-1024}"
    --per-candidate-forward-limit "${POSTERIOR_V8_HOLDOUT_PER_CANDIDATE_BUDGET:-128}"
    --raw-exact-log-rmse-max "${POSTERIOR_V8_HOLDOUT_RAW_LOG_RMSE_MAX:-0.10}"
    --standardized-exact-log-rmse-max "${POSTERIOR_V8_HOLDOUT_STANDARDIZED_LOG_RMSE_MAX:-3.0}"
    --parameter-mode-distance-max "${POSTERIOR_V8_HOLDOUT_PARAMETER_MODE_DISTANCE_MAX:-0.08}"
    --curve-equivalence-log-rmse-max "${POSTERIOR_V8_HOLDOUT_CURVE_EQUIVALENCE_LOG_RMSE_MAX:-0.02}"
    --reference-mode-distance-max "${POSTERIOR_V8_HOLDOUT_REFERENCE_MODE_DISTANCE_MAX:-0.10}"
    --seed "${POSTERIOR_V8_HOLDOUT_SEED:-20260903}"
)

conda run --no-capture-output -n tf \
    python -m utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_holdout_audit \
    "${arguments[@]}"
