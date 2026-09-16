#!/bin/bash
#SBATCH --partition=allcpu
#SBATCH --account=hasylab
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/data/dust/user/zhaiyufe/MaxwellRuns/k1-balanced-full-search-plan-author-v6-unbound-%j.out
#SBATCH --error=/data/dust/user/zhaiyufe/MaxwellRuns/k1-balanced-full-search-plan-author-v6-unbound-%j.err

set -euo pipefail

: "${SLURM_JOB_ID:?all-K1 full-search plan authoring must run through Slurm}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "all-K1 full-search plan authoring cannot be an array task" >&2
    exit 2
fi
case "$(hostname -s)" in
    max-wgs*|max-fs-display*)
        echo "all-K1 full-search plan authoring cannot run on a login host" >&2
        exit 2
        ;;
esac

required_variables=(
    POSTERIOR_V8_SOURCE_ROOT
    POSTERIOR_V8_SOURCE_ARCHIVE
    POSTERIOR_V8_EXPECTED_SOURCE_ARCHIVE_SHA256
    POSTERIOR_V8_BALANCED_DATASET_LAUNCH_PLAN
    POSTERIOR_V8_BALANCED_DATASET_LAUNCH_PLAN_FILE_SHA256
    POSTERIOR_V8_BALANCED_DATASET_LAUNCH_PLAN_SHA256
    POSTERIOR_V8_BALANCED_DATASET_COMPLETION
    POSTERIOR_V8_BALANCED_DATASET_COMPLETION_FILE_SHA256
    POSTERIOR_V8_BALANCED_DATASET_COMPLETION_SHA256
    POSTERIOR_V8_TRAIN_TUNING_RECEIPT
    POSTERIOR_V8_TRAIN_TUNING_RECEIPT_FILE_SHA256
    POSTERIOR_V8_TRAIN_TUNING_RECEIPT_SHA256
    POSTERIOR_V8_TRAIN_TUNING_CLAIM_SHA256
    POSTERIOR_V8_PHASE_C_COMPLETION
    POSTERIOR_V8_PHASE_C_COMPLETION_FILE_SHA256
    POSTERIOR_V8_PHASE_C_COMPLETION_SHA256
    POSTERIOR_V8_THREE_WAY_RECEIPT
    POSTERIOR_V8_THREE_WAY_RECEIPT_FILE_SHA256
    POSTERIOR_V8_THREE_WAY_RECEIPT_SHA256
    POSTERIOR_V8_PHASE_C_EXCLUSION_CLAIM_SHA256
    POSTERIOR_V8_LOCAL_SOBOL_SCHEDULE
    POSTERIOR_V8_CALIBRATION
    POSTERIOR_V8_EXPECTED_CALIBRATION_ARTIFACT_SHA256
    POSTERIOR_V8_EXPECTED_CALIBRATION_FILE_SHA256
    POSTERIOR_V8_FULL_SEARCH_RUN_ROOT
    POSTERIOR_V8_FULL_SEARCH_CANDIDATE_PLAN
    POSTERIOR_V8_FULL_SEARCH_AUTHORING_COMPLETION
)
for variable_name in "${required_variables[@]}"; do
    if [[ -z "${!variable_name:-}" ]]; then
        echo "missing required environment variable: $variable_name" >&2
        exit 2
    fi
done

for dust_path in \
    "$POSTERIOR_V8_SOURCE_ROOT" \
    "$POSTERIOR_V8_SOURCE_ARCHIVE" \
    "$POSTERIOR_V8_BALANCED_DATASET_LAUNCH_PLAN" \
    "$POSTERIOR_V8_BALANCED_DATASET_COMPLETION" \
    "$POSTERIOR_V8_TRAIN_TUNING_RECEIPT" \
    "$POSTERIOR_V8_PHASE_C_COMPLETION" \
    "$POSTERIOR_V8_THREE_WAY_RECEIPT" \
    "$POSTERIOR_V8_LOCAL_SOBOL_SCHEDULE" \
    "$POSTERIOR_V8_CALIBRATION" \
    "$POSTERIOR_V8_FULL_SEARCH_RUN_ROOT" \
    "$POSTERIOR_V8_FULL_SEARCH_CANDIDATE_PLAN" \
    "$POSTERIOR_V8_FULL_SEARCH_AUTHORING_COMPLETION"; do
    case "$dust_path" in
        /data/dust/user/zhaiyufe/*) ;;
        *) echo "all authoring paths must remain under the owner dust root" >&2; exit 2 ;;
    esac
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
    utils.ML_Fitting_1D_GISAXS.PosteriorV8.author_k1_balanced_full_search_plan_v5 \
    --source-root "$POSTERIOR_V8_SOURCE_ROOT" \
    --source-archive "$POSTERIOR_V8_SOURCE_ARCHIVE" \
    --expected-source-archive-sha256 "$POSTERIOR_V8_EXPECTED_SOURCE_ARCHIVE_SHA256" \
    --balanced-dataset-launch-plan "$POSTERIOR_V8_BALANCED_DATASET_LAUNCH_PLAN" \
    --balanced-dataset-launch-plan-file-sha256 "$POSTERIOR_V8_BALANCED_DATASET_LAUNCH_PLAN_FILE_SHA256" \
    --balanced-dataset-launch-plan-sha256 "$POSTERIOR_V8_BALANCED_DATASET_LAUNCH_PLAN_SHA256" \
    --balanced-dataset-completion "$POSTERIOR_V8_BALANCED_DATASET_COMPLETION" \
    --balanced-dataset-completion-file-sha256 "$POSTERIOR_V8_BALANCED_DATASET_COMPLETION_FILE_SHA256" \
    --balanced-dataset-completion-sha256 "$POSTERIOR_V8_BALANCED_DATASET_COMPLETION_SHA256" \
    --train-tuning-receipt "$POSTERIOR_V8_TRAIN_TUNING_RECEIPT" \
    --train-tuning-receipt-file-sha256 "$POSTERIOR_V8_TRAIN_TUNING_RECEIPT_FILE_SHA256" \
    --train-tuning-receipt-sha256 "$POSTERIOR_V8_TRAIN_TUNING_RECEIPT_SHA256" \
    --train-tuning-claim-sha256 "$POSTERIOR_V8_TRAIN_TUNING_CLAIM_SHA256" \
    --phase-c-completion "$POSTERIOR_V8_PHASE_C_COMPLETION" \
    --phase-c-completion-file-sha256 "$POSTERIOR_V8_PHASE_C_COMPLETION_FILE_SHA256" \
    --phase-c-completion-sha256 "$POSTERIOR_V8_PHASE_C_COMPLETION_SHA256" \
    --three-way-receipt "$POSTERIOR_V8_THREE_WAY_RECEIPT" \
    --three-way-receipt-file-sha256 "$POSTERIOR_V8_THREE_WAY_RECEIPT_FILE_SHA256" \
    --three-way-receipt-sha256 "$POSTERIOR_V8_THREE_WAY_RECEIPT_SHA256" \
    --phase-c-exclusion-claim-sha256 "$POSTERIOR_V8_PHASE_C_EXCLUSION_CLAIM_SHA256" \
    --local-sobol-schedule "$POSTERIOR_V8_LOCAL_SOBOL_SCHEDULE" \
    --calibration "$POSTERIOR_V8_CALIBRATION" \
    --expected-calibration-artifact-sha256 "$POSTERIOR_V8_EXPECTED_CALIBRATION_ARTIFACT_SHA256" \
    --expected-calibration-file-sha256 "$POSTERIOR_V8_EXPECTED_CALIBRATION_FILE_SHA256" \
    --search-run-root "$POSTERIOR_V8_FULL_SEARCH_RUN_ROOT" \
    --candidate-plan "$POSTERIOR_V8_FULL_SEARCH_CANDIDATE_PLAN" \
    --completion "$POSTERIOR_V8_FULL_SEARCH_AUTHORING_COMPLETION"
