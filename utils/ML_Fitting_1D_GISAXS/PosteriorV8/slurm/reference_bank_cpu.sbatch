#!/bin/bash
#SBATCH --partition=allcpu
#SBATCH --account=hasylab
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

set -euo pipefail

: "${POSTERIOR_V8_SOURCE_ROOT:?set POSTERIOR_V8_SOURCE_ROOT}"
: "${POSTERIOR_V8_CURVE_INPUT:?set POSTERIOR_V8_CURVE_INPUT}"
: "${POSTERIOR_V8_REFERENCE_OUTPUT:?set POSTERIOR_V8_REFERENCE_OUTPUT}"

cd "$POSTERIOR_V8_SOURCE_ROOT"
export PYTHONPATH="$POSTERIOR_V8_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

arguments=(
    --input "$POSTERIOR_V8_CURVE_INPUT"
    --output "$POSTERIOR_V8_REFERENCE_OUTPUT"
    --curve-id "${POSTERIOR_V8_CURVE_ID:-reference_bank_curve}"
    --source-kind "${POSTERIOR_V8_SOURCE_KIND:-synthetic}"
    --seed "${POSTERIOR_V8_SEED:-20260902}"
    --starts-per-branch "${POSTERIOR_V8_STARTS_PER_BRANCH:-4}"
    --max-nfev "${POSTERIOR_V8_MAX_NFEV:-120}"
    --minimum-components "${POSTERIOR_V8_MINIMUM_COMPONENTS:-1}"
    --maximum-components "${POSTERIOR_V8_MAXIMUM_COMPONENTS:-2}"
    --raw-threshold "${POSTERIOR_V8_RAW_THRESHOLD:-0.05}"
)

if [[ -n "${POSTERIOR_V8_CALIBRATION:-}" ]]; then
    arguments+=(
        --calibration "$POSTERIOR_V8_CALIBRATION"
        --noise-id "${POSTERIOR_V8_NOISE_ID:?set POSTERIOR_V8_NOISE_ID}"
        --q-window-id "${POSTERIOR_V8_Q_WINDOW_ID:?set POSTERIOR_V8_Q_WINDOW_ID}"
        --calibration-point-count \
            "${POSTERIOR_V8_CALIBRATION_POINT_COUNT:?set POSTERIOR_V8_CALIBRATION_POINT_COUNT}"
    )
elif [[ -n "${POSTERIOR_V8_STANDARDIZED_THRESHOLD:-}" ]]; then
    arguments+=(--standardized-threshold "$POSTERIOR_V8_STANDARDIZED_THRESHOLD")
fi

conda run --no-capture-output -n tf \
    python -m utils.ML_Fitting_1D_GISAXS.PosteriorV8.competing_model_search \
    "${arguments[@]}"
