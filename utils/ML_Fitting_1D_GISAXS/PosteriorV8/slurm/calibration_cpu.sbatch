#!/bin/bash
#SBATCH --partition=allcpu
#SBATCH --account=hasylab
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

: "${POSTERIOR_V8_SOURCE_ROOT:?set POSTERIOR_V8_SOURCE_ROOT}"
: "${POSTERIOR_V8_DATASET_DIR:?set POSTERIOR_V8_DATASET_DIR}"
: "${POSTERIOR_V8_CALIBRATION_OUTPUT:?set POSTERIOR_V8_CALIBRATION_OUTPUT}"

cd "$POSTERIOR_V8_SOURCE_ROOT"
export PYTHONPATH="$POSTERIOR_V8_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

shopt -s nullglob
shards=("$POSTERIOR_V8_DATASET_DIR"/*.npz)
if (( ${#shards[@]} == 0 )); then
    echo "no dataset shards found in $POSTERIOR_V8_DATASET_DIR" >&2
    exit 2
fi
shard_args=()
for shard in "${shards[@]}"; do
    shard_args+=(--shard "$shard")
done

conda run --no-capture-output -n tf \
    python -m utils.ML_Fitting_1D_GISAXS.PosteriorV8.calibrate_dataset \
    "${shard_args[@]}" \
    --output "$POSTERIOR_V8_CALIBRATION_OUTPUT" \
    --target-coverage "${POSTERIOR_V8_TARGET_COVERAGE:-0.95}" \
    --minimum-samples-per-stratum "${POSTERIOR_V8_MINIMUM_CALIBRATION_GROUPS:-20}"
