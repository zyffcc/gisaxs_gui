#!/bin/bash
#SBATCH --partition=allgpu
#SBATCH --account=hasylab
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --constraint=GPUx1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

: "${POSTERIOR_V8_SOURCE_ROOT:?set POSTERIOR_V8_SOURCE_ROOT}"
: "${POSTERIOR_V8_DATASET_DIR:?set POSTERIOR_V8_DATASET_DIR}"
: "${POSTERIOR_V8_TRAINING_RUN:?set POSTERIOR_V8_TRAINING_RUN}"
: "${POSTERIOR_V8_HOLDOUT_OUTPUT:?set POSTERIOR_V8_HOLDOUT_OUTPUT}"

cd "$POSTERIOR_V8_SOURCE_ROOT"
export PYTHONPATH="$POSTERIOR_V8_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

module purge
source /etc/profile.d/modules.sh
module load maxwell cuda/12.3

shopt -s nullglob
shards=("$POSTERIOR_V8_DATASET_DIR"/*.npz)
if (( ${#shards[@]} == 0 )); then
    echo "no dataset shards found in $POSTERIOR_V8_DATASET_DIR" >&2
    exit 2
fi

conda run --no-capture-output -n tf \
    python -m utils.ML_Fitting_1D_GISAXS.PosteriorV8.proposal_holdout_audit \
    --shards "${shards[@]}" \
    --training-run "$POSTERIOR_V8_TRAINING_RUN" \
    --output "$POSTERIOR_V8_HOLDOUT_OUTPUT" \
    --range-selection "${POSTERIOR_V8_HOLDOUT_RANGE:-full}" \
    --maximum-examples "${POSTERIOR_V8_HOLDOUT_EXAMPLES:-512}" \
    --batch-size "${POSTERIOR_V8_HOLDOUT_BATCH_SIZE:-64}" \
    --topology-recall-k "${POSTERIOR_V8_HOLDOUT_TOPOLOGY_K:-8}" \
    --joint-branch-recall-k "${POSTERIOR_V8_HOLDOUT_BRANCH_K:-32}" \
    --mixture-components "${POSTERIOR_V8_HOLDOUT_MIXTURES:-12}" \
    --samples-per-mixture "${POSTERIOR_V8_HOLDOUT_SAMPLES_PER_MIXTURE:-2}" \
    --seed "${POSTERIOR_V8_HOLDOUT_SEED:-20260902}"
