#!/bin/bash
#SBATCH --partition=allgpu
#SBATCH --account=hasylab
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --constraint=GPUx4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --signal=B:TERM@180

set -euo pipefail

: "${POSTERIOR_V8_SOURCE_ROOT:?set POSTERIOR_V8_SOURCE_ROOT}"
: "${POSTERIOR_V8_DATASET_DIR:?set POSTERIOR_V8_DATASET_DIR}"
: "${POSTERIOR_V8_TRAINING_OUTPUT:?set POSTERIOR_V8_TRAINING_OUTPUT}"

cd "$POSTERIOR_V8_SOURCE_ROOT"
export PYTHONPATH="$POSTERIOR_V8_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

module purge
source /etc/profile.d/modules.sh
module load maxwell cuda/12.3

shopt -s nullglob
shards=("$POSTERIOR_V8_DATASET_DIR"/*.npz)
if (( ${#shards[@]} == 0 )); then
    echo "no dataset shards found in $POSTERIOR_V8_DATASET_DIR" >&2
    exit 2
fi

arguments=(
    --shards "${shards[@]}"
    --output-dir "$POSTERIOR_V8_TRAINING_OUTPUT"
    --epochs "${POSTERIOR_V8_EPOCHS:-50}"
    --global-batch-size "${POSTERIOR_V8_GLOBAL_BATCH_SIZE:-256}"
    --learning-rate "${POSTERIOR_V8_LEARNING_RATE:-0.0001}"
    --seed "${POSTERIOR_V8_SEED:-20260902}"
    --max-points "${POSTERIOR_V8_MAX_POINTS:-1000}"
    --width "${POSTERIOR_V8_WIDTH:-128}"
    --encoder-blocks "${POSTERIOR_V8_ENCODER_BLOCKS:-6}"
    --mixture-components "${POSTERIOR_V8_MIXTURE_COMPONENTS:-12}"
    --shuffle-buffer "${POSTERIOR_V8_SHUFFLE_BUFFER:-8192}"
    --gradient-clip-norm "${POSTERIOR_V8_GRADIENT_CLIP_NORM:-10.0}"
    --checkpoint-keep "${POSTERIOR_V8_CHECKPOINT_KEEP:-2}"
)

if [[ "${POSTERIOR_V8_MIXED_PRECISION:-1}" == "1" ]]; then
    arguments+=(--mixed-precision)
fi
if [[ -n "${POSTERIOR_V8_STEPS_PER_EPOCH:-}" ]]; then
    arguments+=(--steps-per-epoch "$POSTERIOR_V8_STEPS_PER_EPOCH")
fi
if [[ -n "${POSTERIOR_V8_VALIDATION_STEPS:-}" ]]; then
    arguments+=(--validation-steps "$POSTERIOR_V8_VALIDATION_STEPS")
fi
if [[ "${POSTERIOR_V8_RESUME:-0}" == "1" ]]; then
    arguments+=(--resume)
fi

srun nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
conda run --no-capture-output -n tf \
    python -m utils.ML_Fitting_1D_GISAXS.PosteriorV8.train_proposal \
    "${arguments[@]}"
