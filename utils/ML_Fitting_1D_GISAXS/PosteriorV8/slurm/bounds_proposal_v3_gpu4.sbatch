#!/bin/bash
#SBATCH --partition=allgpu
#SBATCH --account=hasylab
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --constraint=GPUx4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --signal=B:TERM@180
#SBATCH --output=/data/dust/user/zhaiyufe/MaxwellRuns/GISAXS_POSTERIOR_V8_20260902/logs/v3-bounds-train-%j.out
#SBATCH --error=/data/dust/user/zhaiyufe/MaxwellRuns/GISAXS_POSTERIOR_V8_20260902/logs/v3-bounds-train-%j.err

set -euo pipefail

: "${SLURM_JOB_ID:?submit V3 training through Slurm; do not run it on max-wgs}"
: "${POSTERIOR_V8_SOURCE_ROOT:?set POSTERIOR_V8_SOURCE_ROOT}"
: "${POSTERIOR_V8_V4_DATASET_DIR:?set POSTERIOR_V8_V4_DATASET_DIR}"
: "${POSTERIOR_V8_V3_TRAINING_OUTPUT:?set POSTERIOR_V8_V3_TRAINING_OUTPUT}"

case "$(hostname -s)" in
    max-wgs*) echo "V3 training must run on a Slurm worker" >&2; exit 2 ;;
esac

case "$POSTERIOR_V8_V4_DATASET_DIR" in
    /data/dust/user/zhaiyufe/*) ;;
    *) echo "POSTERIOR_V8_V4_DATASET_DIR must be under /data/dust/user/zhaiyufe/" >&2; exit 2 ;;
esac
case "$POSTERIOR_V8_V3_TRAINING_OUTPUT" in
    /data/dust/user/zhaiyufe/*) ;;
    *) echo "POSTERIOR_V8_V3_TRAINING_OUTPUT must be under /data/dust/user/zhaiyufe/" >&2; exit 2 ;;
esac

cd "$POSTERIOR_V8_SOURCE_ROOT"
export PYTHONPATH="$POSTERIOR_V8_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

module purge
source /etc/profile.d/modules.sh
module load maxwell cuda/12.3

shopt -s nullglob
shards=("$POSTERIOR_V8_V4_DATASET_DIR"/*.npz)
if (( ${#shards[@]} == 0 )); then
    echo "no V4 bounds-first shards found in $POSTERIOR_V8_V4_DATASET_DIR" >&2
    exit 2
fi

arguments=(
    --shards "${shards[@]}"
    --output-dir "$POSTERIOR_V8_V3_TRAINING_OUTPUT"
    --epochs "${POSTERIOR_V8_EPOCHS:-50}"
    --global-batch-size "${POSTERIOR_V8_GLOBAL_BATCH_SIZE:-256}"
    --learning-rate "${POSTERIOR_V8_LEARNING_RATE:-0.0001}"
    --seed "${POSTERIOR_V8_SEED:-20260903}"
    --max-points "${POSTERIOR_V8_MAX_POINTS:-1000}"
    --width "${POSTERIOR_V8_WIDTH:-128}"
    --encoder-blocks "${POSTERIOR_V8_ENCODER_BLOCKS:-6}"
    --mixture-components "${POSTERIOR_V8_MIXTURE_COMPONENTS:-12}"
    --shuffle-buffer "${POSTERIOR_V8_SHUFFLE_BUFFER:-8192}"
    --gradient-clip-norm "${POSTERIOR_V8_GRADIENT_CLIP_NORM:-10.0}"
    --checkpoint-keep "${POSTERIOR_V8_CHECKPOINT_KEEP:-2}"
)

if [[ "${POSTERIOR_V8_MIXED_PRECISION:-0}" == "1" ]]; then
    arguments+=(--mixed-precision)
fi
if [[ -n "${POSTERIOR_V8_STEPS_PER_EPOCH:-}" ]]; then
    arguments+=(--steps-per-epoch "$POSTERIOR_V8_STEPS_PER_EPOCH")
fi
if [[ "${POSTERIOR_V8_RESUME:-0}" == "1" ]]; then
    arguments+=(--resume)
fi

srun nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
conda run --no-capture-output -n tf \
    python -m utils.ML_Fitting_1D_GISAXS.PosteriorV8.train_proposal_v3 \
    "${arguments[@]}"
