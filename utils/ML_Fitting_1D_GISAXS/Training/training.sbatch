#!/bin/bash
#SBATCH --partition=allgpu
#SBATCH --time=07:00:00
#SBATCH --nodes=1
#SBATCH --constraint="GPUx4"
#SBATCH --chdir=/home/zhaiyufe/PycharmProjects/ML_Fitting_1D_GISAXS
#SBATCH --job-name=ml1d_gisaxs_training
#SBATCH --output=./Training/HCPoutput/ml1d_training-%N-%j.out
#SBATCH --error=./Training/HCPoutput/ml1d_training-%N-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=yufeng.zhai@desy.de
#SBATCH --signal=B:TERM@120

EPOCHS=${1:-50}
BATCH_SIZE=${2:-64}
LEARNING_RATE=${3:-0.0001}
DATASET_DIR=${4:-/data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS}
MODEL_DIR=${5:-/data/dust/user/zhaiyufe/Models/ML_1D_Fitting_GISAXS}

shift $(( $# < 5 ? $# : 5 ))
EXTRA_TRAIN_ARGS=("$@")

echo "Training parameters:"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Dataset dir: $DATASET_DIR"
echo "Model dir: $MODEL_DIR"
echo "Extra train args: ${EXTRA_TRAIN_ARGS[*]:-}"

module purge
source /etc/profile.d/modules.sh
module load maxwell cuda/12.3
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=0

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
srun which nvidia-smi
srun nvidia-smi

unset LD_PRELOAD
source activate tf || conda activate tf

mkdir -p ./Training/HCPoutput
mkdir -p "$MODEL_DIR"

TRAIN_PID=""
on_term() {
    echo "Received termination signal; forwarding to training process." >&2
    if [[ -n "$TRAIN_PID" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        kill -TERM "$TRAIN_PID" 2>/dev/null || true
        wait "$TRAIN_PID" 2>/dev/null || true
    fi
    exit 143
}
trap on_term TERM INT

python -u Training/train.py \
    --dataset_dir "$DATASET_DIR" \
    --model_dir "$MODEL_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --max_points 1000 \
    "${EXTRA_TRAIN_ARGS[@]}" &
TRAIN_PID=$!
wait "$TRAIN_PID"
