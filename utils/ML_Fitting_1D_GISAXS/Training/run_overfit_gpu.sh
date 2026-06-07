#!/bin/bash
#SBATCH --partition=allgpu
#SBATCH --time=07:00:00
#SBATCH --nodes=1
#SBATCH --constraint="GPUx1"
#SBATCH --chdir=/home/zhaiyufe/PycharmProjects/ML_Fitting_1D_GISAXS
#SBATCH --job-name=ml1d_overfitting_training
#SBATCH --output=./Training/HCPoutput/ml1d_training_overfitting-%N-%j.out
#SBATCH --error=./Training/HCPoutput/ml1d_training_overfitting-%N-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=yufeng.zhai@desy.de

cd /home/zhaiyufe/PycharmProjects/ML_Fitting_1D_GISAXS
mkdir -p ./Training/HCPoutput

module purge
source /etc/profile.d/modules.sh
module load maxwell cuda/12.3
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=0
export TF_FORCE_GPU_ALLOW_GROWTH=true

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
srun which nvidia-smi
srun nvidia-smi

TRAIN_DIR=${TRAIN_DIR:-/data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS_K1/train}
OUTPUT_DIR=${OUTPUT_DIR:-/home/zhaiyufe/PycharmProjects/ML_Fitting_1D_GISAXS/Training/overfit_debug_simple_loss}
NUM_SAMPLES=${NUM_SAMPLES:-64}
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-8}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
PRINT_EVERY=${PRINT_EVERY:-5}
NOISE_FRAC=${NOISE_FRAC:-0.01}
DEBUG_MAX_POINTS=${DEBUG_MAX_POINTS:-256}
PLOT_N=${PLOT_N:-16}
LEARNING_RATE=${LEARNING_RATE:-0.0001}
SKIP_MODEL_SAVE=${SKIP_MODEL_SAVE:-1}
USE_TRUE_GLOBAL_FOR_PLOT=${USE_TRUE_GLOBAL_FOR_PLOT:-1}

echo "Overfit debug parameters:"
echo "TRAIN_DIR=${TRAIN_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "NUM_SAMPLES=${NUM_SAMPLES}"
echo "EPOCHS=${EPOCHS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo "PRINT_EVERY=${PRINT_EVERY}"
echo "NOISE_FRAC=${NOISE_FRAC}"
echo "DEBUG_MAX_POINTS=${DEBUG_MAX_POINTS}"
echo "PLOT_N=${PLOT_N}"
echo "LEARNING_RATE=${LEARNING_RATE}"
echo "SKIP_MODEL_SAVE=${SKIP_MODEL_SAVE}"
echo "USE_TRUE_GLOBAL_FOR_PLOT=${USE_TRUE_GLOBAL_FOR_PLOT}"

CMD=(
  /data/dust/user/zhaiyufe/conda/envs/tf/bin/python -u Training/overfit_debug.py
  --train_dir "${TRAIN_DIR}"
  --num_samples "${NUM_SAMPLES}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --eval_batch_size "${EVAL_BATCH_SIZE}"
  --print_every "${PRINT_EVERY}"
  --noise_frac "${NOISE_FRAC}"
  --debug_max_points "${DEBUG_MAX_POINTS}"
  --simple_k1_loss
  --plot_n "${PLOT_N}"
  --learning_rate "${LEARNING_RATE}"
  --output_dir "${OUTPUT_DIR}"
)

if [[ "${SKIP_MODEL_SAVE}" != "0" ]]; then
  CMD+=(--skip_model_save)
fi

if [[ "${USE_TRUE_GLOBAL_FOR_PLOT}" != "0" ]]; then
  CMD+=(--use_true_global_for_plot)
fi

"${CMD[@]}"
