#!/bin/bash
#SBATCH --partition=allgpu
#SBATCH --account=hasylab
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --constraint=GPUx4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --signal=B:TERM@180
#SBATCH --output=/data/dust/user/zhaiyufe/MaxwellRuns/GISAXS_ONE_CLICK_20260902/logs/v5-grouped-train-%j.out
#SBATCH --error=/data/dust/user/zhaiyufe/MaxwellRuns/GISAXS_ONE_CLICK_20260902/logs/v5-grouped-train-%j.err

set -euo pipefail

: "${SLURM_JOB_ID:?submit V5.1 training through Slurm; do not run it on max-wgs}"
: "${POSTERIOR_V8_SOURCE_ROOT:?set POSTERIOR_V8_SOURCE_ROOT}"
: "${POSTERIOR_V8_V5_TRAIN_DATASETS:?set colon-separated POSTERIOR_V8_V5_TRAIN_DATASETS}"
: "${POSTERIOR_V8_V5_VALIDATION_DATASETS:?set colon-separated POSTERIOR_V8_V5_VALIDATION_DATASETS}"
: "${POSTERIOR_V8_V5_GROUPED_OUTPUT:?set POSTERIOR_V8_V5_GROUPED_OUTPUT}"

case "$(hostname -s)" in
    max-wgs*) echo "V5.1 training must run on a Slurm worker" >&2; exit 2 ;;
esac

case "$POSTERIOR_V8_V5_GROUPED_OUTPUT" in
    /data/dust/user/zhaiyufe/*) ;;
    *) echo "POSTERIOR_V8_V5_GROUPED_OUTPUT must be under /data/dust/user/zhaiyufe/" >&2; exit 2 ;;
esac
if [[ -e "$POSTERIOR_V8_V5_GROUPED_OUTPUT" ]]; then
    echo "refusing to overwrite existing V5.1 training output" >&2
    exit 2
fi

mkdir -p "$(dirname "$POSTERIOR_V8_V5_GROUPED_OUTPUT")"
cd "$POSTERIOR_V8_SOURCE_ROOT"
export PYTHONPATH="$POSTERIOR_V8_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

module purge
source /etc/profile.d/modules.sh
module load maxwell cuda/12.3

IFS=: read -r -a train_datasets <<< "$POSTERIOR_V8_V5_TRAIN_DATASETS"
IFS=: read -r -a validation_datasets <<< "$POSTERIOR_V8_V5_VALIDATION_DATASETS"
if (( ${#train_datasets[@]} == 0 || ${#validation_datasets[@]} == 0 )); then
    echo "train and validation artifact lists must both be non-empty" >&2
    exit 2
fi
for dataset_file in "${train_datasets[@]}" "${validation_datasets[@]}"; do
    case "$dataset_file" in
        /data/dust/user/zhaiyufe/*) ;;
        *) echo "all V5.1 grouped artifacts must be under /data/dust/user/zhaiyufe/" >&2; exit 2 ;;
    esac
    if [[ ! -f "$dataset_file" ]]; then
        echo "checked V5.1 grouped artifact does not exist: $dataset_file" >&2
        exit 2
    fi
done

train_sidecars=()
validation_sidecars=()
if [[ -n "${POSTERIOR_V8_V5_TRAIN_SIDECARS:-}" || -n "${POSTERIOR_V8_V5_VALIDATION_SIDECARS:-}" ]]; then
    if [[ -z "${POSTERIOR_V8_V5_TRAIN_SIDECARS:-}" || -z "${POSTERIOR_V8_V5_VALIDATION_SIDECARS:-}" ]]; then
        echo "train and validation search-sidecar subsets must both be supplied" >&2
        exit 2
    fi
    IFS=: read -r -a train_sidecars <<< "$POSTERIOR_V8_V5_TRAIN_SIDECARS"
    IFS=: read -r -a validation_sidecars <<< "$POSTERIOR_V8_V5_VALIDATION_SIDECARS"
fi
if [[ "${POSTERIOR_V8_FULL_EPOCHS:-0}" != "0" ]] && (( ${#train_sidecars[@]} == 0 || ${#validation_sidecars[@]} == 0 )); then
    echo "full training requires checked train and validation search sidecars" >&2
    exit 2
fi
for sidecar_file in "${train_sidecars[@]}" "${validation_sidecars[@]}"; do
    case "$sidecar_file" in
        /data/dust/user/zhaiyufe/*) ;;
        *) echo "all V5.1 search sidecars must be under /data/dust/user/zhaiyufe/" >&2; exit 2 ;;
    esac
    if [[ ! -f "$sidecar_file" ]]; then
        echo "checked V5.1 search sidecar does not exist: $sidecar_file" >&2
        exit 2
    fi
done

arguments=(
    --output-dir "$POSTERIOR_V8_V5_GROUPED_OUTPUT"
    --warmup-epochs "${POSTERIOR_V8_WARMUP_EPOCHS:-10}"
    --full-epochs "${POSTERIOR_V8_FULL_EPOCHS:-0}"
    --recipes-per-replica "${POSTERIOR_V8_RECIPES_PER_REPLICA:-4}"
    --validation-recipes-per-batch "${POSTERIOR_V8_VALIDATION_RECIPES_PER_BATCH:-16}"
    --learning-rate "${POSTERIOR_V8_LEARNING_RATE:-0.0001}"
    --seed "${POSTERIOR_V8_SEED:-20260903}"
    --width "${POSTERIOR_V8_WIDTH:-128}"
    --encoder-blocks "${POSTERIOR_V8_ENCODER_BLOCKS:-6}"
    --mixture-components "${POSTERIOR_V8_MIXTURE_COMPONENTS:-12}"
    --gradient-clip-norm "${POSTERIOR_V8_GRADIENT_CLIP_NORM:-10.0}"
    --search-yield-weight "${POSTERIOR_V8_SEARCH_YIELD_WEIGHT:-1.0}"
    --pairwise-ranking-weight "${POSTERIOR_V8_PAIRWISE_RANKING_WEIGHT:-1.0}"
    --local-mdn-weight "${POSTERIOR_V8_LOCAL_MDN_WEIGHT:-1.0}"
    --local-coverage-weight "${POSTERIOR_V8_LOCAL_COVERAGE_WEIGHT:-1.0}"
    --operational-top-l-alignment-weight "${POSTERIOR_V8_OPERATIONAL_TOP_L_ALIGNMENT_WEIGHT:-1.0}"
    --local-coverage-temperature "${POSTERIOR_V8_LOCAL_COVERAGE_TEMPERATURE:-0.05}"
    --operational-hit-rms-threshold "${POSTERIOR_V8_OPERATIONAL_HIT_RMS_THRESHOLD:-0.05}"
    --operational-duplicate-rms-threshold "${POSTERIOR_V8_OPERATIONAL_DUPLICATE_RMS_THRESHOLD:-0.02}"
)

for dataset_file in "${train_datasets[@]}"; do
    arguments+=(--train-dataset "$dataset_file")
done
for dataset_file in "${validation_datasets[@]}"; do
    arguments+=(--validation-dataset "$dataset_file")
done
for sidecar_file in "${train_sidecars[@]}"; do
    arguments+=(--train-sidecar "$sidecar_file")
done
for sidecar_file in "${validation_sidecars[@]}"; do
    arguments+=(--validation-sidecar "$sidecar_file")
done

if [[ -n "${POSTERIOR_V8_STEPS_PER_EPOCH:-}" ]]; then
    arguments+=(--steps-per-epoch "$POSTERIOR_V8_STEPS_PER_EPOCH")
fi
if [[ -n "${POSTERIOR_V8_FULL_STEPS_PER_EPOCH:-}" ]]; then
    arguments+=(--full-steps-per-epoch "$POSTERIOR_V8_FULL_STEPS_PER_EPOCH")
fi
if [[ "${POSTERIOR_V8_MIXED_PRECISION:-0}" == "1" ]]; then
    arguments+=(--mixed-precision)
fi
if [[ "${POSTERIOR_V8_NON_DETERMINISTIC_OPS:-0}" == "1" ]]; then
    arguments+=(--non-deterministic-ops)
fi
if [[ "${POSTERIOR_V8_SMOKE:-0}" == "1" ]]; then
    arguments+=(--smoke)
fi
if [[ "${POSTERIOR_V8_DRY_RUN:-0}" == "1" ]]; then
    arguments+=(--dry-run)
fi

srun nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
conda run --no-capture-output -n tf \
    python -m utils.ML_Fitting_1D_GISAXS.PosteriorV8.train_grouped_v5 \
    "${arguments[@]}"
