#!/bin/bash

set -euo pipefail

module purge
source /etc/profile.d/modules.sh
unset LD_PRELOAD

NUM_SAMPLES=${1:-100000}
SAMPLES_PER_SHARD=${2:-1000}
SEED=${3:-42}
NUM_TASKS=${4:-${SLURM_ARRAY_TASK_COUNT:-1}}

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    TASK_ID=${SLURM_ARRAY_TASK_ID}
    OUTPUT_DIR=${5:-/data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS}
    K_VALUES=${6:-1,2,3,4}
    K_PROBS=${7:-}
else
    TASK_ID=${5:-0}
    OUTPUT_DIR=${6:-/data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS}
    K_VALUES=${7:-1,2,3,4}
    K_PROBS=${8:-}
fi

cleanup_assigned_tmp() {
    python - "${OUTPUT_DIR}" "${TASK_ID}" "${NUM_TASKS}" <<'PY'
import re
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
task_id = int(sys.argv[2])
num_tasks = int(sys.argv[3])
pattern = re.compile(r"^\.shard_(\d+)\.(?:tfrecord|npz)\.tmp-.+")
removed = 0

for split in ("train", "val", "test"):
    split_dir = output_dir / split
    if not split_dir.exists():
        continue
    for path in split_dir.glob(".shard_*.tmp-*"):
        match = pattern.match(path.name)
        if match is None:
            continue
        shard_idx = int(match.group(1))
        if shard_idx % num_tasks != task_id:
            continue
        try:
            path.unlink()
            removed += 1
            print(f"removed stale tmp: {path}", flush=True)
        except FileNotFoundError:
            pass

print(f"stale tmp cleanup done for task {task_id}/{num_tasks}: removed {removed}", flush=True)
PY
}

BUILD_PID=""

on_term() {
    echo "Received termination signal; cleaning assigned temporary shard files before exit." >&2
    if [[ -n "${BUILD_PID}" ]] && kill -0 "${BUILD_PID}" 2>/dev/null; then
        echo "Stopping build process ${BUILD_PID}." >&2
        kill -TERM "${BUILD_PID}" 2>/dev/null || true
        wait "${BUILD_PID}" 2>/dev/null || true
    fi
    cleanup_assigned_tmp || true
    exit 143
}

trap on_term TERM INT

echo "Build parameters:"
echo "NUM_SAMPLES=${NUM_SAMPLES}"
echo "SAMPLES_PER_SHARD=${SAMPLES_PER_SHARD}"
echo "SEED=${SEED}"
echo "TASK_ID=${TASK_ID}"
echo "NUM_TASKS=${NUM_TASKS}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "K_VALUES=${K_VALUES}"
echo "K_PROBS=${K_PROBS}"

cleanup_assigned_tmp

CMD=(
    conda run -n tf python TrainSetBuild/build_dataset.py
    --output_dir "${OUTPUT_DIR}"
    --num_samples "${NUM_SAMPLES}"
    --samples_per_shard "${SAMPLES_PER_SHARD}"
    --seed "${SEED}"
    --max_points 1000
    --format tfrecord
    --task_id "${TASK_ID}"
    --num_tasks "${NUM_TASKS}"
    --shard_log_interval 5
    --k_values "${K_VALUES}"
)

if [[ -n "${K_PROBS}" ]]; then
    CMD+=(--k_probs "${K_PROBS}")
fi

"${CMD[@]}" &
BUILD_PID=$!
wait "${BUILD_PID}"
