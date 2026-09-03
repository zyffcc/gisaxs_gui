#!/bin/bash
#SBATCH --partition=allcpu
#SBATCH --account=hasylab
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/data/dust/user/zhaiyufe/MaxwellRuns/GISAXS_POSTERIOR_V8_20260902/logs/v4-audit-%j.out
#SBATCH --error=/data/dust/user/zhaiyufe/MaxwellRuns/GISAXS_POSTERIOR_V8_20260902/logs/v4-audit-%j.err

set -euo pipefail

: "${SLURM_JOB_ID:?submit this audit through Slurm; do not run it on max-wgs}"
: "${POSTERIOR_V8_SOURCE_ROOT:?set POSTERIOR_V8_SOURCE_ROOT}"
: "${POSTERIOR_V8_V4_DATASET_DIR:?set POSTERIOR_V8_V4_DATASET_DIR}"
: "${POSTERIOR_V8_V4_AUDIT_OUTPUT:?set POSTERIOR_V8_V4_AUDIT_OUTPUT}"

case "$(hostname -s)" in
    max-wgs*) echo "V4 dataset audit must run on a Slurm worker" >&2; exit 2 ;;
esac
for path in "$POSTERIOR_V8_V4_DATASET_DIR" "$POSTERIOR_V8_V4_AUDIT_OUTPUT"
do
    case "$path" in
        /data/dust/user/zhaiyufe/*) ;;
        *) echo "dataset and audit paths must be under /data/dust/user/zhaiyufe/" >&2; exit 2 ;;
    esac
done

cd "$POSTERIOR_V8_SOURCE_ROOT"
export PYTHONPATH="$POSTERIOR_V8_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

shopt -s nullglob
shards=("$POSTERIOR_V8_V4_DATASET_DIR"/*.npz)
if (( ${#shards[@]} == 0 )); then
    echo "no immutable V4 bounds-first shards found" >&2
    exit 2
fi

conda run --no-capture-output -n tf \
    python -m utils.ML_Fitting_1D_GISAXS.PosteriorV8.build_bounds_first_shards audit \
    --shards "${shards[@]}" \
    --output "$POSTERIOR_V8_V4_AUDIT_OUTPUT"
