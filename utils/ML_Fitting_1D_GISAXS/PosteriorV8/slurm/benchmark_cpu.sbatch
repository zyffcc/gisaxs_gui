#!/bin/bash
#SBATCH --partition=allcpu
#SBATCH --account=hasylab
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

set -euo pipefail

: "${POSTERIOR_V8_SOURCE_ROOT:?set POSTERIOR_V8_SOURCE_ROOT}"
: "${POSTERIOR_V8_OUTPUT_DIR:?set POSTERIOR_V8_OUTPUT_DIR}"

cd "$POSTERIOR_V8_SOURCE_ROOT"
export PYTHONPATH="$POSTERIOR_V8_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mode="${POSTERIOR_V8_MODE:-oracle_k1}"
noise_mode="${POSTERIOR_V8_NOISE_MODE:-default}"
curves="${POSTERIOR_V8_CURVES:-60}"
starts="${POSTERIOR_V8_STARTS_PER_BRANCH:-32}"
max_nfev="${POSTERIOR_V8_MAX_NFEV:-100}"
amplitude_polish_max_nfev="${POSTERIOR_V8_AMPLITUDE_POLISH_MAX_NFEV:-80}"
points="${POSTERIOR_V8_POINTS:-256}"
seed="${POSTERIOR_V8_SEED:-20260902}"
workers="${POSTERIOR_V8_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"

conda run --no-capture-output -n tf \
    python -m utils.ML_Fitting_1D_GISAXS.PosteriorV8.solver_benchmark \
    --mode "$mode" \
    --noise-mode "$noise_mode" \
    --curves "$curves" \
    --starts-per-branch "$starts" \
    --max-nfev "$max_nfev" \
    --amplitude-polish-max-nfev "$amplitude_polish_max_nfev" \
    --points "$points" \
    --seed "$seed" \
    --workers "$workers" \
    --output-dir "$POSTERIOR_V8_OUTPUT_DIR"
