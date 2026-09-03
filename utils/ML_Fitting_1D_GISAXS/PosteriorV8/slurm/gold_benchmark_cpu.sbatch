#!/bin/bash
#SBATCH --partition=allcpu
#SBATCH --account=hasylab
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail

: "${POSTERIOR_V8_SOURCE_ROOT:?set POSTERIOR_V8_SOURCE_ROOT}"
: "${POSTERIOR_V8_GOLD_OUTPUT:?set POSTERIOR_V8_GOLD_OUTPUT under /data/dust/user/zhaiyufe/}"

cd "$POSTERIOR_V8_SOURCE_ROOT"
export PYTHONPATH="$POSTERIOR_V8_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Defaults intentionally form a one-curve smoke audit.  Formal experiment
# size and budgets are explicit submit-time environment overrides.
curves="${POSTERIOR_V8_GOLD_CURVES:-1}"
k_schedule="${POSTERIOR_V8_GOLD_K_SCHEDULE:-2}"
noise_mode="${POSTERIOR_V8_GOLD_NOISE_MODE:-clean}"
points="${POSTERIOR_V8_GOLD_POINTS:-64}"
seed="${POSTERIOR_V8_GOLD_SEED:-20260902}"
sequential_nfev="${POSTERIOR_V8_GOLD_SEQUENTIAL_MAX_NFEV:-20}"
amplitude_nfev="${POSTERIOR_V8_GOLD_AMPLITUDE_MAX_NFEV:-20}"
joint_exact="${POSTERIOR_V8_GOLD_JOINT_MAX_EXACT:-64}"

conda run --no-capture-output -n tf \
    python -m utils.ML_Fitting_1D_GISAXS.PosteriorV8.gold_solver_benchmark \
    --output "$POSTERIOR_V8_GOLD_OUTPUT" \
    --curves "$curves" \
    --component-schedule "$k_schedule" \
    --noise-mode "$noise_mode" \
    --points "$points" \
    --seed "$seed" \
    --sequential-max-nfev "$sequential_nfev" \
    --amplitude-polish-max-nfev "$amplitude_nfev" \
    --joint-max-exact-evaluations "$joint_exact"
