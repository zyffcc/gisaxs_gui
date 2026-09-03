#!/bin/bash
#SBATCH --partition=allgpu
#SBATCH --account=hasylab
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --constraint=GPUx1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/data/dust/user/zhaiyufe/MaxwellRuns/GISAXS_ONE_CLICK_PAPER_V5_20260903_V5_2_R2/logs/k1-phase-a-v5-gate-%j.out
#SBATCH --error=/data/dust/user/zhaiyufe/MaxwellRuns/GISAXS_ONE_CLICK_PAPER_V5_20260903_V5_2_R2/logs/k1-phase-a-v5-gate-%j.err

set -euo pipefail

: "${SLURM_JOB_ID:?K1 memorization must run through Slurm}"
: "${POSTERIOR_V8_SOURCE_ROOT:?set POSTERIOR_V8_SOURCE_ROOT}"
: "${POSTERIOR_V8_V5_SOURCE_ARCHIVE:?set POSTERIOR_V8_V5_SOURCE_ARCHIVE}"
: "${POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_SHA256:?set expected archive SHA-256}"
: "${POSTERIOR_V8_V5_EXPECTED_SOURCE_MANIFEST_SHA256:?set expected manifest SHA-256}"
: "${POSTERIOR_V8_V5_EXPECTED_SOURCE_TREE_SHA256:?set expected source-tree SHA-256}"
: "${POSTERIOR_V8_V5_CROSS_PLATFORM_PASS_MARKER:?set cross-platform PASS marker}"
: "${POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_REFERENCE_SHA256:?set reference SHA-256}"
: "${POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_REFERENCE_BYTE_COUNT:?set reference byte count}"
: "${POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_MANIFEST_SHA256:?set manifest SHA-256}"
: "${POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_SCIENTIFIC_SHA256:?set scientific SHA-256}"
: "${POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_COMPARISON_SHA256:?set comparison SHA-256}"
: "${POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_GATE_CLAIM_SHA256:?set gate claim SHA-256}"
: "${POSTERIOR_V8_V5_K1_DATASET:?set POSTERIOR_V8_V5_K1_DATASET}"
: "${POSTERIOR_V8_V5_K1_DATASET_BINDING:?set dataset completion binding}"
: "${POSTERIOR_V8_V5_K1_GATE_OUTPUT:?set POSTERIOR_V8_V5_K1_GATE_OUTPUT}"

case "$(hostname -s)" in
    max-wgs*) echo "K1 memorization must run on a Slurm worker" >&2; exit 2 ;;
esac
for dust_path in \
    "$POSTERIOR_V8_SOURCE_ROOT" \
    "$POSTERIOR_V8_V5_SOURCE_ARCHIVE" \
    "$POSTERIOR_V8_V5_CROSS_PLATFORM_PASS_MARKER" \
    "$POSTERIOR_V8_V5_K1_DATASET" \
    "$POSTERIOR_V8_V5_K1_DATASET_BINDING" \
    "$POSTERIOR_V8_V5_K1_GATE_OUTPUT"; do
    case "$dust_path" in
        /data/dust/user/zhaiyufe/*) ;;
        *) echo "K1 dataset and gate output must be under /data/dust/user/zhaiyufe/" >&2; exit 2 ;;
    esac
done
if [[ ! -d "$POSTERIOR_V8_SOURCE_ROOT" || -L "$POSTERIOR_V8_SOURCE_ROOT" ]]; then
    echo "K1 source root must be a real directory" >&2
    exit 2
fi
if [[ ! -f "$POSTERIOR_V8_V5_SOURCE_ARCHIVE" || -L "$POSTERIOR_V8_V5_SOURCE_ARCHIVE" ]]; then
    echo "K1 source archive must be a real file" >&2
    exit 2
fi
if [[ ! -f "$POSTERIOR_V8_V5_CROSS_PLATFORM_PASS_MARKER" \
    || -L "$POSTERIOR_V8_V5_CROSS_PLATFORM_PASS_MARKER" ]]; then
    echo "cross-platform PASS marker must be a regular non-symlink file" >&2
    exit 2
fi
if [[ ! -f "$POSTERIOR_V8_V5_K1_DATASET" || -L "$POSTERIOR_V8_V5_K1_DATASET" ]]; then
    echo "checked K1 grouped dataset does not exist" >&2
    exit 2
fi
if [[ ! -f "$POSTERIOR_V8_V5_K1_DATASET_BINDING" \
    || -L "$POSTERIOR_V8_V5_K1_DATASET_BINDING" ]]; then
    echo "K1 dataset completion binding does not exist" >&2
    exit 2
fi
if [[ -e "$POSTERIOR_V8_V5_K1_GATE_OUTPUT" \
    || -L "$POSTERIOR_V8_V5_K1_GATE_OUTPUT" ]]; then
    echo "refusing to overwrite existing K1 gate output" >&2
    exit 2
fi
if [[ ! -d "$(dirname "$POSTERIOR_V8_V5_K1_GATE_OUTPUT")" \
    || -L "$(dirname "$POSTERIOR_V8_V5_K1_GATE_OUTPUT")" ]]; then
    echo "K1 gate output parent must be a real directory" >&2
    exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
unset PYTHONPATH
GISAXS_JOB_TMP_BASE="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
if [[ ! -d "$GISAXS_JOB_TMP_BASE" || -L "$GISAXS_JOB_TMP_BASE" ]]; then
    echo "K1 gate scratch parent must be a real directory" >&2
    exit 2
fi
GISAXS_JOB_TMP_BASE="$(realpath -e -- "$GISAXS_JOB_TMP_BASE")"
case "$GISAXS_JOB_TMP_BASE" in
    /data/dust/user/zhaiyufe|/data/dust/user/zhaiyufe/*)
        echo "K1 gate scratch must be node-local, not dust" >&2
        exit 2
        ;;
esac
GISAXS_JOB_CACHE_ROOT="$(mktemp -d "$GISAXS_JOB_TMP_BASE/gisaxs-v5-k1-gate-${SLURM_JOB_ID}-XXXXXXXX")"
chmod 0700 "$GISAXS_JOB_CACHE_ROOT"
if [[ ! -d "$GISAXS_JOB_CACHE_ROOT" || -L "$GISAXS_JOB_CACHE_ROOT" \
    || ! -O "$GISAXS_JOB_CACHE_ROOT" \
    || "$(stat -c '%a' -- "$GISAXS_JOB_CACHE_ROOT")" != "700" ]]; then
    echo "K1 gate scratch is not exclusive owner-private storage" >&2
    exit 2
fi
export XDG_CACHE_HOME="$GISAXS_JOB_CACHE_ROOT/xdg"
export KERAS_HOME="$GISAXS_JOB_CACHE_ROOT/keras"
export CUDA_CACHE_PATH="$GISAXS_JOB_CACHE_ROOT/cuda"
export MPLCONFIGDIR="$GISAXS_JOB_CACHE_ROOT/matplotlib"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

if [[ ! "$POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "expected source archive SHA-256 is malformed" >&2
    exit 2
fi
POSTERIOR_V8_JOB_SOURCE_ARCHIVE="$GISAXS_JOB_CACHE_ROOT/source-snapshot.tar"
POSTERIOR_V8_TRUSTED_VERIFIER="$GISAXS_JOB_CACHE_ROOT/package_source_snapshot_v5.py"
POSTERIOR_V8_JOB_SOURCE_ROOT="$GISAXS_JOB_CACHE_ROOT/source"
cp -- "$POSTERIOR_V8_V5_SOURCE_ARCHIVE" "$POSTERIOR_V8_JOB_SOURCE_ARCHIVE"
chmod 0400 "$POSTERIOR_V8_JOB_SOURCE_ARCHIVE"
read -r POSTERIOR_V8_JOB_ARCHIVE_SHA256 _ < <(
    sha256sum -- "$POSTERIOR_V8_JOB_SOURCE_ARCHIVE"
)
if [[ "$POSTERIOR_V8_JOB_ARCHIVE_SHA256" != "$POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_SHA256" ]]; then
    echo "source archive SHA-256 changed before worker preflight" >&2
    exit 2
fi
tar -xOf "$POSTERIOR_V8_JOB_SOURCE_ARCHIVE" \
    snapshot/utils/ML_Fitting_1D_GISAXS/PosteriorV8/package_source_snapshot_v5.py \
    > "$POSTERIOR_V8_TRUSTED_VERIFIER"
chmod 0400 "$POSTERIOR_V8_TRUSTED_VERIFIER"
conda run --no-capture-output -n tf \
    python -I "$POSTERIOR_V8_TRUSTED_VERIFIER" \
    extract \
    --archive "$POSTERIOR_V8_JOB_SOURCE_ARCHIVE" \
    --source-root "$POSTERIOR_V8_JOB_SOURCE_ROOT" \
    --expected-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_SHA256"
conda run --no-capture-output -n tf \
    python -I "$POSTERIOR_V8_TRUSTED_VERIFIER" \
    verify-extracted \
    --archive "$POSTERIOR_V8_JOB_SOURCE_ARCHIVE" \
    --source-root "$POSTERIOR_V8_JOB_SOURCE_ROOT" \
    --expected-archive-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_SHA256" \
    --expected-manifest-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_MANIFEST_SHA256" \
    --expected-source-tree-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_TREE_SHA256"

cd "$POSTERIOR_V8_JOB_SOURCE_ROOT"
run_verified_module() {
    local module="$1"
    shift
    conda run --no-capture-output -n tf \
        python -I -c \
        'import runpy, sys; root, module = sys.argv[1:3]; sys.path.insert(0, root); sys.argv = [module, *sys.argv[3:]]; runpy.run_module(module, run_name="__main__")' \
        "$POSTERIOR_V8_JOB_SOURCE_ROOT" "$module" "$@"
}
POSTERIOR_V8_JOB_PASS_MARKER="$GISAXS_JOB_CACHE_ROOT/PASS.json"
cp -- "$POSTERIOR_V8_V5_CROSS_PLATFORM_PASS_MARKER" "$POSTERIOR_V8_JOB_PASS_MARKER"
chmod 0400 "$POSTERIOR_V8_JOB_PASS_MARKER"
marker_expected_arguments=(
    --source-archive-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_SHA256"
    --source-manifest-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_MANIFEST_SHA256"
    --source-tree-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_TREE_SHA256"
    --reference-file-sha256 "$POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_REFERENCE_SHA256"
    --reference-file-byte-count "$POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_REFERENCE_BYTE_COUNT"
    --reference-manifest-sha256 "$POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_MANIFEST_SHA256"
    --scientific-content-sha256 "$POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_SCIENTIFIC_SHA256"
    --comparison-result-sha256 "$POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_COMPARISON_SHA256"
    --expected-gate-claim-sha256 "$POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_GATE_CLAIM_SHA256"
)
run_verified_module \
    utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_a_cross_platform_v5 \
    --marker "$POSTERIOR_V8_JOB_PASS_MARKER" \
    "${marker_expected_arguments[@]}"
POSTERIOR_V8_JOB_DATASET="$GISAXS_JOB_CACHE_ROOT/input.gvd5"
POSTERIOR_V8_JOB_DATASET_BINDING="$GISAXS_JOB_CACHE_ROOT/input.gvd5.phase-a-binding.json"
cp -- "$POSTERIOR_V8_V5_K1_DATASET" "$POSTERIOR_V8_JOB_DATASET"
cp -- "$POSTERIOR_V8_V5_K1_DATASET_BINDING" "$POSTERIOR_V8_JOB_DATASET_BINDING"
chmod 0400 "$POSTERIOR_V8_JOB_DATASET" "$POSTERIOR_V8_JOB_DATASET_BINDING"
run_verified_module \
    utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_a_dataset_binding_v5 \
    validate \
    --dataset "$POSTERIOR_V8_JOB_DATASET" \
    --marker "$POSTERIOR_V8_JOB_PASS_MARKER" \
    --binding "$POSTERIOR_V8_JOB_DATASET_BINDING" \
    --original-dataset-path "$POSTERIOR_V8_V5_K1_DATASET" \
    "${marker_expected_arguments[@]}"
arguments=(
    --dataset "$POSTERIOR_V8_JOB_DATASET"
    --dataset-root "$GISAXS_JOB_CACHE_ROOT"
    --dataset-binding "$POSTERIOR_V8_JOB_DATASET_BINDING"
    --cross-platform-pass-marker "$POSTERIOR_V8_JOB_PASS_MARKER"
    --original-dataset-path "$POSTERIOR_V8_V5_K1_DATASET"
    --source-archive-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_SHA256"
    --source-manifest-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_MANIFEST_SHA256"
    --source-tree-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_TREE_SHA256"
    --reference-file-sha256 "$POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_REFERENCE_SHA256"
    --reference-file-byte-count "$POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_REFERENCE_BYTE_COUNT"
    --reference-manifest-sha256 "$POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_MANIFEST_SHA256"
    --scientific-content-sha256 "$POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_SCIENTIFIC_SHA256"
    --comparison-result-sha256 "$POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_COMPARISON_SHA256"
    --gate-claim-sha256 "$POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_GATE_CLAIM_SHA256"
    --output-dir "$POSTERIOR_V8_V5_K1_GATE_OUTPUT"
    --source-root "$POSTERIOR_V8_JOB_SOURCE_ROOT"
    --steps "${POSTERIOR_V8_V5_K1_STEPS:-1500}"
    --learning-rate "${POSTERIOR_V8_V5_K1_LEARNING_RATE:-0.003}"
    --seed "${POSTERIOR_V8_V5_K1_SEED:-20260903}"
    --max-final-target-median-rms "${POSTERIOR_V8_V5_K1_MAX_FINAL_RMS:-0.01}"
    --minimum-loss-reduction "${POSTERIOR_V8_V5_K1_MINIMUM_LOSS_REDUCTION:-0.5}"
    --width "${POSTERIOR_V8_V5_K1_WIDTH:-128}"
    --encoder-blocks "${POSTERIOR_V8_V5_K1_ENCODER_BLOCKS:-6}"
)
if [[ "${POSTERIOR_V8_V5_K1_SMOKE:-0}" == "1" ]]; then
    arguments+=(--smoke)
fi
if [[ "${POSTERIOR_V8_V5_K1_DRY_RUN:-0}" == "1" ]]; then
    arguments+=(--dry-run)
fi

srun nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
run_verified_module \
    utils.ML_Fitting_1D_GISAXS.PosteriorV8.run_k1_memorization_gate_v5 \
    "${arguments[@]}"
