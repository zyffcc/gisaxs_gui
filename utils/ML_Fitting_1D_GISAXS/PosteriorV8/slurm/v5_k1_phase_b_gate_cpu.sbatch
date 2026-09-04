#!/bin/bash
#SBATCH --partition=allcpu
#SBATCH --account=hasylab
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/data/dust/user/zhaiyufe/MaxwellRuns/GISAXS_ONE_CLICK_PAPER_V5_20260903_V5_2_R2/logs/k1-phase-b-v5-%j.out
#SBATCH --error=/data/dust/user/zhaiyufe/MaxwellRuns/GISAXS_ONE_CLICK_PAPER_V5_20260903_V5_2_R2/logs/k1-phase-b-v5-%j.err

set -euo pipefail

: "${SLURM_JOB_ID:?K1 Phase-B must run through Slurm}"
: "${POSTERIOR_V8_SOURCE_ROOT:?set POSTERIOR_V8_SOURCE_ROOT}"
: "${POSTERIOR_V8_V5_SOURCE_ARCHIVE:?set POSTERIOR_V8_V5_SOURCE_ARCHIVE}"
: "${POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_SHA256:?set source archive SHA-256}"
: "${POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_BYTE_COUNT:?set source archive size}"
: "${POSTERIOR_V8_V5_EXPECTED_SOURCE_MANIFEST_SHA256:?set source manifest SHA-256}"
: "${POSTERIOR_V8_V5_EXPECTED_SOURCE_TREE_SHA256:?set source tree SHA-256}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_DATASET:?set the checked Phase-A dataset path}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_DATASET_SHA256:?set Phase-A dataset SHA-256}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_DATASET_BYTE_COUNT:?set Phase-A dataset size}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_DATASET_BINDING:?set dataset binding path}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_DATASET_BINDING_SHA256:?set binding SHA-256}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_DATASET_BINDING_BYTE_COUNT:?set binding size}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_CROSS_PLATFORM_PASS_MARKER:?set PASS marker path}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_CROSS_PLATFORM_PASS_MARKER_SHA256:?set marker SHA-256}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_CROSS_PLATFORM_PASS_MARKER_BYTE_COUNT:?set marker size}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_RESULT:?set the Phase-A result path}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_RESULT_SHA256:?set Phase-A result SHA-256}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_RESULT_BYTE_COUNT:?set Phase-A result size}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_MODEL:?set the Phase-A model path}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_MODEL_SHA256:?set Phase-A model SHA-256}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_MODEL_BYTE_COUNT:?set Phase-A model size}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_MODEL_PROVENANCE:?set model provenance path}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_MODEL_PROVENANCE_SHA256:?set provenance SHA-256}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_MODEL_PROVENANCE_BYTE_COUNT:?set provenance size}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_PHASE_A_COMPLETION:?set Phase-A completion path}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_PHASE_A_COMPLETION_SHA256:?set Phase-A completion SHA-256}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_PHASE_A_COMPLETION_BYTE_COUNT:?set Phase-A completion size}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_GATE_CLAIM_SHA256:?set gate claim SHA-256}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_REFERENCE_SHA256:?set reference SHA-256}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_REFERENCE_BYTE_COUNT:?set reference size}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_REFERENCE_MANIFEST_SHA256:?set reference manifest SHA-256}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_SCIENTIFIC_SHA256:?set scientific SHA-256}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_COMPARISON_SHA256:?set comparison SHA-256}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_OUTPUT:?set the exclusive Phase-B output path}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_MODE:?set the audited Phase-B mode}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_STAGE:?set the receipt-bound launch stage}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN:?set the immutable launch plan path}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN_SHA256:?set launch plan self SHA-256}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN_FILE_SHA256:?set launch plan file SHA-256}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN_BYTE_COUNT:?set launch plan size}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN_MODE:?set launch plan mode}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN_DEVICE:?set launch plan device}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN_INODE:?set launch plan inode}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN_MTIME_NS:?set launch plan mtime}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN_CTIME_NS:?set launch plan ctime}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN_NLINK:?set launch plan link count}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_RECEIPT:?set held submission receipt path}"
: "${POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_COMPLETION:?set launch completion path}"

case "$(hostname -s)" in
    max-wgs*) echo "K1 Phase-B must run on a Slurm worker" >&2; exit 2 ;;
esac

run_root="/data/dust/user/zhaiyufe/MaxwellRuns/GISAXS_ONE_CLICK_PAPER_V5_20260903_V5_2_R2"
phase_a_root="$run_root/k1_phase_a_v5_2_dag_v13"
phase_b_root="$run_root/k1_phase_b_v5_2_dag_v3"
expected_dataset="$phase_a_root/datasets/k1-v5-2-phase-a-v5-full-r512-sphere-v0.gvd5"
expected_binding="$expected_dataset.binding-v1.json"
expected_marker="$phase_a_root/audit/sobol-cross-platform-PASS-v1.json"
expected_model_root="$phase_a_root/models/k1-v5-2-phase-a-v5-full-steps1500"
expected_phase_a_completion="$phase_a_root/audit/full-gate-completion-v1.json"
if [[ "$POSTERIOR_V8_V5_K1_PHASE_B_DATASET" != "$expected_dataset" \
    || "$POSTERIOR_V8_V5_K1_PHASE_B_DATASET_BINDING" != "$expected_binding" \
    || "$POSTERIOR_V8_V5_K1_PHASE_B_CROSS_PLATFORM_PASS_MARKER" != "$expected_marker" \
    || "$POSTERIOR_V8_V5_K1_PHASE_B_RESULT" != "$expected_model_root/result.json" \
    || "$POSTERIOR_V8_V5_K1_PHASE_B_MODEL" != "$expected_model_root/model.keras" \
    || "$POSTERIOR_V8_V5_K1_PHASE_B_MODEL_PROVENANCE" != "$expected_model_root/model.provenance.json" \
    || "$POSTERIOR_V8_V5_K1_PHASE_B_PHASE_A_COMPLETION" != "$expected_phase_a_completion" ]]; then
    echo "K1 Phase-B accepts only the new audited Phase-A full-run root" >&2
    exit 2
fi
case "$POSTERIOR_V8_V5_K1_PHASE_B_OUTPUT" in
    "$phase_b_root/results/engineering-smoke-parents2"|"$phase_b_root/results/formal-all512") ;;
    *) echo "K1 Phase-B output escaped the versioned Phase-B root" >&2; exit 2 ;;
esac

for dust_path in \
    "$POSTERIOR_V8_SOURCE_ROOT" \
    "$POSTERIOR_V8_V5_SOURCE_ARCHIVE" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_DATASET" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_DATASET_BINDING" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_CROSS_PLATFORM_PASS_MARKER" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_RESULT" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_MODEL" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_MODEL_PROVENANCE" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_PHASE_A_COMPLETION" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_RECEIPT" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_COMPLETION" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_OUTPUT"; do
    case "$dust_path" in
        /data/dust/user/zhaiyufe/*) ;;
        *) echo "K1 Phase-B persistent paths must be under the owner dust root" >&2; exit 2 ;;
    esac
done
if [[ ! -d "$POSTERIOR_V8_SOURCE_ROOT" || -L "$POSTERIOR_V8_SOURCE_ROOT" ]]; then
    echo "K1 Phase-B source root must be a real directory" >&2
    exit 2
fi
immutable_inputs=(
    "$POSTERIOR_V8_V5_SOURCE_ARCHIVE"
    "$POSTERIOR_V8_V5_K1_PHASE_B_DATASET"
    "$POSTERIOR_V8_V5_K1_PHASE_B_DATASET_BINDING"
    "$POSTERIOR_V8_V5_K1_PHASE_B_CROSS_PLATFORM_PASS_MARKER"
    "$POSTERIOR_V8_V5_K1_PHASE_B_RESULT"
    "$POSTERIOR_V8_V5_K1_PHASE_B_MODEL"
    "$POSTERIOR_V8_V5_K1_PHASE_B_MODEL_PROVENANCE"
    "$POSTERIOR_V8_V5_K1_PHASE_B_PHASE_A_COMPLETION"
    "$POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN"
)
for input_file in "${immutable_inputs[@]}"; do
    if [[ ! -f "$input_file" || -L "$input_file" ]]; then
        echo "missing or symlinked immutable K1 input: $input_file" >&2
        exit 2
    fi
    case "$(stat -c '%A' -- "$input_file")" in
        *w*) echo "immutable K1 input has write bits: $input_file" >&2; exit 2 ;;
    esac
done
if [[ -e "$POSTERIOR_V8_V5_K1_PHASE_B_OUTPUT" \
    || -L "$POSTERIOR_V8_V5_K1_PHASE_B_OUTPUT" ]]; then
    echo "refusing to overwrite existing K1 Phase-B output" >&2
    exit 2
fi
if [[ ! -d "$(dirname "$POSTERIOR_V8_V5_K1_PHASE_B_OUTPUT")" \
    || -L "$(dirname "$POSTERIOR_V8_V5_K1_PHASE_B_OUTPUT")" ]]; then
    echo "K1 Phase-B output parent must be a real directory" >&2
    exit 2
fi

sha_values=(
    "$POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_SHA256"
    "$POSTERIOR_V8_V5_EXPECTED_SOURCE_MANIFEST_SHA256"
    "$POSTERIOR_V8_V5_EXPECTED_SOURCE_TREE_SHA256"
    "$POSTERIOR_V8_V5_K1_PHASE_B_DATASET_SHA256"
    "$POSTERIOR_V8_V5_K1_PHASE_B_DATASET_BINDING_SHA256"
    "$POSTERIOR_V8_V5_K1_PHASE_B_CROSS_PLATFORM_PASS_MARKER_SHA256"
    "$POSTERIOR_V8_V5_K1_PHASE_B_RESULT_SHA256"
    "$POSTERIOR_V8_V5_K1_PHASE_B_MODEL_SHA256"
    "$POSTERIOR_V8_V5_K1_PHASE_B_MODEL_PROVENANCE_SHA256"
    "$POSTERIOR_V8_V5_K1_PHASE_B_PHASE_A_COMPLETION_SHA256"
    "$POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN_SHA256"
    "$POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN_FILE_SHA256"
    "$POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_GATE_CLAIM_SHA256"
    "$POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_REFERENCE_SHA256"
    "$POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_REFERENCE_MANIFEST_SHA256"
    "$POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_SCIENTIFIC_SHA256"
    "$POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_COMPARISON_SHA256"
)
for expected_sha in "${sha_values[@]}"; do
    if [[ ! "$expected_sha" =~ ^[0-9a-f]{64}$ ]]; then
        echo "K1 Phase-B expected SHA-256 is malformed" >&2
        exit 2
    fi
done
byte_values=(
    "$POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_BYTE_COUNT"
    "$POSTERIOR_V8_V5_K1_PHASE_B_DATASET_BYTE_COUNT"
    "$POSTERIOR_V8_V5_K1_PHASE_B_DATASET_BINDING_BYTE_COUNT"
    "$POSTERIOR_V8_V5_K1_PHASE_B_CROSS_PLATFORM_PASS_MARKER_BYTE_COUNT"
    "$POSTERIOR_V8_V5_K1_PHASE_B_RESULT_BYTE_COUNT"
    "$POSTERIOR_V8_V5_K1_PHASE_B_MODEL_BYTE_COUNT"
    "$POSTERIOR_V8_V5_K1_PHASE_B_MODEL_PROVENANCE_BYTE_COUNT"
    "$POSTERIOR_V8_V5_K1_PHASE_B_PHASE_A_COMPLETION_BYTE_COUNT"
    "$POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_REFERENCE_BYTE_COUNT"
    "$POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN_BYTE_COUNT"
)
for expected_bytes in "${byte_values[@]}"; do
    if [[ ! "$expected_bytes" =~ ^[1-9][0-9]*$ ]]; then
        echo "K1 Phase-B expected byte count must be positive" >&2
        exit 2
    fi
done

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
unset PYTHONPATH
GISAXS_JOB_TMP_BASE="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
if [[ ! -d "$GISAXS_JOB_TMP_BASE" || -L "$GISAXS_JOB_TMP_BASE" ]]; then
    echo "K1 Phase-B scratch parent must be a real directory" >&2
    exit 2
fi
GISAXS_JOB_TMP_BASE="$(realpath -e -- "$GISAXS_JOB_TMP_BASE")"
case "$GISAXS_JOB_TMP_BASE" in
    /data/dust/user/zhaiyufe|/data/dust/user/zhaiyufe/*)
        echo "K1 Phase-B scratch must be node-local, not dust" >&2
        exit 2
        ;;
esac
GISAXS_JOB_CACHE_ROOT="$(mktemp -d "$GISAXS_JOB_TMP_BASE/gisaxs-v5-k1-phase-b-${SLURM_JOB_ID}-XXXXXXXX")"
chmod 0700 "$GISAXS_JOB_CACHE_ROOT"
if [[ ! -d "$GISAXS_JOB_CACHE_ROOT" || -L "$GISAXS_JOB_CACHE_ROOT" \
    || ! -O "$GISAXS_JOB_CACHE_ROOT" \
    || "$(stat -c '%a' -- "$GISAXS_JOB_CACHE_ROOT")" != "700" ]]; then
    echo "K1 Phase-B scratch is not exclusive owner-private storage" >&2
    exit 2
fi
export XDG_CACHE_HOME="$GISAXS_JOB_CACHE_ROOT/xdg"
export KERAS_HOME="$GISAXS_JOB_CACHE_ROOT/keras"
export CUDA_CACHE_PATH="$GISAXS_JOB_CACHE_ROOT/cuda"
export MPLCONFIGDIR="$GISAXS_JOB_CACHE_ROOT/matplotlib"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

copy_checked() {
    local source_path="$1"
    local target_path="$2"
    local expected_sha="$3"
    local expected_bytes="$4"
    local label="$5"
    cp -- "$source_path" "$target_path"
    chmod 0400 "$target_path"
    local observed_sha observed_bytes ignored
    read -r observed_sha ignored < <(sha256sum -- "$target_path")
    observed_bytes="$(stat -c '%s' -- "$target_path")"
    if [[ "$observed_sha" != "$expected_sha" || "$observed_bytes" != "$expected_bytes" ]]; then
        echo "$label changed before worker preflight" >&2
        exit 2
    fi
}

verify_checked() {
    local path="$1"
    local expected_sha="$2"
    local expected_bytes="$3"
    local label="$4"
    if [[ ! -f "$path" || -L "$path" ]]; then
        echo "$label is missing or symlinked" >&2
        exit 2
    fi
    case "$(stat -c '%A' -- "$path")" in
        *w*) echo "$label acquired write bits" >&2; exit 2 ;;
    esac
    local observed_sha observed_bytes ignored
    read -r observed_sha ignored < <(sha256sum -- "$path")
    observed_bytes="$(stat -c '%s' -- "$path")"
    if [[ "$observed_sha" != "$expected_sha" || "$observed_bytes" != "$expected_bytes" ]]; then
        echo "$label identity drift detected" >&2
        exit 2
    fi
}

job_archive="$GISAXS_JOB_CACHE_ROOT/source-snapshot.tar"
copy_checked \
    "$POSTERIOR_V8_V5_SOURCE_ARCHIVE" "$job_archive" \
    "$POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_SHA256" \
    "$POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_BYTE_COUNT" "source archive"
trusted_verifier="$GISAXS_JOB_CACHE_ROOT/package_source_snapshot_v5.py"
job_source_root="$GISAXS_JOB_CACHE_ROOT/source"
tar -xOf "$job_archive" \
    snapshot/utils/ML_Fitting_1D_GISAXS/PosteriorV8/package_source_snapshot_v5.py \
    > "$trusted_verifier"
chmod 0400 "$trusted_verifier"
conda run --no-capture-output --prefix /data/dust/user/zhaiyufe/conda/envs/gisaxs-v5-r2 \
    python -I "$trusted_verifier" \
    extract \
    --archive "$job_archive" \
    --source-root "$job_source_root" \
    --expected-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_SHA256"
conda run --no-capture-output --prefix /data/dust/user/zhaiyufe/conda/envs/gisaxs-v5-r2 \
    python -I "$trusted_verifier" \
    verify-extracted \
    --archive "$job_archive" \
    --source-root "$job_source_root" \
    --expected-archive-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_SHA256" \
    --expected-manifest-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_MANIFEST_SHA256" \
    --expected-source-tree-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_TREE_SHA256"

job_input_root="$GISAXS_JOB_CACHE_ROOT/phase-a"
mkdir -m 0700 "$job_input_root"
job_dataset="$job_input_root/input.gvd5"
job_binding="$job_input_root/input.gvd5.binding-v1.json"
job_marker="$job_input_root/PASS.json"
job_result="$job_input_root/result.json"
job_model="$job_input_root/model.keras"
job_provenance="$job_input_root/model.provenance.json"
job_phase_a_completion="$job_input_root/full-gate-completion-v1.json"
copy_checked "$POSTERIOR_V8_V5_K1_PHASE_B_DATASET" "$job_dataset" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_DATASET_SHA256" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_DATASET_BYTE_COUNT" "Phase-A dataset"
copy_checked "$POSTERIOR_V8_V5_K1_PHASE_B_DATASET_BINDING" "$job_binding" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_DATASET_BINDING_SHA256" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_DATASET_BINDING_BYTE_COUNT" "dataset binding"
copy_checked "$POSTERIOR_V8_V5_K1_PHASE_B_CROSS_PLATFORM_PASS_MARKER" "$job_marker" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_CROSS_PLATFORM_PASS_MARKER_SHA256" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_CROSS_PLATFORM_PASS_MARKER_BYTE_COUNT" "PASS marker"
copy_checked "$POSTERIOR_V8_V5_K1_PHASE_B_RESULT" "$job_result" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_RESULT_SHA256" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_RESULT_BYTE_COUNT" "Phase-A result"
copy_checked "$POSTERIOR_V8_V5_K1_PHASE_B_MODEL" "$job_model" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_MODEL_SHA256" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_MODEL_BYTE_COUNT" "Phase-A model"
copy_checked "$POSTERIOR_V8_V5_K1_PHASE_B_MODEL_PROVENANCE" "$job_provenance" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_MODEL_PROVENANCE_SHA256" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_MODEL_PROVENANCE_BYTE_COUNT" "model provenance"
copy_checked "$POSTERIOR_V8_V5_K1_PHASE_B_PHASE_A_COMPLETION" "$job_phase_a_completion" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_PHASE_A_COMPLETION_SHA256" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_PHASE_A_COMPLETION_BYTE_COUNT" "Phase-A completion"

cd "$job_source_root"
run_verified_module() {
    local module="$1"
    shift
    conda run --no-capture-output --prefix /data/dust/user/zhaiyufe/conda/envs/gisaxs-v5-r2 \
        python -I -c \
        'import runpy, sys; root, module = sys.argv[1:3]; sys.path.insert(0, root); sys.argv = [module, *sys.argv[3:]]; runpy.run_module(module, run_name="__main__")' \
        "$job_source_root" "$module" "$@"
}
marker_expected_arguments=(
    --source-archive-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_SHA256"
    --source-manifest-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_MANIFEST_SHA256"
    --source-tree-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_TREE_SHA256"
    --reference-file-sha256 "$POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_REFERENCE_SHA256"
    --reference-file-byte-count "$POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_REFERENCE_BYTE_COUNT"
    --reference-manifest-sha256 "$POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_REFERENCE_MANIFEST_SHA256"
    --scientific-content-sha256 "$POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_SCIENTIFIC_SHA256"
    --comparison-result-sha256 "$POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_COMPARISON_SHA256"
    --expected-gate-claim-sha256 "$POSTERIOR_V8_V5_K1_PHASE_B_EXPECTED_GATE_CLAIM_SHA256"
)
run_verified_module \
    utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_a_dataset_binding_v5 \
    validate \
    --dataset "$job_dataset" \
    --marker "$job_marker" \
    --binding "$job_binding" \
    --original-dataset-path "$POSTERIOR_V8_V5_K1_PHASE_B_DATASET" \
    "${marker_expected_arguments[@]}"

arguments=(
    --dataset "$job_dataset"
    --phase-a-result "$job_result"
    --phase-a-model "$job_model"
    --output-dir "$POSTERIOR_V8_V5_K1_PHASE_B_OUTPUT"
    --source-root "$job_source_root"
    --source-archive "$job_archive"
    --source-archive-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_SHA256"
    --source-archive-byte-count "$POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_BYTE_COUNT"
    --source-manifest-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_MANIFEST_SHA256"
    --source-tree-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_TREE_SHA256"
    --input-root "$GISAXS_JOB_CACHE_ROOT"
    --original-source-root "$POSTERIOR_V8_SOURCE_ROOT"
    --original-source-archive "$POSTERIOR_V8_V5_SOURCE_ARCHIVE"
    --original-dataset-path "$POSTERIOR_V8_V5_K1_PHASE_B_DATASET"
    --dataset-sha256 "$POSTERIOR_V8_V5_K1_PHASE_B_DATASET_SHA256"
    --dataset-byte-count "$POSTERIOR_V8_V5_K1_PHASE_B_DATASET_BYTE_COUNT"
    --dataset-binding "$job_binding"
    --original-dataset-binding "$POSTERIOR_V8_V5_K1_PHASE_B_DATASET_BINDING"
    --dataset-binding-sha256 "$POSTERIOR_V8_V5_K1_PHASE_B_DATASET_BINDING_SHA256"
    --dataset-binding-byte-count "$POSTERIOR_V8_V5_K1_PHASE_B_DATASET_BINDING_BYTE_COUNT"
    --cross-platform-pass-marker "$job_marker"
    --original-cross-platform-pass-marker "$POSTERIOR_V8_V5_K1_PHASE_B_CROSS_PLATFORM_PASS_MARKER"
    --cross-platform-pass-marker-sha256 "$POSTERIOR_V8_V5_K1_PHASE_B_CROSS_PLATFORM_PASS_MARKER_SHA256"
    --cross-platform-pass-marker-byte-count "$POSTERIOR_V8_V5_K1_PHASE_B_CROSS_PLATFORM_PASS_MARKER_BYTE_COUNT"
    --original-phase-a-output-dir "$expected_model_root"
    --original-phase-a-result "$POSTERIOR_V8_V5_K1_PHASE_B_RESULT"
    --phase-a-result-sha256 "$POSTERIOR_V8_V5_K1_PHASE_B_RESULT_SHA256"
    --phase-a-result-byte-count "$POSTERIOR_V8_V5_K1_PHASE_B_RESULT_BYTE_COUNT"
    --original-phase-a-model "$POSTERIOR_V8_V5_K1_PHASE_B_MODEL"
    --phase-a-model-sha256 "$POSTERIOR_V8_V5_K1_PHASE_B_MODEL_SHA256"
    --phase-a-model-byte-count "$POSTERIOR_V8_V5_K1_PHASE_B_MODEL_BYTE_COUNT"
    --original-phase-a-model-provenance "$POSTERIOR_V8_V5_K1_PHASE_B_MODEL_PROVENANCE"
    --phase-a-model-provenance-sha256 "$POSTERIOR_V8_V5_K1_PHASE_B_MODEL_PROVENANCE_SHA256"
    --phase-a-model-provenance-byte-count "$POSTERIOR_V8_V5_K1_PHASE_B_MODEL_PROVENANCE_BYTE_COUNT"
    --phase-a-completion "$job_phase_a_completion"
    --original-phase-a-completion "$POSTERIOR_V8_V5_K1_PHASE_B_PHASE_A_COMPLETION"
    --phase-a-completion-sha256 "$POSTERIOR_V8_V5_K1_PHASE_B_PHASE_A_COMPLETION_SHA256"
    --phase-a-completion-byte-count "$POSTERIOR_V8_V5_K1_PHASE_B_PHASE_A_COMPLETION_BYTE_COUNT"
    --launch-stage "$POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_STAGE"
    --launch-plan "$POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN"
    --launch-plan-sha256 "$POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN_SHA256"
    --launch-plan-file-sha256 "$POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN_FILE_SHA256"
    --launch-plan-byte-count "$POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN_BYTE_COUNT"
    --launch-plan-mode "$POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN_MODE"
    --launch-plan-device "$POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN_DEVICE"
    --launch-plan-inode "$POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN_INODE"
    --launch-plan-mtime-ns "$POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN_MTIME_NS"
    --launch-plan-ctime-ns "$POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN_CTIME_NS"
    --launch-plan-nlink "$POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN_NLINK"
    --launch-receipt "$POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_RECEIPT"
    --launch-completion "$POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_COMPLETION"
    --seed "${POSTERIOR_V8_V5_K1_PHASE_B_SEED:-20260903}"
    --per-candidate-forward-limit "${POSTERIOR_V8_V5_K1_PHASE_B_PER_CANDIDATE_LIMIT:-128}"
    --per-parent-forward-limit "${POSTERIOR_V8_V5_K1_PHASE_B_PER_PARENT_LIMIT:-4096}"
    --progress-interval "${POSTERIOR_V8_V5_K1_PHASE_B_PROGRESS_INTERVAL:-1}"
)

mode="$POSTERIOR_V8_V5_K1_PHASE_B_MODE"
case "$mode" in
    smoke)
        if [[ "$POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_STAGE" != "engineering_smoke" ]]; then
            echo "smoke mode requires the receipt-bound engineering stage" >&2
            exit 2
        fi
        if [[ "$POSTERIOR_V8_V5_K1_PHASE_B_OUTPUT" != "$phase_b_root/results/engineering-smoke-parents2" ]]; then
            echo "smoke mode/output mismatch" >&2
            exit 2
        fi
        arguments+=(--parent-limit "${POSTERIOR_V8_V5_K1_PHASE_B_SMOKE_PARENTS:-2}")
        ;;
    formal)
        : "${POSTERIOR_V8_V5_K1_PHASE_B_FORMAL_ACK:?set to YES for the audited formal DAG}"
        if [[ "$POSTERIOR_V8_V5_K1_PHASE_B_FORMAL_ACK" != "YES" \
            || "$POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_STAGE" != "formal_gate" \
            || "$POSTERIOR_V8_V5_K1_PHASE_B_OUTPUT" != "$phase_b_root/results/formal-all512" ]]; then
            echo "formal K1 Phase-B requires the audited formal acknowledgement/output" >&2
            exit 2
        fi
        ;;
    *)
        echo "POSTERIOR_V8_V5_K1_PHASE_B_MODE must be smoke or formal" >&2
        exit 2
        ;;
esac

for _ in {1..60}; do
    if [[ -f "$POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_COMPLETION" \
        && ! -L "$POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_COMPLETION" \
        && -f "$POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_RECEIPT" \
        && ! -L "$POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_RECEIPT" ]]; then
        break
    fi
    sleep 1
done
if [[ ! -f "$POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_COMPLETION" \
    || -L "$POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_COMPLETION" \
    || ! -f "$POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_RECEIPT" \
    || -L "$POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_RECEIPT" ]]; then
    echo "Phase-B launch completion did not appear after held release" >&2
    exit 2
fi

set +e
run_verified_module \
    utils.ML_Fitting_1D_GISAXS.PosteriorV8.run_k1_phase_b_gate_v5 \
    "${arguments[@]}"
worker_status=$?
set -e

verify_checked "$POSTERIOR_V8_V5_SOURCE_ARCHIVE" \
    "$POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_SHA256" \
    "$POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_BYTE_COUNT" "source archive after worker"
verify_checked "$POSTERIOR_V8_V5_K1_PHASE_B_DATASET" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_DATASET_SHA256" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_DATASET_BYTE_COUNT" "Phase-A dataset after worker"
verify_checked "$POSTERIOR_V8_V5_K1_PHASE_B_DATASET_BINDING" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_DATASET_BINDING_SHA256" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_DATASET_BINDING_BYTE_COUNT" "dataset binding after worker"
verify_checked "$POSTERIOR_V8_V5_K1_PHASE_B_CROSS_PLATFORM_PASS_MARKER" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_CROSS_PLATFORM_PASS_MARKER_SHA256" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_CROSS_PLATFORM_PASS_MARKER_BYTE_COUNT" "PASS marker after worker"
verify_checked "$POSTERIOR_V8_V5_K1_PHASE_B_RESULT" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_RESULT_SHA256" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_RESULT_BYTE_COUNT" "Phase-A result after worker"
verify_checked "$POSTERIOR_V8_V5_K1_PHASE_B_MODEL" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_MODEL_SHA256" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_MODEL_BYTE_COUNT" "Phase-A model after worker"
verify_checked "$POSTERIOR_V8_V5_K1_PHASE_B_MODEL_PROVENANCE" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_MODEL_PROVENANCE_SHA256" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_MODEL_PROVENANCE_BYTE_COUNT" "model provenance after worker"
verify_checked "$POSTERIOR_V8_V5_K1_PHASE_B_PHASE_A_COMPLETION" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_PHASE_A_COMPLETION_SHA256" \
    "$POSTERIOR_V8_V5_K1_PHASE_B_PHASE_A_COMPLETION_BYTE_COUNT" "Phase-A completion after worker"
exit "$worker_status"
