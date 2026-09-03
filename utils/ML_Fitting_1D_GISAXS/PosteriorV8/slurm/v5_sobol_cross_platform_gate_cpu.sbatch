#!/bin/bash
#SBATCH --partition=allcpu
#SBATCH --account=hasylab
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/data/dust/user/zhaiyufe/MaxwellRuns/GISAXS_ONE_CLICK_PAPER_V5_20260903_V5_2_R2/logs/k1-phase-a-v5-cross-platform-%j.out
#SBATCH --error=/data/dust/user/zhaiyufe/MaxwellRuns/GISAXS_ONE_CLICK_PAPER_V5_20260903_V5_2_R2/logs/k1-phase-a-v5-cross-platform-%j.err

set -euo pipefail

: "${SLURM_JOB_ID:?cross-platform gate must run through Slurm}"
: "${POSTERIOR_V8_SOURCE_ROOT:?set POSTERIOR_V8_SOURCE_ROOT}"
: "${POSTERIOR_V8_V5_SOURCE_ARCHIVE:?set POSTERIOR_V8_V5_SOURCE_ARCHIVE}"
: "${POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_SHA256:?set expected archive SHA-256}"
: "${POSTERIOR_V8_V5_EXPECTED_SOURCE_MANIFEST_SHA256:?set expected manifest SHA-256}"
: "${POSTERIOR_V8_V5_EXPECTED_SOURCE_TREE_SHA256:?set expected source-tree SHA-256}"
: "${POSTERIOR_V8_V5_CROSS_PLATFORM_REFERENCE:?set cross-platform reference}"
: "${POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_REFERENCE_SHA256:?set reference SHA-256}"
: "${POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_REFERENCE_BYTE_COUNT:?set reference byte count}"
: "${POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_MANIFEST_SHA256:?set manifest SHA-256}"
: "${POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_SCIENTIFIC_SHA256:?set scientific SHA-256}"
: "${POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_COMPARISON_SHA256:?set comparison SHA-256}"
: "${POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_GATE_CLAIM_SHA256:?set gate claim SHA-256}"
: "${POSTERIOR_V8_V5_CROSS_PLATFORM_CANDIDATE_OUTPUT:?set candidate output}"
: "${POSTERIOR_V8_V5_CROSS_PLATFORM_RESULT_OUTPUT:?set result output}"
: "${POSTERIOR_V8_V5_CROSS_PLATFORM_PASS_MARKER:?set PASS marker output}"

case "$(hostname -s)" in
    max-wgs*) echo "cross-platform gate must run on a Slurm worker" >&2; exit 2 ;;
esac
for dust_path in \
    "$POSTERIOR_V8_SOURCE_ROOT" \
    "$POSTERIOR_V8_V5_SOURCE_ARCHIVE" \
    "$POSTERIOR_V8_V5_CROSS_PLATFORM_REFERENCE" \
    "$POSTERIOR_V8_V5_CROSS_PLATFORM_CANDIDATE_OUTPUT" \
    "$POSTERIOR_V8_V5_CROSS_PLATFORM_RESULT_OUTPUT" \
    "$POSTERIOR_V8_V5_CROSS_PLATFORM_PASS_MARKER"; do
    case "$dust_path" in
        /data/dust/user/zhaiyufe/*) ;;
        *) echo "cross-platform gate paths must be under /data/dust/user/zhaiyufe/" >&2; exit 2 ;;
    esac
done
if [[ ! -d "$POSTERIOR_V8_SOURCE_ROOT" || -L "$POSTERIOR_V8_SOURCE_ROOT" ]]; then
    echo "source root must be a real directory" >&2
    exit 2
fi
for input_file in \
    "$POSTERIOR_V8_V5_SOURCE_ARCHIVE" \
    "$POSTERIOR_V8_V5_CROSS_PLATFORM_REFERENCE"; do
    if [[ ! -f "$input_file" || -L "$input_file" ]]; then
        echo "cross-platform gate inputs must be regular non-symlink files" >&2
        exit 2
    fi
done
for output_file in \
    "$POSTERIOR_V8_V5_CROSS_PLATFORM_CANDIDATE_OUTPUT" \
    "$POSTERIOR_V8_V5_CROSS_PLATFORM_RESULT_OUTPUT" \
    "$POSTERIOR_V8_V5_CROSS_PLATFORM_PASS_MARKER"; do
    if [[ -e "$output_file" || -L "$output_file" ]]; then
        echo "refusing to overwrite a cross-platform gate output" >&2
        exit 2
    fi
    if [[ ! -d "$(dirname "$output_file")" || -L "$(dirname "$output_file")" ]]; then
        echo "cross-platform gate output parent must be a real directory" >&2
        exit 2
    fi
done
if [[ ! "$POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_REFERENCE_BYTE_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "cross-platform reference byte count must be positive" >&2
    exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
unset PYTHONPATH
GISAXS_JOB_TMP_BASE="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
if [[ ! -d "$GISAXS_JOB_TMP_BASE" || -L "$GISAXS_JOB_TMP_BASE" ]]; then
    echo "cross-platform gate scratch parent must be a real directory" >&2
    exit 2
fi
GISAXS_JOB_TMP_BASE="$(realpath -e -- "$GISAXS_JOB_TMP_BASE")"
case "$GISAXS_JOB_TMP_BASE" in
    /data/dust/user/zhaiyufe|/data/dust/user/zhaiyufe/*)
        echo "cross-platform gate scratch must be node-local, not dust" >&2
        exit 2
        ;;
esac
GISAXS_JOB_CACHE_ROOT="$(mktemp -d "$GISAXS_JOB_TMP_BASE/gisaxs-v5-cross-gate-${SLURM_JOB_ID}-XXXXXXXX")"
chmod 0700 "$GISAXS_JOB_CACHE_ROOT"
if [[ ! -d "$GISAXS_JOB_CACHE_ROOT" || -L "$GISAXS_JOB_CACHE_ROOT" \
    || ! -O "$GISAXS_JOB_CACHE_ROOT" \
    || "$(stat -c '%a' -- "$GISAXS_JOB_CACHE_ROOT")" != "700" ]]; then
    echo "cross-platform gate scratch is not exclusive owner-private storage" >&2
    exit 2
fi
export XDG_CACHE_HOME="$GISAXS_JOB_CACHE_ROOT/xdg"
export KERAS_HOME="$GISAXS_JOB_CACHE_ROOT/keras"
export MPLCONFIGDIR="$GISAXS_JOB_CACHE_ROOT/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ ! "$POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "expected source archive SHA-256 is malformed" >&2
    exit 2
fi
POSTERIOR_V8_JOB_SOURCE_ARCHIVE="$GISAXS_JOB_CACHE_ROOT/source-snapshot.tar"
POSTERIOR_V8_TRUSTED_VERIFIER="$GISAXS_JOB_CACHE_ROOT/package_source_snapshot_v5.py"
POSTERIOR_V8_JOB_SOURCE_ROOT="$GISAXS_JOB_CACHE_ROOT/source"
cp -- "$POSTERIOR_V8_V5_SOURCE_ARCHIVE" "$POSTERIOR_V8_JOB_SOURCE_ARCHIVE"
chmod 0400 "$POSTERIOR_V8_JOB_SOURCE_ARCHIVE"
read -r POSTERIOR_V8_JOB_ARCHIVE_SHA256 _ < <(sha256sum -- "$POSTERIOR_V8_JOB_SOURCE_ARCHIVE")
if [[ "$POSTERIOR_V8_JOB_ARCHIVE_SHA256" != "$POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_SHA256" ]]; then
    echo "source archive SHA-256 changed before worker preflight" >&2
    exit 2
fi
tar -xOf "$POSTERIOR_V8_JOB_SOURCE_ARCHIVE" \
    snapshot/utils/ML_Fitting_1D_GISAXS/PosteriorV8/package_source_snapshot_v5.py \
    > "$POSTERIOR_V8_TRUSTED_VERIFIER"
chmod 0400 "$POSTERIOR_V8_TRUSTED_VERIFIER"
conda run --no-capture-output -n tf python -I "$POSTERIOR_V8_TRUSTED_VERIFIER" \
    extract \
    --archive "$POSTERIOR_V8_JOB_SOURCE_ARCHIVE" \
    --source-root "$POSTERIOR_V8_JOB_SOURCE_ROOT" \
    --expected-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_SHA256"
conda run --no-capture-output -n tf python -I "$POSTERIOR_V8_TRUSTED_VERIFIER" \
    verify-extracted \
    --archive "$POSTERIOR_V8_JOB_SOURCE_ARCHIVE" \
    --source-root "$POSTERIOR_V8_JOB_SOURCE_ROOT" \
    --expected-archive-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_SHA256" \
    --expected-manifest-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_MANIFEST_SHA256" \
    --expected-source-tree-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_TREE_SHA256"

POSTERIOR_V8_JOB_REFERENCE="$GISAXS_JOB_CACHE_ROOT/reference.json"
cp -- "$POSTERIOR_V8_V5_CROSS_PLATFORM_REFERENCE" "$POSTERIOR_V8_JOB_REFERENCE"
chmod 0400 "$POSTERIOR_V8_JOB_REFERENCE"
read -r POSTERIOR_V8_JOB_REFERENCE_SHA256 _ < <(sha256sum -- "$POSTERIOR_V8_JOB_REFERENCE")
POSTERIOR_V8_JOB_REFERENCE_BYTE_COUNT="$(wc -c < "$POSTERIOR_V8_JOB_REFERENCE" | tr -d '[:space:]')"
if [[ "$POSTERIOR_V8_JOB_REFERENCE_SHA256" != "$POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_REFERENCE_SHA256" \
    || "$POSTERIOR_V8_JOB_REFERENCE_BYTE_COUNT" != "$POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_REFERENCE_BYTE_COUNT" ]]; then
    echo "cross-platform reference identity changed before worker use" >&2
    exit 2
fi

cd "$POSTERIOR_V8_JOB_SOURCE_ROOT"
run_verified_module() {
    local module="$1"
    shift
    conda run --no-capture-output -n tf \
        python -I -c \
        'import runpy, sys; root, module = sys.argv[1:3]; sys.path.insert(0, root); sys.argv = [module, *sys.argv[3:]]; runpy.run_module(module, run_name="__main__")' \
        "$POSTERIOR_V8_JOB_SOURCE_ROOT" "$module" "$@"
}
run_verified_module \
    utils.ML_Fitting_1D_GISAXS.PosteriorV8.run_sobol_cross_platform_gate_v5 \
    build-and-compare \
    --source-archive-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_SHA256" \
    --source-manifest-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_MANIFEST_SHA256" \
    --source-tree-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_TREE_SHA256" \
    --reference "$POSTERIOR_V8_JOB_REFERENCE" \
    --candidate-output "$POSTERIOR_V8_V5_CROSS_PLATFORM_CANDIDATE_OUTPUT" \
    --result-output "$POSTERIOR_V8_V5_CROSS_PLATFORM_RESULT_OUTPUT" \
    --pass-marker "$POSTERIOR_V8_V5_CROSS_PLATFORM_PASS_MARKER"
chmod 0400 \
    "$POSTERIOR_V8_V5_CROSS_PLATFORM_CANDIDATE_OUTPUT" \
    "$POSTERIOR_V8_V5_CROSS_PLATFORM_RESULT_OUTPUT" \
    "$POSTERIOR_V8_V5_CROSS_PLATFORM_PASS_MARKER"

POSTERIOR_V8_JOB_PASS_MARKER="$GISAXS_JOB_CACHE_ROOT/PASS.json"
cp -- "$POSTERIOR_V8_V5_CROSS_PLATFORM_PASS_MARKER" "$POSTERIOR_V8_JOB_PASS_MARKER"
chmod 0400 "$POSTERIOR_V8_JOB_PASS_MARKER"
run_verified_module \
    utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_a_cross_platform_v5 \
    --marker "$POSTERIOR_V8_JOB_PASS_MARKER" \
    --source-archive-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_SHA256" \
    --source-manifest-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_MANIFEST_SHA256" \
    --source-tree-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_TREE_SHA256" \
    --reference-file-sha256 "$POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_REFERENCE_SHA256" \
    --reference-file-byte-count "$POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_REFERENCE_BYTE_COUNT" \
    --reference-manifest-sha256 "$POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_MANIFEST_SHA256" \
    --scientific-content-sha256 "$POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_SCIENTIFIC_SHA256" \
    --comparison-result-sha256 "$POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_COMPARISON_SHA256" \
    --expected-gate-claim-sha256 "$POSTERIOR_V8_V5_EXPECTED_CROSS_PLATFORM_GATE_CLAIM_SHA256"
