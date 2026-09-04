#!/bin/bash
#SBATCH --partition=allcpu
#SBATCH --account=hasylab
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

: "${SLURM_JOB_ID:?PosteriorV8 regression must run through Slurm}"
: "${POSTERIOR_V8_SOURCE_ROOT:?set POSTERIOR_V8_SOURCE_ROOT}"
: "${POSTERIOR_V8_V5_SOURCE_ARCHIVE:?set POSTERIOR_V8_V5_SOURCE_ARCHIVE}"
: "${POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_SHA256:?set expected archive SHA-256}"
: "${POSTERIOR_V8_V5_EXPECTED_SOURCE_MANIFEST_SHA256:?set expected manifest SHA-256}"
: "${POSTERIOR_V8_V5_EXPECTED_SOURCE_TREE_SHA256:?set expected source-tree SHA-256}"
: "${POSTERIOR_V8_V7_PHASE_A_PLAN:?set Phase-A v7 plan}"
: "${POSTERIOR_V8_V7_PHASE_A_PLAN_SHA256:?set Phase-A v7 plan SHA-256}"
: "${POSTERIOR_V8_V7_PHASE_A_RECEIPT:?set Phase-A v7 submission receipt}"
: "${POSTERIOR_V8_V7_PHASE_A_RELEASE:?set Phase-A v7 release completion}"
: "${POSTERIOR_V8_V7_PHASE_A_STAGE:?set Phase-A v7 stage}"
: "${POSTERIOR_V8_V7_PHASE_A_UPSTREAM_COMPLETION:?set upstream completion}"
: "${POSTERIOR_V8_V7_PHASE_A_COMPLETION:?set stage completion output}"

case "$(hostname -s)" in
    max-wgs*) echo "PosteriorV8 regression must run on a Slurm worker" >&2; exit 2 ;;
esac
for dust_path in "$POSTERIOR_V8_SOURCE_ROOT" "$POSTERIOR_V8_V5_SOURCE_ARCHIVE"; do
    case "$dust_path" in
        /data/dust/user/zhaiyufe/*) ;;
        *) echo "PosteriorV8 source inputs must be under /data/dust/user/zhaiyufe/" >&2; exit 2 ;;
    esac
done
if [[ ! -d "$POSTERIOR_V8_SOURCE_ROOT" || -L "$POSTERIOR_V8_SOURCE_ROOT" ]]; then
    echo "PosteriorV8 source root must be a real directory" >&2
    exit 2
fi
if [[ ! -f "$POSTERIOR_V8_V5_SOURCE_ARCHIVE" || -L "$POSTERIOR_V8_V5_SOURCE_ARCHIVE" ]]; then
    echo "PosteriorV8 source archive must be a real file" >&2
    exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
unset PYTHONPATH
GISAXS_JOB_TMP_BASE="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
if [[ ! -d "$GISAXS_JOB_TMP_BASE" || -L "$GISAXS_JOB_TMP_BASE" ]]; then
    echo "regression scratch parent must be a real directory" >&2
    exit 2
fi
GISAXS_JOB_TMP_BASE="$(realpath -e -- "$GISAXS_JOB_TMP_BASE")"
case "$GISAXS_JOB_TMP_BASE" in
    /data/dust/user/zhaiyufe|/data/dust/user/zhaiyufe/*)
        echo "regression scratch must be node-local, not dust" >&2
        exit 2
        ;;
esac
GISAXS_JOB_CACHE_ROOT="$(mktemp -d "$GISAXS_JOB_TMP_BASE/gisaxs-v5-regression-${SLURM_JOB_ID}-XXXXXXXX")"
chmod 0700 "$GISAXS_JOB_CACHE_ROOT"
if [[ ! -d "$GISAXS_JOB_CACHE_ROOT" || -L "$GISAXS_JOB_CACHE_ROOT" \
    || ! -O "$GISAXS_JOB_CACHE_ROOT" \
    || "$(stat -c '%a' -- "$GISAXS_JOB_CACHE_ROOT")" != "700" ]]; then
    echo "regression scratch is not exclusive owner-private storage" >&2
    exit 2
fi
export XDG_CACHE_HOME="$GISAXS_JOB_CACHE_ROOT/xdg"
export KERAS_HOME="$GISAXS_JOB_CACHE_ROOT/keras"
export CUDA_CACHE_PATH="$GISAXS_JOB_CACHE_ROOT/cuda"
export MPLCONFIGDIR="$GISAXS_JOB_CACHE_ROOT/matplotlib"
export OMP_NUM_THREADS=1
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
POSTERIOR_V8_JOB_STAGING_ROOT="$GISAXS_JOB_CACHE_ROOT/staging"
mkdir -m 0700 "$POSTERIOR_V8_JOB_STAGING_ROOT"
export POSTERIOR_V8_JOB_SOURCE_ARCHIVE POSTERIOR_V8_JOB_SOURCE_ROOT POSTERIOR_V8_JOB_STAGING_ROOT
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
conda run --no-capture-output --prefix /data/dust/user/zhaiyufe/conda/envs/gisaxs-v5-r2 \
    python -I "$POSTERIOR_V8_TRUSTED_VERIFIER" \
    extract \
    --archive "$POSTERIOR_V8_JOB_SOURCE_ARCHIVE" \
    --source-root "$POSTERIOR_V8_JOB_SOURCE_ROOT" \
    --expected-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_SHA256"
conda run --no-capture-output --prefix /data/dust/user/zhaiyufe/conda/envs/gisaxs-v5-r2 \
    python -I "$POSTERIOR_V8_TRUSTED_VERIFIER" \
    verify-extracted \
    --archive "$POSTERIOR_V8_JOB_SOURCE_ARCHIVE" \
    --source-root "$POSTERIOR_V8_JOB_SOURCE_ROOT" \
    --expected-archive-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_SHA256" \
    --expected-manifest-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_MANIFEST_SHA256" \
    --expected-source-tree-sha256 "$POSTERIOR_V8_V5_EXPECTED_SOURCE_TREE_SHA256"

cd "$POSTERIOR_V8_JOB_SOURCE_ROOT"
conda run --no-capture-output --prefix /data/dust/user/zhaiyufe/conda/envs/gisaxs-v5-r2 \
    python -I -c \
    'import sys; root=sys.argv[1]; sys.path.insert(0, root); from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_a_runtime_v7 import run_stage_from_environment; run_stage_from_environment(sys.argv[2])' \
    "$POSTERIOR_V8_JOB_SOURCE_ROOT" "$POSTERIOR_V8_V7_PHASE_A_STAGE"
