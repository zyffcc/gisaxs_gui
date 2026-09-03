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
GISAXS_JOB_CACHE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/gisaxs-v5-regression-${SLURM_JOB_ID}-XXXXXXXX")"
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
# The wrapper has already proved that it is running on a Slurm worker.  The
# suite itself exercises login-node launchers and local temporary paths, so it
# must not inherit the production-job sentinel and mistake those unit-test
# fixtures for live Maxwell operations.  Individual tests can still set the
# sentinel explicitly with monkeypatch when they exercise worker-only guards.
env -u SLURM_JOB_ID conda run --no-capture-output -n tf \
    python -I -c \
    'from pytest import console_main; import sys; root = sys.argv[1]; sys.path.insert(0, root); sys.argv = ["pytest", *sys.argv[2:]]; raise SystemExit(console_main())' \
    "$POSTERIOR_V8_JOB_SOURCE_ROOT" -q \
    utils/ML_Fitting_1D_GISAXS/tests/test_posterior_v8_*.py \
    --disable-warnings \
    -p no:cacheprovider
