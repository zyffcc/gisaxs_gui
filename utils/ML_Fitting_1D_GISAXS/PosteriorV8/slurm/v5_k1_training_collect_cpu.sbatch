#!/bin/bash
#SBATCH --partition=allcpu
#SBATCH --account=hasylab
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/data/dust/user/zhaiyufe/MaxwellRuns/v5-k1-collect-unbound-%j.out
#SBATCH --error=/data/dust/user/zhaiyufe/MaxwellRuns/v5-k1-collect-unbound-%j.err

set -euo pipefail

: "${SLURM_JOB_ID:?submit K1 tuning handoff through Slurm}"
: "${POSTERIOR_V8_SOURCE_ROOT:?set POSTERIOR_V8_SOURCE_ROOT}"
: "${POSTERIOR_V8_K1_TRAINING_PLAN:?set POSTERIOR_V8_K1_TRAINING_PLAN}"
: "${POSTERIOR_V8_K1_TRAINING_PLAN_SHA256:?set POSTERIOR_V8_K1_TRAINING_PLAN_SHA256}"

case "$(hostname -s)" in
    max-wgs*|max-fs-display*) echo "K1 tuning handoff cannot run on the Maxwell login node" >&2; exit 2 ;;
esac
for dust_path in "$POSTERIOR_V8_SOURCE_ROOT" "$POSTERIOR_V8_K1_TRAINING_PLAN"; do
    case "$dust_path" in
        /data/dust/user/zhaiyufe/*) ;;
        *) echo "K1 tuning source and plan must be under the user dust root" >&2; exit 2 ;;
    esac
done
if [[ ! -f "$POSTERIOR_V8_K1_TRAINING_PLAN" ]]; then
    echo "frozen K1 training plan does not exist" >&2
    exit 2
fi

readonly POSTERIOR_V8_PYTHON="/data/dust/user/zhaiyufe/conda/envs/gisaxs-v5-r2/bin/python"
[[ -x "$POSTERIOR_V8_PYTHON" ]] || { echo "dedicated GISAXS Python is unavailable" >&2; exit 2; }
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export POSTERIOR_V8_SCRATCH_BASE="/data/dust/user/zhaiyufe"
if [[ ! -d "$POSTERIOR_V8_SCRATCH_BASE" || -L "$POSTERIOR_V8_SCRATCH_BASE" ]]; then
    echo "worker scratch base must be a real directory" >&2
    exit 2
fi
POSTERIOR_V8_JOB_TMP_ROOT="$(mktemp -d "$POSTERIOR_V8_SCRATCH_BASE/gisaxs-v5-k1-collect-${SLURM_JOB_ID}-XXXXXXXX")"
chmod 700 "$POSTERIOR_V8_JOB_TMP_ROOT"
export POSTERIOR_V8_JOB_TMP_ROOT
POSTERIOR_V8_WRAPPER_MINT="$POSTERIOR_V8_JOB_TMP_ROOT/.posterior-v8-wrapper-mint"
"$POSTERIOR_V8_PYTHON" -I - "$POSTERIOR_V8_WRAPPER_MINT" <<'PY'
import os
import sys

path = sys.argv[1]
descriptor = os.open(
    path,
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0),
    0o600,
)
with os.fdopen(descriptor, "wb") as stream:
    stream.write(os.urandom(32))
    stream.flush()
    os.fsync(stream.fileno())
    os.fchmod(stream.fileno(), 0o400)
PY
POSTERIOR_V8_JOB_STAGING_ROOT="$POSTERIOR_V8_JOB_TMP_ROOT/staging"
mkdir -m 700 "$POSTERIOR_V8_JOB_STAGING_ROOT"
POSTERIOR_V8_JOB_PLAN="$POSTERIOR_V8_JOB_STAGING_ROOT/k1-training-plan.json"
POSTERIOR_V8_JOB_SOURCE_ARCHIVE="$POSTERIOR_V8_JOB_STAGING_ROOT/source-snapshot.tar"
POSTERIOR_V8_TRUSTED_VERIFIER="$POSTERIOR_V8_JOB_STAGING_ROOT/bootstrap/package_source_snapshot_v5.py"
POSTERIOR_V8_JOB_SOURCE_ROOT="$POSTERIOR_V8_JOB_STAGING_ROOT/source"
POSTERIOR_V8_RUNTIME_CACHE_ROOT="$POSTERIOR_V8_JOB_TMP_ROOT/runtime-cache"
mkdir -m 700 "$POSTERIOR_V8_RUNTIME_CACHE_ROOT"
export TMPDIR="$POSTERIOR_V8_RUNTIME_CACHE_ROOT"
export XDG_CACHE_HOME="$POSTERIOR_V8_RUNTIME_CACHE_ROOT/xdg"
export KERAS_HOME="$POSTERIOR_V8_RUNTIME_CACHE_ROOT/keras"
export MPLCONFIGDIR="$POSTERIOR_V8_RUNTIME_CACHE_ROOT/matplotlib"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

POSTERIOR_V8_FROZEN_SOURCE_BINDING="$("$POSTERIOR_V8_PYTHON" -I - \
    "$POSTERIOR_V8_K1_TRAINING_PLAN" \
    "$POSTERIOR_V8_K1_TRAINING_PLAN_SHA256" \
    "$POSTERIOR_V8_SOURCE_ROOT" \
    "$POSTERIOR_V8_JOB_STAGING_ROOT" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tarfile

DUST = Path("/data/dust/user/zhaiyufe")
PLAN_LIMIT = 32 * 1024 * 1024
VERIFIER = "snapshot/utils/ML_Fitting_1D_GISAXS/PosteriorV8/package_source_snapshot_v5.py"


def no_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("K1 plan contains duplicate JSON fields")
        value[key] = item
    return value


def lexical(path, name, *, exists=True, require_dust=True):
    value = Path(os.path.abspath(path))
    if require_dust:
        try:
            value.relative_to(DUST)
        except ValueError as exc:
            raise ValueError(f"{name} must stay under the user dust root") from exc
    current = Path(value.anchor)
    for part in value.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{name} must not traverse a symlink")
        if not current.exists():
            break
    if exists and not value.exists():
        raise FileNotFoundError(f"{name} does not exist")
    return value


def regular_bytes(path, name, limit=None, require_dust=True):
    source = lexical(path, name, require_dust=require_dust)
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError(f"{name} must be a regular file")
        data = stream.read() if limit is None else stream.read(limit + 1)
    if limit is not None and len(data) > limit:
        raise ValueError(f"{name} is unexpectedly large")
    return data


def checked_plan(path, expected, require_dust=True):
    raw = regular_bytes(
        path, "frozen K1 plan", PLAN_LIMIT, require_dust=require_dust
    )
    value = json.loads(raw, object_pairs_hook=no_duplicates)
    if not isinstance(value, dict):
        raise ValueError("K1 plan must contain one JSON object")
    core = dict(value)
    supplied = core.pop("plan_sha256", None)
    observed = hashlib.sha256(
        json.dumps(core, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    if supplied != expected or observed != expected or len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise ValueError("K1 plan does not match its Slurm SHA-256 binding")
    return value, raw


def write_exclusive(path, content):
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
        os.fchmod(stream.fileno(), 0o400)


original_plan = lexical(Path(sys.argv[1]), "frozen K1 plan")
expected_plan_sha256 = sys.argv[2]
exported_source = lexical(Path(sys.argv[3]), "frozen source root")
staging_root = lexical(
    Path(sys.argv[4]), "job staging root", require_dust=False
)
plan, plan_bytes = checked_plan(original_plan, expected_plan_sha256)
if plan["layout"]["plan"] != str(original_plan):
    raise ValueError("Slurm plan path differs from the authenticated plan")
if plan["source"]["root"] != str(exported_source):
    raise ValueError("Slurm source root differs from the authenticated plan")
lexical(Path(plan["layout"]["run_root"]), "K1 run root")
scratch = lexical(
    Path(os.environ["POSTERIOR_V8_SCRATCH_BASE"]),
    "worker scratch base",
    require_dust=True,
)
job_tmp = lexical(
    Path(os.environ["POSTERIOR_V8_JOB_TMP_ROOT"]),
    "wrapper job-private temporary root",
    require_dust=False,
)
job_tmp_status = job_tmp.stat()
expected_prefix = f"gisaxs-v5-k1-collect-{os.environ['SLURM_JOB_ID']}-"
if (
    not job_tmp.is_dir()
    or job_tmp.is_symlink()
    or job_tmp.parent != scratch
    or re.fullmatch(re.escape(expected_prefix) + r"[A-Za-z0-9]{8}", job_tmp.name) is None
    or job_tmp_status.st_uid != os.getuid()
    or stat.S_IMODE(job_tmp_status.st_mode) != 0o700
):
    raise ValueError("wrapper job-private temporary root is not fresh owner-private scratch")
if {path.name for path in job_tmp.iterdir()} != {
    "staging", "runtime-cache", ".posterior-v8-wrapper-mint"
}:
    raise ValueError("wrapper job-private temporary root has unexpected prior content")
mint_status = (job_tmp / ".posterior-v8-wrapper-mint").lstat()
if (
    not stat.S_ISREG(mint_status.st_mode)
    or mint_status.st_uid != os.getuid()
    or stat.S_IMODE(mint_status.st_mode) != 0o400
    or mint_status.st_nlink != 1
    or mint_status.st_size != 32
):
    raise ValueError("wrapper one-shot mint is not fresh private state")
runtime_cache = job_tmp / "runtime-cache"
cache_status = runtime_cache.lstat()
if (
    not stat.S_ISDIR(cache_status.st_mode)
    or runtime_cache.is_symlink()
    or cache_status.st_uid != os.getuid()
    or stat.S_IMODE(cache_status.st_mode) != 0o700
):
    raise ValueError("wrapper runtime cache root is not owner-private")
status = staging_root.lstat()
if (
    staging_root != job_tmp / "staging"
    or not stat.S_ISDIR(status.st_mode)
    or status.st_uid != os.getuid()
    or stat.S_IMODE(status.st_mode) != 0o700
    or any(staging_root.iterdir())
):
    raise ValueError("job staging root is not fresh owner-private Slurm scratch")
(staging_root / "bootstrap").mkdir(mode=0o700)

local_plan = staging_root / "k1-training-plan.json"
write_exclusive(local_plan, plan_bytes)
local_plan_value, _ = checked_plan(
    local_plan, expected_plan_sha256, require_dust=False
)
if local_plan_value != plan:
    raise RuntimeError("K1 plan changed while creating the job-private copy")

archive_sha256 = plan["source"]["archive_sha256"]
archive_path = lexical(Path(plan["source"]["archive_path"]), "frozen source archive")
archive_bytes = regular_bytes(archive_path, "frozen source archive")
if hashlib.sha256(archive_bytes).hexdigest() != archive_sha256:
    raise RuntimeError("source archive changed before job-private staging")
local_archive = staging_root / "source-snapshot.tar"
write_exclusive(local_archive, archive_bytes)
if hashlib.sha256(
    regular_bytes(local_archive, "job-local source archive", require_dust=False)
).hexdigest() != archive_sha256:
    raise RuntimeError("job-local source archive rehash failed")

with tarfile.open(local_archive, "r:*") as archive:
    matches = [member for member in archive.getmembers() if member.name == VERIFIER]
    if len(matches) != 1 or not matches[0].isfile():
        raise ValueError("source archive has no unique regular trusted verifier")
    stream = archive.extractfile(matches[0])
    if stream is None:
        raise ValueError("trusted verifier is unreadable")
    verifier_bytes = stream.read()
expected_verifier = plan["source"]["required_file_sha256"][VERIFIER.removeprefix("snapshot/")]
if hashlib.sha256(verifier_bytes).hexdigest() != expected_verifier:
    raise RuntimeError("trusted verifier differs from the authenticated source bundle")
write_exclusive(staging_root / "bootstrap" / "package_source_snapshot_v5.py", verifier_bytes)
binding = plan["source"]["archive_tree_binding"]
print(archive_sha256, binding["manifest_sha256"], binding["source_tree_sha256"])
PY
)"
read -r POSTERIOR_V8_FROZEN_ARCHIVE_SHA256 POSTERIOR_V8_FROZEN_MANIFEST_SHA256 POSTERIOR_V8_FROZEN_SOURCE_TREE_SHA256 POSTERIOR_V8_FROZEN_EXTRA <<< "$POSTERIOR_V8_FROZEN_SOURCE_BINDING"
if [[ -n "$POSTERIOR_V8_FROZEN_EXTRA" || ! "$POSTERIOR_V8_FROZEN_ARCHIVE_SHA256" =~ ^[0-9a-f]{64}$ || ! "$POSTERIOR_V8_FROZEN_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ || ! "$POSTERIOR_V8_FROZEN_SOURCE_TREE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "authenticated source binding is malformed" >&2
    exit 2
fi
readonly POSTERIOR_V8_FROZEN_ARCHIVE_SHA256 POSTERIOR_V8_FROZEN_MANIFEST_SHA256 POSTERIOR_V8_FROZEN_SOURCE_TREE_SHA256
"$POSTERIOR_V8_PYTHON" -I "$POSTERIOR_V8_TRUSTED_VERIFIER" \
    extract \
    --archive "$POSTERIOR_V8_JOB_SOURCE_ARCHIVE" \
    --source-root "$POSTERIOR_V8_JOB_SOURCE_ROOT" \
    --expected-sha256 "$POSTERIOR_V8_FROZEN_ARCHIVE_SHA256"
"$POSTERIOR_V8_PYTHON" -I "$POSTERIOR_V8_TRUSTED_VERIFIER" \
    verify-extracted \
    --archive "$POSTERIOR_V8_JOB_SOURCE_ARCHIVE" \
    --source-root "$POSTERIOR_V8_JOB_SOURCE_ROOT" \
    --expected-archive-sha256 "$POSTERIOR_V8_FROZEN_ARCHIVE_SHA256" \
    --expected-manifest-sha256 "$POSTERIOR_V8_FROZEN_MANIFEST_SHA256" \
    --expected-source-tree-sha256 "$POSTERIOR_V8_FROZEN_SOURCE_TREE_SHA256"

cd "$POSTERIOR_V8_JOB_SOURCE_ROOT"

module purge
source /etc/profile.d/modules.sh
module load maxwell

"$POSTERIOR_V8_PYTHON" -I -c 'import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); runpy.run_module("utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_training_chain_runtime_v5", run_name="__main__", alter_sys=True)' \
    "$POSTERIOR_V8_JOB_SOURCE_ROOT" \
    collect \
    --plan "$POSTERIOR_V8_JOB_PLAN" \
    --expected-plan-sha256 "$POSTERIOR_V8_K1_TRAINING_PLAN_SHA256" \
    --job-staging-root "$POSTERIOR_V8_JOB_STAGING_ROOT" \
    --job-source-root "$POSTERIOR_V8_JOB_SOURCE_ROOT" \
    --job-source-archive "$POSTERIOR_V8_JOB_SOURCE_ARCHIVE"
