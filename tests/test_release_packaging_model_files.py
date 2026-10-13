"""A checkout must get the bundled 1D Predict model byte for byte.

1D Predict checks every file of its model against a SHA-256 manifest before it runs
(``conditional_v2/conditional_fast_manifest_v2.json``, ``stable_blue_rc_v1/MANIFEST.json``).  The
checksums were taken from the files on disk, where some lines end with CRLF and others with LF.  Git
used to store these files with LF and to convert them again on checkout, so on every clone dozens of
files failed their checksum and 1D Predict could not start.  ``.gitattributes`` now switches the
conversion off for that folder, and the files are stored as they are on disk.

The weights (``*.h5``, ``*.npz``) are tracked too, although ``.gitignore`` ignores those endings in
general: they were added with ``git add -f``, so a clone can run 1D Predict. A model file that is
missing, in git or on disk, is a failure.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
V5 = "modules/Fitting_1D_Model/Workflow_v5"
MANIFESTS = (
    f"{V5}/conditional_v2/conditional_fast_manifest_v2.json",
    f"{V5}/stable_blue_rc_v1/MANIFEST.json",
)
RENORMALIZE = f"git add --renormalize {V5}"
RESTORE = f"git checkout -- {V5}"

needs_git = pytest.mark.skipif(
    not shutil.which("git") or not (ROOT / ".git").exists(),
    reason="not a git checkout (or git is not installed)",
)


def _checksums(manifest: str) -> dict[str, str]:
    """``{path from the repository root: sha256}`` of one model manifest."""
    folder = manifest.rsplit("/", 1)[0]
    files = json.loads((ROOT / manifest).read_text(encoding="utf-8"))["files"]
    return {f"{folder}/{name}": checksum for name, checksum in files.items()}


def _git(*arguments: str, data: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, input=data, capture_output=True, check=True, timeout=120
    ).stdout


def _stored_by_git(folder: str) -> dict[str, bytes]:
    """The bytes git holds (in its index) for each tracked file of ``folder``: what a clone gets."""
    names = {}
    for record in _git("ls-files", "--stage", "-z", "--", folder).split(b"\0"):
        if record:
            entry, path = record.split(b"\t", 1)
            names[path.decode("utf-8")] = entry.split()[1]
    output = _git("cat-file", "--batch", data=b"\n".join(names.values()) + b"\n")
    stored, position = {}, 0
    for path in names:  # one "<id> blob <size>\n<bytes>\n" record per id, in the order asked
        end = output.index(b"\n", position)
        size = int(output[position:end].split()[2])
        stored[path] = output[end + 1:end + 1 + size]
        position = end + 1 + size + 1
    return stored


@needs_git
def test_git_does_not_convert_line_endings_in_the_model_folder():
    for name in (f"{V5}/conditional_v2/predict.py", f"{V5}/stable_blue_rc_v1/protocol.json",
                 f"{V5}/conditional_v2/model_bundle_precision_r2/source/predict_1d.py"):
        answer = _git("check-attr", "text", "--", name).decode("utf-8").strip()
        assert answer.endswith("text: unset"), answer


@needs_git
def test_git_stores_the_model_files_the_manifests_were_made_from():
    """What a clone gets is what the manifests were made from, the weights included.

    This reads what git holds, not the files on disk: it fails until the files have been added
    again without conversion (once, after ``.gitattributes`` was written), and it would fail again
    if they were ever stored converted. A file a manifest names and git does not hold would be
    missing on every clone (a new weight file needs ``git add -f``: ``.gitignore`` ignores its ending).
    """
    stored = _stored_by_git(V5)
    assert stored, "git tracks no file of the bundled model"
    untracked = [path for manifest in MANIFESTS for path in _checksums(manifest) if path not in stored]
    assert not untracked, (
        f"git does not hold {len(untracked)} model files the manifests name, so 1D Predict cannot start "
        f"on a clone. Add them as they are on disk (git add -f): " + ", ".join(untracked)
    )
    different = [
        path
        for manifest in MANIFESTS
        for path, checksum in _checksums(manifest).items()
        if path in stored and hashlib.sha256(stored[path]).hexdigest() != checksum
    ]
    assert not different, (
        f"git stores {len(different)} model files with other line endings than the manifests were "
        f"made from, so 1D Predict cannot start on a clone. Stage them as they are on disk: "
        f"{RENORMALIZE}  -- " + ", ".join(different)
    )


def test_the_single_rc_model_matches_its_manifest():
    """As ``test_imported_assets_remain_identical`` does for the general model."""
    checksums = _checksums(MANIFESTS[1])
    absent = [path for path in checksums if not (ROOT / path).is_file()]
    assert not absent, (
        f"{len(absent)} model files are missing from this folder, so 1D Predict cannot run here. "
        f"git holds them ({RESTORE} brings them back): " + ", ".join(absent)
    )
    for path, checksum in checksums.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == checksum, path
