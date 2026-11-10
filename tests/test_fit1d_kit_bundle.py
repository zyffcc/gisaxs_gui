"""The bundled fit1d kit (``utils/gisaxs1d_gui_kit``): byte-exact, complete, and free of private material.

Fitting's Predict parameters (fit1d) and the method fit1d of 1D Predict import this kit inside a job process
(``fitting/infrastructure/adapters/fit1d_method.py``). ``FILES_SHA256.json`` holds the SHA-256 of every file and
the release builder checks the runtime files against it, so git must hand out the files unchanged
(``.gitattributes``: ``-text``). ``SOURCE.json`` says where the kit came from, what was left out and what was
changed. These checks read files only; the fits themselves are tested in ``test_fit1d_method.py``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path, PurePosixPath

import pytest

ROOT = Path(__file__).resolve().parents[1]
KIT = "utils/gisaxs1d_gui_kit"
KIT_DIR = ROOT / KIT
MANIFEST = "FILES_SHA256.json"
DOCUMENTATION = {"README.md"}
"""In the manifest, but not runtime files: the release package leaves them out."""
REAL_DATA_MODULES = {"__init__.py", "symmetry.py", "yoneda_cut.py"}
"""The only modules of ``app/research/real_data`` the fit imports; the others are validation scripts on measured
data and stay out."""
PRIVATE_TOKENS = ROOT / "docs" / "agents" / "eval" / "local" / "private_tokens.txt"
"""Names of private datasets, people and accounts, one per line, in the git-ignored local folder (AGENTS.md);
absent on a clone, where only the generic checks run."""
ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]{1,2}[A-Za-z_]|(?<![\w.~])/(?:home|data|Users|scratch)/")
TEXT_SUFFIXES = {".py", ".json", ".md", ".txt", ".csv"}

needs_git = pytest.mark.skipif(
    not shutil.which("git") or not (ROOT / ".git").exists(),
    reason="not a git checkout (or git is not installed)",
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest() -> dict[str, str]:
    record = json.loads((KIT_DIR / MANIFEST).read_text(encoding="utf-8"))
    assert record["schema"] == "gisaxs1d_kit_files_v1"
    return record["files"]


def _kit_files() -> list[Path]:
    return sorted(path for path in KIT_DIR.rglob("*")
                  if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"})


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
    if not names:
        return {}
    output = _git("cat-file", "--batch", data=b"\n".join(names.values()) + b"\n")
    stored, position = {}, 0
    for path in names:
        end = output.index(b"\n", position)
        size = int(output[position:end].split()[2])
        stored[path] = output[end + 1:end + 1 + size]
        position = end + 1 + size + 1
    return stored


def _release_lister():
    spec = importlib.util.spec_from_file_location("list_fit1d_release_files", ROOT / "tools" / "list_fit1d_release_files.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------------------------- the bytes --
def test_the_kit_matches_its_manifest():
    manifest = _manifest()
    on_disk = {path.relative_to(KIT_DIR).as_posix(): path for path in _kit_files()}
    absent = [name for name in manifest if name not in on_disk]
    assert not absent, "kit files missing on disk: " + ", ".join(absent)
    changed = [name for name, digest in manifest.items() if _sha(on_disk[name].read_bytes()) != digest]
    assert not changed, f"kit files differ from {MANIFEST}: " + ", ".join(changed)
    unlisted = sorted(set(on_disk) - set(manifest) - {MANIFEST})
    assert not unlisted, f"kit files not in {MANIFEST}: " + ", ".join(unlisted)


@needs_git
def test_git_tracks_the_kit_without_converting_line_endings():
    for name in (f"{KIT}/gisaxs1d_kit/core.py", f"{KIT}/models/models.json", f"{KIT}/{MANIFEST}"):
        answer = _git("check-attr", "text", "--", name).decode("utf-8").strip()
        assert answer.endswith("text: unset"), answer
    # research/ and *.npz are ignored elsewhere in the repository; the kit's own must be tracked
    for name in (f"{KIT}/app/research/fit1d/api.py", f"{KIT}/models/general/weights_numpy.npz"):
        ignored = subprocess.run(["git", "check-ignore", "-q", name], cwd=ROOT, timeout=60)
        assert ignored.returncode == 1, f"git ignores {name}; it must be tracked"


@needs_git
def test_git_stores_the_kit_files_the_manifest_was_made_from():
    stored = _stored_by_git(KIT)
    if not stored:
        pytest.skip("the kit is not in git's index yet")
    manifest = {f"{KIT}/{name}": digest for name, digest in _manifest().items()}
    missing = [path for path in manifest if path not in stored]
    assert not missing, "git does not hold these kit files (add them as they are on disk): " + ", ".join(missing)
    different = [path for path, digest in manifest.items() if _sha(stored[path]) != digest]
    assert not different, f"git stores these kit files with other bytes than {MANIFEST}: " + ", ".join(different)
    extra = sorted(set(stored) - set(manifest) - {f"{KIT}/{MANIFEST}"})
    assert not extra, f"tracked kit files missing from {MANIFEST}: " + ", ".join(extra)


def test_the_text_files_end_their_lines_with_lf():
    crlf = [path.relative_to(KIT_DIR).as_posix() for path in _kit_files()
            if path.suffix in TEXT_SUFFIXES and b"\r" in path.read_bytes()]
    assert not crlf, "kit text files with CR line endings: " + ", ".join(crlf)


# ---------------------------------------------------------------------------------- what is in it --
def test_the_release_builder_accepts_every_runtime_file():
    """The builder's own check (``tools/list_fit1d_release_files.py``) passes and keeps all but the two notes."""
    result = _release_lister().release_files(ROOT)
    shipped = {name[len(KIT) + 1:] for name in result["files"]}
    assert shipped == (set(_manifest()) - DOCUMENTATION) | {MANIFEST}
    assert result["manifest"]["files"] == {name: digest for name, digest in _manifest().items()
                                           if name not in DOCUMENTATION}


def test_the_networks_and_tables_are_those_of_the_source():
    source = json.loads((KIT_DIR / "SOURCE.json").read_text(encoding="utf-8"))
    manifest = _manifest()
    unchanged = {**source["networks_unchanged"], **source["tables_unchanged"]}
    assert set(unchanged) == {name for name in manifest if name.endswith((".npz", ".fft"))}
    assert all(manifest[name] == digest for name, digest in unchanged.items())
    catalogue = json.loads((KIT_DIR / "models" / "models.json").read_text(encoding="utf-8"))
    assert catalogue["default"] == "general" and set(catalogue["models"]) == {"general", "au_np_si"}


def test_the_source_record_names_its_origin_what_was_left_out_and_its_patches():
    source = json.loads((KIT_DIR / "SOURCE.json").read_text(encoding="utf-8"))
    assert source["source"] and source["source_commit"] and source["not_bundled"]
    for patch in source["local_patches"]:
        assert patch["files"] and patch["change"] and patch["reason"]
        assert all((KIT_DIR / name).is_file() for name in patch["files"]), patch["files"]
        assert set(patch["files"]) <= set(source["upstream_sha256_of_patched_files"]) | set(patch.get("added_files", []))


def test_no_measurements_selftests_or_research_notes_are_bundled():
    """Example measurements, results on them and notes that cite them stay out of the repository."""
    names = [PurePosixPath(name) for name in _manifest()]
    assert not [n for n in names if n.parts[0] in {"examples", "gimap"}]
    assert not [n for n in names if n.name.startswith(("selftest", "expected", "timing"))]
    assert not [n for n in names if n.suffix == ".md" and n.as_posix() != "README.md"]
    assert not [n for n in names if n.suffix in {".npz", ".npy", ".csv", ".txt", ".cbf"}
                and n.parts[0] != "models"]
    real_data = {n.name for n in names if n.parts[:3] == ("app", "research", "real_data")}
    assert real_data == REAL_DATA_MODULES


# ------------------------------------------------------------------------------------ privacy --
def _private_tokens() -> set[str]:
    if not PRIVATE_TOKENS.is_file():
        return set()
    lines = PRIVATE_TOKENS.read_text(encoding="utf-8-sig").splitlines()
    return {line.split("#", 1)[0].strip().lower() for line in lines} - {""}


def test_no_absolute_paths_in_the_kit():
    found = []
    for path in _kit_files():
        if path.suffix in TEXT_SUFFIXES:
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if ABSOLUTE_PATH.search(line):
                    found.append(f"{path.relative_to(KIT_DIR).as_posix()}:{number}")
    assert not found, "absolute paths in the kit:\n" + "\n".join(found)


def test_no_private_names_in_the_kit():
    private = _private_tokens()
    if not private:
        pytest.skip("no local list of private names (docs/agents/eval/local/private_tokens.txt; absent on a clone)")
    found = []
    for path in [p for p in _kit_files() if p.suffix in TEXT_SUFFIXES]:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if private & set(re.findall(r"[a-z0-9]+", line.lower())):
                found.append(f"{path.relative_to(ROOT).as_posix()}:{number}")  # the line, never the name
    assert not found, "a private dataset, person or account is named here:\n" + "\n".join(found)
