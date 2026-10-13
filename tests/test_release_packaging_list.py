"""The portable release packs an allow list, never the developer's whole folder.

``tools/build_portable_release.ps1 -ListOnly`` prints the application files: what git tracks plus
the model files 1D Predict needs that git ignores.  A developer's folder also holds personal
settings, the instrument geometry of their own beamtime, reports, research folders, notebooks and
caches; GIMaP imports ``<program>/config/*.json`` into a new user's data folder on the first start,
so a shipped profile would be applied to that person's frames.  The builder once packed everything
but a short list of folders, and shipped them.

The rules are tried on a small repository made in a temporary folder (git never writes to this
one), then the list of this repository is read.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "build_portable_release.ps1"
V5 = "modules/Fitting_1D_Model/Workflow_v5"

pytestmark = pytest.mark.skipif(
    sys.platform != "win32" or not shutil.which("powershell") or not shutil.which("git"),
    reason="the release builder is a Windows PowerShell script that asks git for the file list",
)

# What must never be in a release, written a second time on purpose (not read from the script).
PERSONAL = re.compile(
    r"^config/user_[^/]*\.json$|^config/instrument_profiles\.json$"
    r"|(^|/)(_handoff|research|validation|development)/"
    r"|\.ipynb$|\.py[co]$|(^|/)(__pycache__|\.pytest_cache|\.ruff_cache|\.gimap_cache)/"
    r"|(^|/)(gimap_analysis|AI_Fitting_Output|trainset_jobs|test_output|test_results)/"
    r"|^(runtime|release|work)/",
    re.IGNORECASE,
)

TRACKED = {
    ".gitignore": (
        "config/user_parameters.json\nconfig/user_settings.json\nconfig/instrument_profiles.json\n"
        "research/\nvalidation/\nresults/\n*.h5\n*.npz\n__pycache__/\ngimap_analysis/\ntest.ipynb\n"
        f"{V5}/development/\n"
    ),
    "main.py": "print('GIMaP')\n",
    "Start_GIMaP.bat": "@echo off\n",
    "src/gimap/app.py": "APP = 'GIMaP'\n",
    "src/gimap/shared/version.py": '__version__ = "9.8.7"\n',
    "config/detectors.json": "{}\n",
    "docs/User_Manual.md": "# Manual\n",
    f"{V5}/conditional_v2/predict.py": "print('predict')\r\n",
    f"{V5}/stable_blue_rc_v1/protocol.json": "{}\r\n",
}
# Tracked, but working papers: git knows them, the package leaves them out.
WORKING_PAPERS = {
    "docs/research/outline.md": "working paper\n",
    "work/display.json": "[]\n",
    "GISAXS_HANDOFF_20260909.md": "hand-over note\n",
}
# On disk only: git ignores them, or was never told about them (``_handoff``).
LOCAL = {
    "config/user_parameters.json": '{"Cut": {"center_x": 1.0}}\n',
    "config/user_settings.json": '{"language": "zh"}\n',
    "config/instrument_profiles.json": '{"profiles": [{"name": "my beamtime"}]}\n',
    "research/notes.md": "unpublished\n",
    "validation/run.json": "{}\n",
    "test.ipynb": "{}\n",
    "src/gimap/__pycache__/app.cpython-310.pyc": "cache",
    "TestSAXSdata/gimap_analysis/frame_fit_input.dat": "# output\n",
    f"{V5}/development/evidence/dataset.npz": "development record",
    "_handoff/report.md": "personal report\n",
}
# Weights as a new one is until `git add -f`: ignored by git (*.h5, *.npz, results/), named by the
# manifests, needed to run. (The weights of this repository are tracked; the builder packs both kinds.)
MODEL = {
    f"{V5}/conditional_v2/results/r1/final.weights.h5": "condition weights",
    f"{V5}/conditional_v2/bundle/example_input.npz": "example input",
    f"{V5}/stable_blue_rc_v1/network.npz": "network",
}
MANIFESTS = {
    f"{V5}/conditional_v2/conditional_fast_manifest_v2.json": (
        "predict.py", "results/r1/final.weights.h5", "bundle/example_input.npz"),
    f"{V5}/stable_blue_rc_v1/MANIFEST.json": ("protocol.json", "network.npz"),
}
PACKED = set(TRACKED) | set(MODEL) | set(MANIFESTS) | {"tools/build_portable_release.ps1"}


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-c", "core.autocrlf=false", "-c", "core.safecrlf=false", *arguments],
        cwd=root, check=True, capture_output=True, timeout=120)


def _builder(root: Path, *arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
         str(root / "tools" / "build_portable_release.ps1"), *arguments],
        cwd=root, capture_output=True, timeout=300, encoding="utf-8", errors="replace")


def _list(root: Path, *arguments: str):
    """``-ListOnly`` in ``root``: ``(exit code, listed paths, the summary and problems)``."""
    done = _builder(root, "-ListOnly", *arguments)
    listed = [line.strip() for line in done.stdout.splitlines() if line.strip()]
    return done.returncode, listed, done.stderr


@pytest.fixture
def developer_folder(tmp_path):
    """A small git repository that looks like a developer's folder, personal files included."""
    root = tmp_path / "gimap"
    for name, text in {**TRACKED, **WORKING_PAPERS, **LOCAL, **MODEL}.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
    for name, files in MANIFESTS.items():
        folder = (root / name).parent
        checksums = {
            file: hashlib.sha256((folder / file).read_bytes()).hexdigest() for file in files
        }
        (root / name).write_text(json.dumps({"files": checksums}), encoding="utf-8")
    (root / "tools").mkdir()
    shutil.copy2(SCRIPT, root / "tools" / SCRIPT.name)
    _git(root, "init", "--quiet")
    # ``git add`` with paths: the ignored files stay out, and so does the untracked ``_handoff``.
    _git(root, "add", "--", ".gitignore", "main.py", "Start_GIMaP.bat", "src", "config", "docs",
         "work", "GISAXS_HANDOFF_20260909.md", "modules", "tools")
    # As in this repository: tracked although ``research/`` is ignored.
    _git(root, "add", "--force", "--", "docs/research/outline.md")
    return root


def test_the_list_is_what_git_tracks_plus_the_model_files(developer_folder):
    code, listed, said = _list(developer_folder)
    assert code == 0, said
    assert set(listed) == PACKED
    assert len(listed) == len(set(listed))
    assert not [path for path in listed if PERSONAL.search(path)]
    assert "model files git ignores: 3; tracked but left out (working papers): 3" in said
    # The version is the program's own, so the package cannot be named after another release.
    assert "GIMaP-9.8.7-windows-x64-portable" in said
    assert not (developer_folder / "release").exists()  # nothing is written


def test_the_package_holds_the_list_and_nothing_else(developer_folder, tmp_path, monkeypatch):
    """The whole build on the small repository, with stand-ins for ``conda`` and ``conda-pack``.

    The stand-in "environment" is one file, so nothing large is packed; what is tested is which
    files of the folder go into the ZIP beside it.
    """
    import zipfile

    stand_ins = tmp_path / "stand_ins"
    stand_ins.mkdir()
    conda_pack = stand_ins / "conda_pack_stand_in.py"
    conda_pack.write_text(
        "import sys, zipfile\n"
        "arguments = sys.argv[1:]\n"
        "if '-o' in arguments:\n"
        "    archive = arguments[arguments.index('-o') + 1]\n"
        "    runtime = arguments[arguments.index('--arcroot') + 1]\n"
        "    with zipfile.ZipFile(archive, 'w') as packed:\n"
        "        packed.writestr(runtime + '/pythonw.exe', 'python')\n",
        encoding="utf-8")
    (stand_ins / "python.cmd").write_text(f'@"{sys.executable}" "{conda_pack}" %*\r\n')
    (stand_ins / "conda.cmd").write_text('@echo {"envs": ["C:/stand-in/gimap-stand-in"]}\r\n')
    monkeypatch.setenv("PATH", str(stand_ins), prepend=";")

    # Only the stand-in conda knows this environment: a real one would stop the script here.
    build = _builder(developer_folder, "-EnvironmentName", "gimap-stand-in")
    assert build.returncode == 0, build.stdout + build.stderr
    package = "GIMaP-9.8.7-windows-x64-portable"
    with zipfile.ZipFile(developer_folder / "release" / f"{package}.zip") as archive:
        names = {name[len(package) + 1:] for name in archive.namelist()}
        predict = archive.read(f"{package}/{V5}/conditional_v2/predict.py")
    assert names == PACKED | {"runtime/pythonw.exe"}
    assert not [name for name in names - {"runtime/pythonw.exe"} if PERSONAL.search(name)]
    assert predict == b"print('predict')\r\n"  # the bytes on disk: the manifests are made from them


def test_a_personal_file_that_git_was_told_to_track_stops_the_build(developer_folder):
    personal = ("config/user_parameters.json", "config/instrument_profiles.json",
                "research/notes.md", "test.ipynb")
    _git(developer_folder, "add", "--force", "--", *personal)
    code, _listed, said = _list(developer_folder)
    assert code == 1
    for name in personal:
        problem = rf"PROBLEM: Not allowed in a release \([^)]*\): {re.escape(name)}"
        assert re.search(problem, said), said
    # The build itself stops on the list, before the environment is packed. (The environment
    # named here does not exist, so nothing could be packed even if the check were lost.)
    build = _builder(developer_folder, "-EnvironmentName", "gimap-no-such-environment")
    assert build.returncode != 0
    assert "PROBLEM: Not allowed in a release" in build.stdout
    # PowerShell wraps the text of an error at the console width: compare without white space.
    assert "Nothingwaspacked" in re.sub(r"\s+", "", build.stderr), build.stderr
    assert not (developer_folder / "release").exists()


def test_a_source_file_git_does_not_know_stops_the_build_unless_allowed(developer_folder):
    (developer_folder / "src" / "gimap" / "new_module.py").write_text("NEW = 1\n", encoding="utf-8")
    code, listed, said = _list(developer_folder)
    assert code == 1 and "src/gimap/new_module.py" in said and "not tracked by git" in said
    assert "src/gimap/new_module.py" not in listed
    code, listed, said = _list(developer_folder, "-AllowUntracked")
    assert code == 0, said
    assert "WARNING: 1 file(s) in the program folders are not tracked by git" in said
    assert set(listed) == PACKED


def test_a_model_file_that_is_missing_or_was_changed_is_a_problem(developer_folder):
    """1D Predict checks every file of the manifests before it runs: so does the builder."""
    name = f"{V5}/stable_blue_rc_v1/network.npz"
    (developer_folder / name).unlink()
    code, _listed, said = _list(developer_folder)
    assert code == 1
    assert f"1D Predict needs this model file, and it is missing: {name}" in said
    (developer_folder / name).write_bytes(b"network")
    code, _listed, said = _list(developer_folder)
    assert code == 0, said
    # A checkout that converts line endings changes the bytes the checksum was taken from.
    (developer_folder / V5 / "conditional_v2" / "predict.py").write_bytes(b"print('predict')\n")
    code, _listed, said = _list(developer_folder)
    assert code == 1
    assert "Differs from the checksum in conditional_fast_manifest_v2.json" in said
    assert f"{V5}/conditional_v2/predict.py" in said


def test_this_repository_lists_no_personal_file():
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout: the builder needs git for the file list")
    # The exit code is not asserted: while developing, new source files are untracked, which the
    # builder reports as a problem. The list itself must be clean.
    _code, listed, said = _list(ROOT)
    assert listed, said
    for name in ("main.py", "Start_GIMaP.bat", "src/gimap/app/main_window.py",
                 "config/detectors.json", "docs/User_Manual.md",
                 f"{V5}/conditional_v2/conditional_fast_manifest_v2.json"):
        assert name in listed
    assert not [path for path in listed if PERSONAL.search(path)]
    assert not [path for path in listed if path.startswith("docs/research/")]
    on_disk = [
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / V5).rglob("*")
        if path.suffix in {".h5", ".npz"} and "development" not in path.relative_to(ROOT).parts
    ]
    assert set(on_disk) <= set(listed)  # the weights that are here go into the package
