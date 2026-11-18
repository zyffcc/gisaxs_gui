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

from test_fit1d_release_manifest import install_kit
from test_release_packaging_licenses import make_environment
from tools.collect_licenses import collect
from tools.list_fit1d_release_files import KIT, MANIFEST, release_files

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "build_portable_release.ps1"
KIT_HELPER = ROOT / "tools" / "list_fit1d_release_files.py"
LICENSE_HELPER = ROOT / "tools" / "collect_licenses.py"
APPEND_HELPER = ROOT / "tools" / "append_release_files.py"
NETWORK = "src/gimap/features/fitting/domain/posterior/data/flow_v1"

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
        f"{KIT}/\n"
    ),
    "main.py": "print('GIMaP')\n",
    "Start_GIMaP.bat": "@echo off\n",
    "src/gimap/app.py": "APP = 'GIMaP'\n",
    "src/gimap/shared/version.py": '__version__ = "9.8.7"\n',
    "config/detectors.json": "{}\n",
    "docs/User_Manual.md": "# Manual\n",
    "docs/User_Manual_zh.md": "# Manual (zh)\n",
    "docs/Physics_Manual.md": "# Physics\n",
    "docs/Physics_Manual_zh.md": "# Physics (zh)\n",
    "CITATION.cff": "cff-version: 1.2.0\n",
    "tests/data/external/manifest.json": "{}\n",
    "LICENSE": "MIT License\n",
    "THIRD_PARTY_NOTICES.md": "# Third-party notices\n",
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
    "modules/Example/development/evidence/dataset.npz": "development record",
    "_handoff/report.md": "personal report\n",
}
# Weights of a model of the modules folder as a new one is until `git add -f`: ignored by git (*.h5, *.npz); the builder
# packs them too.
MODEL = {
    "modules/Example/weights.h5": "weights",
}
# The proposal network of Predict parameters: its weights tracked (`git add -f`, as in this repository) and its record
# naming their SHA-256.
NETWORK_FILES = {f"{NETWORK}.npz": "network"}
PACKED = set(TRACKED) | set(MODEL) | set(NETWORK_FILES) | {f"{NETWORK}.json"} | {
    "tools/build_portable_release.ps1", "tools/list_fit1d_release_files.py", "tools/collect_licenses.py",
    "tools/append_release_files.py",
}


def _personal(path: str) -> bool:
    # The single research exception contains only independently verified frozen runtime files.
    return bool(PERSONAL.search(path)) and not path.startswith(f"{KIT}/app/research/")


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
    for name, text in {**TRACKED, **WORKING_PAPERS, **LOCAL, **MODEL, **NETWORK_FILES}.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
    weights = (root / f"{NETWORK}.npz").read_bytes()
    (root / f"{NETWORK}.json").write_text(json.dumps({"sha256": hashlib.sha256(weights).hexdigest()}),
                                          encoding="utf-8")
    (root / "tools").mkdir()
    shutil.copy2(SCRIPT, root / "tools" / SCRIPT.name)
    shutil.copy2(KIT_HELPER, root / "tools" / KIT_HELPER.name)
    shutil.copy2(LICENSE_HELPER, root / "tools" / LICENSE_HELPER.name)
    shutil.copy2(APPEND_HELPER, root / "tools" / APPEND_HELPER.name)
    _git(root, "init", "--quiet")
    # ``git add`` with paths: the ignored files stay out, and so does the untracked ``_handoff``.
    _git(root, "add", "--", ".gitignore", "main.py", "Start_GIMaP.bat", "src", "config", "docs",
         "work", "GISAXS_HANDOFF_20260909.md", "modules", "tools", "LICENSE", "THIRD_PARTY_NOTICES.md",
         "CITATION.cff", "tests")
    # As in this repository: tracked although ``research/`` and ``*.npz`` are ignored.
    _git(root, "add", "--force", "--", "docs/research/outline.md", f"{NETWORK}.npz")
    return root


def test_the_list_is_what_git_tracks_plus_the_model_files(developer_folder):
    code, listed, said = _list(developer_folder)
    assert code == 0, said
    assert set(listed) == PACKED
    assert len(listed) == len(set(listed))
    assert not [path for path in listed if _personal(path)]
    assert "model files git ignores: 1; tracked but left out (working papers): 3" in said
    # The version is the program's own, so the package cannot be named after another release.
    assert "GIMaP-9.8.7-windows-x64-portable" in said
    assert not (developer_folder / "release").exists()  # nothing is written


@pytest.mark.parametrize("with_fit1d", [False, True])
def test_the_package_holds_the_list_and_nothing_else(developer_folder, tmp_path, monkeypatch, with_fit1d):
    """The whole build on the small repository, with stand-ins for ``conda`` and ``conda-pack``.

    The stand-in "environment" is one file, so nothing large is packed; what is tested is which
    files of the folder go into the ZIP beside it.
    """
    import zipfile

    kit_paths = set()
    if with_fit1d:
        install_kit(developer_folder)
        kit_paths = set(release_files(developer_folder)["files"])

    stand_ins = tmp_path / "stand_ins"
    stand_ins.mkdir()
    conda_pack = stand_ins / "conda_pack_stand_in.py"
    conda_pack.write_text(
        "import sys, zipfile, runpy\n"
        "arguments = sys.argv[1:]\n"
        "if '--root' in arguments:\n"
        "    sys.argv = arguments\n"
        "    runpy.run_path(arguments[0], run_name='__main__')\n"
        "elif '-o' in arguments:\n"
        "    archive = arguments[arguments.index('-o') + 1]\n"
        "    runtime = arguments[arguments.index('--arcroot') + 1]\n"
        "    with zipfile.ZipFile(archive, 'w') as packed:\n"
        "        packed.writestr(runtime + '/pythonw.exe', 'python')\n",
        encoding="utf-8")
    (stand_ins / "python.cmd").write_text(f'@"{sys.executable}" "{conda_pack}" %*\r\n')
    # The stand-in environment holds a few licence files, which the build collects into licenses/.
    environment = make_environment(tmp_path / "envs" / "gimap-stand-in")
    (stand_ins / "conda.cmd").write_text('@echo {"envs": ["%s"]}\r\n' % environment.as_posix())
    monkeypatch.setenv("PATH", str(stand_ins), prepend=";")
    temporary = tmp_path / "temp"  # where the build stages licenses/ (and must clean up)
    temporary.mkdir()
    monkeypatch.setenv("TMP", str(temporary))
    monkeypatch.setenv("TEMP", str(temporary))

    # Only the stand-in conda knows this environment: a real one would stop the script here.
    build = _builder(developer_folder, "-EnvironmentName", "gimap-stand-in")
    assert build.returncode == 0, build.stdout + build.stderr
    package = "GIMaP-9.8.7-windows-x64-portable"
    zip_path = developer_folder / "release" / f"{package}.zip"
    with zipfile.ZipFile(zip_path) as archive:
        names = {name[len(package) + 1:] for name in archive.namelist()}
        network = archive.read(f"{package}/{NETWORK}.npz")
        pyqt_gpl = archive.read(f"{package}/licenses/python/PyQt5-5.15.11/GPL-3.0.txt")
        notices = archive.read(f"{package}/licenses/THIRD_PARTY_NOTICES.md")
        if with_fit1d:
            subset = json.loads(archive.read(f"{package}/{KIT}/{MANIFEST}"))
            assert subset == release_files(developer_folder)["manifest"]
            archive.extractall(tmp_path / "extracted")
    licences = {f"licenses/{name}" for name in collect(environment, tmp_path / "expected")["written"]}
    licences |= {"licenses/THIRD_PARTY_NOTICES.md", "licenses/GIMaP/LICENSE"}
    assert names == PACKED | kit_paths | {"runtime/pythonw.exe"} | licences
    assert not [name for name in names - {"runtime/pythonw.exe"} if _personal(name)]
    assert network == b"network"  # the bytes on disk: the record's SHA-256 is made from them
    assert b"GNU GENERAL PUBLIC LICENSE" in pyqt_gpl  # the PyQt5 wheel has no text of its own
    assert notices == (developer_folder / "THIRD_PARTY_NOTICES.md").read_bytes()
    assert not list(temporary.glob("gimap-licenses-*"))  # the staging folder is gone
    # SHA256SUMS.txt beside the ZIP, in the format of sha256sum (the ZIP is small: one part, no installer).
    sums = (developer_folder / "release" / "SHA256SUMS.txt").read_bytes()
    assert sums == f"{hashlib.sha256(zip_path.read_bytes()).hexdigest()}  {package}.zip\n".encode("ascii")
    if with_fit1d:
        assert release_files(tmp_path / "extracted" / package) == release_files(developer_folder)


def test_fit1d_runtime_is_listed_without_tracking_private_examples(developer_folder):
    install_kit(developer_folder)
    # Even accidentally tracked private examples must be filtered out of the public kit subset.
    _git(developer_folder, "add", "--force", "--", f"{KIT}/examples/private_curve.npz")
    code, listed, said = _list(developer_folder)
    assert code == 0, said
    assert set(listed) == PACKED | set(release_files(developer_folder)["files"])
    assert not [path for path in listed if _personal(path)]
    assert not any("private_curve" in path or "selftest" in path or "expected" in path for path in listed)
    assert not (developer_folder / "release").exists()


def test_fit1d_checksum_failure_stops_before_packing(developer_folder):
    install_kit(developer_folder)
    (developer_folder / KIT / "models/general/weights_numpy.npz").write_bytes(b"wrong weights")
    code, listed, said = _list(developer_folder)
    assert code == 1 and "fit1d checksum mismatch" in said
    assert not any(path.startswith(f"{KIT}/") for path in listed)
    assert not (developer_folder / "release").exists()


def test_fit1d_missing_runtime_dependency_stops_before_packing(developer_folder, tmp_path, monkeypatch):
    install_kit(developer_folder)
    stand_ins = tmp_path / "dependency_stand_ins"
    stand_ins.mkdir()
    python = stand_ins / "python_stand_in.py"
    python.write_text(
        "import sys, runpy\n"
        "arguments = sys.argv[1:]\n"
        "if '--root' in arguments:\n"
        "    sys.argv = arguments\n"
        "    runpy.run_path(arguments[0], run_name='__main__')\n",
        encoding="utf-8")
    (stand_ins / "python.cmd").write_text(f'@"{sys.executable}" "{python}" %*\r\n')
    (stand_ins / "conda.cmd").write_text(
        '@echo off\r\nif "%1"=="run" exit /b 1\r\n'
        'echo {"envs": ["C:/stand-in/gimap-stand-in"]}\r\n')
    monkeypatch.setenv("PATH", str(stand_ins), prepend=";")
    build = _builder(developer_folder, "-EnvironmentName", "gimap-stand-in")
    assert build.returncode != 0
    assert "hostruntimeisincompleteorincompatible" in re.sub(r"\s+", "", build.stderr)
    assert not (developer_folder / "release").exists()


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


@pytest.mark.parametrize("name", ["docs/Physics_Manual.md", "CITATION.cff", "docs/manual_images/en/99_new.png"])
def test_a_document_git_does_not_know_stops_the_build_unless_allowed(developer_folder, name):
    """Help opens the manuals and CITATION.cff; the manuals show their pictures: untracked, they would be missing."""
    path = developer_folder / name
    if path.exists():
        _git(developer_folder, "rm", "--cached", "--quiet", "--", name)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"new screenshot")
    code, listed, said = _list(developer_folder)
    assert code == 1 and name in said and "not tracked by git" in said and "Help or the manuals" in said, said
    assert name not in listed
    code, listed, said = _list(developer_folder, "-AllowUntracked")
    assert code == 0, said
    assert "WARNING: 1 document file(s) that Help or the manuals use are not tracked by git" in said


def test_a_file_help_opens_that_is_missing_is_a_problem(developer_folder):
    (developer_folder / "docs" / "Physics_Manual_zh.md").unlink()
    _git(developer_folder, "rm", "--cached", "--quiet", "--", "docs/Physics_Manual_zh.md")
    code, _listed, said = _list(developer_folder)
    assert code == 1
    assert "Help opens this file, and it is not in the package: docs/Physics_Manual_zh.md" in said


def test_a_network_file_that_is_missing_or_was_changed_is_a_problem(developer_folder):
    """Predict parameters checks its weights against the SHA-256 of their record before it runs: so does the builder."""
    name = f"{NETWORK}.npz"
    _git(developer_folder, "rm", "--cached", "--quiet", "--", name)
    code, _listed, said = _list(developer_folder)
    assert code == 1
    assert f"Predict parameters needs this network file, and it is missing: {name}" in said
    _git(developer_folder, "add", "--force", "--", name)
    code, _listed, said = _list(developer_folder)
    assert code == 0, said
    # A checkout that converts line endings changes the bytes the checksum was taken from.
    (developer_folder / name).write_bytes(b"network\r\n")
    code, _listed, said = _list(developer_folder)
    assert code == 1
    assert f"Differs from the SHA-256 in flow_v1.json (changed, or its line endings were converted): {name}" in said


def test_this_repository_lists_no_personal_file():
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout: the builder needs git for the file list")
    # The exit code is not asserted: while developing, new source files are untracked, which the
    # builder reports as a problem. The list itself must be clean.
    _code, listed, said = _list(ROOT)
    assert listed, said
    for name in ("main.py", "Start_GIMaP.bat", "src/gimap/app/main_window.py",
                 "config/detectors.json", "docs/User_Manual.md", f"{NETWORK}.json", f"{NETWORK}.npz"):
        assert name in listed
    assert not [path for path in listed if _personal(path)]
    assert not [path for path in listed if path.startswith("docs/research/")]
    on_disk = [
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src").rglob("*.npz")
    ]
    assert set(on_disk) <= set(listed)  # the weights that are here go into the package
