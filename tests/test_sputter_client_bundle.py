"""The bundled analysis package of the Sputter page (``utils/sputter_client``): complete, byte-exact, private-free.

Analyze and Static Frames of the Sputter page run this package as a subprocess; without a folder chosen by the user or
``GIMAP_SPUTTER_CLIENT``, the page finds it in the program folder (``features/sputter/infrastructure/client.py``). It
is a *release build* of the research code: only the files the page's commands use, without the names of the private
data it was developed on, and with the model metadata reduced to what the code reads. ``MANIFEST.json`` holds the
SHA-256 of every file, so git must hand out the bytes unchanged (``.gitattributes``: ``-text``). A network trained with
the forward code of the package carries the hashes of this code (otherwise its prediction would report that the code
changed since training); one trained with older code keeps its own forward hash, so the package says so as the
research package does (``MANIFEST.json``: ``current_forward_code``).

These tests need numpy only.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "utils" / "sputter_client"
PRIVATE_TOKENS = ROOT / "docs" / "agents" / "eval" / "local" / "private_tokens.txt"
"""The names of the private datasets, people and accounts, and numbers of their instrument geometry: one per line, in
the git-ignored local folder; on a clone the file is absent and only the generic checks run."""
ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]{1,2}[A-Za-z_]|(?<![\w.])/(?:home|data|Users|scratch)/")
TEXT_SUFFIXES = {".py", ".json", ".md", ".txt", ".csv", ".cmd", ".sh"}

pytestmark = pytest.mark.skipif(not PKG.is_dir(), reason="no bundled analysis package (utils/sputter_client)")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files() -> dict:
    return {path.relative_to(PKG).as_posix(): path for path in sorted(PKG.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts and path.name != "MANIFEST.json"}


def _manifest() -> dict:
    return json.loads((PKG / "MANIFEST.json").read_text(encoding="utf-8"))


def _private() -> set:
    if not PRIVATE_TOKENS.is_file():
        return set()
    lines = PRIVATE_TOKENS.read_text(encoding="utf-8-sig").splitlines()
    return {line.split("#", 1)[0].strip().lower() for line in lines} - {""}


def _metadata(path: Path) -> str:
    with np.load(path, allow_pickle=False) as archive:
        return " ".join(str(archive[key]) for key in archive.files if archive[key].dtype.kind in "US")


def test_the_manifest_lists_every_file_with_its_bytes():
    manifest = _manifest()
    assert manifest["release"] is True  # a release build, not the research package
    on_disk = _files()
    assert set(manifest["files"]) == set(on_disk)
    changed = [name for name, path in on_disk.items() if _sha(path) != manifest["files"][name]]
    assert changed == [], "files differ from MANIFEST.json (line endings converted?): " + ", ".join(changed)


def test_the_package_has_what_the_page_runs():
    from src.gimap.features.sputter.infrastructure.client import PACKAGE_PARTS

    assert [str(part) for part in PACKAGE_PARTS if not (PKG / part).exists()] == []
    manifest = _manifest()
    for family in manifest["families"]:
        assert (PKG / "models" / family / "nn" / "model.npz").is_file(), family
        assert (PKG / "code" / "iam" / "iampred" / "families" / f"{family}.json").is_file(), family
    for mode in manifest["static_models"]:
        assert (PKG / "models" / "static_nn_v1" / mode / "model.npz").is_file(), mode
    assert (PKG / "code" / "iam" / "iampred" / "static.py").is_file()  # Static Frames reads its geometry here
    assert not (PKG / "code" / "iam" / "iampred" / "intake_defaults.json").exists()  # no beamline of the private data
    assert (PKG / "LICENSE").read_text(encoding="utf-8").startswith("MIT License")  # THIRD_PARTY_NOTICES, section 5


def test_no_private_names_or_absolute_paths_in_the_package():
    private = _private()
    found = []
    for name, path in _files().items():
        if private & set(re.findall(r"[a-z0-9]+", name.lower())):
            found.append(f"file name {name}")
        if path.suffix in TEXT_SUFFIXES:
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if private & set(re.findall(r"[a-z0-9]+", line.lower())) or ABSOLUTE_PATH.search(line):
                    found.append(f"{name}:{number}")  # the line, never the name
        elif path.suffix == ".npz":
            text = _metadata(path)
            if private & set(re.findall(r"[a-z0-9]+", text.lower())) or ABSOLUTE_PATH.search(text):
                found.append(f"{name}: metadata")
    assert not found, "a private name or an absolute path is here:\n" + "\n".join(found)


def test_every_python_file_compiles_with_this_python():
    """GIMaP runs the package with its own Python (3.10 or newer): no syntax of a newer version in it."""
    bad = []
    for name, path in _files().items():
        if path.suffix == ".py":
            try:
                compile(path.read_text(encoding="utf-8"), name, "exec")
            except SyntaxError as error:
                bad.append(f"{name}:{error.lineno}")
    assert bad == []


def test_every_model_loads_and_gives_candidates():
    """The model metadata of the release keeps what the code reads: every network loads and predicts."""
    manifest = _manifest()
    models = [PKG / "models" / family / "nn" / "model.npz" for family in manifest["families"]]
    models += [PKG / "models" / "static_nn_v1" / mode / "model.npz" for mode in manifest["static_models"]]
    script = ("import json, sys\nimport numpy as np\nsys.path.insert(0, '.')\nfrom iampred import nnmodel as NM\n"
              "out = []\n"
              "for path in json.loads(sys.argv[1]):\n"
              "    P = NM.load(path)\n"
              "    rng = np.random.default_rng(0)\n"
              "    X = rng.normal(size=(1, P.p['f.0.0.weight'].shape[1], 24, 48)).astype(np.float32)\n"
              "    X[:, -1] = 1.0\n"
              "    cands = P.candidates(X, rng.normal(size=(1, len(P.meta['scalars']))))[0]\n"
              "    out.append([len(cands), bool(np.isfinite([c['weight'] for c in cands]).all())])\n"
              "print(json.dumps(out))\n")
    done = subprocess.run([sys.executable, "-I", "-B", "-c", script, json.dumps([str(m) for m in models])],
                          cwd=str(PKG / "code" / "iam"), capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    for model, (count, finite) in zip(models, json.loads(done.stdout)):
        assert count > 0 and finite, model.relative_to(PKG).as_posix()


def _package_environment(scratch: Path) -> dict:
    """The environment the page gives the package (``infrastructure/client.py``): its own folders, and data and work
    folders outside it (``static_nn`` and the commands import the forward model, which makes its work folder)."""
    return dict(os.environ, IAM_GIMAP=str(PKG / "code" / "gimap"), IAM_DIST=str(PKG / "code" / "toolkit"),
                IAM_DATA=str(scratch / "data"), IAM_WORK=str(scratch / "work"), IAM_RAW=str(PKG / "no_raw"),
                IAMPRED_MODELS=str(PKG / "models"))


def test_the_commands_start_without_pytorch(tmp_path):
    """GIMaP installs PyTorch only for Predict; the commands of the page must import without it. (Some research modules
    of the package import it at the top, on paths the page never runs.)"""
    script = ("import sys\nsys.path.insert(0, '.')\n"
              "class NoTorch:\n"
              "    def find_spec(self, name, path=None, target=None):\n"
              "        if name.split('.')[0] == 'torch':\n"
              "            raise ImportError('no PyTorch here')\n"
              "sys.meta_path.insert(0, NoTorch())\n"
              "import iampred.analyze, iampred.static, iampred.intake, iampred.predict, iampred.present, iampred.report\n"
              "print('ok')\n")
    done = subprocess.run([sys.executable, "-I", "-B", "-c", script], cwd=str(PKG / "code" / "iam"), capture_output=True,
                          text=True, timeout=300, env=_package_environment(tmp_path))
    assert done.returncode == 0 and done.stdout.strip() == "ok", done.stderr[-2000:]


def test_the_models_carry_the_hashes_of_this_code(tmp_path):
    manifest = _manifest()
    families = manifest["families"]
    script = ("import json, sys\nsys.path.insert(0, '.')\nfrom iampred import registry as REG\nimport static_nn as SN\n"
              f"out = {{f: [REG.load(f)['family_hash'], REG.forward_hash(), REG.pred_hash()] for f in {families!r}}}\n"
              "out['static'] = SN.forward_hash()\nprint(json.dumps(out))\n")
    done = subprocess.run([sys.executable, "-I", "-B", "-c", script], cwd=str(PKG / "code" / "iam"), capture_output=True,
                          text=True, timeout=300, env=_package_environment(tmp_path))
    assert done.returncode == 0, done.stderr[-2000:]
    expected = json.loads(done.stdout)
    current = manifest["current_forward_code"]
    for family in families:
        with np.load(PKG / "models" / family / "nn" / "model.npz", allow_pickle=False) as archive:
            meta = json.loads(str(archive["meta"]))
        family_hash, forward_hash, pred_hash = expected[family]
        assert [meta["family_hash"], meta["pred_hash"]] == [family_hash, pred_hash], family
        assert (meta["forward_hash"] == forward_hash) is current[family], family
    for mode in manifest["static_models"]:
        with np.load(PKG / "models" / "static_nn_v1" / mode / "model.npz", allow_pickle=False) as archive:
            same = json.loads(str(archive["meta"]))["forward_hash"] == expected["static"]
        assert same is current[f"static_nn_v1/{mode}"], mode
