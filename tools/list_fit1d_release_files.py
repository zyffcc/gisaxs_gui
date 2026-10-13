"""Validate the optional fit1d kit and select its distributable runtime files.

Standard library only: the Windows release builder uses this before conda-pack. A personal kit
also holds example measurements and their reference results; those do not belong in a release.
The returned manifest is a subset to write into the ZIP, not the personal kit's full manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

KIT = "utils/gisaxs1d_gui_kit"
MANIFEST = "FILES_SHA256.json"
SCHEMA = "gisaxs1d_kit_files_v1"
ROOT_FILES = {"KIT_MANIFEST.json", "tablepack.py", "_env.bat",
              "run_panel.bat", "run_panel.sh"}
REQUIRED = {
    "KIT_MANIFEST.json", "tablepack.py", "models/models.json",
    "gisaxs1d_kit/__init__.py", "gisaxs1d_kit/core.py", "gisaxs1d_kit/_paths.py",
    "gisaxs1d_kit/results.py",
    "app/research/fit1d/__init__.py", "app/research/fit1d/api.py", "app/research/fit1d/schema.py",
    *(f"tables_packed/ff_{number}_number_v2.fft" for number in (1, 2, 3)),
}


def _safe_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (not name or not path.parts or "\\" in name or path.is_absolute() or path.as_posix() != name
            or any(part in {".", ".."} for part in path.parts)):
        raise ValueError(f"Invalid fit1d manifest path: {name!r}")
    return path


def runtime_file(name: str) -> bool:
    """A narrow exception for the frozen kit, never a general research-folder exception."""
    path = _safe_name(name)
    if any(re.search(r"\s|:|[\x00-\x1f]", part) for part in path.parts):
        return False
    if name in ROOT_FILES:
        return True
    if path.parts[0] == "gisaxs1d_kit":
        return len(path.parts) == 2 and path.suffix == ".py"
    if path.parts[:2] == ("app", "research"):
        return (path.suffix in {".py", ".json", ".md"}
                and not any(part.startswith(".") or part in {
                    "__pycache__", "_handoff", "validation", "development", "gimap_analysis",
                    "AI_Fitting_Output", "trainset_jobs", "test_output", "test_results",
                } for part in path.parts))
    if name == "models/models.json":
        return True
    if path.parts[0] == "models" and len(path.parts) == 3:
        return (bool(re.fullmatch(r"[A-Za-z0-9_.-]+", path.parts[1]))
                and path.name in {"weights_numpy.npz", "model_config.json", "MODEL_INFO.json"})
    return name in {f"tables_packed/ff_{number}_number_v2.fft" for number in (1, 2, 3)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def release_files(root: Path) -> dict:
    home = root / KIT
    if not home.exists():
        return {"files": [], "manifest": None}
    if home.is_symlink() or not home.is_dir():
        raise ValueError(f"fit1d kit is not a regular directory: {home}")
    manifest_path = home / MANIFEST
    if manifest_path.is_symlink():
        raise ValueError("fit1d manifest must not be a symlink")
    try:
        source = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"fit1d kit needs {MANIFEST}; use the verified kit packer") from exc
    if (not isinstance(source, dict) or source.get("schema") != SCHEMA
            or not isinstance(source.get("files"), dict)):
        raise ValueError("Unsupported fit1d file manifest")
    selected = {}
    for name, checksum in source["files"].items():
        _safe_name(name)
        if not isinstance(checksum, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", checksum):
            raise ValueError(f"Invalid fit1d SHA-256 for {name}")
        if not runtime_file(name):
            continue
        path = home / name
        if not path.is_file() or any(parent.is_symlink() for parent in
                                     (path, *(home / part for part in PurePosixPath(name).parents))):
            raise ValueError(f"fit1d runtime file missing or linked: {name}")
        if _sha256(path) != checksum.lower():
            raise ValueError(f"fit1d checksum mismatch: {name}")
        selected[name] = checksum.lower()
    missing = REQUIRED - selected.keys()
    if missing:
        raise ValueError("fit1d runtime manifest is incomplete: " + ", ".join(sorted(missing)))
    catalogue = json.loads((home / "models/models.json").read_text(encoding="utf-8"))
    models = catalogue.get("models")
    if not isinstance(models, dict) or not models or catalogue.get("default") not in models:
        raise ValueError("fit1d model catalogue has no valid default")
    for item in models.values():
        folder = item.get("dir") if isinstance(item, dict) else None
        if not isinstance(folder, str) or not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", folder):
            raise ValueError("Invalid fit1d model directory in catalogue")
        for leaf in ("weights_numpy.npz", "model_config.json", "MODEL_INFO.json"):
            name = f"models/{folder}/{leaf}"
            if name not in selected:
                raise ValueError(f"fit1d model resource not in the manifest: {name}")
    subset = {"schema": SCHEMA, "files": dict(sorted(selected.items()))}
    return {"files": [f"{KIT}/{name}" for name in sorted(selected)] + [f"{KIT}/{MANIFEST}"],
            "manifest": subset}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = release_files(args.root)
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
