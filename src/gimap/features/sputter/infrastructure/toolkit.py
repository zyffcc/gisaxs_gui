"""Run the Sputter model bundled with GIMaP (``utils/sputter_toolkit``) in GIMaP's own Python.

The toolkit folder holds ``scripts/sputter_struct_v1.py``, the ``sputtergisaxs`` package, ``configs/``,
``experts/polymer`` and ``experts/TS``. It runs as a subprocess of the console ``python.exe`` of the
environment GIMaP runs in (PyTorch is installed there on the first Predict), with ``PYTHONNOUSERSITE=1``.
Progress comes back on stdout (``SPUTTER_PROGRESS``, ``INTAKE_PLAN``, ``INTAKE_CHUNK``, ``INTAKE_EXCLUDED``);
Stop ends the whole process tree. ``GIMAP_SPUTTER_TOOLKIT`` points at another copy of the toolkit
(development only).

Beam centre: the page takes it in GIMaP's pixel coordinates (as Analyze shows it); the toolkit counts
0-based pixel centres, so the values are converted here (``shared/geometry/conventions.py``).

``model_id`` names the model's version (the SHA-256 of its ``MANIFEST.json``): it is part of the results
folder's code, so a GIMaP with another model never reuses the networks' outputs of an older one.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from src.gimap.shared.geometry.conventions import index_center_from_canonical

from ..application import STOPPED, AnalyzeRequest, EnvironmentStatus, Stopped, ToolkitFailed
from .process import child_environment, console_python, run_streaming

ENTRY = Path("scripts") / "sputter_struct_v1.py"
REQUIRED = (ENTRY, Path("sputtergisaxs"), Path("experts") / "polymer", Path("experts") / "TS")
LOCAL_TOOLKIT = Path(__file__).resolve().parents[5] / "utils" / "sputter_toolkit"
OVERRIDE = "GIMAP_SPUTTER_TOOLKIT"
REQUEST_FILE = "gimap_request.json"
NOISE = ("FutureWarning", "UserWarning")


def toolkit_home() -> Path:
    override = os.environ.get(OVERRIDE, "").strip()
    return Path(override) if override else LOCAL_TOOLKIT


WORKER_BYTES = 0.4e9
"""Memory of one frame-reading process of the toolkit (about 0.3 GB measured, with a margin)."""
KEPT_FREE_BYTES = 1.0e9


def default_workers() -> int:
    """Frame-reading processes: one per core (at most 8, one core left free) and no more than the free memory
    holds, so a busy computer reads more slowly instead of running out of memory."""
    by_cores = max(1, min(8, (os.cpu_count() or 2) - 1))
    try:
        import psutil

        available = float(psutil.virtual_memory().available)
    except Exception:  # psutil missing or unreadable: cores only
        return by_cores
    return max(1, min(by_cores, int((available - KEPT_FREE_BYTES) // WORKER_BYTES)))


def _number(value: float) -> str:
    return repr(float(value))


def _fingerprint(home: Path) -> str:
    """Without a ``MANIFEST.json`` (a development copy): the names, sizes and times of the model's files."""
    digest = hashlib.sha256()
    for part in ("scripts", "sputtergisaxs", "experts", "configs"):
        folder = home / part
        for path in sorted(folder.rglob("*")) if folder.is_dir() else []:
            if path.is_file() and "__pycache__" not in path.parts:
                stat = path.stat()
                digest.update(f"{path.relative_to(home).as_posix()}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
    return "unversioned-" + digest.hexdigest()


class BundledToolkit:
    def __init__(self, home: Optional[Path] = None, python: Optional[str] = None, *, run=run_streaming):
        self._home, self.python, self._run_process = home, python or console_python(), run

    @property
    def home(self) -> Path:
        return Path(self._home) if self._home is not None else toolkit_home()

    def problem(self) -> str:
        """"" when the toolkit is complete, else the folder (missing or incomplete)."""
        home = self.home
        return "" if all((home / part).exists() for part in REQUIRED) else str(home)

    def model_id(self) -> str:
        """The model's version: the SHA-256 of its ``MANIFEST.json`` (else a fingerprint of its files)."""
        try:
            return hashlib.sha256((self.home / "MANIFEST.json").read_bytes()).hexdigest()
        except OSError:
            return _fingerprint(self.home)

    def _run(self, args: list[str], on_line: Callable[[str], None], should_stop: Callable[[], object]) -> None:
        home = self.home

        def keep(line: str) -> None:
            if any(word in line for word in NOISE) or line.lstrip().startswith("warnings.warn"):
                return
            on_line(line)

        outcome = self._run_process([self.python, "-u", str(home / ENTRY), *args], on_line=keep,
                                    should_stop=should_stop, env=child_environment(), cwd=str(home))
        if outcome.stopped:
            raise Stopped(STOPPED)
        if outcome.code != 0:
            raise ToolkitFailed(outcome.code, "\n".join(outcome.tail[-15:]))

    def _record(self, request: AnalyzeRequest, out: Path, environment: Optional[EnvironmentStatus]) -> None:
        manifest = self.home / "MANIFEST.json"
        model = self.model_id()
        record = {
            "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "scan_folder": str(request.scan_folder),
            "request": request.analysis_values() | {"rate_nm_s": request.rate_nm_s, "t0_s": request.t0_s,
                                                    "beam_centre_convention": "GIMaP pixel coordinates (corners)"},
            "digest": request.digest(model),
            "model_id": model,
            "toolkit": str(self.home),
            "toolkit_manifest_sha256": model if manifest.is_file() else None,
            "python": self.python,
            "torch": environment.torch_version if environment else None,
            "numpy": environment.numpy_version if environment else None,
            "device": "cpu", "workers": default_workers(),
        }
        out.mkdir(parents=True, exist_ok=True)
        (out / REQUEST_FILE).write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")

    def analyze(self, request: AnalyzeRequest, out_folder: Path, on_line: Callable[[str], None],
                should_stop: Callable[[], object], environment: Optional[EnvironmentStatus] = None) -> Path:
        home, out = self.home, Path(out_folder)
        self._record(request, out, environment)
        args = ["analyze", "--input", str(request.scan_folder), "--out", str(out),
                "--sdd-mm", _number(request.sdd_mm), "--alpha-i-deg", _number(request.alpha_i_deg),
                "--metal", request.metal, "--polymer-type", request.polymer_type or "polymer",
                "--sample-roll-deg", _number(request.sample_roll_deg), "--bin-seconds", _number(request.bin_seconds),
                "--device", "cpu", "--workers", str(default_workers()),
                "--polymer-expert", str(home / "experts" / "polymer"), "--ts-expert", str(home / "experts" / "TS")]
        args += ["--energy-ev", _number(request.energy_ev)] if request.energy_ev is not None else \
            ["--wavelength-nm", _number(request.wavelength_nm)]
        if request.beam_col is not None and request.beam_row is not None:
            col, row = index_center_from_canonical(request.beam_col, request.beam_row)
            args += ["--beam-col", _number(col), "--beam-row", _number(row)]
        if request.mask_file:
            args += ["--mask", str(request.mask_file)]
        if request.refined_folder:
            args += ["--refined", str(request.refined_folder)]
        args += self._thickness_args(request.rate_nm_s, request.t0_s)
        self._run(args, on_line, should_stop)
        return out / "structure.csv"

    @staticmethod
    def _thickness_args(rate_nm_s: Optional[float], t0_s: Optional[float]) -> list[str]:
        args = ["--rate-nm-s", _number(rate_nm_s)] if rate_nm_s is not None else []
        return args + (["--t0-s", _number(t0_s)] if t0_s is not None else [])

    def table(self, out_folder: Path, metal: str, refined_folder: Optional[Path],
              on_line: Callable[[str], None]) -> Path:
        out = Path(out_folder)
        args = ["table", "--binned", str(out / "binned"), "--out", str(out), "--metal", metal]
        if refined_folder:
            args += ["--refined", str(refined_folder)]
        self._run(args, on_line, lambda: False)
        return out / "structure.csv"

    def physical(self, out_folder: Path, rate_nm_s: Optional[float], t0_s: Optional[float],
                 on_line: Callable[[str], None]) -> Path:
        out = Path(out_folder)
        self._run(["physical", "--folder", str(out), *self._thickness_args(rate_nm_s, t0_s)], on_line, lambda: False)
        return out / "structure_physical.csv"


__all__ = ["BundledToolkit", "ENTRY", "LOCAL_TOOLKIT", "OVERRIDE", "REQUEST_FILE", "default_workers",
           "toolkit_home"]
