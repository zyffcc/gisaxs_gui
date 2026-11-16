"""Sputter use cases around Analyze: look at a scan folder, and where the results of a scan go.

The analysis itself is the client analysis package (``client.py``: ``ClientService``). This module holds what every
request shares — the metals, the problems of a request that are not the package's own, ``Stopped`` and the short name
of a scan — and ``SputterService``, which the page holds: the scan header (``ScanInspector``), the results root and
``client``.

**The scan's results folder** is ``<results root>/<scan>-<8 hex digits>``; each Analyze request writes its own
subfolder in it (``ClientService.results_folder``). The digits are a hash of the geometry and of values the page used
to send to the model it had before Analyze (its polymer and time window at their defaults, and that model's version),
kept as they were so that a scan analysed before finds its results again.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Callable, Optional, Protocol

from .scan import ScanInfo

METALS = ("Ag", "Au", "Cu")
PROBLEMS = (
    "Choose the scan folder that holds the CBF frames.",
    "Give the photon energy or the wavelength.",
    "Give the sample-detector distance and the incidence angle (both positive).",
    "Give both beam-centre coordinates, or neither (automatic estimate).",
    "Metal must be Ag, Au or Cu.",
)
FOLDER_VALUES = {"polymer_type": "PSBMA", "bin_seconds": 0.5,
                 "model": "14199672853be149ebd9ef89875d02254e4356185ca0de97d4de3d5116731a90"}
"""Part of the hash of a scan's results folder that no longer changes (see the module's text)."""


class Stopped(RuntimeError):
    pass


def scan_label(folder) -> str:
    """A short name of the scan: the frame names without their numbers, else the folder's name."""
    folder = Path(folder)
    first = next(iter(sorted(folder.glob("*.cbf"))), None) if folder.is_dir() else None
    name = re.sub(r"(_\d+)+$", "", first.stem) if first is not None else ""
    name = name or folder.name or "scan"
    return re.sub(r"[^\w.-]+", "_", name)[:60].strip("._") or "scan"


def scan_results_folder(request, root: Path) -> Path:
    """``<root>/<scan>-<hash>`` of a request (``ClientRequest``): the hash is of its scan folder, metal and geometry."""
    values = {"scan_folder": str(Path(request.scan_folder).resolve()), "sdd_mm": request.sdd_mm,
              "alpha_i_deg": request.alpha_i_deg, "metal": request.metal, "energy_ev": request.energy_ev,
              "wavelength_nm": request.wavelength_nm, "polymer_type": FOLDER_VALUES["polymer_type"],
              "beam_col": request.beam_col, "beam_row": request.beam_row, "sample_roll_deg": request.sample_roll_deg,
              "mask_file": str(Path(request.mask_file).resolve()) if request.mask_file else None,
              "bin_seconds": FOLDER_VALUES["bin_seconds"], "model": FOLDER_VALUES["model"]}
    payload = json.dumps(values, sort_keys=True, default=str).casefold()
    return Path(root) / f"{scan_label(request.scan_folder)}-{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:8]}"


class ScanInspector(Protocol):
    def inspect(self, folder: Path) -> ScanInfo: ...


class SputterService:
    def __init__(self, inspector: ScanInspector, results_root: Callable[[], Path], client=None):
        self.inspector = inspector
        self.results_root = results_root
        """The default results root (``<user data>/sputter``); the user may name another one."""
        self.client = client
        """``ClientService`` (Analyze: the client analysis package, ``application/client.py``), None without it."""

    def inspect(self, folder: Path) -> ScanInfo:
        return self.inspector.inspect(Path(folder))

    def results_folder(self, request, root: Optional[Path] = None) -> Path:
        """The scan's results folder of a request, in ``root`` (None: the default root)."""
        return scan_results_folder(request, Path(root) if root else self.results_root())


__all__ = ["FOLDER_VALUES", "METALS", "PROBLEMS", "ScanInspector", "SputterService", "Stopped", "scan_label",
           "scan_results_folder"]
