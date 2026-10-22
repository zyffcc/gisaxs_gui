"""Sputter use cases: open a finished structure table, or analyze a new scan with the toolkit.

The heavy model (PyTorch networks, optional physical refinement) runs in the
external sputter toolkit through a ``ToolkitRunner`` port, so the GUI process
needs neither PyTorch nor a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Protocol

from ..domain import StructureResult


@dataclass(frozen=True)
class AnalyzeRequest:
    scan_folder: Path
    out_folder: Path
    sdd_mm: float
    alpha_i_deg: float
    metal: str
    energy_ev: Optional[float] = None
    wavelength_nm: Optional[float] = None
    polymer_type: str = "polymer"
    beam_col: Optional[float] = None
    beam_row: Optional[float] = None
    sample_roll_deg: float = 0.0
    mask_file: Optional[Path] = None
    bin_seconds: float = 0.5
    device: str = "cpu"
    workers: int = 4
    refined_folder: Optional[Path] = None

    def problems(self) -> list[str]:
        out = []
        if not str(self.scan_folder) or not Path(self.scan_folder).is_dir():
            out.append("Choose the scan folder that holds the CBF frames.")
        if not str(self.out_folder):
            out.append("Choose an output folder.")
        if (self.energy_ev is None) == (self.wavelength_nm is None):
            out.append("Give the photon energy or the wavelength.")
        if self.sdd_mm <= 0 or self.alpha_i_deg <= 0:
            out.append("Detector distance and incidence angle must be positive.")
        if (self.beam_col is None) != (self.beam_row is None):
            out.append("Give both beam-centre coordinates, or neither (automatic estimate).")
        if self.metal not in ("Ag", "Au", "Cu"):
            out.append("Metal must be Ag, Au or Cu.")
        return out


@dataclass(frozen=True)
class Progress:
    stage: str
    fraction: float
    message: str


def parse_progress(line: str) -> Optional[Progress]:
    """A toolkit progress line: ``SPUTTER_PROGRESS<TAB>stage<TAB>fraction<TAB>message``."""
    parts = line.rstrip("\r\n").split("\t")
    if len(parts) < 4 or parts[0] != "SPUTTER_PROGRESS":
        return None
    try:
        fraction = float(parts[2])
    except ValueError:
        return None
    return Progress(parts[1], fraction, "\t".join(parts[3:]))


STAGES = ("intake", "binning", "features", "network", "done")


def overall_fraction(p: Progress) -> float:
    """Stage-weighted share of the whole run (reading frames dominates)."""
    weights = {"intake": .55, "binning": .1, "features": .2, "network": .15, "done": 0.}
    done = 0.
    for stage in STAGES:
        if stage == p.stage:
            return min(1., done + weights[stage] * max(0., min(1., p.fraction))) if stage != "done" else 1.
        done += weights[stage]
    return done


class ToolkitRunner(Protocol):
    def analyze(self, request: AnalyzeRequest, on_line: Callable[[str], None],
                should_stop: Callable[[], bool]) -> Path: ...

    def table(self, binned_folder: Path, out_folder: Path, metal: str, refined_folder: Optional[Path],
              on_line: Callable[[str], None]) -> Path: ...


class ResultReader(Protocol):
    def read(self, folder: Path, variant: str = "window") -> StructureResult: ...

    def available(self, folder: Path) -> list[str]: ...


class SputterService:
    def __init__(self, runner: ToolkitRunner, reader: ResultReader):
        self.runner = runner
        self.reader = reader

    def open(self, folder: Path, variant: str = "window") -> StructureResult:
        """``window``: per-window network/refined choice; ``selected``: learned selector; ``sequence``: physics path."""
        folder = Path(folder)
        available = self.reader.available(folder)
        return self.reader.read(folder, variant if variant in available else "window")

    def variants(self, folder: Path) -> list[str]:
        return self.reader.available(Path(folder))

    def analyze(self, request: AnalyzeRequest, on_progress: Callable[[str], None],
                should_stop: Callable[[], bool] = lambda: False) -> StructureResult:
        problems = request.problems()
        if problems:
            raise ValueError(" ".join(problems))
        self.runner.analyze(request, on_progress, should_stop)
        return self.reader.read(Path(request.out_folder))

    def merge_refined(self, folder: Path, refined_folder: Path, metal: str,
                      on_progress: Callable[[str], None]) -> StructureResult:
        """Rebuild the table of an analyzed scan with physical-refinement windows (e.g. from the cluster)."""
        folder = Path(folder)
        self.runner.table(folder / "binned", folder, metal, Path(refined_folder), on_progress)
        return self.reader.read(folder)


__all__ = ["AnalyzeRequest", "Progress", "ResultReader", "SputterService", "ToolkitRunner",
           "overall_fraction", "parse_progress"]
