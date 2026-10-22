"""Sputter use cases: predict the structure of a scan in one go, open a result folder, add refined windows.

The model (PyTorch networks) runs in the toolkit bundled with GIMaP (``utils/sputter_toolkit``) through a
``ToolkitRunner`` port, as a subprocess of GIMaP's own Python; ``predict`` first makes sure that Python has
PyTorch (``EnvironmentInstaller``) and installs it when it is missing. Each request gets its own results
folder, ``<results root>/<scan>-<digest of the request and the model's version>``: the same request to the
same model reuses it (a stopped run goes on where it stopped), another geometry or another model gets a new
one (the toolkit reuses whatever outputs it finds in its folder).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Protocol

from ..domain import StructureResult
from .environment import EnvironmentInstaller, EnvironmentStatus, InstallError, install_message
from .scan import ScanInfo

METALS = ("Ag", "Au", "Cu")
VARIANTS = ("physical", "window", "selected", "sequence")
PROBLEMS = (
    "Choose the scan folder that holds the CBF frames.",
    "Give the photon energy or the wavelength.",
    "Give the sample-detector distance and the incidence angle (both positive).",
    "Give both beam-centre coordinates, or neither (automatic estimate).",
    "Metal must be Ag, Au or Cu.",
    "The time window must be positive.",
    "The deposition rate must be positive (or empty).",
)
TOOLKIT_MISSING = ("The Sputter model is not in this GIMaP folder: {folder} is missing or incomplete. Update GIMaP "
                   "(Update_GIMaP.bat or a new download); the model is part of it.")
STOPPED = "Stopped. Finished steps are kept in the results folder and are reused on the next run."
TOOLKIT_FAILED = "The Sputter model stopped with an error (exit code {code}); its last lines are under Details."
NO_TABLE = "{folder} has no {table} (press Predict, or choose a results folder)."
STEP_CHECK = "Checking PyTorch…"
STEP_ANALYZE = "Step 2 of 2: predicting the structure…"
STEP_ANALYZE_ONLY = "Predicting the structure…"


@dataclass(frozen=True)
class AnalyzeRequest:
    scan_folder: Path
    sdd_mm: Optional[float]
    alpha_i_deg: Optional[float]
    metal: str
    energy_ev: Optional[float] = None
    wavelength_nm: Optional[float] = None
    polymer_type: str = "polymer"
    beam_col: Optional[float] = None
    """Beam centre in GIMaP's pixel coordinates (as in Analyze: pixel corners on integers, row 0 at the top)."""
    beam_row: Optional[float] = None
    sample_roll_deg: float = 0.0
    mask_file: Optional[Path] = None
    bin_seconds: float = 0.5
    rate_nm_s: Optional[float] = None
    t0_s: Optional[float] = None
    results_root: Optional[Path] = None
    refined_folder: Optional[Path] = None

    def problems(self) -> list[str]:
        out = []
        if str(self.scan_folder) in ("", ".") or not Path(self.scan_folder).is_dir():
            out.append(PROBLEMS[0])
        if (self.energy_ev is None) == (self.wavelength_nm is None) or (self.energy_ev or self.wavelength_nm or 0) <= 0:
            out.append(PROBLEMS[1])
        if not self.sdd_mm or self.sdd_mm <= 0 or not self.alpha_i_deg or self.alpha_i_deg <= 0:
            out.append(PROBLEMS[2])
        if (self.beam_col is None) != (self.beam_row is None):
            out.append(PROBLEMS[3])
        if self.metal not in METALS:
            out.append(PROBLEMS[4])
        if not self.bin_seconds or self.bin_seconds <= 0:
            out.append(PROBLEMS[5])
        if self.rate_nm_s is not None and self.rate_nm_s <= 0:
            out.append(PROBLEMS[6])
        return out

    def analysis_values(self) -> dict:
        """What the analysis depends on (the rate and t0 only change the physical files, not the analysis)."""
        return {"scan_folder": str(Path(self.scan_folder).resolve()), "sdd_mm": self.sdd_mm,
                "alpha_i_deg": self.alpha_i_deg, "metal": self.metal, "energy_ev": self.energy_ev,
                "wavelength_nm": self.wavelength_nm, "polymer_type": self.polymer_type, "beam_col": self.beam_col,
                "beam_row": self.beam_row, "sample_roll_deg": self.sample_roll_deg,
                "mask_file": str(Path(self.mask_file).resolve()) if self.mask_file else None,
                "bin_seconds": self.bin_seconds}

    def digest(self, model: str = "") -> str:
        """Eight hex digits of what the analysis depends on, and of the model's version (``model_id``)."""
        values = self.analysis_values() | ({"model": model} if model else {})
        payload = json.dumps(values, sort_keys=True, default=str).casefold()
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


def scan_label(folder) -> str:
    """A short name of the scan: the frame names without their numbers, else the folder's name."""
    folder = Path(folder)
    first = next(iter(sorted(folder.glob("*.cbf"))), None) if folder.is_dir() else None
    name = re.sub(r"(_\d+)+$", "", first.stem) if first is not None else ""
    name = name or folder.name or "scan"
    return re.sub(r"[^\w.-]+", "_", name)[:60].strip("._") or "scan"


def results_folder(request: AnalyzeRequest, default_root: Path, model: str = "") -> Path:
    root = Path(request.results_root) if request.results_root else Path(default_root)
    return root / f"{scan_label(request.scan_folder)}-{request.digest(model)}"


@dataclass(frozen=True)
class PredictStep:
    """A step of Predict for the page: ``text`` (a template) and the results folder once it is known."""
    text: str
    folder: Optional[Path] = None
    installing: bool = False


class Stopped(RuntimeError):
    pass


class ToolkitMissing(RuntimeError):
    pass


class ToolkitFailed(RuntimeError):
    """The model's process ended with an error: the message (``TOOLKIT_FAILED``) and its last lines."""

    def __init__(self, code, details: str = ""):
        super().__init__(TOOLKIT_FAILED.format(code=code))
        self.details = details


class ToolkitRunner(Protocol):
    def problem(self) -> str:
        """"" when the bundled toolkit is complete, else the folder that is missing or incomplete."""
        ...

    def model_id(self) -> str:
        """The model's version (a new one gets new results folders)."""
        ...

    def analyze(self, request: AnalyzeRequest, out_folder: Path, on_line: Callable[[str], None],
                should_stop: Callable[[], bool], environment: Optional[EnvironmentStatus] = None) -> Path: ...

    def table(self, out_folder: Path, metal: str, refined_folder: Optional[Path],
              on_line: Callable[[str], None]) -> Path: ...

    def physical(self, out_folder: Path, rate_nm_s: Optional[float], t0_s: Optional[float],
                 on_line: Callable[[str], None]) -> Path: ...


class ResultReader(Protocol):
    def read(self, folder: Path, variant: str = "physical") -> StructureResult: ...

    def available(self, folder: Path) -> list[str]: ...


class ScanInspector(Protocol):
    def inspect(self, folder: Path) -> ScanInfo: ...


class SputterService:
    def __init__(self, runner: ToolkitRunner, reader: ResultReader, installer: EnvironmentInstaller,
                 inspector: ScanInspector, results_root: Callable[[], Path]):
        self.runner, self.reader, self.installer, self.inspector = runner, reader, installer, inspector
        self.results_root = results_root

    # -- reading --------------------------------------------------------------------------------------
    def open(self, folder: Path, variant: Optional[str] = None) -> StructureResult:
        """``physical`` (default when present): one stage path, network sizes; ``window``: per-window choice;
        ``selected``: learned selector; ``sequence``: time-consistent path (exploratory)."""
        folder = Path(folder)
        available = self.reader.available(folder)
        if not available:
            return self.reader.read(folder, "window")  # says what is missing
        chosen = variant if variant in available else available[0]
        return self.reader.read(folder, chosen)

    def variants(self, folder: Path) -> list[str]:
        return self.reader.available(Path(folder))

    def inspect(self, folder: Path) -> ScanInfo:
        return self.inspector.inspect(Path(folder))

    def environment(self) -> EnvironmentStatus:
        return self.installer.status()

    def toolkit_problem(self) -> str:
        folder = self.runner.problem()
        return TOOLKIT_MISSING.format(folder=folder) if folder else ""

    def results_folder(self, request: AnalyzeRequest) -> Path:
        return results_folder(request, self.results_root(), self.runner.model_id())

    # -- predicting -----------------------------------------------------------------------------------
    def predict(self, request: AnalyzeRequest, on_event: Callable[[object], None],
                should_stop: Callable[[], bool] = lambda: False) -> StructureResult:
        """Check PyTorch, install it when it is missing, run the model, read the result: one task."""
        problems = request.problems()
        if problems:
            raise ValueError(" ".join(problems))
        missing = self.toolkit_problem()
        if missing:
            raise ToolkitMissing(missing)
        on_event(PredictStep(STEP_CHECK))
        status = self.installer.verify()
        installed = False
        if not status.ready:
            # a PyTorch that does not load after an installation that was cut off is installed again over it
            repair = status.problem == "import_failed" and self.installer.interrupted()
            if status.problem == "import_failed" and not repair:
                raise InstallError("import_failed", install_message(
                    "import_failed", detail=status.detail, command=self.installer.manual_command(repair=True)),
                    status.detail)
            if should_stop():
                raise InstallError("cancelled", install_message("cancelled"))
            on_event(PredictStep("", installing=True))
            self.installer.install(on_event, should_stop, repair=repair)
            status = self.installer.verify()
            if not status.ready:
                raise InstallError("import_failed", install_message(
                    "import_failed", detail=status.detail or "torch", command=self.installer.manual_command(repair=True)),
                    status.detail)
            installed = True
        if should_stop():
            raise Stopped(STOPPED)
        folder = self.results_folder(request)
        on_event(PredictStep(STEP_ANALYZE if installed else STEP_ANALYZE_ONLY, folder))
        self.runner.analyze(request, folder, on_event, should_stop, status)
        if should_stop():
            raise Stopped(STOPPED)
        return self.open(folder)

    def merge_refined(self, folder: Path, refined_folder: Path, metal: str, on_line: Callable[[str], None],
                      rate_nm_s: Optional[float] = None, t0_s: Optional[float] = None,
                      variant: Optional[str] = None) -> StructureResult:
        """Rebuild the tables of an analyzed scan with physical-refinement windows (e.g. from the cluster)."""
        folder = Path(folder)
        self.runner.table(folder, metal, Path(refined_folder), on_line)
        self.runner.physical(folder, rate_nm_s, t0_s, on_line)
        return self.open(folder, variant)

    def rebuild_physical(self, folder: Path, rate_nm_s: Optional[float], t0_s: Optional[float],
                         on_line: Callable[[str], None], variant: Optional[str] = None) -> StructureResult:
        """New deposition rate or t0: only the physical files are written again (no frame is read)."""
        folder = Path(folder)
        self.runner.physical(folder, rate_nm_s, t0_s, on_line)
        return self.open(folder, variant or "physical")


__all__ = ["AnalyzeRequest", "METALS", "NO_TABLE", "PROBLEMS", "PredictStep", "ResultReader", "STEP_ANALYZE", "STEP_ANALYZE_ONLY",
           "STEP_CHECK", "STOPPED", "ScanInspector", "SputterService", "Stopped", "TOOLKIT_FAILED", "TOOLKIT_MISSING",
           "ToolkitFailed", "ToolkitMissing", "ToolkitRunner", "VARIANTS", "results_folder", "scan_label"]
