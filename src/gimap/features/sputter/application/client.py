"""Analyze: the client analysis package as a second engine of the Sputter workspace.

The package (a self-contained folder: ``code``, ``models``, NumPy / SciPy / Matplotlib only) is not part of GIMaP. It
is found by ``ClientRunner.package`` (the environment variable ``GIMAP_SPUTTER_CLIENT``, a folder the user chose, or
``utils/sputter_client`` of the program folder) and runs as a subprocess of GIMaP's own Python, one command from the
frames of a scan to a result folder: import the frames, find the solutions of every time window, write the report,
draw the measured and the model images (``ClientProgress`` follows its lines).

A request names its imported scan by a label of its own, ``gui_<scan>_<hash>``: the hash is of everything the import
depends on (the folder, metal, substrate, rate, geometry, mask, onset, duration, film thickness), because the package
reuses a scan it imported under the same label whatever it is told later. The output goes to a subfolder of the scan's
results folder, ``iampred-<hash of the request, the mode and the package>``: the same request to the same package
shows the result it has; another one gets a new folder.

``ClientService`` runs a request, opens a result folder, reads the images of a window and keeps the user's choice of a
solution per window next to the result. Frames measured after the deposition go through ``ClientService.static``
(``client_static.py``): one sample at a time into a series, which is opened and shown like any other result.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Protocol

from ..domain import CLIENT_RESULT_TEMPLATES, NO_MAIN, ClientResult, ClientWindow, WindowPatterns
from .use_cases import METALS, PROBLEMS, Stopped, scan_label

SUBSTRATES = ("Si", "TiO2", "PS", "PET", "PSBMA", "PNO", "PEO")
FILM_SUBSTRATES = ("PS", "PET", "PSBMA", "PNO", "PEO")
"""Substrates that are a polymer film on the wafer: its thickness can be given (the package has a value of its own
for each of them)."""
MODES = ("auto", "quick", "refine", "full")
ENGINE_FOLDER = "iampred"
LABEL_PREFIX = "gui_"
RATE_PROBLEM = ("Give the deposition rate (nm/s): the rate at the sample, or the reading of the quartz balance with "
                "QCM reading ticked.")
SUBSTRATE_PROBLEM = "Choose the substrate."
MODE_PROBLEM = "The mode must be auto, quick, refine or full."
DURATION_PROBLEM = "The deposition duration must be positive (or empty)."
FILM_PROBLEM = "The polymer film thickness must be positive (or empty)."
CLIENT_PROBLEMS = (RATE_PROBLEM, SUBSTRATE_PROBLEM, MODE_PROBLEM, DURATION_PROBLEM, FILM_PROBLEM)
CLIENT_MISSING = ("The analysis package was not found. Choose its folder (it holds code and models), or set the "
                  "environment variable GIMAP_SPUTTER_CLIENT to it.")
CLIENT_INCOMPLETE = "{folder} is not an analysis package: {missing} is missing."
CLIENT_FAILED = "The analysis stopped with an error ({status}); its last lines are under Details."
CLIENT_STOPPED = "Stopped. Frames that were read completely are kept and reused by the next run of the same request."
CLIENT_NO_MAIN = ("The analysis finished, but this package does not write the data of the main screen "
                  "(main/main.json): a newer package is needed. Its report is in {folder}.")
MASK_UNREADABLE = "The mask file cannot be read: {file}"
CLIENT_TEMPLATES = (*CLIENT_PROBLEMS, CLIENT_MISSING, CLIENT_INCOMPLETE, CLIENT_FAILED, CLIENT_STOPPED, CLIENT_NO_MAIN,
                    MASK_UNREADABLE, *CLIENT_RESULT_TEMPLATES)
"""Messages of the engine the page says in its language (the package's own lines stay as they are)."""


@dataclass(frozen=True)
class ClientPackage:
    """Where the analysis package is: ``problem`` is "" when it can run."""

    home: Optional[Path] = None
    source: str = ""
    """``environment``, ``chosen`` or ``bundled`` ("" when none was found)."""
    problem: str = ""
    identity: str = ""
    """The package's version (the SHA-256 of its ``MANIFEST.json``, else a fingerprint of its files)."""


@dataclass(frozen=True)
class ClientRequest:
    scan_folder: Path
    metal: str
    substrate: str
    rate_nm_s: Optional[float]
    rate_is_qcm: bool = False
    """The rate is the reading of the quartz balance (the package converts it to the rate at the sample)."""
    alpha_i_deg: Optional[float] = None
    sdd_mm: Optional[float] = None
    energy_ev: Optional[float] = None
    wavelength_nm: Optional[float] = None
    beam_col: Optional[float] = None
    """Beam centre in GIMaP's pixel coordinates (as in Analyze); None for both: estimated from the frames."""
    beam_row: Optional[float] = None
    sample_roll_deg: float = 0.0
    mask_file: Optional[Path] = None
    mode: str = "auto"
    onset_s: Optional[float] = None
    """Time of the scan at which the deposition started (None: the package finds it in the data)."""
    duration_s: Optional[float] = None
    """How long the deposition lasted after its start (None: to the end of the scan)."""
    film_nm: Optional[float] = None
    """Thickness of the polymer film (None: the package's own value for the substrate)."""

    def problems(self) -> list:
        out = []
        if str(self.scan_folder) in ("", ".") or not Path(self.scan_folder).is_dir():
            out.append(PROBLEMS[0])
        if self.rate_nm_s is None or self.rate_nm_s <= 0:
            out.append(RATE_PROBLEM)
        if (self.energy_ev is None) == (self.wavelength_nm is None) or (self.energy_ev or self.wavelength_nm or 0) <= 0:
            out.append(PROBLEMS[1])
        if not self.sdd_mm or self.sdd_mm <= 0 or not self.alpha_i_deg or self.alpha_i_deg <= 0:
            out.append(PROBLEMS[2])
        if (self.beam_col is None) != (self.beam_row is None):
            out.append(PROBLEMS[3])
        if self.metal not in METALS:
            out.append(PROBLEMS[4])
        if self.substrate not in SUBSTRATES:
            out.append(SUBSTRATE_PROBLEM)
        if self.mode not in MODES:
            out.append(MODE_PROBLEM)
        if self.duration_s is not None and self.duration_s <= 0:
            out.append(DURATION_PROBLEM)
        if self.film_nm is not None and self.film_nm <= 0:
            out.append(FILM_PROBLEM)
        return out

    def intake_values(self) -> dict:
        """Everything the imported scan depends on (the package keeps it under the request's label)."""
        film = self.film_nm if self.substrate in FILM_SUBSTRATES else None
        return {"scan_folder": str(Path(self.scan_folder).resolve()), "metal": self.metal, "substrate": self.substrate,
                "rate_nm_s": self.rate_nm_s, "rate_is_qcm": bool(self.rate_is_qcm), "alpha_i_deg": self.alpha_i_deg,
                "sdd_mm": self.sdd_mm, "energy_ev": self.energy_ev, "wavelength_nm": self.wavelength_nm,
                "beam_col": self.beam_col, "beam_row": self.beam_row, "sample_roll_deg": self.sample_roll_deg,
                "mask_file": str(Path(self.mask_file).resolve()) if self.mask_file else None,
                "onset_s": self.onset_s, "duration_s": self.duration_s, "film_nm": film}

    def _hash(self, values: dict) -> str:
        payload = json.dumps(values, sort_keys=True, default=str).casefold()
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]

    def label(self) -> str:
        """The name of the imported scan in the package's data folder: never one of the package's own scans."""
        name = re.sub(r"[^A-Za-z0-9_]+", "_", scan_label(self.scan_folder)).strip("_")[:40] or "scan"
        return f"{LABEL_PREFIX}{name}_{self._hash(self.intake_values())}"

    def digest(self, package: str = "") -> str:
        """Eight hex digits of the request, its mode and the package's version."""
        return self._hash(self.intake_values() | {"mode": self.mode, "package": package})


@dataclass(frozen=True)
class ClientRun:
    """How the package's process ended."""

    code: Optional[int]
    stopped: bool = False
    tail: tuple = ()


@dataclass(frozen=True)
class ClientStarted:
    """Analyze started (or shows the result it has): the output folder and the label of the imported scan."""

    folder: Path
    label: str
    reused: bool = False


class ClientMissing(RuntimeError):
    pass


class ClientFailed(RuntimeError):
    """The package ended without a result: the message (made from ``CLIENT_FAILED`` or ``CLIENT_NO_MAIN``) and the
    last lines of the failed step."""

    def __init__(self, message: str, details: str = ""):
        super().__init__(message)
        self.details = details


class ClientRunner(Protocol):
    def package(self) -> ClientPackage:
        """Where the package is, or why it cannot run."""
        ...

    def choose(self, folder: Optional[Path]) -> None:
        """The folder the user chose (None: forget it)."""
        ...

    def masks(self) -> list:
        """The pixel-mask files the package brings (paths; none without a package)."""
        ...

    def run(self, request: ClientRequest, out_folder: Path, on_line: Callable[[str], None],
            should_stop: Callable[[], object]) -> ClientRun: ...


class ClientReader(Protocol):
    def locate(self, folder: Path) -> Optional[Path]:
        """The output folder meant by a folder the user chose (itself, its parent or its newest result)."""
        ...

    def analysis(self, folder: Path) -> dict:
        """``analysis.json`` of an output folder ({} when it is missing or unreadable)."""
        ...

    def complete(self, folder: Path) -> bool:
        """The output folder holds a finished result with the data of the main screen."""
        ...

    def read(self, folder: Path) -> ClientResult: ...

    def patterns(self, result: ClientResult, window: ClientWindow) -> WindowPatterns: ...

    def choices(self, result: ClientResult) -> dict: ...

    def save_choices(self, result: ClientResult, choices: dict) -> Path: ...

    def failure_details(self, folder: Path, analysis: dict) -> str:
        """The last lines of the log of the step that failed."""
        ...


class ClientService:
    def __init__(self, runner: ClientRunner, reader: ClientReader, static=None):
        """``static``: the analysis of static frames (``client_static.StaticService``; None: not offered)."""
        self.runner, self.reader, self.static = runner, reader, static

    # -- the package ------------------------------------------------------------------------------------------
    def package(self) -> ClientPackage:
        return self.runner.package()

    def package_masks(self) -> list:
        """The pixel masks of the package, to offer by their file names."""
        return [Path(mask) for mask in self.runner.masks()]

    def choose_package(self, folder) -> ClientPackage:
        self.runner.choose(Path(folder) if folder else None)
        return self.runner.package()

    # -- results ----------------------------------------------------------------------------------------------
    def results_folder(self, request: ClientRequest, scan_results: Path) -> Path:
        """The output folder of a request: a subfolder of the scan's results folder."""
        return Path(scan_results) / f"{ENGINE_FOLDER}-{request.digest(self.runner.package().identity)}"

    def locate(self, folder) -> Optional[Path]:
        return self.reader.locate(Path(folder))

    def open(self, folder) -> ClientResult:
        """The result in (or under) ``folder``; raises ``OSError`` / ``ValueError`` with what is missing."""
        folder = Path(folder)
        found = self.reader.locate(folder)
        if found is None:
            raise FileNotFoundError(NO_MAIN.format(folder=folder))
        return self.reader.read(found)

    def patterns(self, result: ClientResult, window: ClientWindow) -> WindowPatterns:
        return self.reader.patterns(result, window)

    def choices(self, result: ClientResult) -> dict:
        """The user's choices kept with the result: only those that name an entry of their window."""
        kept = self.reader.choices(result)
        windows = {window.key: window for window in result.windows}
        return {key: label for key, label in kept.items() if key in windows and label in windows[key].labels()}

    @staticmethod
    def with_choice(window: ClientWindow, label: str, choices: dict) -> dict:
        """``choices`` with ``label`` chosen for ``window``: the window's default is not kept (so a result analysed
        again keeps only what the user decided)."""
        choices = dict(choices)
        if label == window.default_label() or label not in window.labels():
            choices.pop(window.key, None)
        else:
            choices[window.key] = label
        return choices

    def save_choices(self, result: ClientResult, choices: dict) -> Path:
        """Keep the choices next to the result (``main/choices.json``); raises ``OSError`` when nowhere can be
        written to."""
        return self.reader.save_choices(result, choices)

    # -- analysing --------------------------------------------------------------------------------------------
    def analyze(self, request: ClientRequest, scan_results: Path, on_event: Callable[[object], None],
                should_stop: Callable[[], object] = lambda: False) -> ClientResult:
        """Run the package for ``request`` and read the result (the one it already has when the same request was
        analysed by the same package before)."""
        problems = request.problems()
        if problems:
            raise ValueError(" ".join(problems))
        package = self.runner.package()
        if package.problem:
            raise ClientMissing(package.problem)
        out = Path(scan_results) / f"{ENGINE_FOLDER}-{request.digest(package.identity)}"
        label = request.label()
        if self.reader.complete(out):
            on_event(ClientStarted(out, label, reused=True))
            return self.reader.read(out)
        on_event(ClientStarted(out, label))
        run = self.runner.run(request, out, on_event, should_stop)
        if run.stopped or should_stop():
            raise Stopped(CLIENT_STOPPED)
        analysis = self.reader.analysis(out)
        status = str(analysis.get("status") or "")
        if run.code != 0 or status != "ok":
            details = self.reader.failure_details(out, analysis) or "\n".join(run.tail[-15:])
            raise ClientFailed(CLIENT_FAILED.format(status=status or f"exit code {run.code}"), details)
        if not self.reader.complete(out):
            raise ClientFailed(CLIENT_NO_MAIN.format(folder=out), "\n".join(run.tail[-15:]))
        return self.reader.read(out)


__all__ = ["CLIENT_FAILED", "CLIENT_INCOMPLETE", "CLIENT_MISSING", "CLIENT_NO_MAIN", "CLIENT_PROBLEMS",
           "CLIENT_STOPPED", "CLIENT_TEMPLATES", "ClientFailed", "ClientMissing", "ClientPackage", "ClientReader",
           "ClientRequest", "ClientRun", "ClientRunner", "ClientService", "ClientStarted", "ENGINE_FOLDER",
           "FILM_SUBSTRATES", "LABEL_PREFIX", "MASK_UNREADABLE", "MODES", "SUBSTRATES"]
