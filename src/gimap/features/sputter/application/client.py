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
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Protocol

from ..domain import CLIENT_RESULT_TEMPLATES, NO_MAIN, ClientResult, ClientWindow, WindowPatterns
from .live_steps import TRACE_FOLDER, StepTree
from .step_details import DETAIL_TEMPLATES, StepDetail
from .use_cases import METALS, PROBLEMS, Stopped, scan_label

QCM_FACTOR = 0.6
"""The rate at the sample is this share of the quartz balance's reading (the package's own factor)."""
RATE_LOW_NM_S, RATE_HIGH_NM_S = 0.005, 1.0
"""A rate at the sample outside this range is unusual for sputtering: the page asks whether its unit is right."""
SUBSTRATES = ("Si", "TiO2", "PS", "PET", "PSBMA", "PNO", "PEO", "Polymer")
FILM_SUBSTRATES = ("PS", "PET", "PSBMA", "PNO", "PEO", "Polymer")
"""Substrates that are a polymer film on the wafer: its thickness can be given (the package has a value of its own
for each of them). Polymer: a film of another or unknown kind (a block copolymer, say); the package takes the optics
of the polymer its network for the metal was trained on."""
MODES = ("auto", "quick", "refine", "full")
ENGINE_FOLDER = "iampred"
LABEL_PREFIX = "gui_"
RATE_PROBLEM = ("Give the deposition rate (Å/s or nm/s): the rate at the sample, or the reading of the quartz balance "
                "with QCM reading ticked.")
SUBSTRATE_PROBLEM = "Choose the substrate."
MODE_PROBLEM = "The mode must be auto, quick, refine or full."
DURATION_PROBLEM = "The deposition duration must be positive (or empty)."
FILM_PROBLEM = "The polymer film thickness must be positive (or empty)."
CLIENT_PROBLEMS = (RATE_PROBLEM, SUBSTRATE_PROBLEM, MODE_PROBLEM, DURATION_PROBLEM, FILM_PROBLEM)
CLIENT_MISSING = ("The analysis package was not found. Choose its folder (it holds code and models), or set the "
                  "environment variable GIMAP_SPUTTER_CLIENT to it.")
CLIENT_INCOMPLETE = "{folder} is not an analysis package: {missing} is missing."
CLIENT_FALLBACK = "{folder} is not an analysis package ({missing} is missing): the one that comes with GIMaP is used."
CLIENT_FAILED = "The analysis stopped with an error ({status}); its last lines are under Details."
CLIENT_STOPPED = "Stopped. Frames that were read completely are kept and reused by the next run of the same request."
CLIENT_NO_MAIN = ("The analysis finished, but this package does not write the data of the main screen "
                  "(main/main.json): a newer package is needed. Its report is in {folder}.")
MASK_UNREADABLE = "The mask file cannot be read: {file}"
EXPORT_OLD = ("This analysis package cannot export (it is older than the export): update GIMaP or choose a newer "
              "package.")
EXPORT_FAILED = "The export stopped with an error; its last lines are under Details."
EXPORT_NOTHING = "The result shown has no window with a solution to export."
EXPORT_FOLDER = "{scan}_export"
"""The folder the export writes into, under the folder the user picks."""
CLIENT_TEMPLATES = (EXPORT_OLD, EXPORT_FAILED, EXPORT_NOTHING, *CLIENT_PROBLEMS, CLIENT_MISSING, CLIENT_INCOMPLETE, CLIENT_FALLBACK, CLIENT_FAILED, CLIENT_STOPPED, CLIENT_NO_MAIN,
                    MASK_UNREADABLE, *CLIENT_RESULT_TEMPLATES, *DETAIL_TEMPLATES)
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
    note: str = ""
    """Why the package of the program folder runs instead of the folder named (``CLIENT_FALLBACK``); "" otherwise."""


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
    workers: Optional[int] = None
    """Processes the package runs its steps with (None: as many as the computer can take when it starts). Not part of
    the result's name: the same request with another number shows the result it has."""
    cache_limit: Optional[int] = None
    """The most the import caches may take together (bytes; None: ``storage.DEFAULT_LIMIT_BYTES``, 0: no limit). Not
    part of the result's name either."""

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

    def rate_at_sample(self) -> Optional[float]:
        """The deposition rate at the sample (nm/s): a QCM reading times ``QCM_FACTOR``, as the package takes it."""
        if self.rate_nm_s is None:
            return None
        return self.rate_nm_s * (QCM_FACTOR if self.rate_is_qcm else 1.0)

    def unusual_rate(self) -> bool:
        """The rate at the sample is outside what sputtering gives (a unit mixed up, most likely)."""
        rate = self.rate_at_sample()
        return rate is not None and rate > 0 and not RATE_LOW_NM_S <= rate <= RATE_HIGH_NM_S


@dataclass(frozen=True)
class ResultTexts:
    """What Details says about a result: its report (Markdown), the facts of ``analysis.json`` and the logs of its
    steps (``(file name, text)``, in the order of the steps)."""

    report: str = ""
    analysis: dict = field(default_factory=dict)
    logs: tuple = ()


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
    workers: Optional[int] = None
    """The processes the package was asked to run with (None: its runner's own number)."""
    import_trace: Optional[Path] = None
    """Where the import keeps the details of its steps (``<data folder>/<label>.trace``) until they are copied into
    the result's ``trace`` folder (None: not known)."""

    def trace_folders(self) -> tuple:
        """The folders a step's detail file may be in, the result's first."""
        return (Path(self.folder) / TRACE_FOLDER,) + ((Path(self.import_trace),) if self.import_trace else ())


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

    def export(self, result_folder: Path, out_folder: Path, shown: list) -> list:
        """Write the export of a result (``iampred.export``: figures, tables, methods) into ``out_folder``; ``shown``:
        [{"t": s, "label": entry}] — the solution shown in each window. Returns the files written."""
        ...


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

    def detail(self, folders: list, name: str) -> StepDetail:
        """A step's detail file from the first of ``folders`` that has it."""
        ...

    def texts(self, result: ClientResult) -> ResultTexts:
        """The report, ``analysis.json`` and the logs of a result (empty where a file is missing)."""
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

    def texts(self, result: ClientResult) -> ResultTexts:
        """The report, the facts and the logs of a result, for Details."""
        return self.reader.texts(result)

    # -- the steps of a run -----------------------------------------------------------------------------------
    def steps(self, folder) -> Optional[StepTree]:
        """The step tree of a result: as GIMaP kept it after the run, else rebuilt from ``analysis.json`` and the
        detail files of its ``trace`` folder; None for a result without either (an older package), or a reader that
        does not know."""
        found = getattr(self.reader, "steps", None)
        if not callable(found):
            return None
        record, files, analysis = found(Path(folder))
        tree = StepTree.from_record(record) if record is not None else None
        if tree is None and files is not None:
            tree = StepTree.from_files(analysis or {}, files)
        return tree

    def step_detail(self, folders, name: str) -> StepDetail:
        """What a step found: its detail file from the first of ``folders`` that has it (read now, kept nowhere).
        Raises ``OSError`` / ``ValueError`` with the reason."""
        return self.reader.detail([Path(folder) for folder in folders], name)

    def save_steps(self, folder, tree: StepTree) -> None:
        """Keep the step tree of a finished run with its result (nothing when the reader cannot; ``OSError`` when
        the folder cannot be written to)."""
        found = getattr(self.reader, "save_steps", None)
        if callable(found) and tree.reported:
            found(Path(folder), tree.to_record())

    def storage(self):
        """The import caches (``storage.CacheStore``), or None for a runner that keeps none."""
        return getattr(self.runner, "caches", None)

    # -- processes --------------------------------------------------------------------------------------------
    def automatic_workers(self) -> Optional[int]:
        """The processes the package would run with now when the user names no number (the runner's
        ``workers()``: cores and free memory); None for a runner that does not say."""
        workers = getattr(self.runner, "workers", None)
        try:
            return max(1, int(workers())) if callable(workers) else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def cores() -> int:
        """The cores of the computer: the most processes the user can choose."""
        return max(1, os.cpu_count() or 1)

    # -- exporting ------------------------------------------------------------------------------------------------
    def export(self, result: ClientResult, folder, choices: Optional[dict] = None) -> list:
        """Figures (PDF, SVG, PNG 300 dpi), the solution shown per window as a CSV with units, every solution and a
        methods text of ``result`` in ``<folder>/<scan>_export`` — the solutions as shown, the user's choices
        included. Returns the files written; raises ``ClientFailed`` / ``ClientMissing`` / ``ValueError``."""
        shown = [{"t": window.time_s, "label": window.chosen_label(choices or {})} for window in result.windows
                 if window.labels()]
        if not shown:
            raise ValueError(EXPORT_NOTHING)
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(result.scan or Path(result.folder).name)).strip("_") or "result"
        return self.runner.export(Path(result.folder), Path(folder) / EXPORT_FOLDER.format(scan=name), shown)

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
        trace = getattr(self.runner, "import_trace", None)
        on_event(ClientStarted(out, label, workers=request.workers,
                               import_trace=trace(request) if callable(trace) else None))
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


__all__ = ["CLIENT_FAILED", "EXPORT_FAILED", "EXPORT_FOLDER", "EXPORT_NOTHING", "EXPORT_OLD", "CLIENT_FALLBACK", "CLIENT_INCOMPLETE", "CLIENT_MISSING", "CLIENT_NO_MAIN", "CLIENT_PROBLEMS",
           "CLIENT_STOPPED", "CLIENT_TEMPLATES", "ClientFailed", "ClientMissing", "ClientPackage", "ClientReader",
           "ClientRequest", "ClientRun", "ClientRunner", "ClientService", "ClientStarted", "ENGINE_FOLDER",
           "FILM_SUBSTRATES", "LABEL_PREFIX", "MASK_UNREADABLE", "MODES", "QCM_FACTOR", "RATE_HIGH_NM_S",
           "RATE_LOW_NM_S", "ResultTexts", "SUBSTRATES"]
