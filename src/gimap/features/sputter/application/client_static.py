"""Static frames: one sample measured after its deposition, analysed into a series by the client analysis package.

The package's ``static`` command takes the frames of one sample (a folder, one CBF file, or a zip archive with an
optional folder inside it: ``archive.zip::folder/``), its name and its nominal thickness, and an output folder: the
*series*. Each call adds one sample to the series and writes the series again — its report and the data of the main
screen, where the samples stand in the place of the time windows of an in-situ scan (``domain/client.py``). Frames of
the bare substrate, when given, are the substrate term of the sample; without them the sample is taken as the metal
directly on Si. The package has its own geometry and pixel mask for static frames (``StaticDefaults`` says them): a
value of the request that is None leaves the package's.

A sample is named in the series' files by its label: the name with everything but letters, digits, ``-`` and ``_``
replaced. The package keeps the frames it imported under that label and would reuse them whatever it is told later, so
the runner keeps with each import what it was made from and imports again when a request for the same name differs:
the sample is replaced in the series. The same request to the same package shows the series as it is.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Protocol

from ..domain import ClientResult
from .client import (CLIENT_NO_MAIN, ClientFailed, ClientMissing, ClientPackage, ClientReader, ClientRun,
                     ClientStarted)
from .use_cases import Stopped

STATIC_MODES = ("quick", "refine", "full")
REFERENCE_SUFFIX = "_ref"
"""The package imports the frames of the bare substrate of sample ``x`` under ``x_ref``."""
ARCHIVE_MARK = "::"
FRAMES_PROBLEM = "Choose the frames of the sample: a folder, one CBF file or a zip archive."
REFERENCE_PROBLEM = ("The frames of the bare substrate were not found: choose a folder, one CBF file or a zip "
                     "archive, or leave the field empty.")
NAME_PROBLEM = "Give the sample a name with letters or digits."
NAME_RESERVED = "A sample name cannot end in _ref: the frames of a bare substrate are kept under such a name."
NOMINAL_PROBLEM = "Give the nominal thickness of the sample (nm)."
SERIES_PROBLEM = "Choose the folder of the series."
STATIC_MODE_PROBLEM = "The mode must be quick, refine or full."
STATIC_GEOMETRY_PROBLEM = "The X-ray energy and the sample-detector distance must be positive (or empty)."
STATIC_PROBLEMS = (FRAMES_PROBLEM, REFERENCE_PROBLEM, NAME_PROBLEM, NAME_RESERVED, NOMINAL_PROBLEM, SERIES_PROBLEM,
                   STATIC_MODE_PROBLEM, STATIC_GEOMETRY_PROBLEM)
STATIC_UNSUPPORTED = "This analysis package has no entry for static frames: a newer package is needed."
SERIES_NOT_EMPTY = ("{folder} is not empty and holds no series of static frames: choose an empty folder or the "
                    "folder of a series.")
SERIES_ELSEWHERE = ("The series in {folder} was analysed elsewhere and the frames imported for it are not on this "
                    "computer: a sample cannot be added to it. Choose a new folder for the series.")
SERIES_OTHER_MASK = ("The series in {folder} was analysed with another pixel mask, and a series has one mask: "
                     "choose a new folder for the series.")
SAMPLE_EXISTS = ("The data of the series in {folder} already hold a sample named {name}: give this sample another "
                 "name.")
STATIC_FAILED = "The analysis of the static frames stopped with an error ({status}); its last lines are in the log."
STATIC_STOPPED = "Stopped. The samples analysed before stay in the series."
STATIC_TEMPLATES = (*STATIC_PROBLEMS, STATIC_UNSUPPORTED, SERIES_NOT_EMPTY, SERIES_ELSEWHERE, SERIES_OTHER_MASK,
                    SAMPLE_EXISTS, STATIC_FAILED, STATIC_STOPPED)
"""Messages about static frames the page says in its language (the package's own lines stay as they are)."""


def split_frames(spec) -> tuple:
    """(path, folder inside the archive) of a frames field: ``archive.zip::folder/`` or a plain path."""
    path, _mark, inner = str(spec or "").strip().partition(ARCHIVE_MARK)
    return path.strip(), inner.strip()


def join_frames(path, inner: str = "") -> str:
    return f"{path}{ARCHIVE_MARK}{inner}" if inner else str(path)


def frames_found(spec) -> bool:
    """The frames field names a folder, one CBF file or a zip archive that is there."""
    path, inner = split_frames(spec)
    if not path:
        return False
    source = Path(path)
    if source.is_dir():
        return not inner
    suffix = source.suffix.lower()
    return source.is_file() and (suffix == ".zip" or (suffix == ".cbf" and not inner))


def _resolved(spec) -> str:
    path, inner = split_frames(spec)
    return join_frames(Path(path).resolve(), inner) if path else ""


@dataclass(frozen=True)
class StaticDefaults:
    """What the package's static entry uses for a value that is not given (None: the package does not say)."""

    energy_ev: Optional[float] = None
    sdd_mm: Optional[float] = None
    beam_col: Optional[float] = None
    """Beam centre in GIMaP's pixel coordinates, as the page shows it."""
    beam_row: Optional[float] = None
    sample_roll_deg: Optional[float] = None
    alpha_i_deg: Optional[float] = None
    problem: str = ""
    """Why static frames cannot be analysed ("" when they can)."""


@dataclass(frozen=True)
class StaticRequest:
    frames: str
    """A folder, one CBF file or ``archive.zip[::folder inside]``."""
    name: str
    nominal_nm: Optional[float]
    series_folder: Path
    reference_frames: str = ""
    """Frames of the bare substrate ("": the sample is the metal directly on Si)."""
    mode: str = "quick"
    energy_ev: Optional[float] = None
    sdd_mm: Optional[float] = None
    beam_col: Optional[float] = None
    """Beam centre in GIMaP's pixel coordinates (each None: the package's own)."""
    beam_row: Optional[float] = None
    sample_roll_deg: Optional[float] = None
    mask_file: Optional[Path] = None

    def label(self) -> str:
        """The name of the sample in the files of the series."""
        return re.sub(r"[^A-Za-z0-9_-]+", "_", self.name.strip()).strip("_")[:40]

    def problems(self) -> list:
        out = []
        if not frames_found(self.frames):
            out.append(FRAMES_PROBLEM)
        if self.reference_frames.strip() and not frames_found(self.reference_frames):
            out.append(REFERENCE_PROBLEM)
        label = self.label()
        if not re.search(r"[A-Za-z0-9]", label):
            out.append(NAME_PROBLEM)
        elif label.endswith(REFERENCE_SUFFIX):
            out.append(NAME_RESERVED)
        if self.nominal_nm is None or self.nominal_nm <= 0:
            out.append(NOMINAL_PROBLEM)
        if str(self.series_folder) in ("", "."):
            out.append(SERIES_PROBLEM)
        if self.mode not in STATIC_MODES:
            out.append(STATIC_MODE_PROBLEM)
        if any(value is not None and value <= 0 for value in (self.energy_ev, self.sdd_mm)):
            out.append(STATIC_GEOMETRY_PROBLEM)
        return out

    def intake_values(self) -> dict:
        """Everything the imported frames of the sample depend on."""
        return {"frames": _resolved(self.frames), "label": self.label(), "name": self.name.strip(),
                "nominal_nm": self.nominal_nm, "reference_frames": _resolved(self.reference_frames),
                "energy_ev": self.energy_ev, "sdd_mm": self.sdd_mm, "beam_col": self.beam_col,
                "beam_row": self.beam_row, "sample_roll_deg": self.sample_roll_deg,
                "mask_file": str(Path(self.mask_file).resolve()) if self.mask_file else None}

    def _hash(self, values: dict) -> str:
        payload = json.dumps(values, sort_keys=True, default=str).casefold()
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

    def intake_digest(self) -> str:
        return self._hash(self.intake_values())

    def digest(self, package: str = "") -> str:
        """Twelve hex digits of the request, its mode and the package's version."""
        return self._hash(self.intake_values() | {"mode": self.mode, "package": package})


class StaticRunner(Protocol):
    def package(self) -> ClientPackage: ...

    def defaults(self) -> StaticDefaults:
        """The package's own geometry of static frames, or why it cannot analyse them."""
        ...

    def default_series(self) -> Path:
        """Where a series goes when the user names no folder."""
        ...

    def archive_folders(self, archive: Path) -> list:
        """The folders inside a zip archive that hold CBF frames ("" for frames at its top)."""
        ...

    def analysed(self, request: StaticRequest) -> bool:
        """The series holds the sample of exactly this request, analysed by this package."""
        ...

    def replaces(self, request: StaticRequest) -> bool:
        """The series holds a sample of this name analysed from other frames or values: a run removes it first."""
        ...

    def run(self, request: StaticRequest, on_line: Callable[[str], None],
            should_stop: Callable[[], object]) -> ClientRun: ...


class StaticService:
    def __init__(self, runner: StaticRunner, reader: ClientReader):
        self.runner, self.reader = runner, reader

    def defaults(self) -> StaticDefaults:
        return self.runner.defaults()

    def default_series(self) -> Path:
        return self.runner.default_series()

    def archive_folders(self, archive) -> list:
        """Raises ``OSError`` / ``ValueError`` for a file that is not a zip archive."""
        return self.runner.archive_folders(Path(archive))

    def replaces(self, request: StaticRequest) -> bool:
        """Analysing ``request`` replaces a sample of the same name in its series (ask before)."""
        return not request.problems() and self.runner.replaces(request)

    def analyze(self, request: StaticRequest, on_event: Callable[[object], None],
                should_stop: Callable[[], object] = lambda: False) -> ClientResult:
        """Analyse the sample of ``request`` into its series and read the series (as it is when this request was
        analysed by the same package before)."""
        problems = request.problems()
        if problems:
            raise ValueError(" ".join(problems))
        package = self.runner.package()
        if package.problem:
            raise ClientMissing(package.problem)
        unsupported = self.runner.defaults().problem
        if unsupported:
            raise ClientMissing(unsupported)
        out, label = Path(request.series_folder), request.label()
        if self.runner.analysed(request) and self.reader.complete(out):
            on_event(ClientStarted(out, label, reused=True))
            return self.reader.read(out)
        on_event(ClientStarted(out, label))
        run = self.runner.run(request, on_event, should_stop)
        if run.stopped or should_stop():
            raise Stopped(STATIC_STOPPED)
        analysis = self.reader.analysis(out)
        status = str(analysis.get("status") or "")
        if run.code != 0 or status != "ok":
            details = self.reader.failure_details(out, analysis) or "\n".join(run.tail[-15:])
            raise ClientFailed(STATIC_FAILED.format(status=status or f"exit code {run.code}"), details)
        if not self.reader.complete(out):
            raise ClientFailed(CLIENT_NO_MAIN.format(folder=out), "\n".join(run.tail[-15:]))
        return self.reader.read(out)


__all__ = ["ARCHIVE_MARK", "REFERENCE_SUFFIX", "SAMPLE_EXISTS", "SERIES_ELSEWHERE", "SERIES_NOT_EMPTY",
           "SERIES_OTHER_MASK", "STATIC_FAILED", "STATIC_MODES", "STATIC_PROBLEMS", "STATIC_STOPPED",
           "STATIC_TEMPLATES", "STATIC_UNSUPPORTED", "StaticDefaults", "StaticRequest", "StaticRunner",
           "StaticService", "frames_found", "join_frames", "split_frames"]
