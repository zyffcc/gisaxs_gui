"""Run the client analysis package (``iampred``) in GIMaP's own Python, and read what it writes.

**Where the package is** (``locate_package``): the environment variable ``GIMAP_SPUTTER_CLIENT``, else the folder the
user chose (the page keeps it in the preferences), else ``utils/sputter_client`` of the program folder when it is
there. A package holds ``code/iam/iampred/analyze.py``, ``code/gimap``, ``code/toolkit`` and ``models``; it needs
NumPy, SciPy and Matplotlib only.

**The command** is what the package's ``run.cmd analyze`` does: ``python -u -m iampred.analyze …`` in
``<package>/code/iam`` with ``IAM_GIMAP``, ``IAM_DIST``, ``IAM_DATA``, ``IAM_WORK``, ``IAM_RAW``, ``IAMPRED_MODELS``,
``PYTHONPATH``, ``TZ`` and ``OMP_NUM_THREADS`` set; the options of the import follow ``--``. The beam centre is
entered in GIMaP's pixel coordinates and handed over as the 0-based pixel-centre indices the package's toolkit counts
(``shared/geometry/conventions.py``); an empty centre becomes ``--auto-centre``.

**Data and work folders** are in GIMaP's local folder (``cache_store.local_data_dir``, not the roaming user data
folder): ``sputter_client/<package version>/data/<mask>`` (the package keeps one pixel mask per data folder) and
``…/work``; an import finished before in the user data folder is still used there. An import that was cut off leaves
its folder without the package's ``scan.json`` or without GIMaP's marker ``<label>.gimap.json``: the package would
refuse the label for ever ("exists: never overwritten"), so such a folder — GIMaP's own cache of that request — is
removed before the run. Every run touches the marker of its import (its last use) and, before and after the run,
removes the least recently used imports while they exceed the request's limit (``ClientRequest.cache_limit``; never
the import in use, never a result).

**The steps** (``ClientFolderReader.steps``, ``detail``, ``save_steps``): the result's ``trace`` folder
(``trace_files.py``).

**The result** (``ClientFolderReader``): ``analysis.json``, ``main/main.json`` and ``main/w_<t>.npz`` of an output
folder; the user's choices in ``main/choices.json`` (in the user data folder when the result cannot be written to).
For Details (``ClientFolderReader.texts``): ``REPORT.md``, ``analysis.json`` and the ``analyze_*.log`` of every step.

The pixel masks a package brings are the ``.npy`` files of its ``code/assets`` (``ClientPackageRunner.masks``).
Static frames run through ``client_static.py``, which uses this runner's package, environment and process.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from src.gimap.shared.geometry.conventions import index_center_from_canonical
from src.gimap.shared.version import SOFTWARE

from ..application import (CLIENT_FALLBACK, CLIENT_INCOMPLETE, CLIENT_MISSING, ENGINE_FOLDER, EXPORT_FAILED, EXPORT_OLD,
                           LABEL_PREFIX, MASK_UNREADABLE, ClientActivity, ClientFailed, ClientPackage, ClientProgress,
                           ClientRequest, ClientRun, ResultTexts, live_event, parse_live)
from ..domain import (ANALYSIS_FILE, BAD_ARRAYS, BAD_MAIN, MAIN_FILE, MAIN_FOLDER, NO_ARRAYS, NO_MAIN, ClientResult,
                      ClientWindow, WindowPatterns, client_result, window_patterns)
from ..application.storage import DEFAULT_LIMIT_BYTES, PLACE_LOCAL, PLACE_USER, prune
from .cache_store import IMPORT_MARKER, STORAGE_FOLDER, TRACE_SUFFIX, ImportCacheStore
from .process import TAIL_LINES, child_environment, console_python, default_workers, run_streaming
from .trace_files import read_detail, save_steps, steps_record, trace_files

CLIENT_OVERRIDE = "GIMAP_SPUTTER_CLIENT"
LOCAL_PACKAGE = Path(__file__).resolve().parents[5] / "utils" / "sputter_client"
IAM = Path("code") / "iam"
PACKAGE_PARTS = (IAM / "iampred" / "analyze.py", Path("code") / "gimap", Path("code") / "toolkit", Path("models"))
CLIENT_REQUEST_FILE = "gimap_request.json"
CHOICES_FILE = "choices.json"
EXPORT_SHOWN = "shown.json"
CACHES_REMOVED = "Import caches over the limit of {limit}: removed {count} not used for the longest time ({size})"
"""The line Analyze writes to its messages when it removed caches (the package's own lines are English too)."""
PACKAGE_MASK = "package-mask"
MASK_FOLDER = Path("code") / "assets"
LOG_TAIL = 15
TEXT_LIMIT = 400_000
"""The end of a report or log that Details shows (characters; a log of a long run can be larger)."""
REPORT_FILE = "REPORT.md"
LOG_PATTERN = "analyze_*.log"
LOG_LINE_BYTES, LOG_LINE_CHARS = 8192, 300
"""The end of a step log read for its last line, and the most of that line shown."""
LIVE_ARRAYS_LIMIT = 32 << 20
"""The largest file of a live-view event that is read (bytes; the package writes a few hundred kB at most)."""
_SEE_LOG = re.compile(r"\(see (.+?)\)")


def _number(value: float) -> str:
    return repr(float(value))


def _package_identity(home: Path) -> str:
    """The SHA-256 of the package's ``MANIFEST.json``, else the names, sizes and times of its code and models."""
    try:
        return hashlib.sha256((home / "MANIFEST.json").read_bytes()).hexdigest()
    except OSError:
        digest = hashlib.sha256()
        for part in (IAM / "iampred", Path("models")):
            folder = home / part
            for path in sorted(folder.rglob("*")) if folder.is_dir() else []:
                if path.is_file() and "__pycache__" not in path.parts:
                    stat = path.stat()
                    digest.update(f"{path.relative_to(home).as_posix()}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
        return "unversioned-" + digest.hexdigest()


def _missing_part(home: Path) -> Optional[Path]:
    return next((part for part in PACKAGE_PARTS if not (home / part).exists()), None)


def locate_package(chosen: Optional[Path] = None, local: Path = LOCAL_PACKAGE) -> ClientPackage:
    """The package to run: the environment's, the user's, the program folder's — the first that is named. A named
    folder that is not a package (a folder chosen once and moved since, or the wrong one) gives way to the package of
    the program folder, with the reason in ``note``; without that package it is the problem said."""
    override = os.environ.get(CLIENT_OVERRIDE, "").strip()
    named = (Path(override), "environment") if override else (Path(chosen), "chosen") if chosen else None
    if named is None and not Path(local).is_dir():
        return ClientPackage(None, "", CLIENT_MISSING)
    home, source = named or (Path(local), "bundled")
    missing = _missing_part(home)
    if missing is None:
        return ClientPackage(home, source, "", _package_identity(home))
    if named is not None and Path(local).is_dir() and _missing_part(Path(local)) is None:
        return ClientPackage(Path(local), "bundled", "", _package_identity(Path(local)),
                             CLIENT_FALLBACK.format(folder=home, missing=missing.as_posix()))
    return ClientPackage(home, source, CLIENT_INCOMPLETE.format(folder=home, missing=missing.as_posix()))


def _finished(data_root: Path, label: str) -> bool:
    """The import of ``label`` in ``data_root`` finished (the package's ``scan.json`` and GIMaP's marker)."""
    return (Path(data_root) / label / "scan.json").is_file() and (Path(data_root) / (label + IMPORT_MARKER)).is_file()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def live_arrays(path: Optional[Path]) -> dict:
    """The arrays of an event of the live view ({} without a file, or when it is gone, too large or unreadable: the
    view then goes without its picture)."""
    if path is None:
        return {}
    try:
        if Path(path).stat().st_size > LIVE_ARRAYS_LIMIT:
            return {}
        with np.load(path, allow_pickle=False) as arrays:
            return {name: arrays[name] for name in arrays.files}
    except (OSError, ValueError, EOFError, KeyError):
        return {}


def _json(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _tail(path: Optional[Path]) -> str:
    """The text of a file, its end only when it is long ("" when it is missing or unreadable)."""
    try:
        words = Path(path).read_text(encoding="utf-8", errors="replace") if path is not None else ""
    except OSError:
        return ""
    return words if len(words) <= TEXT_LIMIT else "…\n" + words[-TEXT_LIMIT:]


def last_log_line(folder: Path) -> str:
    """The last line with text of the step log written last in a run's folder (its end only; "" without one)."""
    try:
        logs = [path for path in Path(folder).glob(LOG_PATTERN) if path.is_file()]
        newest = max(logs, key=lambda path: path.stat().st_mtime) if logs else None
        if newest is None:
            return ""
        with open(newest, "rb") as handle:
            handle.seek(max(0, newest.stat().st_size - LOG_LINE_BYTES))
            words = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    lines = [line.strip() for line in words.splitlines() if line.strip()]
    return lines[-1][:LOG_LINE_CHARS] if lines else ""


def _gb(size: float) -> str:
    return f"{size / (1 << 30):.1f} GB"


class ClientPackageRunner:
    def __init__(self, storage_root: Callable[[], Path], python: Optional[str] = None, *, run=run_streaming,
                 chosen: Optional[Path] = None, local: Path = LOCAL_PACKAGE, workers: Callable[[], int] = default_workers,
                 legacy_root: Optional[Callable[[], Path]] = None):
        """``storage_root()``: the folder of the import caches (the local one); ``legacy_root()``: where GIMaP kept
        them before (the user data folder) — an import found there is still used."""
        self.storage_root, self.python, self._run_process = storage_root, python or console_python(), run
        self._chosen, self._local, self._workers = chosen, local, workers
        self.legacy_root = legacy_root
        self.caches = ImportCacheStore(self._cache_roots)

    def _cache_roots(self) -> list:
        roots = [(Path(self.storage_root()), PLACE_LOCAL)]
        return roots + ([(Path(self.legacy_root()), PLACE_USER)] if self.legacy_root is not None else [])

    def storage_folder(self, relative) -> Path:
        """``<storage root>/sputter_client/<relative>`` — or the same under the user data folder when only that one
        exists (made before the caches moved)."""
        local = Path(self.storage_root()) / STORAGE_FOLDER / relative
        if self.legacy_root is not None and not local.exists():
            older = Path(self.legacy_root()) / STORAGE_FOLDER / relative
            if older.exists():
                return older
        return local

    def choose(self, folder: Optional[Path]) -> None:
        self._chosen = Path(folder) if folder else None

    def package(self) -> ClientPackage:
        return locate_package(self._chosen, self._local)

    def masks(self) -> list:
        """The pixel masks the package brings (``code/assets/*.npy``), by name."""
        package = self.package()
        if package.problem or package.home is None:
            return []
        try:
            return sorted(path for path in (Path(package.home) / MASK_FOLDER).glob("*.npy") if path.is_file())
        except OSError:
            return []

    def workers(self) -> int:
        return self._workers()

    def launch(self, command: list, package: ClientPackage, data_root: Path, work_root: Path,
               on_line: Callable[[object], None], should_stop: Callable[[], object], *, logs: Optional[Path] = None,
               **environment) -> ClientRun:
        """Run a command of the package in its folder with its environment (and ``environment`` on top). A line of
        the live view (``@live …``) reaches ``on_line`` as a ``LiveEvent`` with its arrays read here, in the
        background; the others as they are. The tail kept for a failure holds the others only. Every few seconds a
        ``ClientActivity`` says how many of its processes work — while none does, with the last line of the newest
        step log in ``logs`` (the run's folder)."""
        env = self.environment(package, data_root, work_root)
        env.update({key: str(value) for key, value in environment.items()})
        tail: list = []

        def line_or_event(line: str) -> None:
            found = parse_live(line)
            if found is None:
                tail.append(line)
                del tail[:-TAIL_LINES]
                on_line(line)
            else:
                on_line(live_event(found[0], live_arrays(found[1])))

        def sampled(sample) -> None:
            line = last_log_line(logs) if sample.busy == 0 and logs is not None else ""
            on_line(ClientActivity(sample.processes, sample.busy, line))

        outcome = self._run_process(command, on_line=line_or_event, should_stop=should_stop, env=env,
                                    cwd=str(Path(package.home) / IAM), on_sample=sampled)
        kept = tail or [line for line in outcome.tail if parse_live(line) is None]
        return ClientRun(outcome.code, bool(outcome.stopped), tuple(kept))

    def export(self, result_folder: Path, out_folder: Path, shown: list) -> list:
        """``python -m iampred.export RESULT OUT --shown <the solutions shown>`` (a package without ``iampred.export``
        says it is too old). The list of the solutions shown is written into OUT for the run and removed after it."""
        package = self.package()
        if package.problem:
            raise FileNotFoundError(package.problem)
        if not (Path(package.home) / IAM / "iampred" / "export.py").is_file():
            raise ClientFailed(EXPORT_OLD)
        out = Path(out_folder)
        out.mkdir(parents=True, exist_ok=True)
        listed = out / EXPORT_SHOWN
        listed.write_text(json.dumps(shown), encoding="utf-8")
        command = [self.python, "-u", "-m", "iampred.export", str(result_folder), str(out), "--shown", str(listed)]
        root = Path(self.storage_root())
        try:
            run = self.launch(command, package, root, root, lambda _line: None, lambda: False)
        finally:
            listed.unlink(missing_ok=True)
        if run.code != 0:
            raise ClientFailed(EXPORT_FAILED, "\n".join(run.tail[-15:]))
        return sorted(path for path in out.iterdir() if path.is_file())

    # -- where the package keeps its data ---------------------------------------------------------------------
    def roots(self, package: ClientPackage, request: ClientRequest) -> tuple:
        """(data folder, work folder) of a request: one data folder per package version and pixel mask, in the local
        folder — or in the user data folder when the request's import was finished there before the caches moved."""
        base = Path(self.storage_root()) / STORAGE_FOLDER / package.identity[:12]
        mask = PACKAGE_MASK
        if request.mask_file:
            try:
                mask = "mask-" + _file_digest(Path(request.mask_file))[:12]
            except OSError as exc:
                raise OSError(MASK_UNREADABLE.format(file=request.mask_file)) from exc
        data, label = base / "data" / mask, request.label()
        if self.legacy_root is not None and not _finished(data, label):
            older = Path(self.legacy_root()) / STORAGE_FOLDER / package.identity[:12] / "data" / mask
            if _finished(older, label):
                data = older
        return data, base / "work"

    def import_trace(self, request: ClientRequest) -> Optional[Path]:
        """Where the import of a request keeps the details of its steps (``<data folder>/<label>.trace``; None when
        the package or the mask cannot be found)."""
        package = self.package()
        if package.problem or not package.identity:
            return None
        try:
            return self.roots(package, request)[0] / (request.label() + TRACE_SUFFIX)
        except OSError:
            return None

    def _prune(self, limit: Optional[int], label: str, on_line: Callable[[object], None]) -> None:
        """Remove the import caches over the limit, the least recently used first (never ``label``'s)."""
        limit = DEFAULT_LIMIT_BYTES if limit is None else limit
        try:
            removed = prune(self.caches, limit, keep=(label,))
        except OSError:  # a folder that cannot be listed: the caches stay as they are
            return
        if removed:
            on_line(CACHES_REMOVED.format(limit=_gb(limit), count=len(removed),
                                          size=_gb(sum(cache.size for cache in removed))))

    def environment(self, package: ClientPackage, data_root: Path, work_root: Path) -> dict:
        home = Path(package.home)
        env = child_environment(
            IAM_GIMAP=str(home / "code" / "gimap"), IAM_DIST=str(home / "code" / "toolkit"), IAM_DATA=str(data_root),
            IAM_WORK=str(work_root), IAM_RAW=str(home / "no_raw"), IAMPRED_MODELS=str(home / "models"),
            TZ="Europe/Berlin", OMP_NUM_THREADS="1")
        env["PYTHONPATH"] = str(home / IAM)  # the package's own modules only (``child_environment`` drops the user's)
        return env

    def command(self, request: ClientRequest, out_folder: Path, data_root: Path, label: str = "") -> list:
        """The command line of a request (``label``: its label when it is known already)."""
        rate = "--qcm-nm-s" if request.rate_is_qcm else "--rate-nm-s"
        workers = request.workers or self._workers()  # the number the user chose, else what the computer takes now
        args = [self.python, "-u", "-m", "iampred.analyze", "--frames", str(request.scan_folder),
                "--label", label or request.label(), "--metal", request.metal, "--substrate", request.substrate,
                rate, _number(request.rate_nm_s), "--alpha-i", _number(request.alpha_i_deg), "--out", str(out_folder),
                "--mode", request.mode, "--workers", str(workers), "--data", str(data_root), "--",
                "--sdd-mm", _number(request.sdd_mm)]
        args += ["--energy-ev", _number(request.energy_ev)] if request.energy_ev is not None else \
            ["--wavelength-nm", _number(request.wavelength_nm)]
        if request.beam_col is not None and request.beam_row is not None:
            col, row = index_center_from_canonical(request.beam_col, request.beam_row)
            args += ["--beam-col", _number(col), "--beam-row", _number(row)]
        else:
            args += ["--auto-centre"]
        args += ["--roll", _number(request.sample_roll_deg)]
        if request.mask_file:
            args += ["--mask", str(request.mask_file)]
        if request.onset_s is not None:
            args += ["--t-dep", _number(request.onset_s)]
        if request.duration_s is not None:
            args += ["--t-sput", _number(request.duration_s)]
        film = request.intake_values()["film_nm"]
        return args + (["--poly-t", _number(film)] if film is not None else [])

    # -- running ------------------------------------------------------------------------------------------------
    def _clear_unfinished(self, data_root: Path, label: str) -> None:
        """Remove the folder of an import of this label that did not finish (GIMaP's own cache of the request)."""
        folder = data_root / label
        if label.startswith(LABEL_PREFIX) and folder.is_dir() and not _finished(data_root, label):
            shutil.rmtree(folder)

    def _record(self, package: ClientPackage, request: ClientRequest, out: Path, command: list, label: str) -> None:
        record = {
            "software_version": SOFTWARE, "engine": ENGINE_FOLDER,
            "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "scan_folder": str(request.scan_folder), "label": label,
            "request": request.intake_values() | {"mode": request.mode,
                                                  "beam_centre_convention": "GIMaP pixel coordinates (corners)"},
            "package": str(package.home), "package_source": package.source, "package_id": package.identity,
            "python": self.python, "command": command,
        }
        out.mkdir(parents=True, exist_ok=True)
        (out / CLIENT_REQUEST_FILE).write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")

    def run(self, request: ClientRequest, out_folder: Path, on_line: Callable[[str], None],
            should_stop: Callable[[], object]) -> ClientRun:
        package = self.package()
        if package.problem:
            raise FileNotFoundError(package.problem)
        out = Path(out_folder)
        data_root, work_root = self.roots(package, request)
        label = request.label()
        for folder in (data_root, work_root):
            folder.mkdir(parents=True, exist_ok=True)
        self._clear_unfinished(data_root, label)
        self.caches.touch(data_root, label)  # used now: the last to go when the caches exceed their limit
        self._prune(request.cache_limit, label, on_line)
        command = self.command(request, out, data_root, label)
        self._record(package, request, out, command, label)
        progress = ClientProgress()

        def keep(line) -> None:
            if isinstance(line, str):  # not an event of the live view
                progress.feed(line)
            if "intake" in progress.done and not (data_root / (label + IMPORT_MARKER)).is_file():
                (data_root / (label + IMPORT_MARKER)).write_text(
                    json.dumps({"label": label, "request": request.intake_values()}, indent=2, default=str),
                    encoding="utf-8")
            on_line(line)

        try:
            return self.launch(command, package, data_root, work_root, keep, should_stop, logs=out)
        finally:  # the new import counts now (it is kept: it is the one in use)
            self._prune(request.cache_limit, label, on_line)


class ClientFolderReader:
    def __init__(self, fallback_root: Optional[Callable[[], Path]] = None):
        self.fallback_root = fallback_root

    def locate(self, folder: Path) -> Optional[Path]:
        folder = Path(folder)
        if (folder / MAIN_FOLDER / MAIN_FILE).is_file() or (folder / ANALYSIS_FILE).is_file():
            return folder
        if folder.name == MAIN_FOLDER and (folder / MAIN_FILE).is_file():
            return folder.parent
        try:
            found = [path for path in folder.glob(f"{ENGINE_FOLDER}-*") if (path / MAIN_FOLDER / MAIN_FILE).is_file()]
        except OSError:
            return None
        return max(found, key=lambda path: (path / MAIN_FOLDER / MAIN_FILE).stat().st_mtime) if found else None

    def analysis(self, folder: Path) -> dict:
        return _json(Path(folder) / ANALYSIS_FILE)

    def complete(self, folder: Path) -> bool:
        folder = Path(folder)
        if self.analysis(folder).get("status") != "ok":
            return False
        return bool(_json(folder / MAIN_FOLDER / MAIN_FILE).get("windows"))

    def read(self, folder: Path) -> ClientResult:
        folder = Path(folder)
        main_file = folder / MAIN_FOLDER / MAIN_FILE
        if not main_file.is_file():
            raise FileNotFoundError(NO_MAIN.format(folder=folder))
        try:
            main = json.loads(main_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(BAD_MAIN.format(folder=folder, reason=f"{type(exc).__name__}: {exc}")) from exc
        return client_result(folder, main, self.analysis(folder))

    def patterns(self, result: ClientResult, window: ClientWindow) -> WindowPatterns:
        if not window.arrays:
            raise FileNotFoundError(NO_ARRAYS.format(window=window.key, folder=result.folder))
        path = Path(result.folder) / MAIN_FOLDER / window.arrays
        if not path.is_file():
            raise FileNotFoundError(NO_ARRAYS.format(window=window.key, folder=result.folder))
        try:
            with np.load(path, allow_pickle=False) as arrays:
                return window_patterns(window, {name: arrays[name] for name in arrays.files})
        except (OSError, EOFError, KeyError) as exc:
            raise ValueError(BAD_ARRAYS.format(window=window.key, reason=f"{type(exc).__name__}: {exc}")) from exc

    # -- the user's choices ---------------------------------------------------------------------------------------
    def _choice_files(self, result: ClientResult) -> list:
        """Where the choices of a result may be: next to it, then (a result that cannot be written to) the user
        data folder."""
        files = [Path(result.folder) / MAIN_FOLDER / CHOICES_FILE]
        if self.fallback_root is not None:
            name = hashlib.sha1(os.path.normcase(os.path.abspath(str(result.folder))).encode("utf-8")).hexdigest()[:16]
            files.append(Path(self.fallback_root()) / STORAGE_FOLDER / "choices" / f"{name}.json")
        return files

    def choices(self, result: ClientResult) -> dict:
        for path in self._choice_files(result):
            kept = _json(path).get("choices")
            if isinstance(kept, dict):
                return {str(key): str(label) for key, label in kept.items()}
        return {}

    def save_choices(self, result: ClientResult, choices: dict) -> Path:
        record = {"schema": 1, "software_version": SOFTWARE, "result_folder": str(result.folder), "scan": result.scan,
                  "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "choices": {str(key): str(label) for key, label in sorted(choices.items())}}
        error: Optional[OSError] = None
        for path in self._choice_files(result):
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(record, indent=2), encoding="utf-8")
                return path
            except OSError as exc:
                error = exc
        raise error or OSError("no place for the choices")

    def texts(self, result: ClientResult) -> ResultTexts:
        """The report (``REPORT.md``), ``analysis.json`` and the logs of the steps, in the order of the steps."""
        folder = Path(result.folder)
        report = _tail(result.file("report_md") or folder / REPORT_FILE)
        analysis = self.analysis(folder)
        steps = analysis.get("steps") if isinstance(analysis.get("steps"), list) else []
        named = [Path(step["log"]).name for step in steps if isinstance(step, dict) and isinstance(step.get("log"), str)
                 and step["log"]]
        try:
            found = {path.name: path for path in folder.glob(LOG_PATTERN) if path.is_file()}
        except OSError:
            found = {}
        order = [name for name in named if name in found] + sorted(name for name in found if name not in named)
        return ResultTexts(report, analysis, tuple((name, _tail(found[name])) for name in dict.fromkeys(order)))

    # -- the steps of the run ------------------------------------------------------------------------------------
    def steps(self, folder: Path) -> tuple:
        """(GIMaP's record of the step tree or None, the names in ``trace/`` or None, ``analysis.json``)."""
        return steps_record(folder), trace_files(folder), self.analysis(folder)

    @staticmethod
    def detail(folders, name: str):
        return read_detail(folders, name)

    @staticmethod
    def save_steps(folder: Path, record: dict) -> Path:
        return save_steps(folder, record)

    def failure_details(self, folder: Path, analysis: dict) -> str:
        """The last lines of the log of the step that failed; a step that names no log (the ``static`` command
        gives its reason as a note, with the log in brackets when there is one) gives its note."""
        steps = analysis.get("steps") if isinstance(analysis.get("steps"), list) else []
        failed = next((step for step in steps if isinstance(step, dict) and step.get("returncode")), None)
        if failed is None:
            return ""
        note = failed.get("note") if isinstance(failed.get("note"), str) else ""
        log = failed.get("log")
        if not isinstance(log, str) or not log:
            named = _SEE_LOG.search(note)
            log = named.group(1) if named is not None else ""
        try:
            lines = Path(log).read_text(encoding="utf-8", errors="replace").splitlines() if log else []
        except OSError:
            lines = []
        tail = "\n".join(line for line in lines[-LOG_TAIL:] if line.strip())
        return "\n".join(part for part in (note, tail) if part)


__all__ = ["CACHES_REMOVED", "CHOICES_FILE", "CLIENT_OVERRIDE", "CLIENT_REQUEST_FILE", "ClientFolderReader",
           "ClientPackageRunner",
           "IMPORT_MARKER", "LOCAL_PACKAGE", "LOG_PATTERN", "PACKAGE_MASK", "PACKAGE_PARTS", "REPORT_FILE",
           "STORAGE_FOLDER", "locate_package"]
