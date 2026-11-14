"""Run the client analysis package (``iampred``) in GIMaP's own Python, and read what it writes.

**Where the package is** (``locate_package``): the environment variable ``GIMAP_SPUTTER_CLIENT``, else the folder the
user chose (the page keeps it in the preferences), else ``utils/sputter_client`` of the program folder when it is
there. A package holds ``code/iam/iampred/analyze.py``, ``code/gimap``, ``code/toolkit`` and ``models``; it needs
NumPy, SciPy and Matplotlib only.

**The command** is what the package's ``run.cmd analyze`` does: ``python -u -m iampred.analyze …`` in
``<package>/code/iam`` with ``IAM_GIMAP``, ``IAM_DIST``, ``IAM_DATA``, ``IAM_WORK``, ``IAM_RAW``, ``IAMPRED_MODELS``,
``PYTHONPATH``, ``TZ`` and ``OMP_NUM_THREADS`` set; the options of the import follow ``--``. The beam centre is
entered in GIMaP's pixel coordinates and handed over as the 0-based pixel-centre indices the package's toolkit counts
(``shared/geometry/conventions.py``), as for the bundled toolkit; an empty centre becomes ``--auto-centre``.

**Data and work folders** are in the user data folder: ``sputter_client/<package version>/data/<mask>`` (the package
keeps one pixel mask per data folder) and ``…/work``. An import that was cut off leaves its folder without the
package's ``scan.json`` or without GIMaP's marker ``<label>.gimap.json``: the package would refuse the label for ever
("exists: never overwritten"), so such a folder — GIMaP's own cache of that request — is removed before the run.

**The result** (``ClientFolderReader``): ``analysis.json``, ``main/main.json`` and ``main/w_<t>.npz`` of an output
folder; the user's choices in ``main/choices.json`` (in the user data folder when the result cannot be written to).

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

from ..application import (CLIENT_INCOMPLETE, CLIENT_MISSING, ENGINE_FOLDER, LABEL_PREFIX, MASK_UNREADABLE,
                           ClientPackage, ClientProgress, ClientRequest, ClientRun)
from ..domain import (ANALYSIS_FILE, BAD_ARRAYS, BAD_MAIN, MAIN_FILE, MAIN_FOLDER, NO_ARRAYS, NO_MAIN, ClientResult,
                      ClientWindow, WindowPatterns, client_result, window_patterns)
from .process import child_environment, console_python, run_streaming
from .toolkit import default_workers

CLIENT_OVERRIDE = "GIMAP_SPUTTER_CLIENT"
LOCAL_PACKAGE = Path(__file__).resolve().parents[5] / "utils" / "sputter_client"
IAM = Path("code") / "iam"
PACKAGE_PARTS = (IAM / "iampred" / "analyze.py", Path("code") / "gimap", Path("code") / "toolkit", Path("models"))
STORAGE_FOLDER = "sputter_client"
CLIENT_REQUEST_FILE = "gimap_request.json"
CHOICES_FILE = "choices.json"
IMPORT_MARKER = ".gimap.json"
PACKAGE_MASK = "package-mask"
MASK_FOLDER = Path("code") / "assets"
LOG_TAIL = 15
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


def locate_package(chosen: Optional[Path] = None, local: Path = LOCAL_PACKAGE) -> ClientPackage:
    """The package to run: the environment's, the user's, the program folder's — the first that is named (a named
    folder that is not a package is the problem said, not skipped)."""
    override = os.environ.get(CLIENT_OVERRIDE, "").strip()
    if override:
        home, source = Path(override), "environment"
    elif chosen:
        home, source = Path(chosen), "chosen"
    elif Path(local).is_dir():
        home, source = Path(local), "bundled"
    else:
        return ClientPackage(None, "", CLIENT_MISSING)
    missing = next((part for part in PACKAGE_PARTS if not (home / part).exists()), None)
    if missing is not None:
        return ClientPackage(home, source, CLIENT_INCOMPLETE.format(folder=home, missing=missing.as_posix()))
    return ClientPackage(home, source, "", _package_identity(home))


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


class ClientPackageRunner:
    def __init__(self, storage_root: Callable[[], Path], python: Optional[str] = None, *, run=run_streaming,
                 chosen: Optional[Path] = None, local: Path = LOCAL_PACKAGE, workers: Callable[[], int] = default_workers):
        self.storage_root, self.python, self._run_process = storage_root, python or console_python(), run
        self._chosen, self._local, self._workers = chosen, local, workers

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
               on_line: Callable[[str], None], should_stop: Callable[[], object], **environment) -> ClientRun:
        """Run a command of the package in its folder with its environment (and ``environment`` on top)."""
        env = self.environment(package, data_root, work_root)
        env.update({key: str(value) for key, value in environment.items()})
        outcome = self._run_process(command, on_line=on_line, should_stop=should_stop, env=env,
                                    cwd=str(Path(package.home) / IAM))
        return ClientRun(outcome.code, bool(outcome.stopped), tuple(outcome.tail))

    # -- where the package keeps its data ---------------------------------------------------------------------
    def roots(self, package: ClientPackage, request: ClientRequest) -> tuple:
        """(data folder, work folder) of a request: one data folder per package version and pixel mask."""
        base = Path(self.storage_root()) / STORAGE_FOLDER / package.identity[:12]
        mask = PACKAGE_MASK
        if request.mask_file:
            try:
                mask = "mask-" + _file_digest(Path(request.mask_file))[:12]
            except OSError as exc:
                raise OSError(MASK_UNREADABLE.format(file=request.mask_file)) from exc
        return base / "data" / mask, base / "work"

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
        args = [self.python, "-u", "-m", "iampred.analyze", "--frames", str(request.scan_folder),
                "--label", label or request.label(), "--metal", request.metal, "--substrate", request.substrate,
                rate, _number(request.rate_nm_s), "--alpha-i", _number(request.alpha_i_deg), "--out", str(out_folder),
                "--mode", request.mode, "--workers", str(self._workers()), "--data", str(data_root), "--",
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
        finished = (folder / "scan.json").is_file() and (data_root / (label + IMPORT_MARKER)).is_file()
        if label.startswith(LABEL_PREFIX) and folder.is_dir() and not finished:
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
        command = self.command(request, out, data_root, label)
        self._record(package, request, out, command, label)
        progress = ClientProgress()

        def keep(line: str) -> None:
            progress.feed(line)
            if "intake" in progress.done and not (data_root / (label + IMPORT_MARKER)).is_file():
                (data_root / (label + IMPORT_MARKER)).write_text(
                    json.dumps({"label": label, "request": request.intake_values()}, indent=2, default=str),
                    encoding="utf-8")
            on_line(line)

        return self.launch(command, package, data_root, work_root, keep, should_stop)


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


__all__ = ["CHOICES_FILE", "CLIENT_OVERRIDE", "CLIENT_REQUEST_FILE", "ClientFolderReader", "ClientPackageRunner",
           "IMPORT_MARKER", "LOCAL_PACKAGE", "PACKAGE_MASK", "PACKAGE_PARTS", "STORAGE_FOLDER", "locate_package"]
