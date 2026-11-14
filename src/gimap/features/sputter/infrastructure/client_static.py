"""Run the client analysis package's ``static`` command and keep its series (``application/client_static.py``).

**The command** is what the package's ``run.cmd static`` does: ``python -u -m iampred.static --frames … --label …
--nominal-nm … --out <series> …`` in ``<package>/code/iam``, with the environment of ``ClientPackageRunner`` and
``IAM_STATIC_DATA``. A geometry option is given only for a value the request holds (the package has its own); the beam
centre goes over as 0-based pixel-centre indices, as for an in-situ scan.

**The package's own values** (``StaticPackageRunner.defaults``) are read from the source of its ``static`` module —
the ``GEOMETRY`` it assigns — without importing it. A package without that module cannot analyse static frames.

**A series** is an output folder. GIMaP writes ``gimap_series.json`` into it: an id, the pixel mask and, for each
sample, what it was analysed from. The frames the package imports for the series are kept in the user data folder,
``sputter_client/static/<series id>``: apart from the in-situ data folders (the pixel masks differ) and one folder
per series (the package knows a sample by its label alone). A series the package made without GIMaP — its
``analysis.json`` names its data folder — is added to in that folder when it is on this computer; a name that folder
already holds is refused there, since GIMaP cannot tell what it was imported from.

**The same name again.** The package imports a label once and then reuses it whatever it is told. So each import of a
GIMaP series has a marker ``<label>.gimap.json`` beside it, written when the import ended: a request for a label whose
marker is missing (the import was cut off) or says other inputs (``StaticPackageRunner.replaces``) removes that
import — and the sample's folder in the series — before the run, so the sample is analysed afresh.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from src.gimap.shared.geometry.conventions import canonical_from_index_center, index_center_from_canonical
from src.gimap.shared.version import SOFTWARE

from ..application import (REFERENCE_SUFFIX, SAMPLE_EXISTS, SERIES_ELSEWHERE, SERIES_NOT_EMPTY, SERIES_OTHER_MASK,
                           STATIC_UNSUPPORTED, ClientPackage, ClientProgress, ClientRun, StaticDefaults,
                           StaticRequest, join_frames, split_frames)
from ..domain import ANALYSIS_FILE, STATIC_KIND
from .client import IAM, IMPORT_MARKER, PACKAGE_MASK, STORAGE_FOLDER, ClientPackageRunner, _file_digest, _json

STATIC_ENTRY = IAM / "iampred" / "static.py"
STATIC_FOLDER = "static"
STATIC_WORK = "static_work"
SERIES_FILE = "gimap_series.json"
DEFAULT_SERIES = "series_1"
_SERIES_ID = re.compile(r"[0-9a-f]{8,32}")


def _number(value: float) -> str:
    return repr(float(value))


def _float(value) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _assigned(source: str, name: str) -> dict:
    """The literal keys and values of a module-level ``name = dict(…)`` or ``name = {…}`` in ``source``."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == name):
            continue
        value = node.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "dict":
            pairs = [(keyword.arg, keyword.value) for keyword in value.keywords if keyword.arg]
        elif isinstance(value, ast.Dict):
            pairs = [(key.value, item) for key, item in zip(value.keys, value.values)
                     if isinstance(key, ast.Constant) and isinstance(key.value, str)]
        else:
            return {}
        found = {}
        for key, item in pairs:
            try:
                found[key] = ast.literal_eval(item)
            except (ValueError, SyntaxError):
                found[key] = None
        return found
    return {}


def static_defaults(package: ClientPackage) -> StaticDefaults:
    """The geometry the package's ``static`` module uses by itself, or why the package has no such module."""
    if package.problem or package.home is None:
        return StaticDefaults(problem=package.problem or STATIC_UNSUPPORTED)
    try:
        source = (Path(package.home) / STATIC_ENTRY).read_text(encoding="utf-8")
    except OSError:
        return StaticDefaults(problem=STATIC_UNSUPPORTED)
    geometry = _assigned(source, "GEOMETRY")
    col, row = _float(geometry.get("db_col")), _float(geometry.get("db_row"))
    if col is not None and row is not None:
        col, row = canonical_from_index_center(col, row)
    else:
        col = row = None
    return StaticDefaults(_float(geometry.get("energy_ev")), _float(geometry.get("sdd_mm")), col, row,
                          _float(geometry.get("sample_roll_deg")), _float(geometry.get("alpha_i_deg")))


def archive_folders(archive: Path) -> list:
    """The folders inside a zip archive that hold CBF frames, each ending in ``/`` ("" for frames at its top)."""
    with zipfile.ZipFile(archive) as opened:
        frames = [name for name in opened.namelist() if name.lower().endswith(".cbf")]
    return sorted({name.rsplit("/", 1)[0] + "/" if "/" in name else "" for name in frames})


class StaticPackageRunner:
    def __init__(self, runner: ClientPackageRunner, series_root: Callable[[], Path]):
        """``runner``: the package, its environment and its process; ``series_root``: where a series goes when the
        user names no folder."""
        self.runner, self.series_root = runner, series_root

    def package(self) -> ClientPackage:
        return self.runner.package()

    def defaults(self) -> StaticDefaults:
        return static_defaults(self.runner.package())

    def default_series(self) -> Path:
        return Path(self.series_root()) / DEFAULT_SERIES

    def archive_folders(self, archive: Path) -> list:
        return archive_folders(Path(archive))

    # -- the series and its data ------------------------------------------------------------------------------
    def _storage(self) -> Path:
        return Path(self.runner.storage_root()) / STORAGE_FOLDER

    def series(self, folder: Path) -> tuple:
        """(data folder, record, GIMaP's own) of the series in ``folder`` — a new record for an empty folder.
        Raises ``ValueError`` for a folder that cannot take a sample."""
        folder = Path(folder)
        record = _json(folder / SERIES_FILE)
        if isinstance(record.get("series_id"), str) and _SERIES_ID.fullmatch(record["series_id"]):
            return self._storage() / STATIC_FOLDER / record["series_id"], record, True
        analysis = _json(folder / ANALYSIS_FILE)
        if analysis.get("kind") == STATIC_KIND:
            root = analysis.get("data_root")
            if isinstance(root, str) and root and Path(root).is_dir():
                return Path(root), {}, False
            raise ValueError(SERIES_ELSEWHERE.format(folder=folder))
        if folder.is_dir() and any(folder.iterdir()):
            raise ValueError(SERIES_NOT_EMPTY.format(folder=folder))
        record = {"series_id": uuid.uuid4().hex[:16]}
        return self._storage() / STATIC_FOLDER / record["series_id"], record, True

    def _mask_key(self, request: StaticRequest) -> str:
        return "mask-" + _file_digest(Path(request.mask_file))[:12] if request.mask_file else PACKAGE_MASK

    def _sample(self, request: StaticRequest) -> dict:
        samples = _json(Path(request.series_folder) / SERIES_FILE).get("samples")
        sample = samples.get(request.label()) if isinstance(samples, dict) else None
        return sample if isinstance(sample, dict) else {}

    def analysed(self, request: StaticRequest) -> bool:
        series = Path(request.series_folder)
        return (self._sample(request).get("digest") == request.digest(self.runner.package().identity)
                and (series / request.label() / "frames.csv").is_file())

    def replaces(self, request: StaticRequest) -> bool:
        """The series holds a sample of this name that was analysed from other frames or values: a run for the
        request removes it first."""
        kept = self._sample(request).get("intake")
        return bool(kept) and kept != request.intake_digest()

    def _prepare(self, request: StaticRequest, root: Path, owned: bool, label: str) -> bool:
        """Before a run: an import of this label that is not the request's (or was cut off) goes, with the sample
        analysed from it. True when something was removed."""
        if not owned:
            if (root / label).exists():
                raise ValueError(SAMPLE_EXISTS.format(folder=request.series_folder, name=label))
            return False
        marker = root / (label + IMPORT_MARKER)
        if _json(marker).get("intake") == request.intake_digest():
            return False
        stale = Path(request.series_folder) / label
        folders = [root / label, root / (label + REFERENCE_SUFFIX)]
        if (stale / "frames.csv").is_file():  # the sample as it was analysed before: no longer in the series
            folders.append(stale)
        removed = False
        for folder in folders:
            if folder.is_dir():
                shutil.rmtree(folder)
                removed = True
        marker.unlink(missing_ok=True)
        return removed

    # -- the command --------------------------------------------------------------------------------------------
    def command(self, request: StaticRequest, data_root: Path) -> list:
        frames, inner = split_frames(request.frames)
        args = [self.runner.python, "-u", "-m", "iampred.static", "--frames", join_frames(Path(frames), inner),
                "--label", request.label(), "--nominal-nm", _number(request.nominal_nm),
                "--out", str(request.series_folder), "--sample", request.name.strip(), "--mode", request.mode,
                "--workers", str(self.runner.workers()), "--data", str(data_root)]
        if request.reference_frames.strip():
            frames, inner = split_frames(request.reference_frames)
            args += ["--reference-frames", join_frames(Path(frames), inner)]
        if request.mask_file:
            args += ["--mask", str(request.mask_file)]
        if request.sdd_mm is not None:
            args += ["--sdd-mm", _number(request.sdd_mm)]
        if request.energy_ev is not None:
            args += ["--energy-ev", _number(request.energy_ev)]
        if request.beam_col is not None:
            args += ["--beam-col", _number(index_center_from_canonical(request.beam_col, 0.0)[0])]
        if request.beam_row is not None:
            args += ["--beam-row", _number(index_center_from_canonical(0.0, request.beam_row)[1])]
        if request.sample_roll_deg is not None:
            args += ["--roll", _number(request.sample_roll_deg)]
        return args

    # -- running ------------------------------------------------------------------------------------------------
    def _write_series(self, folder: Path, record: dict) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / SERIES_FILE).write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")

    def run(self, request: StaticRequest, on_line: Callable[[str], None],
            should_stop: Callable[[], object]) -> ClientRun:
        package = self.runner.package()
        defaults = static_defaults(package)
        if defaults.problem:
            raise FileNotFoundError(defaults.problem)
        series, label = Path(request.series_folder), request.label()
        root, record, owned = self.series(series)
        mask = self._mask_key(request)
        samples = record.get("samples") if isinstance(record.get("samples"), dict) else {}
        others = [name for name in samples if name != label]
        if owned and others and record.get("mask") not in (None, mask):
            raise ValueError(SERIES_OTHER_MASK.format(folder=series))
        if owned and any(not (root / name).is_dir() for name in others):
            raise ValueError(SERIES_ELSEWHERE.format(folder=series))  # the report reads every sample's import
        work = self._storage() / STATIC_WORK
        for folder in (root, work):
            folder.mkdir(parents=True, exist_ok=True)
        if self._prepare(request, root, owned, label):
            samples = {name: samples[name] for name in others}
        if owned and not others and record.get("mask") not in (None, mask):
            (root / "pixmask.json").unlink(missing_ok=True)  # the only sample comes again with another mask
        command = self.command(request, root)
        written = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if owned:
            record = {"schema": 1, "software_version": SOFTWARE, "series_id": record["series_id"], "mask": mask,
                      "samples": samples}
            self._write_series(series, record)
        progress = ClientProgress.static()
        marker = root / (label + IMPORT_MARKER)

        def keep(line: str) -> None:
            progress.feed(line)
            if owned and "intake" in progress.done and not marker.is_file():
                marker.write_text(json.dumps({"label": label, "intake": request.intake_digest(),
                                              "request": request.intake_values()}, indent=2, default=str),
                                  encoding="utf-8")
            on_line(line)

        run = self.runner.launch(command, package, root, work, keep, should_stop, IAM_STATIC_DATA=root)
        if owned and run.code == 0 and not run.stopped:
            record["samples"] = samples | {label: {
                "digest": request.digest(package.identity), "intake": request.intake_digest(), "mode": request.mode,
                "name": request.name.strip(), "package_id": package.identity, "written": written,
                "command": command}}
            self._write_series(series, record)
        return run


__all__ = ["DEFAULT_SERIES", "SERIES_FILE", "STATIC_ENTRY", "STATIC_FOLDER", "StaticPackageRunner", "archive_folders",
           "static_defaults"]
