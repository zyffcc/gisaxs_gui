"""A GIMaP project: what is open in Analyze and Fitting, in one file, to reopen a sample as it was left.

``<name>.gimap`` is JSON: the frames listed in Analyze with its whole set-up (mode, geometry, αi,
masks and corrections, cuts), Single analysis (the curve, its halves, fitting range, left-out
points, the model and the method), In-situ series (the folder of curves, the frames, the halves of
the cut every frame is fitted on, how each starts) and Compare (its settings and series: curve
files by path, maps from Analyze in ``<name>.<generation>.compare.npz`` next to the project). Paths are kept as
they are; what is no longer there is reported when the project opens (``missing``, and the notes
of ``apply``).

Opening a project replaces what the window held: Analyze's frames, the curve of Single analysis, the
In-situ series and Compare's series are those of the project — none when it names none — so a later
Save Project never writes the work of before into it.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from .presentation.i18n import tr
from ..features.fitting.application.models import InSituSourceFrame

PROJECT_FORMAT = "gimap-project"
PROJECT_VERSION = 1
PROJECT_SUFFIX = ".gimap"
PROJECT_FILTER = "GIMaP project (*.gimap);;All files (*)"
NO_SERIES_FOLDER = ""
"""What In-situ series says in its bar with no folder (the text its view starts with): nothing — its frame plot
says what to do."""


def collect(components, page: str = "", path=None) -> dict:
    """The project of the open window (``path``: where it is saved, for Compare's maps next to it)."""
    workspace = components.fitting_workspace
    data = {
        "format": PROJECT_FORMAT, "version": PROJECT_VERSION, "saved": datetime.now().isoformat(timespec="seconds"),
        "page": page,
        "analyze": components.analyze_page.project_state(),
        "fitting": {"single": workspace.fit_page.project_state(), "series": workspace.series_page.project_state()},
    }
    compare = getattr(components, "compare_page", None)
    if compare is not None:
        data["compare"] = compare.project_state(path)
    sputter = getattr(components, "sputter_page", None)
    if sputter is not None:
        data["sputter"] = sputter.project_state()
    return data


def save(components, path, page: str = "") -> Path:
    """Publish a complete project in one replacement, keeping the previous save until then.

    Compare writes each save's arrays to a new generation; its previous arrays must remain intact while
    collecting and serializing the new JSON. Both files are flushed before the JSON becomes visible.
    """
    path = Path(path)
    if path.suffix.lower() != PROJECT_SUFFIX:
        path = path.with_suffix(PROJECT_SUFFIX)
    previous_sidecar = _owned_sidecar(path)
    generation = path.with_name(f"{path.stem}.{uuid4().hex}{PROJECT_SUFFIX}")
    sidecar = generation.with_suffix(".compare.npz")
    temporary = None
    committed = False
    try:
        content = json.dumps(collect(components, page, generation), indent=2, ensure_ascii=False)
        if sidecar.exists():
            with sidecar.open("r+b") as stream:
                os.fsync(stream.fileno())
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        os.close(descriptor)  # Windows cannot replace a temporary file while its handle is open.
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        committed = True
    finally:
        _discard(temporary)
        if not committed:
            _discard(sidecar)
    # Cleanup cannot turn an already committed save into a failure (a viewer may hold an old NPZ open).
    _discard(previous_sidecar)
    return path


def _discard(path: Path | None) -> None:
    if path is not None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _owned_sidecar(path: Path) -> Path | None:
    """Only retire the previous file named by this project, never arbitrary neighboring files.

    Older projects used ``<name>.compare.npz``; copied/edited projects can refer to other projects' arrays.
    Such references remain readable but are not ours to remove.
    """
    try:
        previous = read(path)
    except (OSError, ValueError):
        return None
    name = _part(previous, "compare").get("arrays_file")
    if not isinstance(name, str):
        return None
    legacy = path.with_suffix(".compare.npz").name
    generated = rf"{re.escape(path.stem)}\.[0-9a-f]{{32}}\.compare\.npz"
    if name != legacy and re.fullmatch(generated, name) is None:
        return None
    sidecar = path.parent / name
    return None if sidecar.is_symlink() else sidecar


def read(path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except ValueError as exc:  # not JSON, or not text: the parser's own words would mean nothing to the user
        raise ValueError("not a GIMaP project") from exc
    if not isinstance(data, dict) or data.get("format") != PROJECT_FORMAT:
        raise ValueError("not a GIMaP project")
    version = data.get("version", 0)
    if type(version) is not int or version < 0:
        raise ValueError("not a GIMaP project")
    if version > PROJECT_VERSION:
        raise ValueError("made by a newer GIMaP")
    return data


def _part(data, *keys) -> dict:
    """``data[key][key]…`` when every step is a dict, else an empty one (a project file edited by hand)."""
    for key in keys:
        data = data.get(key) if isinstance(data, dict) else None
    return data if isinstance(data, dict) else {}


def missing(data: dict, path=None) -> list[str]:
    """The files and folders ``data`` names that are no longer there, each once, in the order of the pages:
    Analyze's frames, the curve of Single analysis, the folder of the In-situ series, Compare's curve files
    and its maps next to the project (``path``). Only looked for, nothing is read."""
    named: list[tuple[str, bool]] = []  # (path, it is there)
    files = _part(data, "analyze").get("files")
    named += [(str(item), Path(str(item)).exists()) for item in (files if isinstance(files, list) else ())]
    curve = _part(data, "fitting", "single", "curve").get("path")
    if curve:
        try:
            source = InSituSourceFrame.from_token(str(curve)).path
        except ValueError:
            pass  # apply_project_state reports a malformed mapping, not a missing physical file
        else:
            named.append((str(source), source.is_file()))
    fit_series = _part(data, "fitting", "series")
    folder = fit_series.get("folder")
    if fit_series.get("source_mode") == "selection":
        sources = fit_series.get("sources")
        for source in sources if isinstance(sources, list) else ():
            if isinstance(source, dict) and source.get("path"):
                item = Path(str(source["path"]))
                named.append((str(item), item.is_file()))
    elif folder:
        named.append((str(folder), Path(str(folder)).is_dir()))
    compare = _part(data, "compare")
    series = compare.get("series")
    for entry in series if isinstance(series, list) else ():
        files = entry.get("files") if isinstance(entry, dict) else None
        named += [(str(item), Path(str(item)).exists()) for item in (files if isinstance(files, list) else ())]
    if compare.get("arrays_file") and path is not None:
        sidecar = Path(path).parent / str(compare["arrays_file"])
        named.append((str(sidecar), sidecar.exists()))
    for key in ("scan_folder", "mask", "result_folder"):
        value = _part(data, "sputter").get(key)
        if value:
            named.append((str(value), Path(value).exists()))
    gone, seen = [], set()
    for item, there in named:
        if not there and item not in seen:
            seen.add(item)
            gone.append(item)
    return gone


def _close_curve(page) -> None:
    """Single analysis with no curve: the one of before goes with its fitting range, left-out points, fit and
    solutions (the model stays: the project brings its own), and is not opened again at the next start."""
    close = getattr(page, "close_curve", None)  # the page's own way, when it has one
    if close is not None:
        close()
        return
    if page.fit_running():  # cannot be while a project opens (menus refuse); never a fit of a closed curve
        page.stop_fit()
    session = page.session
    clear = getattr(session, "clear_curve", None)
    if clear is not None:
        clear()
    else:
        session.curve = session.q_range = session.result = session.result_selection = None
        session.solutions, session.solutions_selection, session.excluded = [], None, set()
    # The halves chosen for the curve that went are not those of the next one: the page's own start, unless
    # the project says. (An In-situ series has its own halves, kept in the project: ``fitting.series.side``.)
    session.side = "mean"
    remember = getattr(page, "_remember", None)
    if remember is not None:
        remember(curve=None)
    page._status_text = None  # the line about the curve of before ("Opened …") is not said again
    page.status_label.setText("")
    for spin in (page.range_min_spin, page.range_max_spin):  # the page fills them only with a curve
        spin.blockSignals(True)
        spin.setValue(0.0)
        spin.blockSignals(False)
    page.show_step("curve")  # not the Results of the curve that went: where the page starts


def _close_series(page) -> None:
    """In-situ series with no folder: the curves, fits and stages of before go; the page is as it starts."""
    close = getattr(page, "close_series", None)  # the page's own way, when it has one
    if close is not None:
        close()
        return
    page.folder, page.paths, page.fits, page.queue = "", [], {}, []  # (a page without ``close_series``)
    page._curves, page._preview_curves, page._selected, page._start_model = {}, {}, None, None
    page._previous = page._last_done = page._drawn_with = None
    page.folder_chip.setText(tr(NO_SERIES_FOLDER))
    page.folder_chip.setToolTip("")
    for spin in (page.first_spin, page.last_spin, page.every_spin):
        spin.blockSignals(True)
        spin.setRange(1, 1)
        spin.setValue(1)
        spin.blockSignals(False)
    page.frame_plot.set_title("")
    page.frame_plot.set_curves([])
    page.stages_label.setText("")
    page.stages_label.setToolTip("")
    page._status_text = None  # the line about the series of before is not said again
    page.status_label.setText("")
    page._forget_stages()
    page._frames_changed()  # the empty list, the steps and the empty trend
    page.show_step("curves")  # not the Results of the series that went: where the page starts


def apply(components, data: dict, path=None) -> list[str]:
    """Open a project in the window; returns notes on what could not be restored.

    Every page is told, also about a part the project does not have: its work of before is closed first, so
    what the window holds afterwards is the project and nothing else."""
    notes: list[str] = []
    workspace = components.fitting_workspace
    notes += components.analyze_page.apply_project_state(_part(data, "analyze"))
    _close_curve(workspace.fit_page)  # opened again just below when the project names one that is there
    notes += workspace.fit_page.apply_project_state(_part(data, "fitting", "single"))
    series = _part(data, "fitting", "series")
    folder = str(series.get("folder") or "")
    selection = series.get("source_mode") == "selection"
    if not workspace.series_page.running and not selection and not (folder and Path(folder).is_dir()):
        _close_series(workspace.series_page)
        if series.get("pattern"):  # the page sets it with the folder; with none, the project's all the same
            workspace.series_page.pattern_edit.setText(str(series["pattern"]))
    notes += workspace.series_page.apply_project_state(series)
    compare = getattr(components, "compare_page", None)
    if compare is not None:  # a project without Compare (an older file): no series, not those of before
        notes += compare.apply_project_state(_part(data, "compare"), path)
    sputter = getattr(components, "sputter_page", None)
    if sputter is not None:
        notes += sputter.apply_project_state(_part(data, "sputter"))
    return notes


__all__ = ["PROJECT_FILTER", "PROJECT_FORMAT", "PROJECT_SUFFIX", "apply", "collect", "missing", "read", "save"]
