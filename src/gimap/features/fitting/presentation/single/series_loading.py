"""Background discovery and previews for the in-situ series.

Only snapshots and I/O callables enter workers. A new listing invalidates all old previews,
stage searches and discovery callbacks; cancelled I/O can finish without changing the page.
"""

from __future__ import annotations

import threading
from pathlib import Path

from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.app.presentation.task_runner import TaskRunner

from ...application import DiscoverInSituFramesRequest, LoadCurveRequest
from ...application.series_frames import frame_path, is_frame_pattern, matching
from ...application.series_fit import file_name
from ...application.single_fit import Curve
from .series_text import FIT_INPUT_FOLDER
from .series_imports import inspect_imported_sources

LISTING = "Listing and previewing the series … (Stop to cancel)"
LOADING = "Loading frame …"
CANCELLED = "Series loading cancelled."


def discover_sources(storage, bridge, recipe, folder, pattern, recursive, stop, *, fallback=True):
    """No QWidget access: list curves or grouped detector frames, including Analyze's output folder."""
    items, below, places = {}, None, ""

    def discover(root):
        nonlocal items
        if stop.is_set():
            return []
        if bridge is not None and is_frame_pattern(pattern):
            frames = matching(bridge.list(root), pattern)
            items = dict(bridge.grouped(frames, recipe.sum_count if recipe is not None else 1))
            return list(items)
        return [str(frame.path) for frame in storage.discover_insitu_frames(
            DiscoverInSituFramesRequest(Path(root), pattern, recursive))]

    paths = discover(folder)
    frame_mode = bridge is not None and is_frame_pattern(pattern)
    if fallback and not paths and not recursive and not frame_mode and not stop.is_set():
        root = Path(folder)
        try:
            children = sorted(child for child in root.iterdir() if child.is_dir())
        except OSError:
            children = []
        candidates = [root / FIT_INPUT_FOLDER] + [child / FIT_INPUT_FOLDER for child in children]
        found = [(place, curves) for place in candidates if place.is_dir()
                 for curves in [discover(place)] if curves]
        places = ", ".join(place.relative_to(root).as_posix() for place, _ in found)
        if len(found) == 1:
            below, paths = found[0]
    if fallback and not paths and not frame_mode and bridge is not None and not stop.is_set():
        try:
            frames = bridge.list(folder)
        except (OSError, ValueError, RuntimeError):
            frames = []
        if frames:
            pattern = f"*{frame_path(frames[0][0]).suffix.lower()}"
            frames = matching(frames, pattern)
            items = dict(bridge.grouped(frames, recipe.sum_count if recipe is not None else 1))
            paths = list(items)
    return paths, pattern, below, places, items


def read_units(paths, stated, stop):
    units = {}
    for path in paths:
        if stop.is_set():
            break
        try:
            units[path] = stated(path) if stated is not None else None
        except (OSError, ValueError) as exc:
            units[path] = exc
    return units


class FitSeriesLoadingMixin:
    def setup_series_loading(self):
        self.loading_tasks = TaskRunner(self)
        self.preview_tasks = TaskRunner(self)
        self._listing_busy = False
        self._loading_epoch = 0
        self._loading_stop = threading.Event()
        self._preview_pending = set()
        self._path_versions = {}
        self._disposed = False
        self._restore_listing = None

    def _invalidate_loading(self):
        self._loading_epoch += 1
        self._loading_stop.set()
        self._loading_stop = threading.Event()
        self.loading_tasks.cancel("listing")
        self.stage_tasks.cancel("series-stages")
        self._stages_timer.stop()
        self._preview_pending.clear()
        self._read_errors = {}
        self._frame_cuts = {}
        self._frame_cut_stamps = {}
        self._path_versions = {}
        self._stages_key = None
        self._watch_scan += 1
        self._watch_busy = False

    def _loading_current(self, epoch):
        return not self._disposed and epoch == self._loading_epoch

    def _begin_listing(self, again):
        self._invalidate_loading()
        epoch, stop = self._loading_epoch, self._loading_stop
        folder = self.folder
        sources = self._imported_sources
        inspect = self.view_model.inspect_curve_table if sources is not None else None
        pattern = self.pattern_edit.text().strip() or "*_fit_input.dat"
        recursive = self.subfolders_check.isChecked()
        if sources is None and self.frames_bridge is not None and self._recipe is None:
            self._take_recipe()  # Analyze's recipe is captured on the GUI thread
        recipe, bridge, storage = self._recipe, self.frames_bridge, self.view_model.storage
        loader, make_loader = self.series_curve_loader(), self.make_curve_loader
        stated = getattr(self.view_model, "stated_q_unit", None)
        unit = "angstrom" if sources is None and bridge is not None and is_frame_pattern(pattern) else self.source_unit_setting()
        self._listing_busy = True
        if again:  # the old results were already saved or discarded before this folder request
            self.paths, self._names, self.fits, self._curves = [], None, {}, {}
            self._selected = self._start_model = None
            self._preview_curves, self._unsaved, self._untouched_start = {}, False, False
            self._clear_frame()
            self._forget_stages()
            self._frames_changed()
        self._status(lambda: tr(LISTING))
        self.refresh()

        def work():
            result = (discover_sources(storage, bridge, recipe, folder, pattern, recursive, stop) if sources is None else
                      ([source.token for source in sources], pattern, None, "", {}))
            paths, chosen_pattern, _below, _places, items = result
            frame_mode = sources is None and is_frame_pattern(chosen_pattern)
            errors = inspect_imported_sources(sources, inspect, stop) if sources is not None else {}
            # First frame and the few extra frames used to decide the halves before Start.
            load = loader if not items else make_loader(
                load=loader, bridge=bridge, recipe=recipe, why="", items=items, cache={})
            read = []
            if paths:
                for index in dict.fromkeys((0, len(paths) // 2, len(paths) - 1, min(1, len(paths) - 1))):
                    if stop.is_set():
                        break
                    read.append((index, load(LoadCurveRequest(Path(paths[index]),
                                "angstrom" if frame_mode else unit))))
            units = {} if frame_mode else read_units(paths, stated, stop)
            return result, read, units, load.loaded, errors

        self.loading_tasks.submit("listing", work,
            on_done=lambda result: self._listing_ready(epoch, again, recursive, unit, result),
            on_error=lambda message, _details: self._listing_failed(epoch, message))
        return True  # accepted; the status line reports completion or failure

    def _listing_ready(self, epoch, again, recursive, unit, result):
        if not self._loading_current(epoch):
            return
        self._listing_busy = False
        listing, read, self._stated_units, cuts, errors = result
        self._read_errors.update(errors)
        self._accept_listing(listing, read, again=again, recursive=recursive, unit=unit)
        self._publish_frame_cuts(cuts)
        if self._restore_listing is not None:
            data, self._restore_listing = self._restore_listing, None
            self._restore_series_choices(data)
        if self.imported_mode() and self._read_errors:
            path, reason = next(iter(self._read_errors.items()))
            self._status(lambda: trf("{count} imported curves cannot be read: {name} — {reason}",
                         count=len(self._read_errors), name=file_name(path), reason=tr(reason)), "warning")
        self.refresh()

    def _listing_failed(self, epoch, message):
        if not self._loading_current(epoch):
            return
        self.close_series()
        self._status(lambda: tr("Could not list the curves: {reason}").format(reason=message), "error")
        self.refresh()

    def _cache_preview(self, index, outcome, unit):
        path = self.paths[index]
        if outcome.error is not None:
            self._read_errors[path] = outcome.error.message
            return
        self._read_errors.pop(path, None)
        value = outcome.value
        self._note_curve(index, value.q, value.observation)
        self._preview_curves[(path, unit)] = Curve.from_arrays(
            value.q, value.intensity, value.error, name=file_name(path), path=path,
            unit=value.q_source_unit, unit_from=value.q_unit_from)

    def _request_preview(self, index, key):
        if key in self._preview_pending or self._listing_busy or self._disposed:
            return
        self._preview_pending.add(key)
        epoch, load = self._loading_epoch, self.series_curve_loader()
        version = self._path_versions.get(key[0], 0)
        request = LoadCurveRequest(Path(key[0]), key[1])

        def current():
            return self._loading_current(epoch) and self._path_versions.get(key[0], 0) == version

        def done(outcome):
            if not current() or index >= len(self.paths) or self.paths[index] != key[0]:
                return
            self._preview_pending.discard(key)
            self._cache_preview(index, outcome, key[1])
            self._publish_frame_cuts(load.loaded)
            if self._selected == index:
                self._select(index)
            self.refresh()

        def failed(message, _details):
            if not current():
                return
            self._preview_pending.discard(key)
            self._read_errors[key[0]] = message
            if self._selected == index:
                self._select(index)

        self.preview_tasks.submit(f"preview:{key}", lambda: load(request), on_done=done, on_error=failed)

    def _invalidate_curve(self, path):
        """Watch saw a changed source: old previews and stage reads cannot republish it."""
        self._path_versions[path] = self._path_versions.get(path, 0) + 1
        self._frame_cuts.pop(path, None)
        self._frame_cut_stamps.pop(path, None)
        self._read_errors.pop(path, None)
        for key in [key for key in self._preview_curves if key[0] == path]:
            self._preview_curves.pop(key)
        for key in [key for key in self._preview_pending if key[0] == path]:
            self._preview_pending.discard(key)
            self.preview_tasks.cancel(f"preview:{key}")

    def _cancel_loading(self):
        self._listing_busy = False
        self._restore_listing = None
        self.close_series()
        self._status(lambda: tr(CANCELLED))

    def _dispose_loading(self):
        self._disposed = True
        self._invalidate_loading()
        self.loading_tasks.shutdown(2000)
        self.preview_tasks.shutdown(2000)
