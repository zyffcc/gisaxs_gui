"""The Compare page: several series side by side (layout in ``views/compare_page_view.py``).

Series come from Analyze (its Series map: Send to Compare there, or Add ▸ The Series Map of Analyze) or
from curve files. Every change — a series added, removed or renamed, the q range, shape only, the end
frames — compares again in the background (a second or two); the steps say where things stand.
One user action gives one notice: adding a series (or opening a project) toasts when its comparison is
ready; a change of the settings only updates the status line and the steps. Errors and written files
always toast. Errors of the files and the comparison are said in the interface language (``errors``);
dropped folders and curve files, and Remove All with its Undo, are in ``series_input``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PyQt5.QtCore import QEvent, QPoint, QSignalBlocker, Qt, QTimer
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QFileDialog, QShortcut, QTableWidgetItem, QWidget

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import tr
from src.gimap.app.presentation.stage_text import axis_symbol
from src.gimap.app.presentation.task_runner import TaskRunner
from src.gimap.app.presentation.theme import theme_manager

from ..application import CURVE_SUFFIXES, CompareService, CompareSettings, SeriesData
from .errors import error_text
from .render import CompareRenderMixin
from .series_input import CompareInputMixin, same_series
from .views.compare_page_view import RANGE_DECIMALS, RANGE_LIMITS, ComparePageView

PREFERENCES_KEY = "compare_page"
DELAY_MS = 300
"""Quiet time after a change before comparing again."""


class ComparePage(CompareRenderMixin, CompareInputMixin, QWidget, ComparePageView):
    def __init__(self, service: CompareService, *, task_runner: Optional[TaskRunner] = None, preferences=None,
                 analyze_map: Optional[Callable[[], Optional[tuple]]] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setup_ui(self)
        self.service = service
        self.tasks = task_runner or TaskRunner(self)
        self.preferences = preferences
        self._analyze_map = analyze_map
        self.series: list[SeriesData] = []
        self.comparison = None
        self._compared: list[SeriesData] = []
        """The series ``comparison`` was made from (``series`` may have changed since: saving waits)."""
        self._failure: Optional[tuple] = None
        """``(series, reason)`` of the last comparison that failed, while those are still the page's."""
        self._q_range: Optional[tuple] = None
        self._running = False
        self._again = False
        self._announce = False
        """A series was added (or a project opened): toast when the comparison is ready."""
        self._status_text: Optional[Callable[[], str]] = None
        """Makes the status line again in another interface language (``refresh_language``)."""
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(DELAY_MS)
        self._timer.timeout.connect(self.recompute)
        self._connect()
        self.setAcceptDrops(True)  # a folder or curve files dropped anywhere on the page (``series_input``)
        self.show_step("series")
        self.refresh()
        self._render()  # the empty state

    def _connect(self) -> None:
        self.step_rail.stepChosen.connect(self.show_step)
        self.add_map_action.triggered.connect(self.add_from_analyze)
        self.add_folder_action.triggered.connect(lambda: self.add_folder())
        self.add_files_action.triggered.connect(lambda: self.add_files())
        self.add_button.menu().aboutToShow.connect(self._menu_shown)
        self.empty_state.actionRequested.connect(self._add_menu_at_empty_state)
        self.remove_button.clicked.connect(self.remove_selected)
        self.delete_shortcut = QShortcut(QKeySequence(QKeySequence.Delete), self.series_table)
        self.delete_shortcut.setContext(Qt.WidgetWithChildrenShortcut)  # a name being edited keeps its Delete key
        self.delete_shortcut.activated.connect(self.remove_selected)
        self.clear_button.clicked.connect(self.clear_all)
        self.series_table.itemChanged.connect(self._renamed)
        self.q_low_spin.valueChanged.connect(lambda _value: self._range_edited())
        self.q_high_spin.valueChanged.connect(lambda _value: self._range_edited())
        self.whole_range_button.clicked.connect(self.whole_range)
        self.end_spin.valueChanged.connect(lambda _value: self._schedule())
        self.shape_check.toggled.connect(lambda _on: self._schedule())
        self.x_axis_combo.currentIndexChanged.connect(lambda _index: self._draw_change())
        self.component_combo.currentIndexChanged.connect(lambda _index: self._draw_change())
        self.distance_combo.currentIndexChanged.connect(lambda _index: self._fill_distances() if self.comparison else None)
        self.save_series_action.triggered.connect(lambda: self.save_series_table())
        self.save_frames_action.triggered.connect(lambda: self.save_frames_table())
        self.save_change_action.triggered.connect(lambda: self._save_plot(self.change_plot, "change"))
        self.save_paths_action.triggered.connect(lambda: self._save_plot(self.paths_plot, "paths"))
        self.save_end_action.triggered.connect(lambda: self._save_plot(self.end_plot, "end_states"))
        theme_manager().changed.connect(self._restyle)

    def set_analyze_source(self, provider: Optional[Callable[[], Optional[tuple]]]) -> None:
        """``provider()`` → ``(series_map, name)`` of Analyze's current Series map, or ``None``."""
        self._analyze_map = provider

    def show_step(self, key: str) -> None:
        self.step_rail.set_current(key)
        self.step_stack.setCurrentWidget(self.step_pages[key])

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.FontChange and self.comparison is not None:
            QTimer.singleShot(0, self._refit_names)  # a new font size: the names would be cut at the old width

    def _restyle(self, *_args) -> None:
        """Another theme: the series' names in colours readable on it."""
        try:
            self._color_names()
        except RuntimeError:  # the page is already gone
            self._disconnect_theme()

    def _disconnect_theme(self) -> None:
        try:
            theme_manager().changed.disconnect(self._restyle)
        except (TypeError, RuntimeError):  # not connected
            pass

    def refresh_language(self) -> None:
        """After a switch of the interface language (``i18n.language_changed``): what this page composed with
        ``tr`` at run time — the chip, the steps, the summary, the tables, the details, the components, the
        plot titles, the empty state, the reason a comparison failed and the status line (made by a function
        whenever it holds a composed text or an error) — again."""
        self.refresh()
        self._render()
        if self._status_text is not None:
            self.status_label.setText(self._status_text())

    # -- series ------------------------------------------------------------------------------

    def _unique(self, name: str, *, skip: Optional[int] = None) -> str:
        """``name``, or “name (2)” … when another series has it (``skip``: the row being renamed)."""
        names = {item.name for index, item in enumerate(self.series) if index != skip}
        if name not in names:
            return name
        index = 2
        while f"{name} ({index})" in names:
            index += 1
        return f"{name} ({index})"

    def add_series(self, data: SeriesData) -> SeriesData:
        """Adds ``data`` (its name made unique) and compares again; the same series twice is refused with
        a warning, and the one already there is returned."""
        for existing in self.series:
            if same_series(existing, data):
                self._status(lambda name=existing.name: tr("{name} is already in Compare.").format(name=name), "warning")
                return existing
        data = data.renamed(self._unique(data.name))
        self.series.append(data)
        self._announce = True  # one toast when its comparison is ready
        self._status(lambda name=data.name, rows=data.rows: tr("Added {name}: {frames} frames.").format(
            name=name, frames=rows))
        self.refresh()
        self._schedule()
        return data

    def add_map(self, series_map, name: str) -> Optional[SeriesData]:
        """Analyze's Series map (Send to Compare)."""
        try:
            return self.add_series(self.service.series_from_map(series_map, name))
        except (ValueError, AttributeError) as exc:
            self._status(lambda error=str(exc): tr("Could not add the map: {error}").format(error=error_text(error)),
                         "error")
            return None

    def _menu_shown(self) -> None:
        self.add_map_action.setEnabled(self._analyze_map is not None and self._analyze_map() is not None)

    def _add_menu_at_empty_state(self) -> None:
        """The empty state's Add Series: the same menu as the command bar's, under its button."""
        button = self.empty_state.action_button
        self.add_button.menu().popup(button.mapToGlobal(QPoint(0, button.height())))

    def add_from_analyze(self) -> Optional[SeriesData]:
        found = self._analyze_map() if self._analyze_map is not None else None
        if found is None:
            self._status(lambda: tr("Build a map in Analyze ▸ Series first."), "warning")
            return None
        return self.add_map(*found)

    def add_folder(self, folder=None) -> Optional[SeriesData]:
        folder = folder or QFileDialog.getExistingDirectory(self, tr("Folder of Curve Files"), self._remembered("folder") or "")
        if not folder:
            return None
        self._remember(folder=str(folder))
        try:
            return self.add_series(self.service.series_from_folder(Path(folder)))
        except (ValueError, OSError) as exc:
            self._status(lambda error=str(exc): error_text(error), "error")
            return None

    def add_files(self, paths=None) -> Optional[SeriesData]:
        if paths is None:
            filters = tr("Curve files") + " (" + " ".join(f"*{suffix}" for suffix in CURVE_SUFFIXES) + ")"
            paths, _ = QFileDialog.getOpenFileNames(self, tr("Curve Files of One Series"), self._remembered("folder") or "", filters)
        if not paths:
            return None
        self._remember(folder=str(Path(paths[0]).parent))
        try:
            return self.add_series(self.service.series_from_files([Path(path) for path in paths]))
        except (ValueError, OSError) as exc:
            self._status(lambda error=str(exc): error_text(error), "error")
            return None

    def remove_selected(self) -> None:
        """Remove Selected (or the Delete key in the Series table)."""
        rows = sorted({index.row() for index in self.series_table.selectedIndexes()}, reverse=True)
        rows = [row for row in rows if 0 <= row < len(self.series)]
        if not rows:
            self._status(lambda: tr("Select a series to remove"))
            return
        for row in rows:
            del self.series[row]
        self.refresh()
        self._schedule()

    def _emptied(self) -> None:
        """No series left (Remove All, or the last one removed): the empty state, and the next series
        start from their whole range."""
        self.comparison = None
        self._compared, self._failure = [], None
        self._announce = False
        self._q_range = None
        self._show_range(0.0, 0.0, limits=RANGE_LIMITS)
        self.status_label.setText("")  # the last comparison's line would be stale
        self._status_text = None
        self.refresh()
        self._render()

    def _renamed(self, item: QTableWidgetItem) -> None:
        row = item.row()
        if item.column() != 0 or not 0 <= row < len(self.series):
            return
        old, typed = self.series[row].name, item.text()
        name = " ".join(typed.split())  # a pasted line break or tab: one space
        if name and name != old:
            unique = self._unique(name, skip=row)  # two series of one name: the tables and the record mix them up
            if unique != name:
                self._status(lambda typed=name, given=unique: tr("{name} is already a series: this one is {unique}.").format(
                    name=typed, unique=given))
            name = unique
        if (name or old) != typed:
            with QSignalBlocker(self.series_table):
                item.setText(name or old)  # an emptied name: the old one back
        if not name or name == old:
            return
        self.series[row] = self.series[row].renamed(name)
        self._sync_save()  # the comparison still has the old name until it is made again
        self._schedule()

    # -- settings ------------------------------------------------------------------------------

    def settings(self) -> CompareSettings:
        return CompareSettings(q_range=self._q_range, shape_only=self.shape_check.isChecked(),
                               end_frames=self.end_spin.value())

    def _range_edited(self) -> None:
        low, high = self.q_low_spin.value(), self.q_high_spin.value()
        if high > low:
            self._q_range = (low, high)
            self._schedule()

    def whole_range(self) -> None:
        self._q_range = None
        self._schedule()

    def _show_range(self, low: float, high: float, *, limits: Optional[tuple] = None) -> None:
        """The compared range in the spins; ``limits``: how far they go (the common grid of the series),
        with a step of about a hundredth of it (0.01 Å⁻¹ for q, 1° for χ over 360°)."""
        for spin, value in ((self.q_low_spin, low), (self.q_high_spin, high)):
            with QSignalBlocker(spin):
                if limits is not None:
                    spin.setRange(*limits)
                    span = float(limits[1]) - float(limits[0])
                    if span > 0:
                        spin.setSingleStep(max(10.0 ** -RANGE_DECIMALS, 10.0 ** math.floor(math.log10(span / 100.0))))
                spin.setValue(float(value))

    # -- comparing ------------------------------------------------------------------------------

    def _schedule(self) -> None:
        self._timer.start()

    def recompute(self) -> None:
        if not self.series:
            self._emptied()
            return
        if self._running:
            self._again = True
            return
        self._running, self._again = True, False
        series, settings = list(self.series), self.settings()
        self.status_progress.show()
        self._status(lambda count=len(series): tr("Comparing {n} series …").format(n=count))
        self.tasks.submit("compare", lambda: self.service.compare(series, settings),
                          on_done=lambda comparison: self._computed(series, comparison),
                          on_error=lambda message, _trace: self._failed(series, message))

    def _finished(self) -> bool:
        self._running = False
        self.status_progress.hide()
        if self._again:
            self.recompute()
            return False
        return True

    def _current(self, series) -> bool:
        """``series`` are still the page's (no series added, removed, renamed or reopened meanwhile)."""
        return len(series) == len(self.series) and all(a is b for a, b in zip(series, self.series))

    def comparison_current(self) -> bool:
        """The comparison shown is of the page's series as they are (not before an add, removal or rename)."""
        return self.comparison is not None and self._current(self._compared)

    def failure_reason(self) -> str:
        """Why the page's series (as they are) could not be compared; empty when they have not failed."""
        if self._failure is None or not self.series or not self._current(self._failure[0]):
            return ""
        return self._failure[1]

    def _computed(self, series, comparison) -> None:
        if not self._finished():
            return  # changed meanwhile: comparing again
        if not self._current(series):  # stale (Remove All, a project opened): the steps say where things stand
            self.refresh()
            return
        self.comparison, self._compared, self._failure = comparison, list(series), None
        low, high = _outward(comparison.compared.min(), comparison.compared.max())
        self._show_range(low, high, limits=_outward(np.nanmin(comparison.q), np.nanmax(comparison.q)))
        self.refresh()
        self._render()
        announce, self._announce = self._announce, False
        self._status(lambda: self._compared_text(comparison), "ok", toast=announce)  # settings changes: status line only
        if self.step_rail.current() == "series":
            self.show_step("results")

    @staticmethod
    def _compared_text(comparison) -> str:
        count, groups = len(comparison.results), len(set(comparison.groups))
        if count < 3:
            return tr("{n} series compared.").format(n=count)
        if groups > 1:
            return tr("{n} series compared: {k} groups.").format(n=count, k=groups)
        return tr("{n} series compared: all alike.").format(n=count)

    def _failed(self, series, message: str) -> None:
        if not self._finished():
            return
        if not self._current(series):  # about series no longer here: no error to show
            self.refresh()
            return
        self.comparison, self._compared, self._failure = None, [], (list(series), str(message))
        self._announce = False
        self.refresh()
        self._render()
        self._status(lambda: tr("Could not compare: {reason}").format(reason=error_text(message)), "error")

    def refresh(self) -> None:
        """The series table, the chip and what each step says."""
        with QSignalBlocker(self.series_table):
            self.series_table.setRowCount(len(self.series))
            for row, item in enumerate(self.series):
                name = QTableWidgetItem(item.name)
                name.setToolTip(item.source)
                self.series_table.setItem(row, 0, name)
                for column, text in ((1, str(item.rows)), (2, "Analyze" if item.source == "Analyze" else Path(item.source).name)):
                    cell = QTableWidgetItem(text)
                    cell.setFlags(cell.flags() & ~Qt.ItemIsEditable)
                    cell.setToolTip(item.source)
                    self.series_table.setItem(row, column, cell)
        frames = sum(item.rows for item in self.series)
        has = bool(self.series)
        self.remove_button.setVisible(has)
        self.clear_button.setVisible(has)
        self.series_hint.setVisible(len(self.series) < 2)
        self.chip.setText(tr("{n} series · {frames} frames").format(n=len(self.series), frames=frames) if has
                          else tr("Nothing to compare yet — add a series"))
        self.step_rail.set_state("series", "ok" if has else "pending",
                                 tr("{n} series").format(n=len(self.series)) if has else tr("Add a series"))
        self.step_intro["series"].setText(tr(
            "Each series is one run or one sample: its frames in order. Two or more are compared with each other; "
            "one alone is described (odd frames, stages)."))
        comparison = self.comparison
        x_label = comparison.x_label if comparison is not None else (self.series[0].x_label if self.series else "q (Å⁻¹)")
        axis = axis_symbol(x_label)
        if comparison is None:
            if self.failure_reason():  # the series listed cannot be compared: say so, not “add a series”
                for key in ("compare", "results"):
                    self.step_rail.set_state(key, "error", tr("Could not compare"))
            elif has:  # a comparison is on its way (one is always scheduled after a change)
                self.step_rail.set_state("compare", "pending", tr("Comparing …"))
                self.step_rail.set_state("results", "busy", tr("Comparing …"))
            else:
                for key in ("compare", "results"):
                    self.step_rail.set_state(key, "pending", tr("After a series is added"))
            self._show_waiting()
        else:
            span = f"{comparison.compared.min():.4g}–{comparison.compared.max():.4g}"
            self.step_rail.set_state("compare", "ok", tr("{axis} {span} · {what}").format(
                axis=axis, span=span, what=tr("shape") if comparison.shape_only else tr("shape and level")))
            self.step_rail.set_state("results", "ok", self._results_detail(comparison))
        self.range_label.setText(tr("{axis} range").format(axis=axis))
        tip = tr("The range of {label} compared; leave out a noisy edge or a detector artefact").format(label=x_label)
        for widget in (self.q_low_spin, self.q_high_spin, self.whole_range_button):
            widget.setEnabled(has)
        for spin in (self.q_low_spin, self.q_high_spin):
            spin.setToolTip(tip)
        shape_only = comparison.shape_only if comparison is not None else self.shape_check.isChecked()
        self.distance_label.setText(tr("How different (percent of intensity, shape):") if shape_only else
                                    tr("How different (percent of intensity, shape and level):"))
        self.step_intro["compare"].setText(tr(
            "The curves are compared on the q range every series covers. Odd frames are found first and left out."))
        self.step_intro["results"].setText(tr(
            "Every series: its odd frames, its stages, by which frame half and 90 % of its change had happened, and "
            "how far its end state is from the others."))
        self._sync_save()

    def _sync_save(self) -> None:
        """Save: shown once there is a comparison, and waiting (disabled) while the series have changed
        since — its tables would pair the comparison with other series."""
        self.save_button.setVisible(self.comparison is not None)
        self.save_button.setEnabled(self.comparison_current())

    @staticmethod
    def _results_detail(comparison) -> str:
        """The Results step's one line: the groups from three series on."""
        count, groups = len(comparison.results), len(set(comparison.groups))
        if count < 3:
            return tr("{n} series compared").format(n=count)
        return tr("{n} groups").format(n=groups) if groups > 1 else tr("All alike")

    # -- status, preferences, project ---------------------------------------------------------

    def _status(self, text, level: str = "info", *, toast: Optional[bool] = None, action=None) -> None:
        """The status line; a toast for warnings and errors (``toast`` decides otherwise), shown wherever
        the user is in the window. ``action``: ``(title, callback)`` on the toast. ``text``: the line, or a
        function that makes it — made again after a switch of the interface language (a composed text, an
        error: always a function)."""
        self._status_text = text if callable(text) else None
        text = text() if callable(text) else text
        self.status_label.setText(text)
        if toast is None:
            toast = level in ("warning", "error")
        if toast and self.window().isVisible():
            if action is None:
                show_toast(self.window(), text, level=level)
            else:
                show_toast(self.window(), text, level=level, action=action)

    def _remembered(self, key: str):
        stored = self.preferences.get(PREFERENCES_KEY, {}) if self.preferences is not None else {}
        return stored.get(key) if isinstance(stored, dict) else None

    def _remember(self, **values) -> None:
        if self.preferences is None:
            return
        stored = self.preferences.get(PREFERENCES_KEY, {})
        stored = dict(stored) if isinstance(stored, dict) else {}
        stored.update(values)
        self.preferences.set(PREFERENCES_KEY, stored)

    def project_state(self, project_path=None) -> dict:
        """The settings and the series: curve files by path; Analyze maps in ``<project>.compare.npz``."""
        state = {"q_range": list(self._q_range) if self._q_range else None, "shape_only": self.shape_check.isChecked(),
                 "end_frames": self.end_spin.value(), "series": []}
        arrays = {}
        for index, item in enumerate(self.series):
            if item.paths:
                state["series"].append({"name": item.name, "files": list(item.paths)})
                continue
            arrays[f"x_{index}"], arrays[f"image_{index}"] = item.x, item.image
            state["series"].append({"name": item.name, "arrays": index, "labels": list(item.labels),
                                    "x_label": item.x_label, "source": item.source})
        if arrays and project_path is not None:
            sidecar = Path(project_path).with_suffix(".compare.npz")
            np.savez_compressed(sidecar, **arrays)
            state["arrays_file"] = sidecar.name
        elif arrays:
            state["series"] = [entry for entry in state["series"] if "files" in entry]
        return state

    def apply_project_state(self, data: dict, project_path=None) -> list[str]:
        notes: list[str] = []
        self.series.clear()
        stored = None
        if data.get("arrays_file") and project_path is not None:
            try:
                stored = np.load(Path(project_path).parent / data["arrays_file"])
            except OSError:
                notes.append(tr("Compare: {name} is missing").format(name=data["arrays_file"]))
        for entry in data.get("series") or ():
            try:
                if "files" in entry:
                    item = self.service.series_from_files([Path(path) for path in entry["files"]], entry.get("name"))
                elif stored is not None:
                    index = int(entry["arrays"])
                    item = SeriesData(entry.get("name", "series"), stored[f"x_{index}"], stored[f"image_{index}"],
                                      tuple(entry.get("labels") or ()), entry.get("x_label", "q (Å⁻¹)"),
                                      entry.get("source", "Analyze"))
                else:
                    continue
            except (ValueError, OSError, KeyError) as exc:
                notes.append(tr("Compare: {name} could not be read ({error})").format(
                    name=entry.get("name"), error=error_text(exc)))
                continue
            self.series.append(item)
        with QSignalBlocker(self.shape_check):
            self.shape_check.setChecked(bool(data.get("shape_only", True)))
        with QSignalBlocker(self.end_spin):
            self.end_spin.setValue(int(data.get("end_frames", 10)))
        rng = data.get("q_range")
        if not self.series:  # nothing to compare: no status line, range or notice left from before
            self._emptied()
            self._q_range = tuple(rng) if rng else None
            return notes
        self._q_range = tuple(rng) if rng else None
        self.comparison, self._compared, self._failure = None, [], None
        self.refresh()
        self._render()
        self._announce = True
        self._schedule()
        return notes

    def dispose(self) -> None:
        self._disconnect_theme()
        self._timer.stop()
        for plot in (self.change_plot, self.paths_plot, self.end_plot):
            plot.dispose()


def _outward(low: float, high: float) -> tuple[float, float]:
    """``low`` and ``high`` rounded outwards to the decimals the spins show, so an edge point is never
    rounded out of the compared range."""
    scale = 10.0 ** RANGE_DECIMALS
    low, high = float(low), float(high)
    down, up = math.floor(low * scale) / scale, math.ceil(high * scale) / scale
    if down > low:  # the product rounded up: one step further out
        down -= 1.0 / scale
    if up < high:
        up += 1.0 / scale
    return down, up


__all__ = ["ComparePage"]
