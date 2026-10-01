"""The Compare page: several series side by side (layout in ``views/compare_page_view.py``).

Series come from Analyze (its Series map: Send to Compare there, or Add ▸ The Series Map of Analyze) or
from curve files. Every change — a series added, removed or renamed, the q range, shape only, the end
frames — compares again in the background (a second or two); the steps say where things stand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PyQt5.QtCore import QSignalBlocker, Qt, QTimer
from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem, QWidget

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import tr
from src.gimap.app.presentation.task_runner import TaskRunner

from ..application import CURVE_SUFFIXES, CompareService, CompareSettings, SeriesData
from .render import CompareRenderMixin
from .views.compare_page_view import ComparePageView

PREFERENCES_KEY = "compare_page"
DELAY_MS = 300
"""Quiet time after a change before comparing again."""


class ComparePage(CompareRenderMixin, QWidget, ComparePageView):
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
        self._q_range: Optional[tuple] = None
        self._running = False
        self._again = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(DELAY_MS)
        self._timer.timeout.connect(self.recompute)
        self._connect()
        self.show_step("series")
        self.refresh()

    def _connect(self) -> None:
        self.step_rail.stepChosen.connect(self.show_step)
        self.add_map_action.triggered.connect(self.add_from_analyze)
        self.add_folder_action.triggered.connect(lambda: self.add_folder())
        self.add_files_action.triggered.connect(lambda: self.add_files())
        self.add_button.menu().aboutToShow.connect(self._menu_shown)
        self.remove_button.clicked.connect(self.remove_selected)
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

    def set_analyze_source(self, provider: Optional[Callable[[], Optional[tuple]]]) -> None:
        """``provider()`` → ``(series_map, name)`` of Analyze's current Series map, or ``None``."""
        self._analyze_map = provider

    def show_step(self, key: str) -> None:
        self.step_rail.set_current(key)
        self.step_stack.setCurrentWidget(self.step_pages[key])

    # -- series ------------------------------------------------------------------------------

    def _unique(self, name: str) -> str:
        names = {item.name for item in self.series}
        if name not in names:
            return name
        index = 2
        while f"{name} ({index})" in names:
            index += 1
        return f"{name} ({index})"

    def add_series(self, data: SeriesData) -> SeriesData:
        data = data.renamed(self._unique(data.name))
        self.series.append(data)
        self._status(tr("Added {name}: {frames} frames.").format(name=data.name, frames=data.rows), "ok")
        self.refresh()
        self._schedule()
        return data

    def add_map(self, series_map, name: str) -> Optional[SeriesData]:
        """Analyze's Series map (Send to Compare)."""
        try:
            return self.add_series(self.service.series_from_map(series_map, name))
        except (ValueError, AttributeError) as exc:
            self._status(tr("Could not add the map: {error}").format(error=exc), "error")
            return None

    def _menu_shown(self) -> None:
        self.add_map_action.setEnabled(self._analyze_map is not None and self._analyze_map() is not None)

    def add_from_analyze(self) -> Optional[SeriesData]:
        found = self._analyze_map() if self._analyze_map is not None else None
        if found is None:
            self._status(tr("Build a map in Analyze ▸ Series first."), "warning")
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
            self._status(str(exc), "error")
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
            self._status(str(exc), "error")
            return None

    def remove_selected(self) -> None:
        rows = sorted({index.row() for index in self.series_table.selectedIndexes()}, reverse=True)
        for row in rows:
            if 0 <= row < len(self.series):
                del self.series[row]
        if rows:
            self.refresh()
            self._schedule()

    def clear_all(self) -> None:
        self.series.clear()
        self.comparison = None
        self.refresh()
        self._render()

    def _renamed(self, item: QTableWidgetItem) -> None:
        row = item.row()
        if item.column() != 0 or not 0 <= row < len(self.series):
            return
        name = item.text().strip()
        if not name or name == self.series[row].name:
            return
        self.series[row] = self.series[row].renamed(name)
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

    def _show_range(self, low: float, high: float) -> None:
        for spin, value in ((self.q_low_spin, low), (self.q_high_spin, high)):
            with QSignalBlocker(spin):
                spin.setValue(float(value))

    # -- comparing ------------------------------------------------------------------------------

    def _schedule(self) -> None:
        self._timer.start()

    def recompute(self) -> None:
        if not self.series:
            self.comparison = None
            self.refresh()
            self._render()
            return
        if self._running:
            self._again = True
            return
        self._running, self._again = True, False
        series, settings = list(self.series), self.settings()
        self.status_progress.show()
        self._status(tr("Comparing {n} series …").format(n=len(series)))
        self.tasks.submit("compare", lambda: self.service.compare(series, settings),
                          on_done=lambda comparison: self._computed(series, comparison),
                          on_error=lambda message, _trace: self._failed(message))

    def _finished(self) -> bool:
        self._running = False
        self.status_progress.hide()
        if self._again:
            self.recompute()
            return False
        return True

    def _computed(self, series, comparison) -> None:
        if not self._finished() or [item.name for item in series] != [item.name for item in self.series]:
            return
        self.comparison = comparison
        self._show_range(float(comparison.compared.min()), float(comparison.compared.max()))
        self.refresh()
        self._render()
        text = tr(comparison.group_text()) or tr("{n} series compared.").format(n=len(series))
        self._status(text, "ok")
        if self.step_rail.current() == "series":
            self.show_step("results")

    def _failed(self, message: str) -> None:
        if self._finished():
            self.comparison = None
            self.refresh()
            self._render()
            self._status(tr("Could not compare: {reason}").format(reason=message), "error")

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
        if comparison is None:
            self.step_rail.set_state("compare", "pending", tr("After a series is added"))
            self.step_rail.set_state("results", "busy" if self._running else "pending",
                                     tr("Comparing …") if self._running else tr("After a series is added"))
        else:
            span = f"{comparison.compared.min():.4g}–{comparison.compared.max():.4g}"
            self.step_rail.set_state("compare", "ok", tr("q {span} · {what}").format(
                span=span, what=tr("shape") if comparison.shape_only else tr("shape and level")))
            groups = len(set(comparison.groups))
            self.step_rail.set_state("results", "ok", tr(comparison.group_text()) if groups > 1 else
                                     tr("{n} series compared").format(n=len(comparison.results)))
        self.step_intro["compare"].setText(tr(
            "The curves are compared on the q range every series covers. Odd frames are found first and left out."))
        self.step_intro["results"].setText(tr(
            "Every series: its odd frames, its stages, by which frame half and 90 % of its change had happened, and "
            "how far its end state is from the others."))
        self.save_button.setVisible(comparison is not None)

    # -- status, preferences, project ---------------------------------------------------------

    def _status(self, text: str, level: str = "info") -> None:
        self.status_label.setText(text)
        if level in ("ok", "warning", "error") and self.isVisible():
            show_toast(self.window(), text, level=level)

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
                notes.append(tr("Compare: {name} could not be read ({error})").format(name=entry.get("name"), error=exc))
                continue
            self.series.append(item)
        rng = data.get("q_range")
        self._q_range = tuple(rng) if rng else None
        with QSignalBlocker(self.shape_check):
            self.shape_check.setChecked(bool(data.get("shape_only", True)))
        with QSignalBlocker(self.end_spin):
            self.end_spin.setValue(int(data.get("end_frames", 10)))
        self.comparison = None
        self.refresh()
        self._render()
        if self.series:
            self._schedule()
        return notes

    def dispose(self) -> None:
        self._timer.stop()
        for plot in (self.change_plot, self.paths_plot, self.end_plot):
            plot.dispose()


__all__ = ["ComparePage"]
