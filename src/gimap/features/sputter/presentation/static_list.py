"""The sample list of After Deposition (a mixin of ``SputterPage``; layout in ``views/sputter_samples_view.py``).

**Add Samples…** (the bar) offers a folder — a folder of samples or one sample's folder — or one CBF file; what it
holds is read by ``application/static_samples.py``: each sample with its nominal thickness, deposition method and
bare substrate as the folder names say them, grouped into series. Samples already listed (the same frames) are not
added twice. A sample the names say too little about is listed with the reason, not ticked.

**The list** shows each series — its bare substrate and how many of its samples are analysed — with its samples:
ticked (Analyze runs it) or not, the name and the nominal thickness (double-click either to change it; a sample that
lacked its thickness is ticked once it has one) and the state: waiting, at work, failed (the reason in the tooltip),
the fit deviation of an analysed sample, or why it is left out. Choosing a sample shows its series on the main screen
at that sample, and moving along the series there chooses its row. A right-click offers Analyze (the sample or the
series), Show Result, the bare substrate of the series (a folder, or none: the metal on Si), Open Folder and Remove.

The list is part of a project and is remembered with the other values of After Deposition. Each series is analysed
into its own folder, ``<results folder>/<series>`` (Details ▸ Static Frames).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QSignalBlocker, Qt, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QFileDialog, QMenu, QTreeWidgetItem

from src.gimap.app.presentation.i18n import tr

from ..application import (NO_SAMPLE_FRAMES, NO_THICKNESS, OTHER_METAL, StaticRequest, StaticSample, complete,
                           find_samples, merged, series_of, series_slug)
from .texts import text

ADD_FOLDER = "Folder…"
ADD_FILE = "CBF File…"
ADD_FOLDER_TITLE = "Choose a Folder of Samples or One Sample's Folder"
ADD_FILE_TITLE = "Choose a CBF File"
ADDED = "{added} samples added in {series} series."
ADDED_LEFT_OUT = "{added} samples added in {series} series; {left} left out (see the list)."
ADDED_NONE = "No CBF frames were found in {folder}."
ALREADY = "These samples are in the list already."
SUMMARY_EMPTY = "No samples yet: Add Samples… takes a folder of samples."
SUMMARY = "{samples} samples in {series} series · {analysed} analysed"
SUMMARY_LEFT_OUT = "{samples} samples in {series} series · {analysed} analysed · {left} left out"
COUNT = "{count}"
SERIES_ROW = "{series} — bare: {bare} · {analysed}/{count} analysed"
BARE_NONE = "none, on Si"
STATE_WAITING, STATE_RUNNING, STATE_FAILED, STATE_STOPPED = "Waiting", "At work…", "Failed", "Stopped"
STATE_DONE = "Fit {fit}"
STATE_LEFT_OUT = "Left out"
NOTE_WORDS = {NO_THICKNESS: "Thickness?", NO_SAMPLE_FRAMES: "No frames", OTHER_METAL: "Not Au"}
"""The state column's words for a sample left out (the whole reason is in the tooltip)."""
SAMPLE_TIP = "{source}\n{frames} CBF frames · bare substrate: {bare}"
ANALYZE_SAMPLE, ANALYZE_SERIES = "Analyze This Sample", "Analyze This Series"
SHOW_RESULT, OPEN_FOLDER, REMOVE, REMOVE_SERIES = "Show Result", "Open Folder", "Remove", "Remove This Series"
SET_BARE, NO_BARE = "Bare Substrate…", "No Bare Substrate (on Si)"
BARE_TITLE = "Choose the Folder of the Bare Substrate"
NOT_ANALYSED = "Not analysed yet: tick it and press Analyze."
LIST_TEMPLATES = (ADD_FOLDER, ADD_FILE, ADD_FOLDER_TITLE, ADD_FILE_TITLE, ADDED, ADDED_LEFT_OUT, ADDED_NONE, ALREADY,
                  SUMMARY_EMPTY, SUMMARY, SUMMARY_LEFT_OUT, SERIES_ROW, BARE_NONE, STATE_WAITING, STATE_RUNNING,
                  STATE_FAILED, STATE_STOPPED, STATE_DONE, STATE_LEFT_OUT, *NOTE_WORDS.values(), SAMPLE_TIP,
                  ANALYZE_SAMPLE, ANALYZE_SERIES, SHOW_RESULT, OPEN_FOLDER, REMOVE, REMOVE_SERIES, SET_BARE, NO_BARE,
                  BARE_TITLE, NOT_ANALYSED)
"""The run-time texts of the list that need Chinese."""
SOURCE_ROLE = Qt.UserRole
SERIES_ROLE = Qt.UserRole + 1


def _fit(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.2f}"


def _same(first, second) -> bool:
    try:
        return Path(first).resolve() == Path(second).resolve()
    except OSError:
        return Path(first) == Path(second)


def _nominal(value: Optional[float]) -> str:
    return "" if value is None else f"{value:g}"


class StaticListMixin:
    """Needs the widgets of ``views/sputter_samples_view.py`` and of the bar's static inputs, ``service``,
    ``_stored`` / ``_store``, ``_notify``, ``show_client_result`` and ``_static_service``."""

    def _init_static_list(self) -> None:
        self.static_samples: list = []
        self._sample_states: dict = {}
        """source → (state, message): waiting, running, failed or stopped while Analyze works through the list."""
        self._series_results: dict = {}
        """series folder → its result (None: nothing analysed into it yet), read once and again after a run."""
        self._list_filling = False

    def _connect_static_list(self) -> None:
        menu = QMenu(self.static_add_button)
        menu.addAction(tr(ADD_FOLDER), self._choose_samples_folder)
        menu.addAction(tr(ADD_FILE), self._choose_samples_file)
        self.static_add_button.setMenu(menu)
        self.static_list_button.toggled.connect(lambda _shown: self._show_samples_panel())
        tree = self.samples_tree
        tree.itemChanged.connect(self._sample_item_changed)
        tree.currentItemChanged.connect(lambda item, _previous: self._sample_chosen(item))
        tree.itemDoubleClicked.connect(self._sample_double_clicked)
        tree.customContextMenuRequested.connect(self._samples_context_menu)
        self.samples_select_all.triggered.connect(lambda: self._tick_all(True))
        self.samples_select_none.triggered.connect(lambda: self._tick_all(False))
        self.samples_remove_left_out.triggered.connect(
            lambda: self.set_static_samples([sample for sample in self.static_samples if sample.included]))
        self.samples_remove_all.triggered.connect(lambda: self.set_static_samples([]))
        self.window_slider.valueChanged.connect(lambda _index: self._choose_row_of_window())

    # -- adding -------------------------------------------------------------------------------------------------
    def _choose_samples_folder(self) -> None:
        start = self._stored().get("static_add_folder") or str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, tr(ADD_FOLDER_TITLE), start)
        if folder:
            self.add_static_samples(folder)

    def _choose_samples_file(self) -> None:
        start = self._stored().get("static_add_folder") or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, tr(ADD_FILE_TITLE), start, "CBF frames (*.cbf);;All files (*)")
        if path:
            self.add_static_samples(path)

    def add_static_samples(self, path) -> list:
        """Add the samples ``path`` holds; returns those added."""
        found = find_samples(path)
        stored = self._stored()
        stored["static_add_folder"] = str(Path(path).parent if Path(path).is_file() else Path(path))
        self._store(stored)
        if not found:
            self._notify(text(ADDED_NONE, folder=str(path)), error=True)
            return []
        listed, known = merged(self.static_samples, found)
        added = listed[len(self.static_samples):]
        if not added:
            self._notify(tr(ALREADY))
            return []
        self.set_static_samples(complete(listed, {}))
        left = sum(1 for sample in added if not sample.included)
        series = len({sample.series for sample in added})
        self._notify(text(ADDED_LEFT_OUT if left else ADDED, added=len(added), series=series, left=left))
        shown = [sample for _series, group in series_of(self.static_samples) for sample in group if sample in added]
        first = next((sample for sample in shown if sample.included), shown[0] if shown else added[0])
        self._choose_sample_row(first.source)
        return added

    def set_static_samples(self, samples: list, *, remember: bool = True) -> None:
        """The list as ``samples`` (the states of samples no longer listed are forgotten)."""
        self.static_samples = list(samples)
        sources = {sample.source for sample in self.static_samples}
        self._sample_states = {key: value for key, value in self._sample_states.items() if key in sources}
        self._series_results = {}
        self._fill_samples()
        if remember:
            self._remember_samples()

    def _remember_samples(self) -> None:
        stored = self._stored()
        stored["static_samples"] = [sample.to_dict() for sample in self.static_samples]
        self._store(stored)

    def _restore_static_samples(self) -> None:
        stored = self._stored().get("static_samples")
        samples = [StaticSample.from_dict(values) for values in stored] if isinstance(stored, list) else []
        self.set_static_samples([sample for sample in samples if sample is not None], remember=False)

    # -- the series and their results ------------------------------------------------------------------------------
    def static_results_folder(self) -> Path:
        """The folder the series go to (Details ▸ Static Frames, else the package's default place)."""
        chosen = self.static_series.path().strip()
        if chosen:
            return Path(chosen)
        static = self._static_service()
        return static.default_series().parent if static is not None else Path("static")

    def series_folder(self, series: str) -> Path:
        return self.static_results_folder() / series_slug(series)

    def _series_result(self, series: str):
        folder = self.series_folder(series)
        key = str(folder)
        if key not in self._series_results:
            client = self.service.client
            result = None
            if client is not None and folder.is_dir():
                try:
                    result = client.open(folder)
                except (OSError, ValueError):
                    result = None
            self._series_results[key] = result if result is not None and result.static else None
        return self._series_results[key]

    def forget_series_result(self, series: str) -> None:
        self._series_results.pop(str(self.series_folder(series)), None)

    def sample_label(self, sample: StaticSample) -> str:
        return StaticRequest(frames=sample.source, name=sample.name, nominal_nm=sample.nominal_nm,
                             series_folder=self.series_folder(sample.series)).label()

    def sample_window(self, sample: StaticSample):
        """The window of the sample in its series' result (None: not analysed), as the request would analyse it."""
        result = self._series_result(sample.series)
        if result is None:
            return None
        label = self.sample_label(sample)
        window = next((window for window in result.windows if window.label == label), None)
        if window is None or window.nominal_nm != sample.nominal_nm:
            return None
        return window

    def _sample_fit(self, sample: StaticSample) -> Optional[float]:
        window = self.sample_window(sample)
        if window is None or not window.entries:
            return None
        return min((entry.fit_deviation for entry in window.entries if entry.fit_deviation is not None), default=None)

    # -- the tree --------------------------------------------------------------------------------------------------
    def _fill_samples(self) -> None:
        tree = self.samples_tree
        current = self._current_source()
        self._list_filling = True
        blocker = QSignalBlocker(tree)
        tree.clear()
        for series, samples in series_of(self.static_samples):
            parent = QTreeWidgetItem(tree)
            parent.setData(0, SERIES_ROLE, series)
            parent.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
            for sample in samples:
                item = QTreeWidgetItem(parent)
                item.setData(0, SOURCE_ROLE, sample.source)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable | Qt.ItemIsEditable)
                item.setCheckState(0, Qt.Checked if sample.included else Qt.Unchecked)
            parent.setExpanded(True)
        del blocker
        self._list_filling = False
        self._show_sample_texts()
        self._show_samples_panel()
        if current:
            self._choose_sample_row(current, show=False)

    def _show_sample_texts(self) -> None:
        """Every text of the list and the bar's summary line (again after a run or a switch of the language)."""
        tree = self.samples_tree
        self._list_filling = True
        blocker = QSignalBlocker(tree)
        by_source = {sample.source: sample for sample in self.static_samples}
        analysed_all = 0
        for row in range(tree.topLevelItemCount()):
            parent = tree.topLevelItem(row)
            series = parent.data(0, SERIES_ROLE)
            samples = [by_source[parent.child(index).data(0, SOURCE_ROLE)] for index in range(parent.childCount())]
            analysed = 0
            for index, sample in enumerate(samples):
                done = self._show_sample_row(parent.child(index), sample)
                analysed += done
            analysed_all += analysed
            bare = Path(samples[0].reference).name if samples and samples[0].reference else tr(BARE_NONE)
            words = text(SERIES_ROW, series=series, bare=bare, analysed=analysed, count=len(samples))
            parent.setText(0, words)
            parent.setToolTip(0, "\n".join((words, str(self.series_folder(series)))))
            parent.setFirstColumnSpanned(True)
        del blocker
        self._list_filling = False
        samples = self.static_samples
        left = sum(1 for sample in samples if not sample.included)
        series = len({sample.series for sample in samples})
        self.samples_count.setText(text(COUNT, count=len(samples)) if samples else "")
        if not samples:
            self.static_summary.set_full_text(tr(SUMMARY_EMPTY))
        else:
            self.static_summary.set_full_text(text(SUMMARY_LEFT_OUT if left else SUMMARY, samples=len(samples),
                                                   series=series, analysed=analysed_all, left=left))

    def _show_sample_row(self, item: QTreeWidgetItem, sample: StaticSample) -> bool:
        """The texts of one sample's row; True when it is analysed."""
        item.setText(0, sample.name)
        item.setText(1, _nominal(sample.nominal_nm) or "?")
        state, message = self._sample_states.get(sample.source, ("", ""))
        fit = self._sample_fit(sample) if not state else None
        bare = Path(sample.reference).name if sample.reference else tr(BARE_NONE)
        item.setToolTip(0, text(SAMPLE_TIP, source=sample.source, frames=sample.frames, bare=bare))
        item.setToolTip(1, item.toolTip(0))
        words, role = "", "muted"
        if state == "waiting":
            words = tr(STATE_WAITING)
        elif state == "running":
            words, role = message or tr(STATE_RUNNING), "info"
        elif state == "failed":
            words, role = tr(STATE_FAILED), "danger"
        elif state == "stopped":
            words = tr(STATE_STOPPED)
        elif sample.note:
            words, role = tr(NOTE_WORDS.get(sample.note, STATE_LEFT_OUT)), "warning"
        elif fit is not None:
            words, role = text(STATE_DONE, fit=_fit(fit)), "success"
        item.setText(2, words)
        item.setToolTip(2, message if state == "failed" else tr(sample.note) if sample.note else "")
        colour = {"info": "info", "danger": "danger", "warning": "warning", "success": "success"}.get(role)
        item.setData(2, Qt.ForegroundRole, self._role_brush(colour) if colour else None)
        return fit is not None

    def _role_brush(self, role: str):
        from PyQt5.QtGui import QBrush

        from src.gimap.app.presentation.theme.engine import theme_color
        try:
            return QBrush(theme_color(role))
        except (KeyError, ValueError):
            return None

    def _show_samples_panel(self) -> None:
        """The list beside the views: with After Deposition chosen and its switch on; its hint while it is empty."""
        static = self.analysis_kind() == "static"
        self.samples_panel.setVisible(static and self.static_list_button.isChecked())
        empty = not self.static_samples
        self.samples_tree.setVisible(not empty)
        self.samples_hint.setVisible(empty)
        self.samples_menu_button.setEnabled(not empty)

    def set_sample_state(self, sample: StaticSample, state: str, message: str = "") -> None:
        """The state of a sample while Analyze works ("": back to what its series says)."""
        if state:
            self._sample_states[sample.source] = (state, message)
        else:
            self._sample_states.pop(sample.source, None)
        item = self._sample_item(sample.source)
        if item is not None:
            self._list_filling = True
            blocker = QSignalBlocker(self.samples_tree)
            self._show_sample_row(item, sample)
            del blocker
            self._list_filling = False

    # -- rows ------------------------------------------------------------------------------------------------------
    def _sample_item(self, source: str) -> Optional[QTreeWidgetItem]:
        tree = self.samples_tree
        for row in range(tree.topLevelItemCount()):
            parent = tree.topLevelItem(row)
            for index in range(parent.childCount()):
                if parent.child(index).data(0, SOURCE_ROLE) == source:
                    return parent.child(index)
        return None

    def _current_source(self) -> str:
        item = self.samples_tree.currentItem()
        return item.data(0, SOURCE_ROLE) or "" if item is not None else ""

    def _sample_of(self, item) -> Optional[StaticSample]:
        source = item.data(0, SOURCE_ROLE) if item is not None else None
        return next((sample for sample in self.static_samples if sample.source == source), None) if source else None

    def _series_samples(self, series: str) -> list:
        return [sample for sample in self.static_samples if sample.series == series]

    def _choose_sample_row(self, source: str, *, show: bool = True) -> None:
        item = self._sample_item(source)
        if item is None:
            return
        if show:
            self.samples_tree.setCurrentItem(item)
        else:
            blocker = QSignalBlocker(self.samples_tree)
            self.samples_tree.setCurrentItem(item)
            del blocker

    def _replace_sample(self, sample: StaticSample, **changes) -> StaticSample:
        changed = replace(sample, **changes)
        self.static_samples = [changed if listed.source == sample.source else listed for listed in self.static_samples]
        return changed

    def _sample_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._list_filling:
            return
        sample = self._sample_of(item)
        if sample is None:  # a series row: Qt ticks its samples itself
            return
        changes = {}
        if column == 0:
            ticked = item.checkState(0) == Qt.Checked
            if ticked != sample.included:
                changes["included"] = ticked
            name = item.text(0).strip()
            if name and name != sample.name:
                changes["name"] = name
        elif column == 1:
            try:
                nominal = float(item.text(1).replace(",", ".").strip())
            except ValueError:
                nominal = None
            if nominal is not None and nominal > 0 and nominal != sample.nominal_nm:
                changes["nominal_nm"] = nominal
                if sample.note == NO_THICKNESS:  # the thickness the name did not give: the sample can be analysed
                    changes.update(note="", included=True)
        if changes:
            changed = self._replace_sample(sample, **changes)
            self._remember_samples()
            self._fill_samples() if "nominal_nm" in changes or "name" in changes else self._show_sample_texts()
            self._choose_sample_row(changed.source, show=False)
        else:
            self._show_sample_texts()

    def _sample_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        if self._sample_of(item) is not None and column in (0, 1):
            self.samples_tree.editItem(item, column)

    def _tick_all(self, ticked: bool) -> None:
        self.static_samples = [replace(sample, included=ticked and not sample.note) for sample in self.static_samples]
        self._remember_samples()
        self._fill_samples()

    # -- what the main screen shows ------------------------------------------------------------------------------
    def _sample_chosen(self, item) -> None:
        if self._list_filling or self.static_analyzing():
            return
        sample = self._sample_of(item)
        series = sample.series if sample is not None else (item.data(0, SERIES_ROLE) if item is not None else None)
        if not series:
            return
        result = self._series_result(series)
        if result is None:
            return
        window = self.sample_window(sample) if sample is not None else None
        self.show_client_result(result, sample=window.label if window is not None else "")
        self._remember_client_result(result.folder)

    def _choose_row_of_window(self) -> None:
        """Moving along a static series on the main screen chooses the sample's row."""
        result = self.client_result
        if result is None or not result.static or self._list_filling:
            return
        window = self.client_window()
        if window is None:
            return
        for sample in self.static_samples:
            if _same(self.series_folder(sample.series), result.folder) and self.sample_label(sample) == window.label:
                self._choose_sample_row(sample.source, show=False)
                return

    def _samples_context_menu(self, position) -> None:
        item = self.samples_tree.itemAt(position)
        if item is None:
            return
        sample = self._sample_of(item)
        series = sample.series if sample is not None else item.data(0, SERIES_ROLE)
        busy = self.running()
        menu = QMenu(self.samples_tree)
        if sample is not None:
            menu.addAction(tr(ANALYZE_SAMPLE), lambda: self.analyze_static([sample])).setEnabled(
                not busy and not sample.note and sample.nominal_nm is not None)
            shown = menu.addAction(tr(SHOW_RESULT), lambda: self._sample_chosen(item))
            shown.setEnabled(self.sample_window(sample) is not None)
        menu.addAction(tr(ANALYZE_SERIES), lambda: self.analyze_static(
            [listed for listed in self._series_samples(series) if listed.ready()])).setEnabled(not busy)
        menu.addSeparator()
        menu.addAction(tr(SET_BARE), lambda: self._choose_bare(series))
        menu.addAction(tr(NO_BARE), lambda: self._set_bare(series, ""))
        menu.addSeparator()
        if sample is not None:
            menu.addAction(tr(OPEN_FOLDER), lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(
                sample.source if Path(sample.source).is_dir() else str(Path(sample.source).parent))))
            menu.addAction(tr(REMOVE), lambda: self.set_static_samples(
                [listed for listed in self.static_samples if listed.source != sample.source])).setEnabled(not busy)
        else:
            menu.addAction(tr(REMOVE_SERIES), lambda: self.set_static_samples(
                [listed for listed in self.static_samples if listed.series != series])).setEnabled(not busy)
        menu.exec_(self.samples_tree.viewport().mapToGlobal(position))

    def _choose_bare(self, series: str) -> None:
        samples = self._series_samples(series)
        start = samples[0].reference if samples and samples[0].reference else (
            str(Path(samples[0].source).parent) if samples else str(Path.home()))
        folder = QFileDialog.getExistingDirectory(self, tr(BARE_TITLE), start)
        if folder:
            self._set_bare(series, folder)

    def _set_bare(self, series: str, folder: str) -> None:
        self.static_samples = [replace(sample, reference=folder) if sample.series == series else sample
                               for sample in self.static_samples]
        self._remember_samples()
        self._show_sample_texts()

    def refresh_static_list_language(self) -> None:
        menu = self.static_add_button.menu()
        if menu is not None:
            for action, words in zip(menu.actions(), (ADD_FOLDER, ADD_FILE)):
                action.setText(tr(words))
        self._show_sample_texts()


__all__ = ["LIST_TEMPLATES", "StaticListMixin"]
