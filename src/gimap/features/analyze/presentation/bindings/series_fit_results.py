"""Browse actual batch fit outcomes in Series without opening exported files."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QSignalBlocker, QTimer, Qt
from PyQt5.QtWidgets import QTableWidgetItem

from src.gimap.app.presentation.i18n import tr, trf

from ...application import FIT_NONE, FrameFitPreview


def parameter_label(column: str) -> str:
    """Table keys are scientific data; explicitly label length parameters and reciprocal units."""
    text = column.replace("(1/A)", "(Å⁻¹)")
    if column.split()[-1:] and column.split()[-1] in ("R", "D", "h"):
        text += " (nm)"
    return {"chi2": "χ²", "log rmse": "log RMSE"}.get(text, text)


def _number(value) -> str:
    if isinstance(value, (int, float, np.number)):
        return f"{value:.6g}" if math.isfinite(float(value)) else ""
    return str(value if value is not None else "")


class SeriesFitResultsMixin:
    def _connect_series_fit_results(self) -> None:
        self._series_fit_run = None
        self._series_fit_entries = []
        self._series_fit_follow = True
        self._series_fit_finished = False
        self._series_fit_pick_epoch = 0
        self._series_fit_timer = QTimer(self)
        self._series_fit_timer.setSingleShot(True)
        self._series_fit_timer.setInterval(250)
        self._series_fit_timer.timeout.connect(self._series_fit_render)
        self.series_fit_table.currentCellChanged.connect(self._series_fit_selected)
        self.series_fit_parameter.currentIndexChanged.connect(lambda _index: self._series_fit_draw_trend())

    def _series_fit_start(self, run) -> None:
        if run.choices.fit == FIT_NONE:
            return
        self.view_model.release_batch_previews(self._series_fit_run)
        self._series_fit_timer.stop()
        self._series_fit_run = run
        self._series_fit_entries = []
        self._series_fit_follow = True
        self._series_fit_finished = False
        self.series_fit_results.show()
        self.series_fit_table.clearContents()
        self.series_fit_table.setRowCount(0)
        self.series_fit_plot.set_curves([])
        self.series_fit_trend_plot.set_curves([])
        self.series_fit_note.setText("")
        self._series_fit_render()

    def _series_fit_changed(self, *, finished: bool = False) -> None:
        if self._series_fit_run is None or self._series_fit_run is not self._batch_run:
            return  # another run, or Clear while workers still finish their files
        self._series_fit_finished = bool(finished)
        if finished:
            self._series_fit_timer.stop()
            self._series_fit_render()
        elif not self._series_fit_timer.isActive():
            self._series_fit_timer.start()

    def _reset_series_fit(self) -> None:
        self._series_fit_timer.stop()
        self.tasks.cancel("fit-preview")
        self._series_fit_pick_epoch += 1
        self.view_model.release_batch_previews(self._series_fit_run)
        self._series_fit_run = None
        self._series_fit_entries = []
        self.series_fit_results.hide()
        self.series_fit_table.setRowCount(0)
        self.series_fit_plot.set_curves([])
        self.series_fit_trend_plot.set_curves([])

    def _series_fit_render(self) -> None:
        run = self._series_fit_run
        if run is None:
            return
        table = self.series_fit_table
        current = table.currentRow()
        picked = ((self._series_fit_entries[current].ref, self._series_fit_entries[current].number)
                  if 0 <= current < len(self._series_fit_entries) else None)
        entries = list(run.fit_previews)
        known = {entry.label for entry in entries}
        for failed in run.failed:
            if failed["frame"] not in known:
                entries.append(FrameFitPreview(str(failed["frame"]), None, (Path(failed.get("file") or "."), 0), False,
                                               {"frame": failed["frame"], "note": failed["why"]}, note=failed["why"]))
        entries.sort(key=lambda entry: entry.number if entry.number is not None else math.inf)
        self._series_fit_entries = entries
        columns = run.fit_table.columns() if run.fit_table is not None else ["frame", "note"]
        columns = ["frame", "state", *[column for column in columns if column != "frame"]]
        labels = {"frame": tr("Frame"), "state": tr("State"), "halves": tr("Halves"), "points": tr("Points"),
                  "model": tr("Model"), "note": tr("Note")}
        with QSignalBlocker(table):
            table.setColumnCount(len(columns))
            table.setHorizontalHeaderLabels([labels.get(column, parameter_label(column)) for column in columns])
            table.setRowCount(len(entries))
            for row, entry in enumerate(entries):
                for index, column in enumerate(columns):
                    value = tr("Fitted" if entry.ok else "Failed") if column == "state" else entry.values.get(column, "")
                    if column == "halves":
                        value = tr({"mean": "Mean of both halves", "positive": "q > 0 half", "negative": "q < 0 half"}.get(value, value))
                    elif column == "note":
                        value = tr(str(value))
                    item = QTableWidgetItem(_number(value))
                    item.setToolTip(trf("{file}\nFrame {n}", file=entry.ref[0], n=entry.ref[1] + 1) if column == "frame"
                                    else _number(value))
                    if column == "state" and not entry.ok:
                        item.setData(Qt.UserRole, "failed")
                    table.setItem(row, index, item)
            table.resizeColumnsToContents()
            # Long source names and reasons can be copied whole without widening the panel.
            for index in range(len(columns)):
                table.setColumnWidth(index, min(260, table.columnWidth(index)))
            wanted = next((index for index, entry in enumerate(entries) if (entry.ref, entry.number) == picked), -1)
            if self._series_fit_follow or wanted < 0:
                wanted = len(entries) - 1
            if wanted >= 0:
                table.setCurrentCell(wanted, 0)
        fitted = sum(entry.ok for entry in entries)
        state = tr("Stopped" if run.stopped else "Complete" if self._series_fit_finished else "Running")
        self.series_fit_summary.setText(trf("{state}: {done}/{total} frames · {fitted} fitted · {failed} failed",
                                           state=state, done=len(entries), total=run.requested, fitted=fitted,
                                           failed=len(entries) - fitted))
        if self._series_fit_follow and wanted >= 0:
            table.scrollToItem(table.item(wanted, 0))
        self._series_fit_parameters(columns)
        self._series_fit_draw_selected()

    def _series_fit_selected(self, row: int, *_previous) -> None:
        if row < 0:
            return
        self._series_fit_follow = False
        self._series_fit_draw_selected()

    def _series_fit_draw_selected(self) -> None:
        row = self.series_fit_table.currentRow()
        plot = self.series_fit_plot
        if not 0 <= row < len(self._series_fit_entries):
            plot.set_curves([])
            plot.set_empty_text(tr("Fits appear here as each frame finishes."))
            return
        entry = self._series_fit_entries[row]
        self._series_fit_pick_epoch += 1
        epoch, run = self._series_fit_pick_epoch, self._series_fit_run
        self.tasks.cancel("fit-preview")
        if entry.cache_key is not None:
            plot.set_curves([])
            plot.set_empty_text(tr("Loading fitted points…"))
            plot.set_title(entry.label)
            self.series_fit_note.setText(tr("Loading fitted points…"))

            def show(curves):
                if self._series_fit_run is run and self._series_fit_pick_epoch == epoch:
                    self._series_fit_draw_curves(entry, curves)

            def failed(message, _details):
                if self._series_fit_run is run and self._series_fit_pick_epoch == epoch:
                    note = trf("Could not load this fit preview: {error}", error=message)
                    self.series_fit_note.setText(note)
                    plot.set_empty_text(note)

            self.tasks.submit("fit-preview", lambda: self.view_model.batch_preview_curves(run, entry),
                              on_done=show, on_error=failed)
            return
        self._series_fit_draw_curves(entry, entry.curves)

    def _series_fit_draw_curves(self, entry, previews) -> None:
        plot = self.series_fit_plot
        plot.set_title(entry.label)
        curves, markers = [], []
        for curve in previews:
            if curve.observed.size == curve.x.size:
                curves.append((tr("Points fitted") + (f" · {curve.name}" if len(previews) > 1 else ""), curve.x, curve.observed))
                markers.append(True)
            curves.append((tr("Fit") + f" · {curve.name}", curve.x, curve.fitted))
            markers.append(False)
        plot.set_curves(curves, markers=markers)
        plot.plot.setLabel("bottom", "|q| (Å⁻¹)")
        plot.plot.setLabel("left", tr("Intensity"))
        note = tr(entry.note) if entry.note else ""
        if not entry.ok:
            plot.set_empty_text(note or tr("No fitted curve for this frame."))
        else:
            used = entry.values.get("halves") or ""
            halves = tr({"mean": "Mean of both halves", "positive": "q > 0 half", "negative": "q < 0 half"}.get(used, used))
            if halves:
                note = trf("{halves} · {points} points fitted · q in Å⁻¹; sizes in nm", halves=halves,
                           points=entry.values.get("points", "")) + (f" · {note}" if note else "")
            if previews and all(not curve.observed.size for curve in previews):
                note += " " + tr("Model only: the fit did not return its observed points.")
        self.series_fit_note.setText(note)

    def _series_fit_parameters(self, columns) -> None:
        combo = self.series_fit_parameter
        picked = combo.currentData()
        available = []
        for column in columns:
            if column in ("frame", "state", "points") or column.endswith(" err"):
                continue
            if any(isinstance(entry.values.get(column), (int, float, np.number)) and
                   np.isfinite(float(entry.values[column])) for entry in self._series_fit_entries):
                available.append(column)
        with QSignalBlocker(combo):
            combo.clear()
            for column in available:
                combo.addItem(parameter_label(column), column)
            if picked in available:
                combo.setCurrentIndex(available.index(picked))
        self._series_fit_draw_trend()

    def _series_fit_draw_trend(self) -> None:
        key = self.series_fit_parameter.currentData()
        plot = self.series_fit_trend_plot
        if key is None:
            plot.set_curves([])
            plot.set_empty_text(tr("No fitted parameter values yet."))
            return
        entries = self._series_fit_entries
        x = np.asarray([entry.number if entry.number is not None else index + 1 for index, entry in enumerate(entries)], dtype=float)
        y = np.asarray([entry.values.get(key, math.nan) if entry.ok else math.nan for entry in entries], dtype=float)
        plot.set_curves([(parameter_label(key), x, y)], markers=[True])
        plot.set_title(parameter_label(key))
        plot.plot.setLabel("bottom", tr("Frame number"))
        plot.plot.setLabel("left", parameter_label(key))

    def _series_fit_language(self) -> None:
        self.series_fit_results.setTitle(tr("Results: batch fitting"))
        self.series_fit_tabs.setTabText(0, tr("Fit curve"))
        self.series_fit_tabs.setTabText(1, tr("Parameter trend"))
        self._series_fit_render()


__all__ = ["SeriesFitResultsMixin", "parameter_label"]
