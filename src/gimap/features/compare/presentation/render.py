"""Showing a comparison: the summary, the tables, the three plots, and saving them."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem

from src.gimap.app.presentation.components.curve_plot import CURVE_COLORS
from src.gimap.app.presentation.i18n import tr
from src.gimap.app.presentation.stage_text import change_text, odd_reason

from ..application import METHOD


def series_color(index: int) -> str:
    return CURVE_COLORS[index % len(CURVE_COLORS)]


class CompareRenderMixin:
    """Needs the widgets of ``ComparePageView``, ``series``, ``comparison``, ``service`` and ``_status``."""

    def _render(self) -> None:
        comparison = self.comparison
        self.method_label.setText(tr(METHOD))
        if comparison is None:
            self.summary_label.setText(tr("Add a series: the comparison appears here."))
            for table in (self.results_table, self.distance_table):
                table.setRowCount(0)
            self.details_label.setText("")
            for plot in (self.change_plot, self.paths_plot, self.end_plot):
                plot.set_curves([])
            self.component_combo.clear()
            return
        self.summary_label.setText(self._summary())
        self._fill_results()
        self._fill_distances()
        self.details_label.setText(self._details())
        with_signals_blocked = self.component_combo.blockSignals(True)
        current = max(0, self.component_combo.currentIndex())
        self.component_combo.clear()
        for index in range(comparison.results[0].scores.shape[1]):
            share = 100.0 * float(comparison.explained[index])
            self.component_combo.addItem(tr("Component {n} ({share:.0f} %)").format(n=index + 1, share=share), index)
        self.component_combo.setCurrentIndex(min(current, self.component_combo.count() - 1))
        self.component_combo.blockSignals(with_signals_blocked)
        self._draw_change()
        self._draw_paths()
        self._draw_end()

    # -- text and tables ------------------------------------------------------------------------

    def _label(self, index: int, row: Optional[int]) -> str:
        """A short name of a frame: “frame 89” for a frame of a multi-frame file, else the file's stem."""
        labels = self.series[index].labels
        if row is None:
            return "—"
        label = labels[row] if row < len(labels) else str(row + 1)
        if " #" in label:
            return tr("frame {n}").format(n=label.rsplit(" #", 1)[1].split(" ")[0])
        stem = Path(label).stem
        return stem if len(stem) <= 22 else stem[:9] + "…" + stem[-12:]

    def _summary(self) -> str:
        comparison = self.comparison
        lines = []
        if len(comparison.results) >= 3:
            lines.append(tr("Groups by the end state: {groups}").format(groups=tr(comparison.group_text())))
        if len(comparison.results) >= 2:
            distance = comparison.distance.copy()
            np.fill_diagonal(distance, np.nan)
            mean = np.nanmean(distance, axis=1)
            far = int(np.nanargmax(mean))
            lines.append(tr("{name} differs most from the others (end states {low:.0f}–{high:.0f} % apart).").format(
                name=comparison.names[far], low=np.nanmin(distance[far]), high=np.nanmax(distance[far])))
            start = comparison.start_distance.copy()
            np.fill_diagonal(start, np.nan)
            first = int(np.nanargmax(np.nanmean(start, axis=1)))
            lines.append(tr("At the start, {name} differs most ({low:.0f}–{high:.0f} %).").format(
                name=comparison.names[first], low=np.nanmin(start[first]), high=np.nanmax(start[first])))
        share = 100.0 * float(comparison.explained[0])
        lines.append(tr("The first main change carries {share:.0f} % of all the change between frames.").format(share=share))
        odd = sum(len(result.odd) for result in comparison.results)
        if odd:
            lines.append(tr("{n} odd frames left out (Odd frames and stages, below).").format(n=odd))
        return "\n".join(lines)

    def _fill_results(self) -> None:
        comparison = self.comparison
        table = self.results_table
        table.setRowCount(len(comparison.results))
        for row, result in enumerate(comparison.results):
            cells = [result.name, str(result.rows), str(len(result.odd)),
                     str(result.stages.count) if result.stages else "—",
                     self._label(row, result.half_row), self._label(row, result.ninety_row), str(comparison.groups[row])]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column == 0:
                    item.setForeground(_brush(series_color(row)))
                table.setItem(row, column, item)

    def _fill_distances(self) -> None:
        comparison = self.comparison
        table = self.distance_table
        names = comparison.names
        table.setColumnCount(len(names) + 1)
        table.setHorizontalHeaderLabels([""] + names)
        table.setRowCount(len(names))
        values = comparison.start_distance if self.distance_combo.currentData() == "start" else comparison.distance
        for row, name in enumerate(names):
            table.setItem(row, 0, QTableWidgetItem(name))
            for column in range(len(names)):
                value = values[row, column]
                table.setItem(row, column + 1, QTableWidgetItem("" if row == column else f"{value:.0f} %"))

    def _details(self) -> str:
        comparison = self.comparison
        lines = []
        for index, result in enumerate(comparison.results):
            stages = result.stages
            if stages is not None:
                ranges = ", ".join(f"{self._label(index, first)}–{self._label(index, last)}" for first, last in stages.ranges())
                lines.append(tr("{name}: {count} stages ({ranges})").format(name=result.name, count=stages.count, ranges=ranges)
                             if stages.count > 1 else tr("{name}: one stage").format(name=result.name))
                lines += ["   " + change_text(change) for change in stages.stage_changes()]
            for frame in result.odd:
                lines.append("   " + tr("odd: {label} — {why}").format(label=self._label(index, frame.row), why=odd_reason(frame)))
        return "\n".join(lines)

    # -- plots ------------------------------------------------------------------------------------

    def _draw_change(self) -> None:
        comparison = self.comparison
        if comparison is None:
            return
        component = max(0, self.component_combo.currentData() or 0)
        share = (self.x_axis_combo.currentData() or "frame") == "share"
        curves, colors = [], []
        for index, result in enumerate(comparison.results):
            odd = {frame.row for frame in result.odd}
            rows = np.array([row for row in range(result.rows) if row not in odd])
            if not rows.size or component >= result.scores.shape[1]:
                continue
            x = rows / max(result.rows - 1, 1) if share else rows + 1.0
            curves.append((result.name, x.astype(float), result.scores[rows, component]))
            colors.append(series_color(index))
        self.change_plot.set_labels("share of the series" if share else "frame", f"component {component + 1}")
        self.change_plot.set_curves(curves, colors)

    def _draw_paths(self) -> None:
        comparison = self.comparison
        if comparison.results[0].scores.shape[1] < 2:
            self.paths_plot.set_title(tr("One main change only: no second direction"))
            self.paths_plot.set_curves([])
            return
        self.paths_plot.set_title(tr("Paths through the two main changes"))
        curves, colors, markers = [], [], []
        for index, result in enumerate(comparison.results):
            odd = {frame.row for frame in result.odd}
            rows = [row for row in range(result.rows) if row not in odd]
            curves.append((result.name, result.scores[rows, 0], result.scores[rows, 1]))
            colors.append(series_color(index))
            markers.append(True)
        self.paths_plot.set_curves(curves, colors, markers=markers)

    def _draw_end(self) -> None:
        comparison = self.comparison
        curves = [(result.name, comparison.q, result.end_curve) for result in comparison.results]
        self.end_plot.set_title(tr("End states (mean of the last {n} frames)").format(n=comparison.end_frames))
        self.end_plot.set_labels(comparison.x_label, "Intensity")
        self.end_plot.set_curves(curves, [series_color(index) for index in range(len(curves))])

    # -- saving -------------------------------------------------------------------------------------

    def _folder(self) -> str:
        for item in self.series:
            if item.source and item.source != "Analyze":
                return item.source
        return self._remembered("folder") or ""

    def save_series_table(self, path=None) -> Optional[Path]:
        return self._save_table(path, "compare_series.csv", tr("Save Table of Every Series"), self.service.export_series_table)

    def save_frames_table(self, path=None) -> Optional[Path]:
        return self._save_table(path, "compare_frames.csv", tr("Save Table of Every Frame"), self.service.export_frames_table)

    def _save_table(self, path, name: str, title: str, write) -> Optional[Path]:
        if self.comparison is None:
            return None
        path = path or QFileDialog.getSaveFileName(self, title, str(Path(self._folder()) / name), "CSV (*.csv)")[0]
        if not path:
            return None
        try:
            written = write(self.series, self.comparison, Path(path))
        except (OSError, ValueError) as exc:
            self._status(tr("Could not save: {error}").format(error=exc), "error")
            return None
        self._status(tr("Saved {name} and its record.").format(name=written.name), "ok")
        return written

    def _save_plot(self, plot, name: str, path=None) -> Optional[Path]:
        if self.comparison is None:
            return None
        path = path or QFileDialog.getSaveFileName(self, tr("Save Plot"), str(Path(self._folder()) / f"compare_{name}.png"),
                                                   "PNG image (*.png);;SVG vector (*.svg)")[0]
        if not path:
            return None
        from pyqtgraph import exporters

        exporter = exporters.SVGExporter(plot.plot) if str(path).lower().endswith(".svg") else exporters.ImageExporter(plot.plot)
        if isinstance(exporter, exporters.ImageExporter):
            exporter.parameters()["width"] = 1600
        exporter.export(str(path))
        self._status(tr("Saved {name}.").format(name=Path(path).name), "ok")
        return Path(path)


def _brush(color: str):
    from PyQt5.QtGui import QBrush, QColor

    return QBrush(QColor(color))


__all__ = ["CompareRenderMixin", "series_color"]
