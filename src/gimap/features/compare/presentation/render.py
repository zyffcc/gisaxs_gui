"""Showing a comparison: the summary, the tables, the three plots, and saving them.

Long series names (“lyx_cu_peo1_20pl_2p5fr_116_00002”) are shortened for the legends, the result tables,
the summary and the rail (``display_names``: “peo1_116”); the editable Series table, the tooltips, the
CSV and the JSON keep the full names.

The change plot marks each series' odd frames with a × in its colour; the paths plot marks where each path
starts (a hollow ○: its first kept frame) and ends (a filled ■: its last). These marks have no legend entry of
their own (``CurvePlot.set_curves(legend=…)``). Plots are saved on the light plot palette (``export_plot``).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem

from src.gimap.app.presentation.components import text_color
from src.gimap.app.presentation.components.curve_plot import CURVE_COLORS, Marker
from src.gimap.app.presentation.components.plot_export import export_plot
from src.gimap.app.presentation.i18n import tr
from src.gimap.app.presentation.stage_text import axis_symbol, change_text, odd_reason

from ..application import METHOD
from .errors import error_text
from .views.compare_page_view import EMPTY_TEXT, EMPTY_TITLE, GROUP_COLUMN, fit_columns, fit_to_rows

FAILED_TITLE = "Could not compare these series"
FAILED_HINT = ("Remove the series that does not fit (Series step) or widen the range (Whole Range, Compare step): "
               "the series are compared again by themselves.")
WAITING_TEXT = "The comparison appears here when it is ready."
START_MARK = Marker("o", 9.0, hollow=True)
"""Where a path starts: the series' first kept frame."""
END_MARK = Marker("s", 9.0)
"""Where a path ends: the series' last kept frame."""
ODD_MARK = Marker("x", 10.0)
"""An odd frame on the change plot (left out of the comparison), in the colour of its series (big enough to
read as a ×, not a dot, on a line of the same colour)."""

_COPY = re.compile(r"(.*?)( \(\d+\))?", re.DOTALL)
"""A name and the “ (2)” that made it unique (a pasted line break is part of the name)."""


def series_color(index: int) -> str:
    return CURVE_COLORS[index % len(CURVE_COLORS)]


def display_names(names: Sequence[str]) -> list[str]:
    """Short names: each name split on ``_`` without the parts every name shares (“peo1_116” from
    “lyx_cu_peo1_20pl_2p5fr_116_00002”); a copy's “ (2)” is kept. The full names when a short one would be
    empty (one series) or two would be the same."""
    names = [str(name) for name in names]
    split = []
    for name in names:
        match = _COPY.fullmatch(name)
        base, copy = (match.group(1), match.group(2) or "") if match else (name, "")
        split.append((base.split("_"), copy))
    if not split:
        return []
    shared = set.intersection(*(set(parts) for parts, _copy in split))
    bases = ["_".join(part for part in parts if part not in shared) for parts, _copy in split]
    short = [base + copy for base, (_parts, copy) in zip(bases, split)]
    if not all(bases) or len(set(short)) < len(short):
        return names
    return short


def _same_text(low: float, high: float) -> bool:
    return f"{low:.0f}" == f"{high:.0f}"


class CompareRenderMixin:
    """Needs the widgets of ``ComparePageView``, ``series``, ``comparison`` with ``_compared`` (the series it
    was made from), ``failure_reason()``, ``service`` and ``_status``."""

    def _render(self) -> None:
        comparison = self.comparison
        self.method_label.setText(tr(METHOD))
        has = comparison is not None
        self.plot_stack.setCurrentWidget(self.plots_page if has else self.empty_page)
        for combo in (self.x_axis_combo, self.component_combo):
            combo.setEnabled(has)
        for widget in (self.results_table, self.details_section):
            widget.setVisible(has)
        if not has:
            self._shown_names = []
            self._show_waiting()
            self.results_table.setRowCount(0)
            self.results_table.setColumnHidden(GROUP_COLUMN, True)
            self.distance_table.setRowCount(0)
            self.distance_table.setColumnCount(0)
            for table in (self.results_table, self.distance_table):
                fit_to_rows(table)
            self.distance_row.hide()
            self.distance_table.hide()
            self.details_label.setText("")
            for plot in (self.change_plot, self.paths_plot, self.end_plot):
                plot.set_curves([])
                plot.reset_view()
            self.component_combo.clear()
            return
        self._shown_names = display_names(comparison.names)
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

    def _show_waiting(self) -> None:
        """No comparison to show: the empty state and the summary say why — no series yet, a comparison on
        its way, or the reason the listed series could not be compared."""
        reason = self.failure_reason()  # the English of the error (“no common q range”): said in zh too
        failure = error_text(reason) if reason else ""
        if failure:
            self.empty_state.set_content(tr(FAILED_TITLE), failure + "\n" + tr(FAILED_HINT), "")
            self.summary_label.setText(tr("Could not compare: {reason}").format(reason=failure))
        elif self.series:
            self.empty_state.set_content(tr("Comparing …"), tr(WAITING_TEXT), "")
            self.summary_label.setText(tr(WAITING_TEXT))
        else:
            self.empty_state.set_content(tr(EMPTY_TITLE), tr(EMPTY_TEXT), tr("Add Series"))
            self.summary_label.setText(tr("Add a series: the comparison appears here."))

    def _shown(self, index: int) -> str:
        """The name shown for series ``index`` of the comparison (short when the names are long)."""
        names = getattr(self, "_shown_names", None) or []
        return names[index] if index < len(names) else self.comparison.results[index].name

    # -- text and tables ------------------------------------------------------------------------

    def _label(self, index: int, row: Optional[int]) -> str:
        """A short name of a frame: “frame 89” for a frame of a multi-frame file, else the file's stem
        (``index``: a series of the comparison, whose frames may no longer be the page's)."""
        labels = self._compared[index].labels if index < len(self._compared) else ()
        if row is None:
            return "—"
        label = labels[row] if row < len(labels) else str(row + 1)
        if " #" in label:
            return tr("frame {n}").format(n=label.rsplit(" #", 1)[1].split(" ")[0])
        stem = Path(label).stem
        return stem if len(stem) <= 22 else stem[:9] + "…" + stem[-12:]

    def _summary(self) -> str:
        comparison = self.comparison
        count = len(comparison.results)
        names = [self._shown(index) for index in range(count)]
        lines = []
        if count == 2:
            lines.append(tr("{a} and {b}: end states {end:.0f} % apart ({start:.0f} % at the start).").format(
                a=names[0], b=names[1], end=float(comparison.distance[0, 1]), start=float(comparison.start_distance[0, 1])))
        elif count >= 3:
            lines += self._group_lines(names)
            if len(set(comparison.groups)) > 1:
                lines.append(self._apart_line(comparison.distance, names, start=False))
            lines.append(self._apart_line(comparison.start_distance, names, start=True))
        share = 100.0 * float(comparison.explained[0])
        lines.append(tr("The first main change carries {share:.0f} % of all the change between frames.").format(share=share))
        odd = sum(len(result.odd) for result in comparison.results)
        if odd:
            lines.append(tr("{n} odd frames left out (Odd frames and stages, below).").format(n=odd))
        return "\n".join(lines)

    def _group_lines(self, names: list[str]) -> list[str]:
        members: dict[int, list[str]] = {}
        for name, group in zip(names, self.comparison.groups):
            members.setdefault(int(group), []).append(name)
        if len(members) == 1:
            return [tr("No clear groups: the end states are all alike.")]
        return [tr("Group {n}: {names}").format(n=group, names=", ".join(found)) for group, found in sorted(members.items())]

    @staticmethod
    def _apart_line(matrix: np.ndarray, names: list[str], *, start: bool) -> str:
        """Which series differs most, with one value when its distances print the same."""
        values = np.array(matrix, dtype=float)
        np.fill_diagonal(values, np.nan)
        apart = values[np.isfinite(values)]
        if apart.size and _same_text(float(apart.min()), float(apart.max())):  # no series stands out
            return (tr("At the start, every two series are {value:.0f} % apart.") if start else
                    tr("The end states of every two series are {value:.0f} % apart.")).format(value=float(apart.min()))
        far = int(np.nanargmax(np.nanmean(values, axis=1)))
        low, high = float(np.nanmin(values[far])), float(np.nanmax(values[far]))
        if _same_text(low, high):
            return (tr("At the start, {name} differs most ({value:.0f} %).") if start else
                    tr("{name} differs most from the others (end states {value:.0f} % apart).")).format(
                name=names[far], value=low)
        return (tr("At the start, {name} differs most ({low:.0f}–{high:.0f} %).") if start else
                tr("{name} differs most from the others (end states {low:.0f}–{high:.0f} % apart).")).format(
            name=names[far], low=low, high=high)

    def _fill_results(self) -> None:
        comparison = self.comparison
        table = self.results_table
        table.setRowCount(len(comparison.results))
        for row, result in enumerate(comparison.results):
            cells = [self._shown(row), str(comparison.groups[row]), str(result.rows), str(len(result.odd)),
                     str(result.stages.count) if result.stages else "—",
                     self._label(row, result.half_row), self._label(row, result.ninety_row)]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column == 0:
                    item.setToolTip(result.name)
                table.setItem(row, column, item)
        table.setColumnHidden(GROUP_COLUMN, len(comparison.results) < 3)
        self._color_names(table)
        fit_columns(table)
        fit_to_rows(table)

    def _color_names(self, *tables) -> None:
        """The series' names in the colour of their curves — lighter on a dark theme, where the curve colours
        are too faint for text (again on every switch of the theme)."""
        for table in tables or (self.results_table, self.distance_table):
            for row in range(table.rowCount()):
                item = table.item(row, 0)
                if item is not None:
                    item.setForeground(_brush(text_color(series_color(row))))

    def _fill_distances(self) -> None:
        comparison = self.comparison
        table = self.distance_table
        count = len(comparison.results)
        self.distance_row.setVisible(count >= 2)
        table.setVisible(count >= 2)
        names = [self._shown(index) for index in range(count)]
        table.setColumnCount(count + 1)
        table.setHorizontalHeaderLabels([""] + names)
        for column, result in enumerate(comparison.results, start=1):
            table.horizontalHeaderItem(column).setToolTip(result.name)
        table.setRowCount(count)
        values = comparison.start_distance if self.distance_combo.currentData() == "start" else comparison.distance
        for row, name in enumerate(names):
            item = QTableWidgetItem(name)
            item.setToolTip(comparison.results[row].name)
            table.setItem(row, 0, item)
            for column in range(count):
                table.setItem(row, column + 1, QTableWidgetItem("—" if row == column else f"{values[row, column]:.0f} %"))
        self._color_names(table)
        fit_columns(table)
        fit_to_rows(table)

    def _refit_names(self) -> None:
        """The columns fitted to their names and titles again (after a font change)."""
        if self.comparison is None:
            return
        for table in (self.results_table, self.distance_table):
            fit_columns(table)
            fit_to_rows(table)

    def _details(self) -> str:
        comparison = self.comparison
        axis = axis_symbol(comparison.x_label)
        lines = []
        for index, result in enumerate(comparison.results):
            stages = result.stages
            if stages is not None:
                ranges = ", ".join(f"{self._label(index, first)}–{self._label(index, last)}" for first, last in stages.ranges())
                lines.append(tr("{name}: {count} stages ({ranges})").format(name=result.name, count=stages.count, ranges=ranges)
                             if stages.count > 1 else tr("{name}: one stage").format(name=result.name))
                lines += ["   " + change_text(change, axis) for change in stages.stage_changes()]
            for frame in result.odd:
                lines.append("   " + tr("odd: {label} — {why}").format(label=self._label(index, frame.row),
                                                                       why=odd_reason(frame, axis)))
        return "\n".join(lines)

    # -- plots ------------------------------------------------------------------------------------

    def _draw_change(self) -> None:
        comparison = self.comparison
        if comparison is None:
            return
        component = max(0, self.component_combo.currentData() or 0)
        share = (self.x_axis_combo.currentData() or "frame") == "share"
        lines, marks = [], []  # ``(curve, colour)``: a line per series; its odd frames as ×, over every line
        for index, result in enumerate(comparison.results):
            if component >= result.scores.shape[1]:
                continue
            odd = sorted({frame.row for frame in result.odd})
            rows = np.setdiff1d(np.arange(result.rows), odd).astype(int)
            name, color = self._shown(index), series_color(index)
            for kept, chosen, label in ((lines, rows, name), (marks, np.array(odd, dtype=int), f"{name} (odd)")):
                if chosen.size:
                    x = chosen / max(result.rows - 1, 1) if share else chosen + 1.0
                    kept.append(((label, x.astype(float), result.scores[chosen, component]), color))
        shown = lines + marks
        self.change_plot.set_labels("share of the series" if share else "frame", f"component {component + 1}")
        self.change_plot.set_curves([curve for curve, _color in shown], [color for _curve, color in shown],
                                    markers=[False] * len(lines) + [ODD_MARK] * len(marks),
                                    legend=[True] * len(lines) + [False] * len(marks))

    def _draw_paths(self) -> None:
        comparison = self.comparison
        self.paths_plot.set_title(tr("Paths through the two main changes"))
        if comparison.results[0].scores.shape[1] < 2:
            self.paths_plot.set_curves([])  # its empty text (the view): a second main change is needed
            return
        clouds, ends = [], []  # ``(curve, colour, marker)``: the kept frames as dots; where each path starts and ends
        for index, result in enumerate(comparison.results):
            odd = {frame.row for frame in result.odd}
            rows = [row for row in range(result.rows) if row not in odd]
            name, color = self._shown(index), series_color(index)
            clouds.append(((name, result.scores[rows, 0], result.scores[rows, 1]), color, True))
            for row, mark, which in ((rows[:1], START_MARK, "first"), (rows[-1:], END_MARK, "last")):
                if row:
                    ends.append(((f"{name} ({which})", result.scores[row, 0], result.scores[row, 1]), color, mark))
        shown = clouds + ends  # the marks over every cloud
        self.paths_plot.set_curves([curve for curve, _color, _mark in shown], [color for _curve, color, _mark in shown],
                                   markers=[mark for _curve, _color, mark in shown],
                                   legend=[True] * len(clouds) + [False] * len(ends))

    def _draw_end(self) -> None:
        comparison = self.comparison
        curves = [(self._shown(index), comparison.q, result.end_curve) for index, result in enumerate(comparison.results)]
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
        if not self.comparison_current():  # a series added, removed or renamed since: the rows would not match
            self._status(lambda: tr("The series changed: save when they are compared again."), "warning")
            return None
        path = path or QFileDialog.getSaveFileName(self, title, str(Path(self._folder()) / name), "CSV (*.csv)")[0]
        if not path:
            return None
        try:
            written = write(list(self._compared), self.comparison, Path(path))  # the series it was made from
        except (OSError, ValueError) as exc:
            self._status(lambda error=str(exc): tr("Could not save: {error}").format(error=error), "error")
            return None
        self._written(lambda: tr("Saved {name} and its record.").format(name=written.name), written)
        return written

    def _save_plot(self, plot, name: str, path=None) -> Optional[Path]:
        if self.comparison is None:
            return None
        path = path or QFileDialog.getSaveFileName(self, tr("Save Plot"), str(Path(self._folder()) / f"compare_{name}.png"),
                                                   "PNG image (*.png);;SVG vector (*.svg)")[0]
        if not path:
            return None
        try:
            export_plot(plot, path)  # PNG 1600 px wide or SVG on the light plot palette; OSError: nothing written
        except (OSError, ValueError, RuntimeError) as exc:
            self._status(lambda error=str(exc): tr("Could not save: {error}").format(error=error), "error")
            return None
        self._written(lambda name=Path(path).name: tr("Saved {name}.").format(name=name), Path(path))
        return Path(path)

    def _written(self, text, path: Path) -> None:
        """A file was written: say so (``text``: the line, or a function that makes it), with Open Folder."""
        folder = Path(path).resolve().parent
        self._status(text, "ok", toast=True,
                     action=(tr("Open Folder"), lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))))


def _brush(color: str):
    from PyQt5.QtGui import QBrush, QColor

    return QBrush(QColor(color))


__all__ = ["END_MARK", "FAILED_HINT", "FAILED_TITLE", "ODD_MARK", "START_MARK", "WAITING_TEXT", "CompareRenderMixin",
           "display_names", "series_color"]
