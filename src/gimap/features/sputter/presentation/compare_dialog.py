"""Compare Scans: several Analyze results side by side (layout in ``views/sputter_compare_view.py``).

The list holds the result shown on the main screen and the results analysed under the results root (newest first);
a result is read when its box is first ticked. Every ticked result is one colour in every plot: one small plot per
quantity any of them has (``domain/compare.py`` ``CURVES``) against the nominal deposit or the time since the onset,
and one column of the growth table. Its name (the scan and the substrate, until the user changes it) names it in the
plots, the table and the exported files. Export… writes the two tables (CSV, with where each value comes from) and the
plots (PNG at 300 dpi, PDF, light colours) into a folder.

The dialog is not modal: the main screen stays usable beside it. It follows the language and the theme.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QDialog, QFileDialog, QListWidgetItem, QSizePolicy, QTableWidgetItem

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.components.table_copy import enable_table_copy
from src.gimap.app.presentation.i18n import DATA_HEADERS, apply_to, current_language, language_changed, tr, trf
from src.gimap.app.presentation.theme.engine import theme_manager

from ..application import (COMPARE_AXES, COMPARE_FILES, COMPARE_NO_SCAN, COMPARE_READ, compare_curves, compare_series,
                           growth_rows, info_rows)
from .client_texts import shown_number
from .texts import message_text, text
from .views.sputter_compare_view import CompareDialogView

SCAN_COLOURS = ("#1f77b4", "#d62728", "#2a9d5c", "#9467bd", "#e08a1e", "#17becf", "#8c564b", "#e377c2")
"""One data colour per ticked result (readable in both themes); after eight they repeat."""
READING_MARK = "s"
"""Readings from the report are drawn as open squares, the model's values as dots joined by a line."""
NO_PLOTTING = "Plotting is unavailable: {error}"
LOAD_FAILED = "{name}: {error}"
EXPORTED = "Comparison written to {folder}"
NOTHING_TO_EXPORT = "Tick at least one result to export."
QUANTITY_HEADER = "Quantity"
TEMPLATES = (NO_PLOTTING, LOAD_FAILED, EXPORTED, NOTHING_TO_EXPORT, QUANTITY_HEADER)
FOLDER_ROLE = Qt.UserRole
NAMED_ROLE = Qt.UserRole + 1
"""True once the user changed the name: reading the result then keeps it."""


class ComparePlots:
    """One small Matplotlib plot per quantity; ``drawn`` holds {key: {name: (x, values)}} as drawn."""

    def __init__(self, layout):
        self.figure = self.canvas = None
        self.drawn: dict = {}
        self._last: Optional[tuple] = None
        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
        except Exception as exc:  # matplotlib missing: the table stays
            self.error = str(exc)
            return
        self.error = ""
        self.figure = Figure(figsize=(9, 5.5), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas, 1)

    def draw(self, scans, axis: str) -> None:
        self._last = (list(scans), axis)
        self.drawn = {}
        if self.figure is None:
            return
        from src.gimap.app.presentation.theme.figures import restyle_figure
        from src.gimap.shared.figures import cjk_fallback

        figure = self.figure
        figure.clear()
        found = compare_curves(scans)
        if scans and found:
            columns = len(found) if len(found) <= 4 else 3 if len(found) <= 9 else 4
            rows = math.ceil(len(found) / columns)
            axes = figure.subplots(rows, columns, squeeze=False)
            for index, (key, name, unit, source) in enumerate(found):
                ax = axes[index // columns][index % columns]
                self.drawn[key] = {}
                for number, scan in enumerate(scans):
                    xs, values = compare_series(scan, key, axis)
                    if not xs:
                        continue
                    colour = SCAN_COLOURS[number % len(SCAN_COLOURS)]
                    style = dict(ls="none", marker=READING_MARK, mfc="none") if source == COMPARE_READ else \
                        dict(marker="o", ms=3, lw=1.2)
                    ax.plot(xs, values, color=colour, label=scan.name, **style)
                    self.drawn[key][scan.name] = (list(xs), list(values))
                title = tr(name) + (f" ({unit})" if unit else "")
                ax.set_title(title, fontsize=8.5)
                if index // columns == rows - 1 or index + columns >= len(found):
                    ax.set_xlabel(tr(dict(COMPARE_AXES)[axis]), fontsize=8)
                ax.tick_params(labelsize=7.5)
            for index in range(len(found), rows * columns):
                axes[index // columns][index % columns].set_visible(False)
            handles = [line for line in axes[0][0].get_lines()]
            if handles:
                figure.legend(handles, [line.get_label() for line in handles], loc="upper center",
                              ncol=min(4, len(handles)), fontsize=8, frameon=False)
            figure.subplots_adjust(left=0.07, right=0.98, top=0.86 if rows > 1 else 0.78, bottom=0.1, hspace=0.55,
                                   wspace=0.32)
        restyle_figure(figure)
        cjk_fallback(figure)  # drawn on a change of the list or the axis only: every text is looked at
        self.canvas.draw_idle()

    def redraw(self) -> None:
        if self._last is not None:
            self.draw(*self._last)

    def save(self, path: Path) -> Path:
        from src.gimap.app.presentation.theme.figures import exported_colors

        with exported_colors(self.figure):
            self.figure.savefig(path, dpi=300)
        return path


class CompareDialog(QDialog, CompareDialogView):
    def __init__(self, service, parent=None, *, folders=(), shown=None, ticked=()):
        """``service``: ``CompareService``; ``folders``: result folders to list (newest first); ``shown``: the folder
        of the result on the main screen (listed first, ticked); ``ticked``: other folders ticked at the start."""
        super().__init__(parent)
        self.setup_ui(self)
        self.service = service
        self.scans: dict = {}
        """Folder (str) -> ComparedScan, for the results read so far."""
        self.plots = ComparePlots(self.curves_layout)
        enable_table_copy(self.growth_table)
        self.growth_table.setProperty(DATA_HEADERS, True)
        listed = [Path(shown)] if shown is not None else []
        listed += [Path(folder) for folder in folders if shown is None or Path(folder).resolve() != Path(shown).resolve()]
        wanted = {str(Path(folder).resolve()) for folder in ticked} | ({str(Path(shown).resolve())} if shown else set())
        self.scan_list.blockSignals(True)
        for folder in listed:
            self._add_item(folder, str(folder.resolve()) in wanted)
        self.scan_list.blockSignals(False)
        self.scan_list.itemChanged.connect(self._item_changed)
        self.add_button.clicked.connect(self._choose_folder)
        self.remove_button.clicked.connect(self._remove_selected)
        self.axis_switch.activated.connect(lambda _index: self.refresh())
        self.export_button.clicked.connect(lambda: self.export())
        self.close_button.clicked.connect(self.close)
        language_changed().connect(self.refresh_language)
        theme_manager().changed.connect(self._theme_changed)
        self.refresh_language()

    # -- the list ------------------------------------------------------------------------------------------------
    def _add_item(self, folder: Path, ticked: bool) -> QListWidgetItem:
        item = QListWidgetItem(_folder_name(folder))
        item.setData(FOLDER_ROLE, str(Path(folder).resolve()))
        item.setData(NAMED_ROLE, False)
        item.setToolTip(str(folder))
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEditable)
        item.setCheckState(Qt.Checked if ticked else Qt.Unchecked)
        self.scan_list.addItem(item)
        if ticked:
            self._read(item)
        return item

    def _read(self, item: QListWidgetItem) -> bool:
        """Read the result of a ticked item once (unticked again when it cannot be read)."""
        key = item.data(FOLDER_ROLE)
        if key in self.scans:
            return True
        try:
            scan = self.service.load(Path(key), item.text() if item.data(NAMED_ROLE) else "")
        except (OSError, ValueError) as exc:
            self.scan_list.blockSignals(True)
            item.setCheckState(Qt.Unchecked)
            self.scan_list.blockSignals(False)
            show_toast(self, text(LOAD_FAILED, name=item.text(), error=message_text(exc)), level="error")
            return False
        self.scans[key] = scan
        if not item.data(NAMED_ROLE):
            self.scan_list.blockSignals(True)
            item.setText(scan.name)
            self.scan_list.blockSignals(False)
        return True

    def add_folder(self, folder, *, name: str = "") -> bool:
        """List ``folder`` ticked (or tick it when it is listed), with ``name`` when given. False when it holds no
        result."""
        key = str(Path(folder).resolve())
        item = next((self.scan_list.item(row) for row in range(self.scan_list.count())
                     if self.scan_list.item(row).data(FOLDER_ROLE) == key), None)
        self.scan_list.blockSignals(True)
        if item is None:
            item = self._add_item(Path(folder), False)
        if name:
            item.setText(name)
            item.setData(NAMED_ROLE, True)
        item.setCheckState(Qt.Checked)
        self.scan_list.blockSignals(False)
        found = self._read(item)
        self.refresh()
        return found

    def _item_changed(self, item: QListWidgetItem) -> None:
        key = item.data(FOLDER_ROLE)
        scan = self.scans.get(key)
        if scan is not None and item.text() != scan.name:  # renamed
            item.setData(NAMED_ROLE, True)
            self.scans[key] = type(scan)(item.text(), scan.result, scan.readings, scan.choices)
        if item.checkState() == Qt.Checked:
            self._read(item)
        self.refresh()

    def _choose_folder(self) -> None:
        start = self.scan_list.item(0).data(FOLDER_ROLE) if self.scan_list.count() else ""
        folder = QFileDialog.getExistingDirectory(self, tr("Add the Folder of a Result"), start)
        if folder:
            self.add_folder(Path(folder))

    def _remove_selected(self) -> None:
        for item in self.scan_list.selectedItems():
            self.scans.pop(item.data(FOLDER_ROLE), None)
            self.scan_list.takeItem(self.scan_list.row(item))
        self.refresh()

    def ticked(self) -> list:
        """The compared scans of the ticked results, in the order of the list (with the names the list shows)."""
        out = []
        for row in range(self.scan_list.count()):
            item = self.scan_list.item(row)
            scan = self.scans.get(item.data(FOLDER_ROLE))
            if item.checkState() == Qt.Checked and scan is not None:
                out.append(scan if scan.name == item.text() else type(scan)(item.text(), scan.result, scan.readings,
                                                                             scan.choices))
        return out

    def axis(self) -> str:
        return self.axis_switch.currentData() or "nominal"

    # -- what is shown -------------------------------------------------------------------------------------------
    def refresh(self) -> None:
        scans = self.ticked()
        self.empty_label.setVisible(not scans or self.plots.figure is None)
        self.empty_label.setText(tr(COMPARE_NO_SCAN) if self.plots.figure is not None
                                 else text(NO_PLOTTING, error=self.plots.error))
        if self.plots.canvas is not None:
            self.plots.canvas.setVisible(bool(scans))
        self.plots.draw(scans, self.axis())
        self._fill_growth(scans)
        self.export_button.setEnabled(bool(scans))

    def _fill_growth(self, scans) -> None:
        rows = growth_rows(scans)
        infos = info_rows(scans) if scans else []
        table = self.growth_table
        table.clear()
        table.setColumnCount(2 + len(scans))
        table.setRowCount(len(infos) + len(rows))
        table.setHorizontalHeaderLabels([tr(QUANTITY_HEADER), tr("Unit"), *(scan.name for scan in scans)])
        for row, (name, texts) in enumerate(infos):  # how each result was analysed: names and codes stay as they are
            table.setItem(row, 0, QTableWidgetItem(tr(name)))
            table.setItem(row, 1, QTableWidgetItem(""))
            for column, words in enumerate(texts):
                table.setItem(row, 2 + column, QTableWidgetItem(words))
        for row, (name, fields, unit, values) in enumerate(rows, start=len(infos)):
            shown = trf(name, **{key: tr(value) for key, value in fields.items()}) if fields else tr(name)
            table.setItem(row, 0, QTableWidgetItem(shown))
            table.setItem(row, 1, QTableWidgetItem(unit))
            for column, value in enumerate(values):
                item = QTableWidgetItem(shown_number(value))
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                table.setItem(row, 2 + column, item)

    def refresh_language(self) -> None:
        apply_to(self, current_language())
        self.refresh()

    def _theme_changed(self, *_args) -> None:
        self.plots.redraw()

    # -- Export… -------------------------------------------------------------------------------------------------
    def export(self, folder=None) -> list:
        """Write the tables and the plots into ``folder`` (asked for when None); returns the files written."""
        scans = self.ticked()
        if not scans:
            show_toast(self, tr(NOTHING_TO_EXPORT), level="warning")
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, tr("Export the Comparison to a Folder"),
                                                      str(scans[0].folder.parent))
            if not folder:
                return []
        folder = Path(folder)
        written = self.service.write_tables(scans, self.axis(), folder)
        if self.plots.figure is not None:
            for name in COMPARE_FILES[2:]:
                written.append(self.plots.save(folder / name))
        show_toast(self, text(EXPORTED, folder=str(folder)), level="success",
                   action=(tr("Open Folder"), lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))))
        return written

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt)
        try:
            language_changed().disconnect(self.refresh_language)
            theme_manager().changed.disconnect(self._theme_changed)
        except (TypeError, RuntimeError):
            pass
        super().closeEvent(event)


def _folder_name(folder: Path) -> str:
    """The name of a result before it is read: its scan folder's name without the hash."""
    parent = Path(folder).parent.name if Path(folder).name.startswith("iampred-") else Path(folder).name
    head, _sep, tail = parent.rpartition("-")
    return head if head and len(tail) == 8 else parent


__all__ = ["CompareDialog", "ComparePlots", "SCAN_COLOURS", "TEMPLATES"]
