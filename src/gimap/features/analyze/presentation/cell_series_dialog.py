"""Track the indexed rings through the selected physical frames, fit their cells, plot and export."""

from pathlib import Path
from threading import Event

import numpy as np
from PyQt5.QtCore import QObject, QSignalBlocker, Qt, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QDialog, QFileDialog

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import language_changed, tr, trf

from ..application.cell_series import refine_series
from .series_texts import MAP_TIME
from .views.cell_series_view import CellSeriesView


class _Progress(QObject):
    changed = pyqtSignal(int, int)


def open_cell_series(panel, phases, results, indexing_source):
    previous = getattr(panel, "_cell_series_dialog", None)
    if previous is not None:
        previous.close()
    dialog = CellSeriesDialog(panel, panel._page)
    panel._cell_series_dialog = dialog
    dialog.finished.connect(lambda _: setattr(panel, "_cell_series_dialog", None))
    panel.finished.connect(dialog.close)
    dialog.show()
    dialog.start(tuple(phases), results, indexing_source)
    return dialog


class CellSeriesDialog(QDialog, CellSeriesView):
    def __init__(self, parent, page):
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self._page = page
        self._record = None
        self._source = None
        self._cancel = Event()
        self._key = f"cell_series_{id(self)}"
        self._progress = _Progress()
        self._progress.changed.connect(self._running)
        self._done = self._total = 0
        self._error = None
        self.setup_ui()
        self.job.cancelRequested.connect(self.cancel)
        self.phase_combo.currentIndexChanged.connect(self._phase_changed)
        self.parameter_combo.currentIndexChanged.connect(self._plot)
        self.export_button.clicked.connect(self._export)
        self.finished.connect(self.cancel)
        language_changed().connect(self._language)

    def start(self, phases, results, source):
        model = self._page.view_model
        numbers = model.frame_numbers()
        first, last = self._page.series_first_spin.value(), self._page.series_last_spin.value()
        step = max(1, self._page.series_step_spin.value())
        frames = model.all_frames().within(numbers, first, last or None)[::step]
        template = model.request_for(Path("."), summed=())
        self._source = source["analysis"]
        self._source_line()
        self.phase_combo.addItems([r["phase"] for r in results if r.get("seeds")])
        self._running(0, len(frames))
        event = self._cancel
        progress = self._progress.changed.emit

        def check_cancel():
            if event.is_set():
                raise InterruptedError("Cancelled")

        self._page.tasks.submit(
            self._key, lambda: refine_series(frames, numbers, template, model.analyze, phases, results, source,
                                             check_cancel=check_cancel, progress=progress),
            on_done=self._ready, on_error=self._failed)

    def cancel(self, *_):
        self._cancel.set()
        self._page.tasks.cancel(self._key)
        if self._record is None:
            self.job.set_state("cancelled", tr("Cancelled"), progress=0)

    def _running(self, done, total):
        if self._cancel.is_set() or self._record is not None:
            return
        self._done, self._total = done, total
        self.job.set_state("running", trf("Refined {done} of {total} frames", done=done, total=total),
                           progress=done / max(1, total))

    def _failed(self, reason, _details):
        self._error = reason
        self.job.set_state("failed", tr(reason), progress=0)

    def _ready(self, record):
        self._record = record
        self.job.hide()
        self.export_button.setEnabled(True)
        self._phase_changed()

    def _language(self, *_):
        self._source_line()
        if self._record is not None:
            self._plot()
        elif self._error:
            self.job.set_state("failed", tr(self._error), progress=0)
        elif not self._cancel.is_set():
            self._running(self._done, self._total)

    def _source_line(self):
        if self._source:
            source = self._source["source_file"]
            self.source_label.setText(trf("Indexed on {file}, frame {number}", file=Path(source).name,
                                          number=self._source["frame_index"] + 1))
            self.source_label.setToolTip(source)

    def _phase_changed(self, *_):
        if self._record is None:
            return
        name = self.phase_combo.currentText()
        keys = {key for row in self._record["frames"] for key in row["fits"].get(name, {}).get("values", {})}
        with QSignalBlocker(self.parameter_combo):
            self.parameter_combo.clear()
            for key in ("a", "b", "c", "alpha", "beta", "gamma"):
                if key in keys:
                    self.parameter_combo.addItem({"alpha": "α", "beta": "β", "gamma": "γ"}.get(key, key), key)
        self._plot()

    def _plot(self, *_):
        if self._record is None:
            return
        import pyqtgraph as pg

        rows = self._record["frames"]
        name = self.phase_combo.currentText()
        fits = [row["fits"].get(name, {}) for row in rows]
        times = bool(self._record["time_from"])
        x = np.array([r["time_s"] if times else r["frame"] for r in rows], dtype=float)
        key = self.parameter_combo.currentData()
        for plot, param, symbol, color in ((self.length_plot, key, self.parameter_combo.currentText(), "#377ac2"),
                                            (self.volume_plot, "V", "V", "#a15cba")):
            unit = "Å³" if param == "V" else "°" if param in ("alpha", "beta", "gamma") else "Å"
            plot.set_labels(tr("Time (s)") if times else tr("Frame"), f"{symbol} ({unit})" if symbol else "")
            y = np.array([fit.get("values", {}).get(param, np.nan) for fit in fits], dtype=float)
            err = np.array([fit.get("errors", {}).get(param, np.nan) for fit in fits], dtype=float)
            ok = np.isfinite(y) & np.isfinite(err)
            plot.set_curves([(symbol, x[ok], y[ok])] if ok.any() else [], colors=[color], markers=[True],
                            legend=[False])
            if ok.any():
                steps = np.diff(np.unique(x))
                cap = 0.10 * np.min(steps) if len(steps) else 0.
                bars = pg.ErrorBarItem(x=x[ok], y=y[ok], top=err[ok], bottom=err[ok], beam=cap,
                                       pen=pg.mkPen(color, width=1.5))
                plot.plot.addItem(bars)
                plot.add_extra(bars)
            else:
                plot.set_empty_text(tr("No determined cell for this phase."))
        count = sum(bool(fit.get("ok")) for fit in fits)
        message = trf("Cell determined in {count} of {total} frames. Error bars show 1σ.", count=count, total=len(rows))
        if self._record["time_from"] in ("exposure", "start_exposure"):
            message += " " + tr(MAP_TIME[self._record["time_from"]])
        if count < len(rows):
            i = next(i for i, fit in enumerate(fits) if not fit.get("ok"))
            reason = rows[i].get("reason") or fits[i].get("reason") or ""
            message += " " + trf("Frame {number}: {reason}", number=rows[i]["frame"], reason=tr(reason))
        self.details_label.setText(message)

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(self, tr("Export indexed cell series"), "cell_series.csv",
                                               tr("CSV files (*.csv)"))
        if not path:
            return
        try:
            saved = self._page.view_model.write_cell_series(Path(path), self._record)
        except (OSError, ValueError) as exc:
            show_toast(self, str(exc), level="error")
            return
        show_toast(self, trf("Saved {name} and its settings record.", name=saved.name),
                   action=(tr("Open Folder"), lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(saved.parent)))))
