"""Open a recorded model on its verified curve and compare its values after a new fit."""

from concurrent.futures import CancelledError
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QAbstractItemView, QDialog, QDialogButtonBox, QFileDialog, QLabel, QTableWidgetItem, QVBoxLayout

from src.gimap.app.presentation.i18n import language_changed, tr, trf
from src.gimap.app.presentation.reproduction import ReproductionProgress, reproduction_error_text
from src.gimap.shared.reproduction import verified_path, verify_run
from ...application.models import LoadCurveRequest
from ...application.saved_fit import read_saved_fit
from ...application.single_fit import SIDES, parameter_text
from ..views.step_widgets import read_only_table
from .session import path_name


def open_saved_fit(page, path=None) -> bool:
    if not path:
        path, _ = QFileDialog.getOpenFileName(page, tr("Open Saved Fit"), str(page._remembered("folder") or ""),
                                              tr("Saved fit records (*.json)"))
    if not path:
        return False
    progress = ReproductionProgress(page, path, title="Open Saved Fit", operation="Checking and opening {name}")
    cancelled = progress.cancelled.is_set

    def read():
        with verify_run(cancelled):
            saved = read_saved_fit(path)
            if cancelled():
                raise CancelledError()
            request = LoadCurveRequest(Path(saved.source.token), saved.unit, keep_missing_sigma=True)
            outcome = page.view_model.read_series_curve(request)
            if cancelled():
                raise CancelledError()
            if outcome.error is not None:
                raise ValueError(outcome.error.message)
            # A file replaced while the curve reader was open must still pass verification.
            verified_path({"path": str(saved.source.path.resolve()), "sha256": saved.source_sha256}, saved.path.parent)
            curve = page._loaded_curve(outcome.value, saved.source.token, saved.source.display_name,
                                       verified_sha256=saved.source_sha256)
            return saved, curve, outcome.value

    progress.runner.submit("open_saved_fit", read, on_done=progress.finished_run, on_error=progress.failed_run)
    progress.exec_()
    outcome, error = progress.result, progress.error
    progress.deleteLater()
    if cancelled():
        page._status(lambda: tr("Opening the saved fit was cancelled."), "info")
        return False
    if outcome is None:
        page._status(lambda: trf("Could not open {name}: {reason}", name=Path(path).name,
                                  reason=reproduction_error_text(error)), "error")
        return False
    saved, curve, loaded = outcome
    page.view_model.accept_loaded_curve(saved.source.display_name, loaded)
    page._show_loaded_curve(curve, loaded, saved.source, saved.side, saved.unit)
    session = page.session
    session.saved_fit = saved
    session.model, session.edited = saved.model, True
    session.excluded = set(saved.excluded)
    session.set_range(saved.q_range, record=False)
    session.clear_history()
    page.set_method(saved.method)
    page._remember_curve()
    page._model_changed(editor=True)
    page._render_all()
    page.show_step("results")
    page._status(lambda: trf("Opened {name}. Change the fit settings, then fit again to compare.",
                             name=saved.path.name), "ok")
    return True


def _selection(count, side, q_range):
    limits = f"{q_range[0]:.5g}–{q_range[1]:.5g} nm⁻¹" if q_range else tr("Full q range")
    return trf("{count} points · {side} · {range}", count=count, side=tr(dict(SIDES)[side]), range=limits)


def _parameter_rows(model, bounds, errors):
    rows = {}
    for path, parameter in model.parameters():
        owner, key = path
        family = None if owner == "globals" else model.components[owner].family
        name = path_name(model, path, full=True)
        value = parameter_text(key, parameter.value, errors.get(path))
        if path in bounds:
            value = trf("{value} (at bound)", value=value)
        rows[(owner, family, key)] = name, value
    return rows


def compare_saved_fit(page):
    saved = page.session.saved_fit
    if saved is None:
        return
    session = page.session
    result = session.result if session.result_is_current() else None
    saved_bounds = {p for p, _ in saved.model.parameters() if f"{p[0]}:{p[1]}" in saved.bounds}
    saved_errors = {p: saved.errors.get(f"{p[0]}:{p[1]}") for p, _ in saved.model.parameters()}
    dialog = QDialog(page)
    dialog.setObjectName("savedFitComparison")
    dialog.resize(640, 520)
    layout = QVBoxLayout(dialog)
    label = QLabel(dialog)
    label.setTextFormat(Qt.PlainText)
    label.setWordWrap(True)
    layout.addWidget(label)
    table = read_only_table(("", "", ""), dialog, "savedFitParameters", QAbstractItemView.ExtendedSelection)
    layout.addWidget(table, 1)
    buttons = QDialogButtonBox(QDialogButtonBox.Close, dialog)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)

    def refresh():
        before = _parameter_rows(saved.model, saved_bounds, saved_errors)
        after = _parameter_rows(session.model, result.at_bounds if result else (), result.errors if result else {})
        dialog.setWindowTitle(tr("Compare with Saved Fit"))
        data = page._data()
        current = _selection(len(data.q) if data is not None else 0, session.side, session.q_range)
        previous = _selection(saved.count, saved.side, saved.q_range)
        if saved.quality is not None:
            metric = "χ²ᵣ" if saved.weighting == "sigma" else "var ln I"
            previous += f" · {metric} {saved.quality:.3g}"
        if result is not None:
            current += f" · {page._chi_name(result)} {result.chi2_reduced:.3g}"
        label.setText("\n".join((trf("Saved in {name} · {selection}", name=saved.path.name, selection=previous),
                                 trf("Current selection · {selection}", selection=current),
                                 tr("± gives the least-squares error where available."))))
        table.setHorizontalHeaderLabels([tr("Parameter"), tr("Saved value"), tr("Current value")])
        keys = list(dict.fromkeys((*before, *after)))
        table.setRowCount(len(keys))
        for row, key in enumerate(keys):
            name = (after.get(key) or before[key])[0]
            for col, value in enumerate((name, before.get(key, ("", "—"))[1], after.get(key, ("", "—"))[1])):
                table.setItem(row, col, QTableWidgetItem(value))
        buttons.button(QDialogButtonBox.Close).setText(tr("Close"))

    refresh()
    language_changed().connect(refresh)
    try:
        dialog.exec_()
    finally:
        language_changed().disconnect(refresh)
        dialog.deleteLater()
