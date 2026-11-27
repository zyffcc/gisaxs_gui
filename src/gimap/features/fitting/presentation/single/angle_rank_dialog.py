"""Fitting ▸ Rank Incidence Angles…: which angle to measure next, from the fit in Single analysis, and the record of
the angle chosen (``application/angle_choice.py``; Physics and Methods 9.20).

Rank Angles scores every candidate with the model and the points of the page; the table gives each angle's gain
(nats of information added to what was measured so far), its expected counts and the relative error each parameter
could reach after it. The row of the largest gain is picked; Record Choice writes the angle of the picked row, every
candidate and the fit into ``angle_decisions.jsonl`` next to the curve.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import QTableWidgetItem

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import tr, trf

from ..views.angle_rank_view import build_angle_dialog
from ...application.angle_choice import (
    NEEDS_ANGLES, NEEDS_PARAMETERS, NEEDS_SUBSTRATE, AngleRequest, append_decision, decision_record, parameter_label,
    rank_angles, with_burial,
)

BAD_LIST = "Write the angles as numbers above 0 and below 90, separated by commas."
BAD_MADE = "Write what was measured as angle:exposure pairs, separated by commas (0.4:10, 0.1:1)."
NO_CURVE = "Load a curve and fit it in Single analysis first."
RANKED = "Best next: αi {angle}° (gain {gain} nats). {count} angles scored in {seconds} s."
RECORDED = "αi {angle}° recorded in {name}"
KNOWN = {template: template for template in (NEEDS_ANGLES, NEEDS_PARAMETERS, NEEDS_SUBSTRATE)}


def _angles(text: str) -> tuple[float, ...]:
    values = tuple(float(part) for part in text.replace(";", ",").split(",") if part.strip())
    if not values or any(not 0.0 < value < 90.0 for value in values):
        raise ValueError(BAD_LIST)
    return values


def _made(text: str) -> tuple[tuple[float, float], ...]:
    pairs = []
    for part in text.replace(";", ",").split(","):
        if not part.strip():
            continue
        angle, _sep, exposure = part.partition(":")
        pairs.append((float(angle), float(exposure)))
        if not (0.0 < pairs[-1][0] < 90.0 and pairs[-1][1] > 0):
            raise ValueError(BAD_MADE)
    return tuple(pairs)


def _percent(value: float) -> str:
    if not math.isfinite(value):
        return "—"
    return f"{100 * value:.3g} %"


class AngleRankDialog:
    """The window of one Fitting workspace; ``page``: Single analysis (its ``session``)."""

    def __init__(self, page, parent) -> None:
        self.page = page
        self.rows: list[dict] = []
        self.request: Optional[AngleRequest] = None
        build_angle_dialog(self, parent)
        self.angle_rank_button.clicked.connect(lambda: self.rank())
        self.angle_record_button.clicked.connect(lambda: self.record())
        self.angle_table.itemSelectionChanged.connect(self._picked)

    def show(self) -> None:
        self.angle_dialog.show()
        self.angle_dialog.raise_()

    def dispose(self) -> None:
        self.angle_dialog.close()
        self.angle_dialog.deleteLater()

    # -- ranking ---------------------------------------------------------------------------------

    def _request(self) -> AngleRequest:
        try:
            angles = _angles(self.angle_list_edit.text())
        except ValueError as exc:
            raise ValueError(BAD_LIST) from exc
        try:
            made = _made(self.angle_made_edit.text())
        except ValueError as exc:
            raise ValueError(BAD_MADE) from exc
        request = AngleRequest(angles, self.angle_exposure_spin.value(), self.angle_curve_exposure_spin.value(), made,
                               self.angle_floor_spin.value() / 100.0, self.angle_beam_spin.value(),
                               self.angle_length_spin.value())
        if self.angle_burial_check.isChecked():
            request = with_burial(request, delta=float(self.angle_film_delta_edit.text()),
                                  beta=float(self.angle_film_beta_edit.text()),
                                  fraction=self.angle_fraction_spin.value() / 100.0, depth_nm=self.angle_depth_spin.value())
        return request

    def rank(self) -> bool:
        """Score the candidates; ``False`` (and the reason in the window) when it cannot."""
        import time

        session = self.page.session
        data = session.data() if session.curve is not None else None
        if data is None:
            self._say(tr(NO_CURVE))
            return False
        try:
            request = self._request()
            started = time.perf_counter()
            rows = rank_angles(session.model, data, request)
        except ValueError as exc:
            self._say(tr(KNOWN.get(str(exc), str(exc))))
            return False
        self.rows, self.request = rows, request
        self._fill(rows, len(session.model.components))
        best = min(rows, key=lambda row: row["rank"])
        self._say(trf(RANKED, angle=f"{best['alpha_i_deg']:g}", gain=f"{best['gain']:.2f}", count=len(rows),
                      seconds=f"{time.perf_counter() - started:.2f}"))
        return True

    def _fill(self, rows: list[dict], components: int) -> None:
        table = self.angle_table
        keys = list(rows[0]["bounds"]) if rows else []
        headers = [tr("Rank"), tr("αi (°)"), tr("Gain"), tr("Counts"),
                   *(trf("σ/θ of {name}", name=tr(parameter_label(key, components))) for key in keys)]
        table.blockSignals(True)
        table.clear()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            cells = [str(row["rank"]), f"{row['alpha_i_deg']:g}",
                     f"{row['gain']:.2f}" if math.isfinite(row["gain"]) else "—", f"{row['counts']:.3g}",
                     *(_percent(row["bounds"][key]) for key in keys)]
            for c, text in enumerate(cells):
                table.setItem(r, c, QTableWidgetItem(text))
        table.blockSignals(False)
        best = min(range(len(rows)), key=lambda r: rows[r]["rank"])
        table.selectRow(best)
        self._picked()

    def _picked(self) -> None:
        self.angle_record_button.setEnabled(bool(self.rows) and self._picked_row() is not None)

    def _picked_row(self) -> Optional[int]:
        rows = self.angle_table.selectionModel().selectedRows() if self.angle_table.selectionModel() else []
        return rows[0].row() if rows else None

    # -- the record ------------------------------------------------------------------------------

    def record(self) -> Optional[Path]:
        """Append the choice to ``angle_decisions.jsonl`` next to the curve; returns its path."""
        index = self._picked_row()
        session = self.page.session
        if index is None or not self.rows or self.request is None or session.curve is None:
            return None
        chosen = float(self.rows[index]["alpha_i_deg"])
        curve = str(session.curve.path or "")
        folder = Path(curve).parent if curve else Path.cwd()
        record = decision_record(session.model, curve, self.request, self.rows, chosen, self.angle_note_edit.text())
        try:
            path = append_decision(folder, record)
        except OSError as exc:
            self._say(trf("Could not write the record: {error}", error=str(exc)))
            return None
        text = trf(RECORDED, angle=f"{chosen:g}", name=path.name)
        self._say(text)
        show_toast(self.angle_dialog, text, level="ok", action=(tr("Open Folder"), lambda: _open_folder(path.parent)))
        return path

    def _say(self, text: str) -> None:
        self.angle_status.setText(text)


def _open_folder(folder: Path) -> None:
    from PyQt5.QtCore import QUrl
    from PyQt5.QtGui import QDesktopServices

    QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))


__all__ = ["AngleRankDialog"]
