"""Manual Refinement behavior for Calibration."""

from __future__ import annotations

import logging
import math


from PyQt5.QtCore import QSignalBlocker, Qt
from PyQt5.QtWidgets import (
    QMessageBox,
)

from src.gimap.app.presentation.i18n import tr


LOGGER = logging.getLogger(__name__)
# A press this close to the center marker (screen pixels) starts a drag.
MARKER_GRAB_PX = 12.0


class ManualRefinementMixin:
    """Own manual refinement presentation behavior."""

    def fit_selected_ring(self) -> None:
        if (
            self.result is None
            or self.experimental_ring_combo.currentData() is None
            or self.theory_ring_combo.currentData() is None
        ):
            return
        try:
            distance = self.view_model.manual_ring_distance(
                float(self.experimental_ring_combo.currentData()),
                float(self.theory_ring_combo.currentData()),
            )
            self.manual_distance.setValue(distance)
            self.stage_label.setText(
                "Manual distance updated from the selected experimental/theoretical ring pair."
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Manual Refinement", str(exc))

    def _preview_press(self, event) -> None:
        """Start dragging the center marker in manual mode.

        Zoom and pan of the toolbar own the mouse while active, and only a press on the marker
        (within ``MARKER_GRAB_PX`` screen pixels) moves the center: a click elsewhere on the
        image never changes the fitted geometry.
        """
        if self.toolbar.mode or not self.manual_group.isChecked():
            return
        if event.button not in (None, 1) or not self._press_hits_center_marker(event):
            return
        self._dragging_center = True

    def _preview_move(self, event) -> None:
        if self.toolbar.mode:
            self._dragging_center = False
            return
        if not self._dragging_center:
            # A move cursor over the marker shows that it can be dragged.
            if self.manual_group.isChecked() and self._press_hits_center_marker(event):
                self.canvas.setCursor(Qt.SizeAllCursor)
            else:
                self.canvas.unsetCursor()
            return
        if (
            event.inaxes is self.axes
            and event.xdata is not None
            and event.ydata is not None
        ):
            self.manual_x.setValue(event.xdata)
            self.manual_y.setValue(event.ydata)

    def _preview_release(self, _event) -> None:
        self._dragging_center = False

    def _press_hits_center_marker(self, event) -> bool:
        if event.inaxes is not self.axes or event.x is None or event.y is None:
            return False
        marker_x, marker_y = self.axes.transData.transform(
            (self.manual_x.value(), self.manual_y.value())
        )
        # Matplotlib event coordinates are physical pixels on high-DPI screens.
        scale = float(getattr(self.canvas, "device_pixel_ratio", 1.0) or 1.0)
        return math.hypot(event.x - marker_x, event.y - marker_y) <= MARKER_GRAB_PX * scale

    def _remember_fitted_geometry(self) -> None:
        """Keep the fitted centre, distance and warnings of every candidate of a new result.

        Committing manual values writes them into the selected candidate in place, so this
        copy is what 'Reset to fitted', manual mode off and the next commit go back to. An
        imported result's selected candidate is an object of its own, not one of
        ``candidates``, so it is kept too: each object gets its own values back.
        """
        if self.result is None:
            self._fitted_geometry = None
            return
        unique = {
            id(candidate): candidate
            for candidate in (*self.result.candidates, self.result.selected_candidate)
            if candidate is not None
        }
        self._fitted_geometry = (
            self.result,
            [
                (c, (c.center_x_px, c.center_y_px, c.distance_mm, list(c.warnings)))
                for c in unique.values()
            ],
        )

    def _restore_fitted_geometry(self) -> None:
        """Write the fitted geometry back onto each remembered candidate of the current result."""
        saved = getattr(self, "_fitted_geometry", None)
        if self.result is None or saved is None or saved[0] is not self.result:
            return
        for candidate, (x, y, distance, warnings) in saved[1]:
            candidate.center_x_px = x
            candidate.center_y_px = y
            candidate.distance_mm = distance
            candidate.warnings = list(warnings)

    def _load_fitted_into_manual_fields(self) -> None:
        """Put the selected fitted centre and distance into the manual fields (no redraw)."""
        if self.result is None:
            return
        self._restore_fitted_geometry()
        candidate = self.result.selected_candidate
        for spin, value in (
            (self.manual_x, candidate.center_x_px),
            (self.manual_y, candidate.center_y_px),
            (self.manual_distance, candidate.distance_mm),
        ):
            blocker = QSignalBlocker(spin)
            spin.setValue(value)
            del blocker

    def reset_manual_to_fitted(self) -> None:
        """Reload the selected fitted solution into the manual fields and the overlay."""
        if self.result is None:
            return
        self._restore_fitted_geometry()
        self._show_candidate(self.result.selected_candidate)
        self.stage_label.setText(tr("Manual values reset to the fitted solution."))

    def _manual_values_changed(self) -> bool:
        """Whether the manual fields differ from the fitted solution as the fields show it."""
        fitted = self.result.selected_candidate
        for spin, value in (
            (self.manual_x, fitted.center_x_px),
            (self.manual_y, fitted.center_y_px),
            (self.manual_distance, fitted.distance_mm),
        ):
            decimals = spin.decimals()
            if abs(spin.value() - round(value, decimals)) > 0.5 * 10.0 ** -decimals:
                return True
        return False

    def _commit_manual_values(self) -> None:
        """Make the result match the window: the fitted solution, plus the manual values only
        while manual mode is on and they really differ (no silent change to a fitted result)."""
        if self.result is None:
            return
        self._restore_fitted_geometry()
        self.view_model.commit_manual_refinement(
            manual_enabled=self.manual_group.isChecked() and self._manual_values_changed(),
            center_x_px=self.manual_x.value(),
            center_y_px=self.manual_y.value(),
            distance_mm=self.manual_distance.value(),
        )

    def _sync_main_window_geometry(self) -> None:
        """Let Analyze see the applied calibration (the confirmation is Apply's toast)."""
        if self.result is None or self.main_window is None:
            return
        # Analyze reads the recorded instrument profile on its next frame.
        analyze = getattr(getattr(self.main_window, "components", None), "analyze_page", None)
        if analyze is not None and hasattr(analyze, "_refresh_profiles"):
            analyze._refresh_profiles()
