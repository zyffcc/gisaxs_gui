"""Result Preview behavior for Calibration."""

from __future__ import annotations

import logging

from typing import Optional

import numpy as np


from matplotlib.patches import Ellipse

from PyQt5.QtCore import QSignalBlocker, QTimer

from src.gimap.app.presentation.theme.figures import restyle_figure

from ...application import (
    CalibrationCandidate,
)


from ..preview_style import (
    CENTER_COLOR,
    DETECTED_RING_COLOR,
    MATCHED_RING_COLOR,
    UNMATCHED_RING_COLOR,
)

LOGGER = logging.getLogger(__name__)
# Two-column selected solution only while the candidate table keeps at least this width.
CANDIDATES_MIN_WIDTH = 260


class ResultPreviewMixin:
    """Own result preview presentation behavior."""

    def _two_column_solution_width(self) -> int:
        """Width the selected-solution group needs for two title/value pairs per row."""
        metrics = self.result_form_widget.fontMetrics()
        titles = [title.sizeHint().width() for title, _value in self.result_rows[:6]]
        form = self.resultForm.contentsMargins()
        group = self.resultGroupLayout.contentsMargins()
        return (
            max(titles[0::2])
            + metrics.horizontalAdvance("10000.000 mm")  # distance, the widest left value
            + max(titles[1::2])
            + metrics.horizontalAdvance("10000.000 px")
            + 3 * self.resultForm.horizontalSpacing()
            + form.left() + form.right() + group.left() + group.right()
            + 12  # the group box frame
        )

    def _arrange_result_form(self, initial: bool = False) -> None:
        """Two title/value pairs per row when the Results section is wide enough, else one.

        Sizes of the results splitter are set initially, when the arrangement changes and when
        two columns no longer fit their pane (other label texts after a language switch); in
        between the user's own split stays.
        """
        try:
            total = self.results_splitter.width()
            current = self.results_splitter.sizes()[0]
        except RuntimeError:  # the dialog is being destroyed
            return
        if total <= 0:
            return
        needed = self._two_column_solution_width()
        two_columns = total - needed >= CANDIDATES_MIN_WIDTH
        changed = two_columns != self._result_two_columns
        if changed:
            self._result_two_columns = two_columns
            self.place_result_rows(two_columns)
        if two_columns and (initial or changed or current < needed):
            solution = needed
        elif not two_columns and (initial or changed):
            solution = min(max(int(total * 0.36), 260), 360)
        else:
            return
        self.results_splitter.setSizes([solution, max(1, total - solution)])

    def _result_language_changed(self, _language: str = "") -> None:
        """Other title widths and line heights: refit once the texts are swapped."""
        QTimer.singleShot(0, self._refit_results)

    def _refit_results(self) -> None:
        self._arrange_result_form()
        self._fit_results_height()

    def _fit_results_height(self) -> None:
        """Give Results the few pixels a longer solution (a wrapped warning) needs, taken from
        the preview only down to its minimum, and never after the user moved the split."""
        try:
            if self._results_user_sized or self.calibration_results_section.isHidden():
                return
            viewport = self.result_scroll.viewport()
            form = self.result_form_widget
            needed = (
                form.heightForWidth(viewport.width())
                if form.hasHeightForWidth()
                else form.sizeHint().height()
            )
            overflow = needed - viewport.height()
            preview, results = self.right_splitter.sizes()
        except RuntimeError:  # the dialog is being destroyed
            return
        spare = preview - self.calibration_preview_panel.minimumSizeHint().height()
        grow = min(overflow, spare)
        if grow > 0:
            self.right_splitter.setSizes([preview - grow, results + grow])

    def _show_candidate(self, candidate: CalibrationCandidate) -> None:
        self.result_labels["Beam center X"].setText(f"{candidate.center_x_px:.3f} px")
        self.result_labels["Beam center Y"].setText(f"{candidate.center_y_px:.3f} px")
        self.result_labels["Distance"].setText(f"{candidate.distance_mm:.3f} mm")
        self.result_labels["Detector rotation"].setText(f"{candidate.detector_rotation_deg:.3f}°")
        self.result_labels["Matched rings"].setText(str(candidate.matched_ring_count))
        self.result_labels["RMS residual"].setText(f"{candidate.rms_residual_px:.3f} px")
        self.result_labels["Confidence"].setText(candidate.confidence)
        self.result_labels["Warning"].setText(" ".join(candidate.warnings) or "None")
        blockers = [
            QSignalBlocker(widget)
            for widget in (
                self.manual_x,
                self.manual_y,
                self.manual_distance,
            )
        ]
        self.manual_x.setValue(candidate.center_x_px)
        self.manual_y.setValue(candidate.center_y_px)
        self.manual_distance.setValue(candidate.distance_mm)
        del blockers
        standard_name = self.view_model.standard_display_name(candidate.standard_key)
        self.preview_info_label.setText(
            f"{self.view_model.source_name(self.image.source_path) if self.image else ''}  ·  "
            f"{standard_name}  ·  {candidate.distance_mm:.2f} mm  ·  "
            f"{candidate.matched_ring_count} matched rings  ·  {candidate.confidence} confidence"
        )
        self._populate_manual_rings(candidate)
        self.redraw_preview()
        self._fit_results_height()

    def _clear_result_labels(self) -> None:
        for label in self.result_labels.values():
            label.setText("—")

    def _display_candidate(self) -> Optional[CalibrationCandidate]:
        return self.view_model.display_candidate(
            manual_enabled=self.manual_group.isChecked(),
            center_x_px=self.manual_x.value(),
            center_y_px=self.manual_y.value(),
            distance_mm=self.manual_distance.value(),
        )

    def _prepared_preview(self) -> tuple:
        """Return a cached, resolution-adaptive detector preview."""
        if self.image is None:
            raise ValueError("No calibration image is loaded.")
        log_scale = self.log_check.isChecked()
        key = (id(self.image.data), log_scale)
        cached = self._preview_cache.get(key)
        if cached is not None:
            return cached
        data = np.asarray(self.image.data, dtype=np.float32)
        height, width = data.shape
        max_preview_pixels = 1_400_000
        stride = max(1, int(np.ceil(np.sqrt(data.size / max_preview_pixels))))
        sampled = data[::stride, ::stride]
        invalid = ~np.isfinite(sampled)
        if self.image.mask is not None:
            invalid |= np.asarray(self.image.mask, dtype=bool)[::stride, ::stride]
        valid = ~invalid
        display = np.zeros(sampled.shape, dtype=np.float32)
        if log_scale:
            display[valid] = np.log1p(np.maximum(sampled[valid], 0.0))
        else:
            display[valid] = sampled[valid]
        values = display[valid]
        if values.size:
            percentile_sample = values[:: max(1, values.size // 250_000)]
            vmin, vmax = np.percentile(percentile_sample, (1.0, 99.7))
        else:
            vmin, vmax = 0.0, 1.0
        result = (
            display,
            invalid,
            (-0.5, width - 0.5, height - 0.5, -0.5),
            float(vmin),
            float(max(vmax, vmin + 1e-6)),
            height,
            width,
        )
        self._preview_cache[key] = result
        return result

    @staticmethod
    def _ellipse_intersects_image(
        center_x: float,
        center_y: float,
        radius_x: float,
        radius_y: float,
        width: int,
        height: int,
    ) -> bool:
        if radius_x <= 0 or radius_y <= 0:
            return False
        nearest_x = float(np.clip(center_x, 0.0, width - 1.0))
        nearest_y = float(np.clip(center_y, 0.0, height - 1.0))
        minimum = np.hypot(
            (nearest_x - center_x) / radius_x,
            (nearest_y - center_y) / radius_y,
        )
        maximum = max(
            np.hypot((x - center_x) / radius_x, (y - center_y) / radius_y)
            for x, y in (
                (0.0, 0.0),
                (width - 1.0, 0.0),
                (0.0, height - 1.0),
                (width - 1.0, height - 1.0),
            )
        )
        return minimum <= 1.02 and maximum >= 0.98

    def redraw_preview(self) -> None:
        old_xlim, old_ylim = self.axes.get_xlim(), self.axes.get_ylim()
        had_image = bool(self.axes.images)
        self.axes.clear()
        # The empty-state text is a Qt label over the canvas: matplotlib's font has no CJK glyphs.
        self.preview_empty_label.setVisible(self.image is None)
        # One empty-state message: the info line above the canvas returns with an image or result.
        self.preview_info_label.setVisible(self.image is not None or self.result is not None)
        if self.image is None:
            # An empty state, not a bare 0–1 axis.
            self.axes.set_axis_off()
            self.overlay_legend.setVisible(False)
            restyle_figure(self.figure)
            self.canvas.draw_idle()
            return
        display, invalid, extent, vmin, vmax, height, width = self._prepared_preview()
        self.axes.imshow(
            display,
            cmap="viridis",
            origin="upper",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        clean_preview = self.clean_preview_button.isChecked()
        if not clean_preview and self.mask_check.isChecked() and invalid.any():
            overlay = np.ma.masked_where(~invalid, invalid.astype(float))
            self.axes.imshow(
                overlay,
                cmap="Reds",
                alpha=0.30,
                origin="upper",
                extent=extent,
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
        candidate = self._display_candidate()
        if candidate is not None and not clean_preview:
            cx, cy = candidate.center_x_px, candidate.center_y_px
            if self.center_check.isChecked():
                self.axes.axvline(cx, color=CENTER_COLOR, linestyle="-.", linewidth=1.2, alpha=0.95)
                self.axes.axhline(cy, color=CENTER_COLOR, linestyle="-.", linewidth=1.2, alpha=0.95)
            if self.rings_check.isChecked() and self.result is not None:
                for radius in candidate.detected_peak_radii_px:
                    if not self._ellipse_intersects_image(cx, cy, radius, radius, width, height):
                        continue
                    self.axes.add_patch(
                        Ellipse(
                            (cx, cy),
                            2 * radius,
                            2 * radius,
                            fill=False,
                            edgecolor=DETECTED_RING_COLOR,
                            linestyle=":",
                            linewidth=0.8,
                            alpha=0.62,
                        )
                    )
                for ring in self.view_model.theoretical_ring_overlays(candidate):
                    ellipse_width = ring.width_px
                    ellipse_height = ring.height_px
                    if not self._ellipse_intersects_image(
                        cx,
                        cy,
                        0.5 * ellipse_width,
                        0.5 * ellipse_height,
                        self.image.data.shape[1],
                        self.image.data.shape[0],
                    ):
                        continue
                    self.axes.add_patch(
                        Ellipse(
                            (cx, cy),
                            ellipse_width,
                            ellipse_height,
                            fill=False,
                            edgecolor=MATCHED_RING_COLOR if ring.matched else UNMATCHED_RING_COLOR,
                            linestyle="-" if ring.matched else "--",
                            linewidth=1.5 if ring.matched else 0.8,
                            alpha=0.95 if ring.matched else 0.65,
                        )
                    )
        self.overlay_legend.setVisible(candidate is not None and not clean_preview)
        self.axes.set_xlabel("Detector X (pixel)")
        self.axes.set_ylabel("Detector Y (pixel)")
        self.axes.set_aspect("equal", adjustable="box")
        if had_image and not self._reset_preview_view:
            self.axes.set_xlim(old_xlim)
            self.axes.set_ylim(old_ylim)
        else:
            self.axes.set_xlim(-0.5, width - 0.5)
            self.axes.set_ylim(height - 0.5, -0.5)
            self._reset_preview_view = False
        # Axes.clear() restores matplotlib's label colours: re-apply the theme's.
        restyle_figure(self.figure)
        self.canvas.draw_idle()

    def _populate_manual_rings(self, candidate: CalibrationCandidate) -> None:
        self.experimental_ring_combo.clear()
        for radius in candidate.detected_peak_radii_px:
            self.experimental_ring_combo.addItem(f"{radius:.2f} px", radius)
        self._populate_theory_rings()

    def _populate_theory_rings(self) -> None:
        self.theory_ring_combo.clear()
        key = (
            self.result.selected_candidate.standard_key
            if self.result
            else self.standard_combo.currentData()
        )
        for index, q in enumerate(self.view_model.standard_q_values(key)):
            self.theory_ring_combo.addItem(f"{index + 1}: q={q:.5f} Å⁻¹", q)
