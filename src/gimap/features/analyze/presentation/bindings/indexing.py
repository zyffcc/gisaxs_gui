"""The Phases window of Analyze and the reflection overlay on the image and on I(q).

The overlay follows the view: detector pixels, the q∥–qz map or the cake (q, χ); I(q) gets a dashed line at each
powder ring of every phase shown. ``_sync_reflections`` is called whenever the image or the curves are drawn."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from src.gimap.app.presentation.i18n import tr, trf

from ...application import GISAXS, X_AXIS_TWO_THETA, two_theta_deg
from ..indexing_model import IndexingModel
from ..reflection_layer import ReflectionLayer, Spots, marker_size
from ..views.indexing_view import make_phases_button

VIEW_DETECTOR, VIEW_Q_MAP, VIEW_CAKE = range(3)
CURVE_LABELS = 8
"""Ring lines on I(q) labelled per phase at most (the strongest)."""


def _tip(overlay, group, channel: str = "") -> str:
    text = f"{overlay.spec.name} {group.label}  d = {group.d:.4f} Å  q = {group.q:.4f} Å⁻¹"
    if overlay.fibre:
        text += f"  q∥ = {group.q_parallel:.4f}, qz = {group.qz:.4f}  χ = {group.chi_deg:.1f}°"
    text += f"  m = {group.multiplicity}"
    return text + (f"  ({channel})" if channel else "")


class IndexingMixin:
    """Needs ``detector_view``, ``top_plot``, ``view_combo``, ``view_model``, ``tasks`` and ``run_analysis``."""

    _indexing_panel = None

    def _setup_indexing(self) -> None:
        self.indexing = IndexingModel()
        self.phases_button = make_phases_button(self.detector_view)
        self.detector_view.toolbar_layout.insertWidget(
            self.detector_view.toolbar_layout.indexOf(self.sources_button) + 1, self.phases_button)
        self.phases_button.toggled.connect(self._phases_toggled)
        self._image_reflections = ReflectionLayer(self.detector_view.plot)
        self._curve_reflections = ReflectionLayer(self.top_plot.plot)
        self._after_indexing_analysis = None
        """Called once when the next analysis is shown (a pole figure after the ring was moved)."""
        self.analysisShown.connect(self._indexing_analysis_shown)

    def _indexing_analysis_shown(self, _analysis) -> None:
        if self._indexing_panel is not None and self._indexing_panel.isVisible():
            self._indexing_panel.refresh()
        pending, self._after_indexing_analysis = self._after_indexing_analysis, None
        if pending is not None:
            pending()

    def _clear_reflections(self) -> None:
        if hasattr(self, "_image_reflections"):
            self._image_reflections.clear()
            self._curve_reflections.clear()

    # -- the window ------------------------------------------------------------------------

    def open_indexing(self):
        """Show the Phases window (made at the first call) and return it."""
        if self._indexing_panel is None:
            from ..indexing_panel import IndexingPanel

            self._indexing_panel = IndexingPanel(self)
            self._indexing_panel.finished.connect(lambda _result: self._phases_closed())
        panel = self._indexing_panel
        panel.refresh()
        panel.show()
        panel.raise_()
        if not self.phases_button.isChecked():
            self.phases_button.setChecked(True)
        return panel

    def _phases_toggled(self, checked: bool) -> None:
        if checked:
            self.open_indexing()
        elif self._indexing_panel is not None and self._indexing_panel.isVisible():
            self._indexing_panel.hide()

    def _phases_closed(self) -> None:
        if self.phases_button.isChecked():
            self.phases_button.blockSignals(True)
            self.phases_button.setChecked(False)
            self.phases_button.blockSignals(False)

    def indexing_changed(self) -> None:
        """The phases or the overlay settings changed: draw again, and the window's tables."""
        self._sync_reflections()
        if self._indexing_panel is not None:
            self._indexing_panel.refresh()

    def _dispose_indexing(self) -> None:
        if self._indexing_panel is not None:
            self._indexing_panel.close()
            self._indexing_panel.deleteLater()
            self._indexing_panel = None

    # -- the overlay -----------------------------------------------------------------------

    def _indexing_geometry(self):
        analysis = self.view_model.state.analysis
        if analysis is None or analysis.reduction is None or analysis.reduction.kind == GISAXS:
            return None
        return analysis.geometry

    def indexing_overlays(self):
        """The visible overlays for the frame shown, or ``()`` (no phases, no GIWAXS frame, or an error)."""
        geometry = self._indexing_geometry()
        if geometry is None or not self.indexing.active:
            return ()
        try:
            return self.indexing.visible_overlays(geometry)
        except ValueError as error:
            self._status(trf("Phases: {error}", error=tr(str(error))), "warning")
            return ()

    def _sync_reflections(self) -> None:
        if not hasattr(self, "_image_reflections"):
            return
        overlays = self.indexing_overlays()
        labels = self.indexing.settings.label_fraction > 0
        if not overlays:
            self._image_reflections.clear()
            self._curve_reflections.clear()
            return
        view = self.view_combo.currentIndex()
        spots, arcs, lines = [], [], []
        for overlay in overlays:
            group_spots, group_arcs, group_lines = self._view_marks(overlay, view)
            spots += group_spots
            arcs += group_arcs
            lines += group_lines
        legend = [(overlay.spec.name, overlay.spec.color) for overlay in overlays]
        self._image_reflections.draw(spots, arcs=arcs, lines=lines, labels=labels, legend=legend)
        self._curve_reflections.draw([], lines=self._curve_lines(overlays), labels=labels)

    def _view_marks(self, overlay, view: int):
        strongest = overlay.strongest() or 1.0
        fraction = self.indexing.settings.label_fraction
        color = overlay.spec.color
        spots = Spots(color)
        arcs, lines = [], []
        if not overlay.fibre:
            for ring, (q_par, qz, x, y) in zip(overlay.groups, overlay.arcs):
                if view == VIEW_DETECTOR:
                    arcs.append((color, x, y))
                elif view == VIEW_Q_MAP:
                    arcs.append((color, q_par, qz))
                else:
                    lines.append((color, ring.q, ring.label if ring.intensity >= fraction * strongest else ""))
            return [spots], arcs, lines
        for marker in overlay.markers:
            group = overlay.groups[marker.group]
            if view == VIEW_DETECTOR:
                x, y = marker.x_px, marker.y_px
            elif view == VIEW_Q_MAP:
                x, y = marker.q_parallel, marker.qz
            else:
                x, y = math.hypot(marker.q_parallel, marker.qz), math.degrees(math.atan2(marker.q_parallel, marker.qz))
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            spots.x.append(float(x))
            spots.y.append(float(y))
            spots.sizes.append(marker_size(group.intensity, strongest))
            spots.hollow.append(marker.channel != "T" or not marker.seen)
            spots.tips.append(_tip(overlay, group, "" if marker.channel == "T" else tr("reflected channel")))
            if marker.channel == "T" and marker.q_parallel >= 0 and group.intensity >= fraction * strongest:
                spots.labels.append((float(x), float(y), group.label, group.intensity))
        return [spots], arcs, lines

    def _curve_lines(self, overlays) -> list:
        analysis = self.view_model.state.analysis
        two_theta = self.view_model.state.giwaxs.x_axis == X_AXIS_TWO_THETA and analysis is not None
        lines = []
        for overlay in overlays:
            strongest = max((ring.intensity for ring in overlay.rings), default=1.0) or 1.0
            ranked = sorted(overlay.rings, key=lambda ring: -ring.intensity)[:CURVE_LABELS]
            labelled = {id(ring) for ring in ranked if ring.intensity >= self.indexing.settings.label_fraction * strongest}
            for ring in overlay.rings:
                x = ring.q
                if two_theta and analysis.geometry is not None:
                    x = float(two_theta_deg(np.array([ring.q]), analysis.geometry.wavelength_angstrom)[0])
                if np.isfinite(x):
                    lines.append((overlay.spec.color, x, ring.label if id(ring) in labelled else ""))
        return lines

    def reflection_count(self) -> Optional[int]:
        """Items of the overlay on the image (tests)."""
        return self._image_reflections.shown if hasattr(self, "_image_reflections") else None


__all__ = ["IndexingMixin"]
