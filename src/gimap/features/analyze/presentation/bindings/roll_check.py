"""Two detector picks on a GISAXS feature, evaluated with the geometry of the displayed frame."""

from src.gimap.app.presentation.components.shape_layer import LINE
from src.gimap.app.presentation.i18n import trf

from ...application.readout import detector_point, rod_rotation_deg

ROD = "check-specular-rod"
BAND = "check-exit-angle"
HINTS = {
    ROD: "Click two points far apart on the specular ridge. Esc cancels.",
    BAND: "Click two points far apart on the same horizon or Yoneda band. Esc cancels.",
}


class RollCheckMixin:
    def _connect_roll_check(self):
        self._roll_target = None
        self._roll_readout = None
        self.check_rod_action.triggered.connect(lambda: self._pick_roll_feature(ROD))
        self.check_band_action.triggered.connect(lambda: self._pick_roll_feature(BAND))
        self.shape_layer.shapePicked.connect(self._roll_picked)
        self.shape_layer.drawingChanged.connect(self._roll_drawing_changed)

    def _refresh_roll_check(self):
        analysis = self.view_model.state.analysis
        ready = self.view_model.is_gisaxs(analysis) and analysis.geometry is not None
        self.roll_button.setVisible(ready)
        if self._roll_target is not analysis or self.view_combo.currentIndex() != 0:
            if self.shape_layer.purpose in HINTS:
                self.shape_layer.cancel_drawing()
            self._roll_readout = None
            self._roll_target = None
        self.roll_result.setVisible(bool(ready and self._roll_readout))
        if self._roll_readout:
            template, values = self._roll_readout
            self.roll_result.setText(trf(template, **values))

    def _pick_roll_feature(self, purpose):
        if self.view_combo.currentIndex() != 0:
            self.view_combo.setCurrentIndex(0)
            self._view_chosen(0)
        self.shape_layer.start_drawing(LINE, purpose=purpose)
        self._roll_target = self.view_model.state.analysis
        self._roll_readout = HINTS[purpose], {}
        self._refresh_roll_check()

    def _roll_drawing_changed(self, _kind):
        if self._roll_readout and self._roll_readout[0] in HINTS.values():
            self._roll_readout = None
            self.roll_result.hide()

    def _roll_picked(self, purpose, _kind, points):
        if purpose not in HINTS:
            return
        analysis = self.view_model.state.analysis
        if analysis is None or analysis is not self._roll_target or self.view_combo.currentIndex() != 0:
            return
        first, second = points
        height, width = analysis.shape
        if first == second or any(not (0 <= x < width and 0 <= y < height) for x, y in points):
            self._roll_readout = "Choose two distinct points inside the detector image.", {}
        else:
            geometry = analysis.geometry
            p1, p2 = (detector_point(geometry, x, y, analysis.kind) for x, y in points)
            if purpose == ROD:
                self._roll_readout = (
                    "Rod qy at the two points: {a} and {b} Å⁻¹ (target 0). "
                    "The ridge direction gives rot3 = {roll}°.",
                    dict(a=f"{p1['qy']:.5g}", b=f"{p2['qy']:.5g}",
                         roll=f"{rod_rotation_deg(geometry, first, second):.3f}"),
                )
            else:
                self._roll_readout = (
                    "Exit angles: {a}° and {b}°. Their difference is {delta}° (target 0).",
                    dict(a=f"{p1['alpha_f_deg']:.4f}", b=f"{p2['alpha_f_deg']:.4f}",
                         delta=f"{p2['alpha_f_deg'] - p1['alpha_f_deg']:.4f}"),
                )
        self._refresh_roll_check()
