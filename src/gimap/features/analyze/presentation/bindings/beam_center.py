"""Beam-centre control of Analyze: show where the centre comes from, change it quickly.

The calibrated centre of the instrument profile is used for every new frame,
so a routine measurement needs no action.  When the centre is wrong for a
beamtime, the user drags the cross on the image, clicks "Pick on Image" or
types coordinates; that session centre then applies to every following frame
of the same detector (frame shape) until it is reset or saved into the profile.
A frame of another shape keeps its own header or profile centre and says so
(``application/geometry_resolution.py``).
"""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
)

from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.app.presentation.theme import set_state

from ..texts import message_text
from ..view_model import CENTER_HEADER, CENTER_SESSION

CENTER_TEXT = {
    "profile": "Beam {x}, {y} px · profile",
    CENTER_SESSION: "Beam {x}, {y} px · this session",
    CENTER_HEADER: "Beam {x}, {y} px · file header",
}
"""The beam-centre button, one template per source (whole sentences translate better than pieces)."""


class CenterDialog(QDialog):
    """Two coordinates in canonical detector pixels (row 0 at the top)."""

    def __init__(self, center: tuple[float, float], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Beam Centre")
        layout = QVBoxLayout(self)
        note = QLabel(
            "Direct-beam position in detector pixels: x from the left edge, "
            "y from the top edge of the image as shown.",
            self,
        )
        note.setWordWrap(True)
        note.setProperty("gimapRole", "muted")
        layout.addWidget(note)
        form = QFormLayout()
        self.x_spin = self._spin(center[0])
        self.y_spin = self._spin(center[1])
        form.addRow("x (px)", self.x_spin)
        form.addRow("y (px)", self.y_spin)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _spin(self, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox(self)
        spin.setDecimals(2)
        spin.setRange(-100000.0, 100000.0)
        spin.setSingleStep(0.5)
        spin.setValue(float(value))
        return spin

    def center(self) -> tuple[float, float]:
        return self.x_spin.value(), self.y_spin.value()


class BeamCenterMixin:
    """Own the beam-centre chip, its menu and the draggable target on the image."""

    def _connect_center(self) -> None:
        self.pick_center_action.toggled.connect(self.detector_view.set_pick_mode)
        self.detector_view.pickModeChanged.connect(self._pick_mode_changed)
        self.detector_view.beamCenterMoved.connect(self.set_session_center)
        self.enter_center_action.triggered.connect(self._enter_center)
        self.header_center_action.triggered.connect(self._use_header_center)
        self.symmetry_center_action.triggered.connect(self.refine_center_x)
        self.reset_center_action.triggered.connect(self.reset_center)
        self.save_center_action.triggered.connect(self.save_center)

    def _pick_mode_changed(self, enabled: bool) -> None:
        if self.pick_center_action.isChecked() != enabled:
            self.pick_center_action.blockSignals(True)
            self.pick_center_action.setChecked(enabled)
            self.pick_center_action.blockSignals(False)
        if enabled:
            self._status(tr("Click the direct-beam position on the image (Esc cancels)."))

    def set_session_center(self, x: float, y: float) -> None:
        """Use (x, y) for this and every following frame of this shape in the session."""
        self.view_model.set_beam_center(x, y)
        self._status(trf(
            "Beam centre ({x}, {y}) px for this session; Beam centre ▸ Save to Profile keeps it for good.",
            x=f"{x:.1f}", y=f"{y:.1f}"), "warning")
        self.run_analysis()

    def _enter_center(self) -> None:
        state = self.view_model.center_state()
        center = state["center"] or state["header_center"] or (0.0, 0.0)
        dialog = CenterDialog(center, self)
        if dialog.exec_() == QDialog.Accepted:
            self.set_session_center(*dialog.center())

    def _use_header_center(self) -> None:
        header = self.view_model.center_state()["header_center"]
        if header is not None:
            self.set_session_center(*header)

    def refine_center_x(self) -> None:
        """GISAXS: centre column from the symmetry of the horizontal cut, for this session."""
        try:
            result = self.view_model.refine_center_x()
        except ValueError as exc:
            self._status(trf("Could not refine the beam centre: {error}", error=message_text(exc)), "warning")
            return
        self._status(trf(
            "Beam centre x {old} → {new} px from the symmetry of the horizontal cut (this session; "
            "Beam centre ▸ Save to Profile keeps it).", old=f"{result.initial_x_px:.2f}", new=f"{result.x_px:.2f}"),
            "warning")
        self.run_analysis()

    def reset_center(self) -> None:
        self.view_model.clear_beam_center()
        self._status(tr("Beam centre back to the instrument profile."), "ok")
        self.run_analysis()

    def save_center(self) -> None:
        try:
            profile = self.view_model.save_center_to_profile()
        except (ValueError, OSError) as exc:
            self._status(trf("Could not save the beam centre: {error}", error=message_text(exc)), "error")
            return
        self._refresh_profiles()
        center = profile.geometry.beam_center_x_px, profile.geometry.beam_center_y_px
        self._status(trf("Saved beam centre ({x}, {y}) px to profile “{name}”.", x=f"{center[0]:.1f}",
                         y=f"{center[1]:.1f}", name=profile.name), "ok")
        self.run_analysis()

    def _update_center_control(self) -> None:
        state = self.view_model.center_state()
        center, source = state["center"], state["source"]
        button = self.center_button
        button.setEnabled(center is not None)
        if center is None:
            button.setText(tr("Beam centre"))
            set_state(button, "centerSource", None)
        else:
            x, y = f"{center[0]:.1f}", f"{center[1]:.1f}"
            template = CENTER_TEXT.get(source) or ("Beam {x}, {y} px · {source}" if source else "Beam {x}, {y} px")
            button.setText(trf(template, x=x, y=y, source=source))
            set_state(button, "centerSource", source)
        header = state["header_center"]
        self.header_center_action.setEnabled(header is not None and center != header)
        self.header_center_action.setText(
            trf("Use File Header Centre ({x}, {y})", x=f"{header[0]:.1f}", y=f"{header[1]:.1f}")
            if header is not None
            else tr("Use File Header Centre (none in this file)")
        )
        # Picking and dragging work in detector pixels, not on the q map.
        self.pick_center_action.setEnabled(center is not None and self.view_combo.currentIndex() == 0)
        self.symmetry_center_action.setEnabled(
            center is not None and self.view_model.is_gisaxs(self.view_model.state.analysis)
        )
        self.symmetry_button.setEnabled(self.symmetry_center_action.isEnabled())
        self.reset_center_action.setEnabled(state["overridden"])
        profile_name = state["profile_name"]
        self.save_center_action.setEnabled(
            profile_name is not None and source in (CENTER_SESSION, CENTER_HEADER)
        )
        self.save_center_action.setText(
            trf("Save to Profile “{name}”", name=profile_name) if profile_name else tr("Save to Profile")
        )
        button.setToolTip(tr(
            "Beam centre of this frame and where it comes from. Drag the cross on the image, "
            "or use this menu to pick, type, reset or save it."
        ))


__all__ = ["BeamCenterMixin", "CenterDialog"]
