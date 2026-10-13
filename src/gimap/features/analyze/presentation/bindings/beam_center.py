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

from threading import Event

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import language_changed, tr, trf
from src.gimap.app.presentation.theme import set_state

from ..center_dialog import CenterDialog
from ..texts import message_text
from ..view_model import CENTER_HEADER, CENTER_SESSION

CENTER_TEXT = {
    "profile": "Beam {x}, {y} px · profile",
    CENTER_SESSION: "Beam {x}, {y} px · this session",
    CENTER_HEADER: "Beam {x}, {y} px · file header",
}
"""The beam-centre button, one template per source (whole sentences translate better than pieces)."""


class BeamCenterMixin:
    """Own the beam-centre chip, its menu and the draggable target on the image."""

    def _connect_center(self) -> None:
        self._center_refinement = None
        self._center_analysis_pending = False
        self._geometry_dialog = None
        self._center_dialog = None
        self._live_preview_request = None
        self.tasks.busy_changed.connect(self._geometry_preview_tasks_changed)
        self.analysisShown.connect(self._geometry_editor_results)
        self.analysisFailed.connect(self._geometry_editor_failed)
        language_changed().connect(self._refresh_geometry_editor_language)
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
            x=f"{x:.4f}", y=f"{y:.4f}"), "warning")
        self.run_analysis()

    def _enter_center(self) -> None:
        if self._raise_geometry_editor("_center_dialog"):
            return
        target = self._geometry_editor_target()
        if target is None:
            return
        state = self.view_model.center_state()
        center = state["center"] or state["header_center"] or (0.0, 0.0)
        dialog = CenterDialog(center, self, center_step_px=self._center_step_px())
        dialog.applyCenterRequested.connect(lambda: self._apply_geometry_editor_center(dialog))
        self._show_geometry_editor("_center_dialog", dialog, target)

    def _use_header_center(self) -> None:
        header = self.view_model.center_state()["header_center"]
        if header is not None:
            self.set_session_center(*header)

    def refine_center_x(self) -> None:
        """Locate the horizontal symmetry axis in the background, then apply it to this session."""
        if self._center_refinement is not None:
            self._cancel_center_refinement()
            self._status(tr("Beam centre search cancelled."))
            return
        analysis = self.view_model.state.analysis
        if self._center_analysis_pending or analysis is None or analysis.geometry is None:
            return
        request = self.view_model.request()
        cancelled = Event()
        self._center_refinement = (analysis, request, cancelled)
        self._shown_status = None
        self._update_center_control()
        self._status(tr("Finding the beam centre: automatic coarse search, then fine refinement …"))
        model = self.view_model
        self.tasks.submit(
            "symmetry-center",
            lambda: model.find_center_x(analysis, cancelled=cancelled.is_set),
            on_done=self._center_refined,
            on_error=self._center_refine_failed,
        )

    def _cancel_center_refinement(self) -> None:
        """Forget a search when the frame/settings change, or when the user cancels it."""
        pending = getattr(self, "_center_refinement", None)
        if pending is not None:
            pending[2].set()
            self.tasks.cancel("symmetry-center")
            self._center_refinement = None
            self._update_center_control()

    def _current_center_refinement(self):
        pending = self._center_refinement
        if pending is None:
            return None
        analysis, request, _cancelled = pending
        if self.view_model.state.analysis is not analysis or self.view_model.request() != request:
            self._cancel_center_refinement()
            return None
        return pending

    def _center_refined(self, result) -> None:
        pending = self._current_center_refinement()
        if pending is None:
            return
        analysis = pending[0]
        self._center_refinement = None
        self.view_model.set_beam_center(result.x_px, analysis.geometry.beam_center_y_px)
        text = trf(
            "Beam centre x: coarse {coarse} → refined {fine} px (this session). "
            "Numerical tolerance {tolerance} px; this is not measured accuracy.",
            coarse=f"{result.coarse_x_px:.4f}", fine=f"{result.x_px:.4f}",
            tolerance=f"{result.tolerance_px:.4f}",
        )
        show_toast(self.window(), text, level="ok")
        self.run_analysis()

    def _center_refine_failed(self, error: str, _details: str) -> None:
        if self._current_center_refinement() is None:
            return
        self._center_refinement = None
        self._update_center_control()
        self._status(trf("Could not refine the beam centre: {error}", error=message_text(error)), "warning")

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
        self._status(trf("Saved beam centre ({x}, {y}) px to profile “{name}”.", x=f"{center[0]:.4f}",
                         y=f"{center[1]:.4f}", name=profile.name), "ok")
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
            x, y = f"{center[0]:.4f}", f"{center[1]:.4f}"
            template = CENTER_TEXT.get(source) or ("Beam {x}, {y} px · {source}" if source else "Beam {x}, {y} px")
            button.setText(trf(template, x=x, y=y, source=source))
            set_state(button, "centerSource", source)
        header = state["header_center"]
        self.header_center_action.setEnabled(header is not None and center != header)
        self.header_center_action.setText(
            trf("Use File Header Centre ({x}, {y})", x=f"{header[0]:.4f}", y=f"{header[1]:.4f}")
            if header is not None
            else tr("Use File Header Centre (none in this file)")
        )
        # Picking and dragging work in detector pixels, not on the q map.
        self.pick_center_action.setEnabled(center is not None and self.view_combo.currentIndex() == 0)
        searching = getattr(self, "_center_refinement", None) is not None
        self.symmetry_center_action.setText(tr("Cancel Centre Search" if searching else "Refine x by Symmetry"))
        self.symmetry_button.setText(tr("Cancel Centre Search" if searching else "Make Left and Right Symmetric"))
        self.symmetry_center_action.setEnabled(searching or (
            center is not None and self.view_model.is_gisaxs(self.view_model.state.analysis)
            and not getattr(self, "_center_analysis_pending", False)
        ))
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
