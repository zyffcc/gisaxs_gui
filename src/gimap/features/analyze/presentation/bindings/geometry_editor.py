"""Lifetime, session application and feedback of modeless geometry editors."""

import math

from PyQt5.QtCore import QTimer, Qt

from src.gimap.app.presentation.i18n import trf

from ..texts import message_text

EDITOR_ATTRIBUTES = ("_geometry_dialog", "_center_dialog")
LIVE_PREVIEW_DELAY_MS = 150


class GeometryEditorMixin:
    def _geometry_editor_target(self):
        analysis = self.view_model.state.analysis
        request = self.view_model.request()
        if analysis is None or request is None or (
            analysis.path != request.path or analysis.frame_index != request.frame_index
        ):
            return None
        return analysis.path, analysis.frame_index, analysis.shape

    def _geometry_editor_is_current(self, dialog):
        request = self.view_model.request()
        path, frame, shape = dialog._analyze_target
        analysis = self.view_model.state.analysis
        return request is not None and (request.path, request.frame_index) == (path, frame) and (
            analysis is None or (analysis.path, analysis.frame_index, analysis.shape) == (path, frame, shape)
        )

    def _raise_geometry_editor(self, attribute):
        dialog = getattr(self, attribute, None)
        if dialog is None:
            return False
        if not self._geometry_editor_is_current(dialog):
            dialog.close()
            return False
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        return True

    def _show_geometry_editor(self, attribute, dialog, target):
        dialog._analyze_target = target
        dialog._editor_message = None
        setattr(self, attribute, dialog)
        dialog.setModal(False)
        dialog.setAttribute(Qt.WA_DeleteOnClose)
        dialog.finished.connect(lambda _result: self._geometry_editor_closed(attribute, dialog))
        dialog.center_step_spin.valueChanged.connect(self._remember_center_step)
        dialog._preview_pending = False
        dialog._preview_timer = QTimer(dialog)
        dialog._preview_timer.setSingleShot(True)
        dialog._preview_timer.setInterval(LIVE_PREVIEW_DELAY_MS)
        dialog._preview_timer.timeout.connect(lambda: self._flush_geometry_preview(dialog))
        dialog.live_preview_check.toggled.connect(lambda enabled: self._toggle_geometry_preview(dialog, enabled))
        for spin in self._geometry_editor_center_spins(dialog):
            spin.valueChanged.connect(lambda _value, dialog=dialog: self._queue_geometry_preview(dialog))
            spin.lineEdit().textEdited.connect(lambda _text, dialog=dialog: self._queue_geometry_preview(dialog))
        self._geometry_editor_results(self.view_model.state.analysis, completed=False)
        dialog.show()

    def _geometry_editor_closed(self, attribute, dialog):
        self._cancel_geometry_preview(dialog)
        if getattr(self, attribute, None) is dialog:
            setattr(self, attribute, None)

    def _close_geometry_editors(self):
        for attribute in EDITOR_ATTRIBUTES:
            dialog = getattr(self, attribute, None)
            if dialog is not None:
                setattr(self, attribute, None)
                dialog.close()
        self._live_preview_request = None

    def _invalidate_geometry_editors(self):
        for attribute in EDITOR_ATTRIBUTES:
            dialog = getattr(self, attribute, None)
            if dialog is not None and not self._geometry_editor_is_current(dialog):
                dialog.close()

    def _set_geometry_editor_status(self, dialog, template, level="info", **values):
        dialog._editor_message = (template, level, values)
        self._render_geometry_editor_status(dialog)

    def _render_geometry_editor_status(self, dialog):
        template, level, values = dialog._editor_message
        if "error" in values:
            values = {**values, "error": message_text(values["error"])}
        dialog.set_status(trf(template, **values), level)

    def _refresh_geometry_editor_language(self, _language=None):
        for attribute in EDITOR_ATTRIBUTES:
            dialog = getattr(self, attribute, None)
            if dialog is not None and dialog._editor_message is not None:
                self._render_geometry_editor_status(dialog)

    def _geometry_editor_results(self, analysis, *, completed=True):
        if completed:
            self._live_preview_request = None
        self._invalidate_geometry_editors()
        if analysis is None:
            return
        for attribute in EDITOR_ATTRIBUTES:
            dialog = getattr(self, attribute, None)
            if dialog is None:
                continue
            geometry = analysis.geometry
            dialog.apply_button.setEnabled(geometry is not None)
            dialog.live_preview_check.setEnabled(geometry is not None)
            if geometry is None:
                self._cancel_geometry_preview(dialog)
            if geometry is None:
                self._set_geometry_editor_status(dialog, "Save a geometry profile before applying the centre.", "warning")
            else:
                self._set_geometry_editor_status(
                    dialog, "Curves use beam centre ({x}, {y}) px.", "ok",
                    x=f"{geometry.beam_center_x_px:.4f}", y=f"{geometry.beam_center_y_px:.4f}",
                )

    def _geometry_editor_failed(self, message):
        self._live_preview_request = None
        for attribute in EDITOR_ATTRIBUTES:
            dialog = getattr(self, attribute, None)
            if dialog is not None and self._geometry_editor_is_current(dialog):
                self._set_geometry_editor_status(dialog, "Could not update the curves: {error}", "error", error=message)

    def _apply_geometry_editor_center(self, dialog):
        if not self._geometry_editor_is_current(dialog):
            dialog.close()
            return
        x, y = dialog.center()
        self._cancel_geometry_previews()
        self.view_model.set_beam_center(x, y, shape=dialog._analyze_target[2])
        self._set_geometry_editor_status(
            dialog, "Updating curves for beam centre ({x}, {y}) px …", x=f"{x:.4f}", y=f"{y:.4f}",
        )
        self.run_analysis()

    @staticmethod
    def _geometry_editor_center_spins(dialog):
        if hasattr(dialog, "center_x_spin"):
            return dialog.center_x_spin, dialog.center_y_spin
        return dialog.x_spin, dialog.y_spin

    @staticmethod
    def _cancel_geometry_preview(dialog):
        dialog._preview_timer.stop()
        dialog._preview_pending = False

    def _cancel_geometry_previews(self):
        for attribute in EDITOR_ATTRIBUTES:
            dialog = getattr(self, attribute, None)
            if dialog is not None:
                self._cancel_geometry_preview(dialog)
        self._live_preview_request = None

    def _toggle_geometry_preview(self, dialog, enabled):
        for spin in self._geometry_editor_center_spins(dialog):
            spin.setKeyboardTracking(enabled)
        if enabled:
            self._queue_geometry_preview(dialog)
        else:
            self._cancel_geometry_preview(dialog)

    def _queue_geometry_preview(self, dialog):
        if not dialog.live_preview_check.isChecked() or not dialog.live_preview_check.isEnabled():
            return
        if not self._geometry_editor_is_current(dialog):
            dialog.close()
            return
        spins = self._geometry_editor_center_spins(dialog)
        if not all(spin.hasAcceptableInput() for spin in spins):
            self._cancel_geometry_preview(dialog)
            return
        centre = tuple(spin.value() for spin in spins)
        if centre == self._geometry_editor_applied_center(dialog):
            self._cancel_geometry_preview(dialog)
            return
        # Two open editors share the same session centre: keep the latest user edit.
        for attribute in EDITOR_ATTRIBUTES:
            other = getattr(self, attribute, None)
            if other is not None and other is not dialog:
                self._cancel_geometry_preview(other)
        dialog._preview_pending = True
        dialog._preview_timer.start(LIVE_PREVIEW_DELAY_MS)

    def _geometry_editor_applied_center(self, dialog):
        request = self.view_model.request()
        if request.beam_center is not None and request.beam_center_shape in (None, dialog._analyze_target[2]):
            return request.beam_center
        analysis = self.view_model.state.analysis
        geometry = analysis.geometry if analysis is not None else None
        return (geometry.beam_center_x_px, geometry.beam_center_y_px) if geometry is not None else None

    def _flush_geometry_preview(self, dialog):
        if not dialog._preview_pending:
            return
        if not dialog.live_preview_check.isChecked() or not dialog.live_preview_check.isEnabled() or (
            not all(spin.hasAcceptableInput() for spin in self._geometry_editor_center_spins(dialog))
        ) or not self._geometry_editor_is_current(dialog):
            self._cancel_geometry_preview(dialog)
            return
        centre = tuple(spin.value() for spin in self._geometry_editor_center_spins(dialog))
        if centre == self._geometry_editor_applied_center(dialog):
            self._cancel_geometry_preview(dialog)
            return
        if self.tasks.is_busy():
            return  # busy_changed resumes the newest edit after the running reduction.
        self._apply_geometry_editor_center(dialog)
        self._live_preview_request = self.view_model.request()

    def _geometry_preview_tasks_changed(self, busy):
        if busy:
            return
        for attribute in EDITOR_ATTRIBUTES:
            dialog = getattr(self, attribute, None)
            if dialog is not None and dialog._preview_pending and not dialog._preview_timer.isActive():
                dialog._preview_timer.start(0)

    def _is_live_center_analysis(self, analysis):
        request = getattr(self, "_live_preview_request", None)
        geometry = analysis.geometry
        return request is not None and geometry is not None and (
            analysis.path, analysis.frame_index,
            (geometry.beam_center_x_px, geometry.beam_center_y_px),
        ) == (request.path, request.frame_index, request.beam_center)

    def _center_step_px(self):
        settings = self.view_model.settings
        try:
            value = float(settings.get("analyze", "center_step_px", 1.0)) if settings is not None else 1.0
        except (TypeError, ValueError):
            value = 1.0
        return min(1000.0, max(0.0001, value)) if math.isfinite(value) else 1.0

    def _remember_center_step(self, value):
        if getattr(self, "_syncing_center_step", False):
            return
        self._syncing_center_step = True
        try:
            for attribute in EDITOR_ATTRIBUTES:
                dialog = getattr(self, attribute, None)
                if dialog is not None:
                    dialog.center_step_spin.setValue(value)
        finally:
            self._syncing_center_step = False
        settings = self.view_model.settings
        if settings is None:
            return
        try:
            settings.set("analyze", "center_step_px", float(value))
            settings.save()
        except OSError:
            # The adjustment still works if a read-only settings store cannot keep its step.
            pass
