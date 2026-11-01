"""Start-up and display-control state of the curve-first Fitting workspace."""

from __future__ import annotations

from ..binding_primitives import is_matplotlib_available


class FittingUiLifecycleMixin:
    """Initialize the curve controls and restore them from a session.

    There is no window-wide debug key: the parameter-trigger metadata stays available to code and tests as
    ``param_trigger_manager.debug_dump_meta()`` (a Ctrl+Alt+M shortcut would catch AltGr+M, the µ of German
    and Polish keyboards)."""

    def _initialize_ui(self):
        self._initialize_fit_checkboxes()
        self._set_default_parameters()
        if not is_matplotlib_available():
            self.status_updated.emit("Warning: matplotlib not available. Plots are disabled.")

    def _initialize_fit_checkboxes(self):
        """Curve display defaults: signed q, linear x, log intensity (scattering data)."""
        for name, checked in (("fitLogXCheckBox", False), ("fitLogYCheckBox", True), ("fitNormCheckBox", False)):
            widget = getattr(self.ui, name, None)
            if widget is not None:
                widget.blockSignals(True)
                widget.setChecked(checked)
                widget.blockSignals(False)
        combo = getattr(self.ui, "fitQViewModeComboBox", None)
        if combo is not None:
            combo.blockSignals(True)
            combo.setCurrentIndex(max(0, combo.findData("signed")))
            combo.blockSignals(False)

    def _restore_fit_checkboxes(self, session_data):
        try:
            for name, key, default in (
                ("fitLogXCheckBox", "fit_log_x", False),
                ("fitLogYCheckBox", "fit_log_y", True),
                ("fitNormCheckBox", "fit_norm", False),
            ):
                widget = getattr(self.ui, name, None)
                if widget is not None:
                    widget.blockSignals(True)
                    widget.setChecked(bool(session_data.get(key, default)))
                    widget.blockSignals(False)
            combo = getattr(self.ui, "fitQViewModeComboBox", None)
            if combo is not None:
                q_view_mode = session_data.get("fit_q_view_mode") or self._q_view_mode_from_legacy(
                    session_data.get("fit_q_branch", "both"),
                    session_data.get("fit_q_combination", "separate"),
                )
                combo.blockSignals(True)
                combo.setCurrentIndex(max(0, combo.findData(q_view_mode)))
                combo.blockSignals(False)
            self._update_q_view_hint()
        except Exception:
            pass


__all__ = ["FittingUiLifecycleMixin"]
