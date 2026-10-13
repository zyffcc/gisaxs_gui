"""Start-up and display-control state of the curve-first Fitting workspace."""

from __future__ import annotations

from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QShortcut

from ..binding_primitives import is_matplotlib_available


class FittingUiLifecycleMixin:
    """Initialize the curve controls and restore them from a session."""

    def _setup_meta_debug_shortcut(self):
        """Ctrl+Alt+M prints the parameter-trigger metadata to the fitting log."""
        try:
            shortcut = QShortcut(QKeySequence("Ctrl+Alt+M"), self.ui)

            def _dump():
                snapshot = self.param_trigger_manager.debug_dump_meta(verbose=False)
                self._add_fitting_message("==== META SNAPSHOT ====", "INFO")
                for widget_id, data in snapshot.items():
                    self._add_fitting_message(f"{widget_id}: {data}", "INFO")

            shortcut.activated.connect(_dump)
        except Exception as exc:
            print(f"Failed to register meta debug shortcut: {exc}")

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
