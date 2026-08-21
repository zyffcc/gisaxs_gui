"""Bind the inline In-situ page to existing fitting workflow commands."""

from __future__ import annotations

from PyQt5.QtWidgets import QWidget

from ..binding_primitives import _ai_catalog


class InsituPageBindingMixin:
    """Own page composition and Qt signal routing, not scientific work."""

    def _connect_insitu_series_page(self):
        if getattr(self, "_insitu_series_page_connected", False):
            return
        page = getattr(self.ui, "fittingInsituSeriesPage", None)
        if page is None:
            return
        page.capture_recipe_requested.connect(self._capture_current_insitu_recipe)
        page.return_to_single_requested.connect(self._open_single_analysis_page)
        page.error_occurred.connect(self._add_fitting_error)
        widgets = page.workflow_widgets()
        self._insitu_workflow_widgets = widgets
        self._insitu_workflow_canvas_image = page.ui.imageCanvas
        self._insitu_workflow_canvas_curve = page.ui.curveCanvas

        for key in (
            "auto_show",
            "auto_cut",
            "auto_fit",
            "use_previous",
            "full_auto_fit",
            "auto_refine",
        ):
            widgets[key].toggled.connect(self._refresh_insitu_workflow_step_styles)
        widgets["run_mode"].currentTextChanged.connect(
            lambda _text: self._update_insitu_run_mode_ui()
        )
        widgets["profile"].currentTextChanged.connect(self._set_ai_profile)
        widgets["sequence_browse"].clicked.connect(self._browse_insitu_sequence_folder)
        widgets["start"].clicked.connect(self._start_insitu_workflow)
        widgets["process"].clicked.connect(self._start_insitu_sequence_processing)
        widgets["pause"].clicked.connect(self._pause_insitu_workflow)
        widgets["stop"].clicked.connect(self._stop_insitu_workflow)
        widgets["trend"].clicked.connect(self._open_insitu_trend_monitor)
        widgets["heatmap"].clicked.connect(self._open_insitu_heatmap)
        widgets["export"].clicked.connect(self._export_insitu_workflow_results)
        widgets["clear_cache"].clicked.connect(self._clear_insitu_session_cache)
        widgets["open_cache"].clicked.connect(self._open_insitu_cache_folder)

        profile_combo = widgets["profile"]
        profile_combo.blockSignals(True)
        profile_combo.clear()
        profile_combo.addItems(list(_ai_catalog(self).profile_names()))
        profile_combo.blockSignals(False)
        profile = str(
            self._ai_run_settings().get(
                "profile", _ai_catalog(self).default_profile_name
            )
        )
        profile_combo.setCurrentText(profile)
        self._populate_insitu_sequence_folder_default()
        self._update_insitu_run_mode_ui()
        self._refresh_insitu_workflow_step_styles()
        self._refresh_insitu_workflow_status()
        page.render_recipe(self.fitting_view_model.insitu.recipe)
        page.render_workflow(self.fitting_view_model.insitu.state)
        self._insitu_series_page_connected = True

    def _open_insitu_series_page(self):
        workspace = getattr(self.ui, "fittingWorkspace", None)
        if workspace is not None:
            workspace.show_context("insitu")

    def _open_single_analysis_page(self):
        workspace = getattr(self.ui, "fittingWorkspace", None)
        if workspace is not None:
            workspace.show_context("single")

    def _insitu_workflow_parent_widget(self):
        page = getattr(self.ui, "fittingInsituSeriesPage", None)
        if isinstance(page, QWidget):
            return page
        for candidate in (
            getattr(self, "main_window", None),
            getattr(self, "parent", None),
            getattr(self.ui, "centralwidget", None),
        ):
            if isinstance(candidate, QWidget):
                return candidate
        return None


__all__ = ["InsituPageBindingMixin"]
