"""Feature-owned binding between the Fitting workspace and ViewModel."""

from __future__ import annotations


from collections import defaultdict


from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QWidget


from .parameter_trigger import UniversalParameterTriggerManager


from .binding_primitives import (
    _create_default_fitting_view_model,
    IndependentFitWindow,
)

from .bindings.lifecycle import LifecycleMixin
from .bindings.roi_controls import RoiControlsMixin
from .bindings.signal_connections import SignalConnectionsMixin
from .bindings.curve_files import CurveFilesMixin
from .bindings.insitu_setup import InsituSetupMixin
from .bindings.insitu_execution import InsituExecutionMixin
from .bindings.insitu_monitoring import InsituMonitoringMixin
from .bindings.fit_graphics_events import FitGraphicsEventsMixin
from .bindings.curve_plot import CurvePlotMixin
from .bindings.curve_view import CurveViewMixin
from .bindings.message_log import MessageLogMixin
from .bindings.particle_editor import ParticleEditorMixin
from .bindings.particle_state import ParticleStateMixin
from .bindings.ai_workspace import AiWorkspaceMixin
from .bindings.ai_execution import AiExecutionMixin
from .bindings.ai_review import AiReviewMixin
from .bindings.fit_export import FitExportMixin
from .bindings.manual_fitting import ManualFittingMixin
from .bindings.plot_refresh import PlotRefreshMixin
from .bindings.workflow_feedback import WorkflowFeedbackMixin

__all__ = ["FittingViewBinding", "IndependentFitWindow"]


class FittingViewBinding(
    LifecycleMixin,
    RoiControlsMixin,
    SignalConnectionsMixin,
    CurveFilesMixin,
    InsituSetupMixin,
    InsituExecutionMixin,
    InsituMonitoringMixin,
    FitGraphicsEventsMixin,
    CurvePlotMixin,
    CurveViewMixin,
    MessageLogMixin,
    ParticleEditorMixin,
    ParticleStateMixin,
    AiWorkspaceMixin,
    AiExecutionMixin,
    AiReviewMixin,
    FitExportMixin,
    ManualFittingMixin,
    PlotRefreshMixin,
    WorkflowFeedbackMixin,
    QObject,
):
    """Translate Fitting Qt events and render ViewModel results."""

    status_updated = pyqtSignal(str)

    progress_updated = pyqtSignal(int)

    parameters_changed = pyqtSignal(dict)

    fitting_completed = pyqtSignal(dict)

    def __init__(self, ui, parent=None, fitting_view_model=None):
        super().__init__(parent)
        self.ui = ui
        self.parent = parent
        parent_widget = parent if isinstance(parent, QWidget) else None
        if parent_widget is None and parent is not None:
            parent_getter = getattr(parent, "parent", None)
            candidate = parent_getter() if callable(parent_getter) else parent_getter
            parent_widget = candidate if isinstance(candidate, QWidget) else None
        self.main_window = parent_widget or (ui if isinstance(ui, QWidget) else None)
        if fitting_view_model is None:
            fitting_view_model = _create_default_fitting_view_model()
        self.fitting_view_model = fitting_view_model
        self.preferences = fitting_view_model.context.preferences

        self.q = None
        self.I = None
        self.I_fitting = None

        self.display_mode = "normal"
        self.data_source = None
        self._q_full_min = None
        self._q_full_max = None
        self._roi_min = None
        self._roi_max = None
        self.q_ROI = None
        self.I_ROI = None
        self._updating_roi_controls = False
        self._roi_controls_enabled = True
        self._last_axis_filter_mode = "all"
        self._slider_is_source = False
        self._points_num_default = 50
        self._points_num_current = 50
        self._interp_method_default = "Linear"

        self._fitting_messages_max_lines = 500
        self._detached_fitting_dialog = None
        self._detached_append = None
        self._fitting_browser_original_height = None
        self.has_fitting_data = False

        self.current_parameters = {}

        self.fitting_results = {}

        self._insitu_last_refine_ui_update = 0.0
        self._insitu_last_refine_log_update = 0.0
        self._insitu_last_trend_refresh = 0.0
        self._insitu_trend_refresh_pending = False
        self.fitting = None

        self.independent_fit_window = None

        self.current_1d_data = None
        self.current_1d_file_path = None
        self._imported_1d_q_unit = "angstrom"

        self._preview_resize_refit_pending = False

        try:
            default_points = None
            try:
                gp_val = self.fitting_view_model.get_setting("fitting", "fit.points_num", None)
                if gp_val is not None:
                    default_points = int(gp_val)
            except Exception:
                default_points = None
            try:
                us_val = int(self.preferences.get("fit.points_num", 50))
            except Exception:
                us_val = 50
            self._points_num_default = int(default_points if default_points is not None else us_val)
            self._points_num_current = self._points_num_default
            try:
                self._interp_method_default = self.preferences.get("fit.interp_method", "Linear")
            except Exception:
                pass
        except Exception:
            pass
        try:
            self._setup_fitting_region_controls()
        except Exception:
            pass

        self._curve_graphics_scene = None
        self._current_fit_canvas = None
        self._current_fit_figure = None

        self._initialized = False
        self._initializing = True

        self._display_mode = "normal"
        self._has_fitting_data = False
        self._fitting_mode_active = False
        self._last_active_particle_ids = []
        self._particle_widgets = {}
        self._particle_parameter_meta_ids = defaultdict(list)
        self._recycled_particle_ids = []
        self._particle_widget_style_template = ""
        self._particle_widget_style_source_name = ""
        self._particle_widget_style_fallback = (
            "background-color: #ffffff;border: 1px solid #d8dee8;border-radius: 7px;"
        )
        self._particle_container_layout = None
        self._particle_add_button = None
        self._particle_show_checkboxes = {}
        self._dynamic_show_layout = None
        self._dynamic_show_container = None
        self._particle_checkbox_host_name = ""
        self._ai_job_thread = None
        self._ai_job_worker = None
        self._ai_output_dir = None
        self._ai_input_csv = None
        self._ai_action_buttons = []
        self._ai_stop_button = None
        self._ai_open_output_button = None
        self._ai_export_output_button = None
        self._ai_results_dialog = None
        self._ai_candidate_rows = []
        self._ai_log_lines = []
        self._ai_excluded_input_q = set()
        self._ai_input_data_dialog = None
        self._ai_input_data_table = None
        self._ai_input_data_summary = None
        self._ai_input_dialog_arrays = None

        self._insitu_workflow_timer = None
        self._insitu_workflow_state = "Idle"
        self._insitu_workflow_queue = []
        self._insitu_workflow_seen = set()
        self._insitu_workflow_file_sizes = {}
        self._insitu_workflow_processing_file = None
        self._insitu_workflow_busy = False
        self._insitu_workflow_stop_requested = False
        self._insitu_workflow_processed_count = 0
        self._insitu_workflow_failed_count = 0
        self._insitu_workflow_results = []
        self._insitu_workflow_last_fit_params = None
        self._insitu_workflow_current_record = None
        self._insitu_workflow_refine_thread = None
        self._insitu_workflow_refine_worker = None
        self._insitu_workflow_ai_record = None
        self._insitu_workflow_ai_then_refine = False
        self._insitu_trend_dialog = None
        self._insitu_trend_table = None
        self._insitu_trend_combo = None
        self._insitu_trend_plot_holder = None
        self._insitu_workflow_canvas_curve = None
        self._insitu_heatmap_dialog = None
        self._insitu_heatmap_widgets = {}
        self._insitu_heatmap_q = None
        self._insitu_heatmap_data = None
        self._insitu_heatmap_count = 0
        self._insitu_heatmap_capacity = 0
        self._insitu_heatmap_artist = None
        self._insitu_heatmap_colorbar = None
        self._insitu_heatmap_refresh_pending = False

        self._default_signal_mode = "changed"
        self._signal_mode_overrides = {"fitFittingRegionSlider": "changed"}

        self.model_params_manager = self.fitting_view_model.storage.model_parameters
        if self.model_params_manager is None:
            raise ValueError("FittingViewBinding requires model parameter storage")

        self.param_trigger_manager = UniversalParameterTriggerManager(
            self,
            settings_repository=self.fitting_view_model.context.settings,
        )

        self._loading_parameters = False
        self._initializing = False

        # ==========================
        # ==========================
        self._param_debounce_ms = 220
        self._param_abs_eps = 1e-12
        self._param_rel_eps = 1e-10
        self._roi_debounce_ms = 140
        self._roi_update_timer = None

        self._auto_k_enabled = False

        self._setup_particle_shape_connector()

        self._load_auto_k_enabled()

    def initialize(self):
        """No description."""
        if self._initialized:
            return

        self._initialize_ui()
        self._connect_insitu_series_page()
        self._setup_connections()
        self._sync_fitting_workflow()
        self._initialized = True
        self._initializing = False
        self._setup_meta_debug_shortcut()
