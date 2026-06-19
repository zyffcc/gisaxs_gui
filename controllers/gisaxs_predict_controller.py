"""GISAXS Predict controller responsible for displaying GISAXS data."""

from __future__ import annotations

import os
import sys
import subprocess
import re
import json
import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, Qt, QSignalBlocker, QRectF, QEvent, QTimer
from PyQt5.QtGui import QImage, QPixmap, QKeySequence
from PyQt5.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QGraphicsScene,
    QLabel,
    QShortcut,
    QTabWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QPushButton,
    QSizePolicy,
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QDialog,
    QTextBrowser,
)

from core.global_params import global_params
from ui.responsive_layout import (
    apply_density_profile,
    install_adaptive_window_profile,
    move_window_to_cursor_screen,
)
from utils.path_utils import normalize_path
from .fitting_controller import AsyncImageLoader, is_matplotlib_available, is_fabio_available
from .multifile_predict_results import (
    MultiFilePredictResultsWidget,
    MultiFilePredictManager,
    PredictResult,
    PredictStatus,
    ExportDialog
)


class GisaxsPredictController(QObject):
    """GISAXS prediction controller handling data import, display, and prediction."""

    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    parameters_changed = pyqtSignal(dict)
    prediction_completed = pyqtSignal(dict)
    model_load_finished = pyqtSignal(object, str, str)

    _DEFAULT_COLORMAPS = [
        "viridis",
        "cividis",
        "plasma",
        "magma",
        "inferno",
        "turbo",
        "jet",
        "coolwarm",
        "gray",
    ]

    _mpl_cm = None

    def __init__(self, ui, parent=None) -> None:
        super().__init__(parent)
        self.ui = ui
        self.parent = parent
        self.main_window = parent.parent if hasattr(parent, "parent") else None

        self.current_parameters: Dict[str, Optional[str]] = {}
        self.prediction_results: Dict[str, object] = {}

        # 初始化运行时状态
        self._initialized = False
        self._ui_updating = False
        self._synchronizing = False

        self._graphics_scene: Optional[QGraphicsScene] = None
        self._current_pixmap: Optional[QPixmap] = None
        self._current_image: Optional[np.ndarray] = None
        self._view_zoom_steps = 0
         

        # Predict-2D view state
        self._predict_scene: Optional[QGraphicsScene] = None
        self._predict_pixmap: Optional[QPixmap] = None
        self._predict_zoom_steps = 0

        self._index_to_file: Dict[int, str] = {}
        self._available_indices: List[int] = []
        self._sequence_indices: List[int] = []
        self._current_file_index: Optional[int] = None
        self._folder_entries: List[Tuple[str, int]] = []

        self._load_request_seq = 0
        self._latest_display_request = 0
        self._active_loaders: Dict[int, AsyncImageLoader] = {}
        self._pending_contexts: Dict[int, Dict[str, object]] = {}

        # Module system state
        self._modules_by_name: Dict[str, Dict[str, object]] = {}
        self._modules_by_id: Dict[str, Dict[str, object]] = {}
        self._current_module: Optional[Dict[str, object]] = None
        self._module_edit_watch_timer: Optional[QTimer] = None
        self._module_edit_watch_path: Optional[str] = None
        self._module_edit_watch_mtime: Optional[float] = None
        self._module_edit_watch_ticks: int = 0
        self._current_mask: Optional[np.ndarray] = None
        self._current_model: Optional[object] = None
        self._model_loading: bool = False
        self._model_cancel_requested: bool = False
        self._model_loader_thread = None
        self._model_status_label: Optional[QLabel] = None
        self._status_text_window: Optional[QDialog] = None
        self._status_text_window_browser: Optional[QTextBrowser] = None
        self._cancel_shortcut: Optional[QShortcut] = None
        self._predict_tabs: Optional[QTabWidget] = None
        self._predict_panel: Optional[QWidget] = None
        self._predict_panel_layout: Optional[QVBoxLayout] = None
        self._predict_import_button: Optional[QPushButton] = None
        self._predict_tab_specs: List[Dict[str, object]] = []
        self._predict_current_kind: Optional[str] = None
        self._predict_current_image: Optional[np.ndarray] = None
        self._predict_current_curve: Optional[np.ndarray] = None
        self._predict_curve_controls: Dict[str, object] = {}
        self._current_step_index: int = 0

        # 多文件预测相关
        self._multifile_results_widget: Optional[MultiFilePredictResultsWidget] = None
        self._multifile_manager: Optional[MultiFilePredictManager] = None
        self._multifile_prediction_active: bool = False
        self._multifile_batch_map: Dict[str, List[str]] = {}

        # 读取全局参数
        self._set_default_parameters()
        self._load_saved_parameters()
        self.model_load_finished.connect(self._on_model_load_finished)

    # ------------------------------------------------------------------
    # 初始化 & UI
    # ------------------------------------------------------------------
    def _load_saved_parameters(self) -> None:
        try:
            saved = global_params.get_module_parameters("gisaxs_predict")
            if saved:
                self.current_parameters.update(saved)
        except Exception:
            pass

    def initialize(self) -> None:
        if self._initialized:
            return

        self._setup_display_resources()
        self._setup_status_text_browser()
        self._setup_connections()
        self._initialize_ui()
        # 初始化模块选择和列表
        self._initialize_modules_ui()
        # 初始化模型状态指示灯与快捷键
        self._init_model_status_ui()
        # 初始化多文件预测UI
        self._setup_multifile_ui()
        self._initialized = True

    def _setup_display_resources(self) -> None:
        view = getattr(self.ui, "gisaxsImageGraphicsView", None)
        if view is None:
            return
        self._graphics_scene = QGraphicsScene(view)
        view.setScene(self._graphics_scene)
        view.setTransformationAnchor(view.AnchorUnderMouse)
        view.setDragMode(view.ScrollHandDrag)

        self._populate_colormap_combos()

        # Setup predict2dGraphicsView scene as well
        pview = getattr(self.ui, "predict2dGraphicsView", None)
        if pview is not None and self._predict_scene is None:
            self._predict_scene = QGraphicsScene(pview)
            pview.setScene(self._predict_scene)
            pview.setTransformationAnchor(pview.AnchorUnderMouse)
            pview.setDragMode(pview.ScrollHandDrag)

    def _setup_status_text_browser(self) -> None:
        browser = getattr(self.ui, "predictStatusTextBrowser", None)
        scroll_area = getattr(self.ui, "predictStatusScrollArea", None)
        top_panel = getattr(self.ui, "widget_2", None)
        if browser is None:
            return

        if top_panel is not None:
            top_panel.setMaximumHeight(16777215)
        if scroll_area is not None:
            scroll_area.setVisible(False)
            scroll_area.setMaximumHeight(0)
        browser.setMinimumHeight(120)
        browser.setMaximumHeight(180)
        browser.setOpenExternalLinks(False)
        browser.setContextMenuPolicy(Qt.CustomContextMenu)
        browser.customContextMenuRequested.connect(self._show_status_text_context_menu)

    def _show_status_text_context_menu(self, pos) -> None:
        browser = getattr(self.ui, "predictStatusTextBrowser", None)
        if browser is None:
            return
        menu = browser.createStandardContextMenu(pos)
        menu.addSeparator()
        menu.addAction("Open in Separate Window", self._open_status_text_window)
        menu.exec_(browser.mapToGlobal(pos))

    def _open_status_text_window(self) -> None:
        source = getattr(self.ui, "predictStatusTextBrowser", None)
        if source is None:
            return
        if self._status_text_window is not None:
            if not self._status_text_window.isVisible():
                move_window_to_cursor_screen(self._status_text_window)
            self._status_text_window.show()
            self._status_text_window.raise_()
            self._status_text_window.activateWindow()
            return

        win = QDialog(self.main_window)
        win.setWindowTitle("Predict Log")
        win.resize(900, 560)
        layout = QVBoxLayout(win)
        viewer = QTextBrowser(win)
        viewer.setReadOnly(True)
        viewer.setLineWrapMode(QTextBrowser.NoWrap)
        viewer.setPlainText(source.toPlainText())
        layout.addWidget(viewer)
        self._status_text_window = win
        self._status_text_window_browser = viewer
        install_adaptive_window_profile(
            win,
            lambda profile, screen, window=win: self._apply_floating_screen_profile(window, profile),
            apply_window_minimum=False,
        )
        win.finished.connect(self._on_status_text_window_closed)
        move_window_to_cursor_screen(win)
        win.show()

    def _on_status_text_window_closed(self) -> None:
        self._status_text_window = None
        self._status_text_window_browser = None

    def _apply_floating_screen_profile(self, window, profile) -> None:
        try:
            apply_density_profile(window, profile)
        except Exception:
            pass

    def _set_predict_main_tab(self, target_label: str) -> None:
        tabs = getattr(self.ui, "gisaxsPredictImageShowTabWidget", None)
        if tabs is None:
            return
        target = (target_label or "").strip().lower()
        try:
            for i in range(tabs.count()):
                text = tabs.tabText(i)
                if isinstance(text, str) and text.strip().lower() == target:
                    blocker = QSignalBlocker(tabs)
                    tabs.setCurrentIndex(i)
                    del blocker
                    return
        except Exception:
            return

    def _populate_colormap_combos(self) -> None:
        combos = []
        gisaxs_combo = getattr(self.ui, "gisaxsImageColormapCombox", None)
        if gisaxs_combo is not None:
            combos.append(gisaxs_combo)

        predict_combo = getattr(self.ui, "predict2dLabelCombox", None)
        if predict_combo is not None:
            combos.append(predict_combo)

        if not combos:
            return

        # Ensure the active colormap is present in the options even if defaults change later
        options = list(self._DEFAULT_COLORMAPS)
        active = self.current_parameters.get("colormap") or options[0]
        if active not in options:
            options.insert(0, active)

        for combo in combos:
            blocker = QSignalBlocker(combo)
            combo.clear()
            combo.addItems(options)
            # Set the active selection without emitting change events
            idx = combo.findText(active)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            del blocker

    def _initialize_ui(self) -> None:
        self._ui_updating = True
        try:
            framework_combo = getattr(self.ui, "gisaxsPredictFrameworkCombox", None)
            if framework_combo is not None:
                self._populate_framework_combo(framework_combo)
                idx = framework_combo.findText(self.current_parameters.get("framework", ""))
                framework_combo.setCurrentIndex(idx if idx >= 0 else 0)
                self._refresh_framework_status()

            mode = self.current_parameters.get("mode", "single_file")
            single_btn = getattr(self.ui, "gisaxsPredictSingleFileRadioButton", None)
            multi_btn = getattr(self.ui, "gisaxsPredictMultiFilesRadioButton", None)
            if single_btn is not None and multi_btn is not None:
                if mode == "multi_files":
                    multi_btn.setChecked(True)
                else:
                    single_btn.setChecked(True)

            self._set_line_edit("gisaxsPredictChooseGisaxsFileValue", os.path.basename(self.current_parameters.get("input_file", "")))
            self._set_line_edit("gisaxsPredictChooseFolderValue", self.current_parameters.get("input_folder", ""))
            self._set_line_edit("gisaxsPredictExportFolderValue", self.current_parameters.get("export_path", ""))

            stack_text = self.current_parameters.get("stack_value", "1")
            if mode == "multi_files":
                stack_text = self.current_parameters.get("range_value", "") or stack_text
            self._set_line_edit("gisaxsPredictStackValue", stack_text)
            self._set_line_edit("gisaxsImageShowingValue", self.current_parameters.get("showing_value", ""))

            auto_scale = bool(self.current_parameters.get("auto_scale", True))
            self._configure_color_spin("gisaxsImageVminValue")
            self._configure_color_spin("gisaxsImageVmaxValue")
            self._configure_color_spin("predict2dVminValue")
            self._configure_color_spin("predict2dVmaxValue")
            self._set_checkbox("gisaxsImageAutoScaleCheckBox", auto_scale)
            self._set_checkbox("gisaxsImageLogScaleCheckBox", bool(self.current_parameters.get("gisaxs_log_scale", False)))
            self._set_double_spin("gisaxsImageVminValue", self.current_parameters.get("vmin"))
            self._set_double_spin("gisaxsImageVmaxValue", self.current_parameters.get("vmax"))

            predict_auto_scale = bool(self.current_parameters.get("predict_auto_scale", True))
            self._set_checkbox("predict2dAutoScaleCheckBox", predict_auto_scale)
            self._set_double_spin("predict2dVminValue", self.current_parameters.get("predict_vmin"))
            self._set_double_spin("predict2dVmaxValue", self.current_parameters.get("predict_vmax"))

            colormap = self.current_parameters.get("colormap") or self._DEFAULT_COLORMAPS[0]
            self._set_combobox_text("gisaxsImageColormapCombox", colormap)
            self._set_combobox_text("predict2dLabelCombox", colormap)

            self._set_checkbox("predict2dLogScaleCheckBox", bool(self.current_parameters.get("predict_log_scale", False)))

            btn = getattr(self.ui, "gisaxsImageSaveButton", None)
            if btn is not None:
                btn.setVisible(False)
            btn = getattr(self.ui, "predict2SaveButton", None)
            if btn is not None:
                btn.setVisible(False)

            self._update_mode_controls(mode)

            # Default to GISAXS tab on initial load
            self._set_predict_main_tab("GISAXS")
            self._refresh_predict_readiness()

        finally:
            self._ui_updating = False

    def _setup_connections(self) -> None:
        btn = getattr(self.ui, "gisaxsPredictChooseFolderButton", None)
        if btn:
            btn.clicked.connect(self._choose_gisaxs_folder)

        btn = getattr(self.ui, "gisaxsPredictChooseGisaxsFileButton", None)
        if btn:
            btn.clicked.connect(self._choose_gisaxs_file)

        btn = getattr(self.ui, "gisaxsPredictExportFolderButton", None)
        if btn:
            btn.clicked.connect(self._choose_export_folder)

        btn = getattr(self.ui, "gisaxsPredictPredictButton", None)
        if btn:
            btn.clicked.connect(self._run_gisaxs_predict)

        btn = getattr(self.ui, "gisaxsPredictStopButton", None)
        if btn:
            btn.clicked.connect(self._stop_gisaxs_predict)

        btn = getattr(self.ui, "gisaxsPredictShowMultiFileResultsButton", None)
        if btn:
            btn.clicked.connect(self._show_multifile_results_window)

        # Inline import button on stack row (new ui name)
        inline_import = getattr(self.ui, "gisaxsPredictImportimagesButton", None)
        if inline_import:
            inline_import.clicked.connect(self._on_import_images_clicked)

        single_btn = getattr(self.ui, "gisaxsPredictSingleFileRadioButton", None)
        multi_btn = getattr(self.ui, "gisaxsPredictMultiFilesRadioButton", None)
        if single_btn:
            single_btn.toggled.connect(self._on_mode_changed)
        if multi_btn:
            multi_btn.toggled.connect(self._on_mode_changed)

        file_edit = getattr(self.ui, "gisaxsPredictChooseGisaxsFileValue", None)
        if file_edit is not None:
            file_edit.returnPressed.connect(self._handle_file_line_edit_committed)
        stack_edit = getattr(self.ui, "gisaxsPredictStackValue", None)
        if stack_edit is not None:
            stack_edit.returnPressed.connect(self._on_stack_field_committed)
        showing_edit = getattr(self.ui, "gisaxsImageShowingValue", None)
        if showing_edit is not None:
            showing_edit.returnPressed.connect(self._on_showing_value_committed)

        cb = getattr(self.ui, "gisaxsImageAutoScaleCheckBox", None)
        if cb:
            cb.toggled.connect(self._on_auto_scale_toggled)

        cb = getattr(self.ui, "gisaxsImageLogScaleCheckBox", None)
        if cb:
            cb.toggled.connect(self._on_gisaxs_log_scale_toggled)

        btn = getattr(self.ui, "gisaxsImageAutoScaleResetButton", None)
        if btn:
            btn.clicked.connect(self._on_auto_scale_reset)

        btn = getattr(self.ui, "gisaxsImageExportButton", None)
        if btn:
            btn.clicked.connect(self._export_gisaxs_image)

        self._connect_double_spin("gisaxsImageVminValue", self._on_vmin_changed)
        self._connect_double_spin("gisaxsImageVmaxValue", self._on_vmax_changed)

        predict_auto_cb = getattr(self.ui, "predict2dAutoScaleCheckBox", None)
        if predict_auto_cb:
            predict_auto_cb.toggled.connect(self._on_predict_auto_scale_toggled)
        predict_auto_reset = getattr(self.ui, "predict2dAutoScaleResetButton", None)
        if predict_auto_reset:
            predict_auto_reset.clicked.connect(self._on_predict_auto_scale_reset)
        self._connect_double_spin("predict2dVminValue", self._on_predict_vmin_changed)
        self._connect_double_spin("predict2dVmaxValue", self._on_predict_vmax_changed)

        combo = getattr(self.ui, "gisaxsImageColormapCombox", None)
        if combo:
            combo.currentTextChanged.connect(self._on_colormap_changed)

        p_combo = getattr(self.ui, "predict2dLabelCombox", None)
        if p_combo:
            p_combo.currentTextChanged.connect(self._on_colormap_changed)

        framework_combo = getattr(self.ui, "gisaxsPredictFrameworkCombox", None)
        if framework_combo:
            framework_combo.currentTextChanged.connect(lambda _=None: (self._refresh_framework_status(), self._refresh_predict_readiness()))

        zoom_in = getattr(self.ui, "gisaxsImageZoomInButton", None)
        zoom_out = getattr(self.ui, "gisaxsImageZoomOutButton", None)
        zoom_reset = getattr(self.ui, "gisaxsImageZoomResetButton", None)
        if zoom_in:
            zoom_in.clicked.connect(self._zoom_in)
        if zoom_out:
            zoom_out.clicked.connect(self._zoom_out)
        if zoom_reset:
            zoom_reset.clicked.connect(self._zoom_reset)

        p_zoom_in = getattr(self.ui, "predict2dZoomInButton", None)
        p_zoom_out = getattr(self.ui, "predict2dZoomOutButton", None)
        p_zoom_reset = getattr(self.ui, "predict2dZoomResetButton", None)
        if p_zoom_in:
            p_zoom_in.clicked.connect(self._predict_zoom_in)
        if p_zoom_out:
            p_zoom_out.clicked.connect(self._predict_zoom_out)
        if p_zoom_reset:
            p_zoom_reset.clicked.connect(self._predict_zoom_reset)

        # Predict-2D controls
        cb = getattr(self.ui, "predict2dLogScaleCheckBox", None)
        if cb:
            cb.toggled.connect(self._on_predict_log_scale_toggled)
        btn = getattr(self.ui, "predict2dExportButton", None)
        if btn:
            btn.clicked.connect(self._on_predict_export_clicked)

        # Module select combobox
        module_combo = getattr(self.ui, "gisaxsPredictModuleSelectCombox", None)
        if module_combo:
            module_combo.currentTextChanged.connect(self._on_module_selected)
            module_combo.installEventFilter(self)

        # Module action buttons
        btn_edit = getattr(self.ui, "gisaxsPredictEditButton", None)
        if btn_edit:
            btn_edit.clicked.connect(self._on_edit_module_clicked)
        btn_reload = getattr(self.ui, "gisaxsPredictReloadConfigButton", None)
        if btn_reload:
            btn_reload.clicked.connect(self._on_reload_module_config_clicked)
        btn_import = getattr(self.ui, "gisaxsPredictModelImportButton", None)
        if btn_import:
            btn_import.clicked.connect(self._on_model_import_clicked)

    def _init_model_status_ui(self) -> None:
        text_label = getattr(self.ui, "gisaxsPredictModelStatusTextLabel", None)
        if text_label is not None:
            self._model_status_label = text_label
            self._set_model_status_color("gray", "Not loaded")
            self._refresh_predict_readiness()
            return

        # Fallback for older generated layouts: create a status label in the button row.
        layout = getattr(self.ui, "horizontalLayout_15", None)
        if layout is None:
            return
        if self._model_status_label is None:
            lbl = QLabel("Not loaded")
            lbl.setMinimumWidth(76)
            lbl.setToolTip("Model status")
            self._model_status_label = lbl
            try:
                layout.addWidget(lbl)
            except Exception:
                pass
        self._set_model_status_color("gray", "Not loaded")

        # Create predict panel with tabs on the right side (under the same row)
        try:
            if self._predict_panel is None:
                panel = QWidget()
                vlayout = QVBoxLayout(panel)
                tabs = QTabWidget(panel)
                vlayout.addWidget(tabs)
                hlayout = QHBoxLayout()
                btn = QPushButton("Import")
                btn.setToolTip("Import/Reload Model")
                btn.clicked.connect(self._on_model_import_clicked)
                hlayout.addStretch(1)
                hlayout.addWidget(btn)
                vlayout.addLayout(hlayout)
                layout.addWidget(panel)
                self._predict_panel = panel
                self._predict_panel_layout = vlayout
                self._predict_tabs = tabs
                self._predict_import_button = btn
                try:
                    panel.setVisible(False)
                except Exception:
                    pass
        except Exception:
            pass

        # Register Ctrl+C to cancel loading
        parent_widget = getattr(self.ui, "widget_4", None) or getattr(self.ui, "centralwidget", None) or self.main_window
        try:
            if parent_widget is not None:
                self._cancel_shortcut = QShortcut(QKeySequence("Ctrl+C"), parent_widget)
                self._cancel_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
                self._cancel_shortcut.activated.connect(self._on_cancel_loading_shortcut)
        except Exception:
            pass

    def _set_model_status_color(self, color: str, tooltip: str = "") -> None:
        text_map = {
            "green": "Loaded",
            "red": "Loading",
            "gray": "Not loaded",
        }
        status_text = tooltip if tooltip in ("Loaded", "Not loaded", "Canceled") else text_map.get(color, tooltip or "Not loaded")
        style = (
            "QLabel {"
            f"background-color: {color};"
            "border: 1px solid #94a3b8;"
            "border-radius: 6px;"
            "color: white;"
            "font-weight: 600;"
            "padding: 4px 8px;"
            "}"
        )
        labels = []
        if self._model_status_label is not None:
            labels.append(self._model_status_label)
        for name in ("gisaxsPredictModelStatusTextLabel",):
            label = getattr(self.ui, name, None)
            if label is not None and label not in labels:
                labels.append(label)
        for label in labels:
            label.setStyleSheet(style)
            label.setText(status_text)
            if tooltip:
                label.setToolTip(tooltip)
        self._refresh_predict_readiness()

    def _on_cancel_loading_shortcut(self) -> None:
        if not self._model_loading:
            return
        self._model_cancel_requested = True
        self._set_model_status_color("gray", "Canceled")
        self.status_updated.emit("Model load cancel requested (Ctrl+C).")
        self.progress_updated.emit(0)
        # Re-enable import button now for UX; background thread may still finish but will be ignored
        btn_import = getattr(self.ui, "gisaxsPredictModelImportButton", None)
        if btn_import:
            btn_import.setEnabled(True)

    def _setup_multifile_ui(self) -> None:
        """初始化多文件预测的外置窗口，不改动主窗口布局"""
        try:
            if getattr(self, "_multifile_window", None) is not None:
                return

            # 创建一个无模式的外置对话框，独立显示多文件结果
            win = QDialog(self.main_window)
            win.setWindowTitle("Multi-File Results")
            win.setModal(False)
            win.setMinimumSize(700, 600)
            win.resize(820, 680)
            try:
                win.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            except Exception:
                pass

            outer = QVBoxLayout(win)
            outer.setContentsMargins(10, 8, 10, 8)
            outer.setSpacing(8)
            install_adaptive_window_profile(
                win,
                lambda profile, screen, window=win: self._apply_floating_screen_profile(window, profile),
                apply_window_minimum=False,
            )

            # === 1. 当前文件显示区域 ===
            current_file_frame = QFrame(win)
            current_file_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
            current_file_frame.setStyleSheet(
                """
                QFrame {
                    background-color: #ffffff;
                    border: 1px solid #ced4da;
                    border-radius: 6px;
                    margin: 2px;
                }
                """
            )
            current_file_layout = QVBoxLayout(current_file_frame)
            current_file_layout.setContentsMargins(8, 6, 8, 6)
            current_file_layout.setSpacing(4)

            current_file_title = QLabel("Current File", current_file_frame)
            current_file_title.setStyleSheet(
                """
                QLabel {
                    font-weight: bold;
                    font-size: 11px;
                    color: #495057;
                    border-bottom: 1px solid #dee2e6;
                    padding-bottom: 3px;
                    margin-bottom: 3px;
                }
                """
            )
            current_file_layout.addWidget(current_file_title)

            self._current_file_label = QLabel("No file selected", current_file_frame)
            self._current_file_label.setStyleSheet(
                """
                QLabel {
                    background-color: #f8f9fa;
                    border: 1px solid #e9ecef;
                    border-radius: 4px;
                    padding: 8px;
                    font-size: 10px;
                    color: #6c757d;
                    font-family: 'Consolas', 'Courier New', monospace;
                }
                """
            )
            self._current_file_label.setWordWrap(True)
            self._current_file_label.setMinimumHeight(50)
            current_file_layout.addWidget(self._current_file_label)
            outer.addWidget(current_file_frame)

            # === 2. 多文件结果列表区域 ===
            results_frame = QFrame(win)
            results_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
            results_frame.setStyleSheet(
                """
                QFrame {
                    background-color: #ffffff;
                    border: 1px solid #ced4da;
                    border-radius: 6px;
                    margin: 2px;
                }
                """
            )
            results_layout = QVBoxLayout(results_frame)
            results_layout.setContentsMargins(8, 6, 8, 6)
            results_layout.setSpacing(4)

            results_title = QLabel("Multi-File Results", results_frame)
            results_title.setStyleSheet(
                """
                QLabel {
                    font-weight: bold;
                    font-size: 11px;
                    color: #495057;
                    border-bottom: 1px solid #dee2e6;
                    padding-bottom: 3px;
                    margin-bottom: 3px;
                }
                """
            )
            results_layout.addWidget(results_title)

            self._multifile_results_widget = MultiFilePredictResultsWidget(parent=results_frame)
            self._multifile_results_widget.setStyleSheet(
                """
                MultiFilePredictResultsWidget {
                    border: none;
                    background-color: transparent;
                }
                """
            )
            results_layout.addWidget(self._multifile_results_widget)
            outer.addWidget(results_frame, stretch=1)

            # === 3. 快捷操作按钮区域 ===
            actions_frame = QFrame(win)
            actions_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
            actions_frame.setStyleSheet(
                """
                QFrame {
                    background-color: #ffffff;
                    border: 1px solid #ced4da;
                    border-radius: 6px;
                    margin: 2px;
                }
                """
            )
            actions_layout = QVBoxLayout(actions_frame)
            actions_layout.setContentsMargins(8, 6, 8, 6)
            actions_layout.setSpacing(6)

            actions_title = QLabel("Quick Actions", actions_frame)
            actions_title.setStyleSheet(
                """
                QLabel {
                    font-weight: bold;
                    font-size: 11px;
                    color: #495057;
                    border-bottom: 1px solid #dee2e6;
                    padding-bottom: 3px;
                    margin-bottom: 3px;
                }
                """
            )
            actions_layout.addWidget(actions_title)

            buttons_layout = QHBoxLayout()
            buttons_layout.setSpacing(8)
            clear_button = QPushButton("Clear All", actions_frame)
            clear_button.setMinimumHeight(28)
            clear_button.clicked.connect(self._clear_multifile_results)
            export_all_button = QPushButton("Export All", actions_frame)
            export_all_button.setMinimumHeight(28)
            export_all_button.clicked.connect(self._export_all_results)
            buttons_layout.addWidget(clear_button)
            buttons_layout.addWidget(export_all_button)
            actions_layout.addLayout(buttons_layout)
            outer.addWidget(actions_frame)

            # 连接信号
            self._multifile_results_widget.result_selected.connect(self._on_multifile_result_selected)
            self._multifile_results_widget.result_double_clicked.connect(self._on_multifile_result_selected)
            self._multifile_results_widget.export_requested.connect(self._on_multifile_export_requested)

            # 创建多文件管理器
            if self._multifile_manager is None:
                self._multifile_manager = MultiFilePredictManager(self)
                self._multifile_manager.prediction_started.connect(self._on_multifile_prediction_started)
                self._multifile_manager.prediction_completed.connect(self._on_multifile_prediction_completed)
                self._multifile_manager.result_updated.connect(self._on_multifile_result_updated)
                self._multifile_manager.progress_updated.connect(self._on_multifile_progress_updated)

            # 初始不显示，仅在切换到 multi_files 模式时显示
            self._multifile_window = win
            self._append_status_message("Multi-file external window initialized", level="INFO")

        except Exception as e:
            self._append_status_message(f"Failed to setup multi-file UI: {e}", level="ERROR")

    def _show_multifile_results_window(self) -> None:
        if getattr(self, "_multifile_window", None) is None:
            self._setup_multifile_ui()
        win = getattr(self, "_multifile_window", None)
        if win is None:
            QMessageBox.information(self.main_window, "Multi-File Results", "The multi-file results window is not available yet.")
            return
        if self._multifile_results_widget is not None:
            self._multifile_results_widget.setVisible(True)
        if not win.isVisible():
            move_window_to_cursor_screen(win)
        win.show()
        try:
            win.raise_()
            win.activateWindow()
        except Exception:
            pass

    def _clear_multifile_results(self) -> None:
        """清空所有多文件结果"""
        if self._multifile_results_widget:
            self._multifile_results_widget.clear_all_results()
            
    def _export_all_results(self) -> None:
        """导出所有结果"""
        if self._multifile_results_widget:
            all_results = self._multifile_results_widget.get_all_results()
            if all_results:
                self._multifile_results_widget.onExportClicked()
            else:
                QMessageBox.information(self.main_window, "Export", "No results to export.")

    def _stop_gisaxs_predict(self) -> None:
        if not self._multifile_prediction_active:
            self._append_status_message("No active multi-file prediction to stop.", level="INFO")
            return
        if self._multifile_manager:
            self._multifile_manager.cancel_prediction()
            self._append_status_message("Stopping multi-file prediction after the current file...", level="WARN")
        stop_btn = getattr(self.ui, "gisaxsPredictStopButton", None)
        if stop_btn:
            stop_btn.setEnabled(False)

    def _adjust_predict_layout_for_mode(self, mode: str) -> None:
        """根据模式调整预测布局"""
        # 显示/隐藏外置的多文件窗口
        try:
            win = getattr(self, "_multifile_window", None)
            if win is not None and mode == "multi_files" and win.isVisible():
                win.raise_()
        except Exception:
            pass
        
        # 更新当前文件标签的可见性
        if hasattr(self, '_current_file_label'):
            if mode == "multi_files":
                self._current_file_label.setVisible(True)
                if not self._current_file_label.text() or self._current_file_label.text() == "Current: No file selected":
                    self._current_file_label.setText("No file selected")
            else:
                self._current_file_label.setVisible(False)

    def _update_current_file_display(self, file_path: str, stack_count: int = 1) -> None:
        """更新当前文件显示"""
        if hasattr(self, '_current_file_label'):
            if file_path:
                file_name = os.path.basename(file_path)
                suffix = f" ({stack_count} files stacked)" if stack_count and stack_count > 1 else " (1 file)"
                self._current_file_label.setText(f"{file_name}{suffix}")
                self._current_file_label.setToolTip(file_path)
            else:
                self._current_file_label.setText("No file selected")
                self._current_file_label.setToolTip("")

    def _connect_line_edit(self, name: str, slot) -> None:
        widget = getattr(self.ui, name, None)
        if widget is None:
            return
        widget.returnPressed.connect(slot)

    def _connect_double_spin(self, name: str, slot) -> None:
        widget = getattr(self.ui, name, None)
        if widget is None:
            return
        widget.editingFinished.connect(slot)

    # ------------------------------------------------------------------
    # 参数与持久化
    # ------------------------------------------------------------------
    def _set_default_parameters(self) -> None:
        self.current_parameters = {
            "framework": "tensorflow 2.15.0",
            "mode": "single_file",
            "input_file": "",
            "input_folder": "",
            "export_path": "",
            "stack_value": "1",
            "range_value": "",
            "showing_value": "",
            "auto_scale": True,
            "vmin": None,
            "vmax": None,
            "predict_auto_scale": True,
            "predict_vmin": None,
            "predict_vmax": None,
            "colormap": self._DEFAULT_COLORMAPS[0],
            "gisaxs_log_scale": False,
            "predict_log_scale": False,
            "predict_curve_logx": False,
            "predict_curve_logy": False,
            "predict_curve_autoscale": True,
            "predict_curve_xmin": None,
            "predict_curve_xmax": None,
            "predict_curve_ymin": None,
            "predict_curve_ymax": None,
            # module selection
            "module_name": "",
            "module_model_path": "",
        }

    def _persist_parameters(self) -> None:
        if self._synchronizing:
            return
        self._synchronizing = True
        try:
            global_params.set_module_parameters("gisaxs_predict", dict(self.current_parameters))
            self.parameters_changed.emit(dict(self.current_parameters))
        finally:
            self._synchronizing = False

    def get_parameters(self) -> Dict[str, object]:
        return dict(self.current_parameters)

    def set_parameters(self, parameters: Dict[str, object]) -> None:
        if not parameters:
            return
        self.current_parameters.update(parameters)
        if self._initialized:
            self._initialize_ui()

    # ------------------------------------------------------------------
    # 文件与模式处理
    # ------------------------------------------------------------------
    def _choose_gisaxs_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self.main_window, "Select GISAXS Folder", "")
        if not folder:
            return
        folder = normalize_path(folder)
        self.current_parameters["input_folder"] = folder
        self._set_line_edit("gisaxsPredictChooseFolderValue", folder)
        self._scan_directory_for_cbf(folder)
        self._persist_parameters()
        self._refresh_predict_readiness()

    def _choose_gisaxs_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Select GISAXS File",
            self.current_parameters.get("input_folder", ""),
            "GISAXS Files (*.cbf);;All Files (*)",
        )
        if file_path:
            file_path = normalize_path(file_path)
            self._handle_new_file_selection(file_path)

    def _choose_export_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self.main_window,
            "Select Export Folder",
            self.current_parameters.get("export_path", ""),
        )
        if not folder:
            return
        folder = normalize_path(folder)
        self.current_parameters["export_path"] = folder
        self._set_line_edit("gisaxsPredictExportFolderValue", folder)
        self._persist_parameters()
        self._append_status_message(f"Export folder selected: {folder}")

    def _prompt_export_folder(self, title: str = "Select Export Folder") -> str:
        folder = QFileDialog.getExistingDirectory(
            self.main_window,
            title,
            self.current_parameters.get("export_path", "") or "",
        )
        if not folder:
            return ""
        folder = normalize_path(folder)
        self.current_parameters["export_path"] = folder
        self._set_line_edit("gisaxsPredictExportFolderValue", folder)
        self._persist_parameters()
        return folder

    def _handle_file_line_edit_committed(self) -> None:
        widget = getattr(self.ui, "gisaxsPredictChooseGisaxsFileValue", None)
        if not widget:
            return
        text = normalize_path(widget.text())
        if not text:
            return
        if os.path.isabs(text) and os.path.exists(text):
            self._handle_new_file_selection(text)
            return

        folder = normalize_path(self.current_parameters.get("input_folder", ""))
        candidate = normalize_path(os.path.join(folder, text) if folder else text)
        if os.path.exists(candidate):
            self._handle_new_file_selection(candidate)
            return
        self._append_status_message(f"Unable to locate file: {text}", level="WARN")
        QMessageBox.warning(self.main_window, "File Not Found", f"Unable to locate file: {text}")

    def _on_import_images_clicked(self) -> None:
        # Behaves like pressing Enter in the file input: try to load the typed file
        self._sync_pending_text_fields()
        self._handle_file_line_edit_committed()

    def _handle_new_file_selection(self, file_path: str) -> None:
        file_path = normalize_path(file_path)
        if not os.path.exists(file_path):
            QMessageBox.warning(self.main_window, "File Not Found", file_path)
            return
        if not file_path.lower().endswith(".cbf"):
            QMessageBox.warning(self.main_window, "Unsupported Format", "Only CBF files are supported.")
            return

        folder = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        index = self._extract_index(base_name)

        self.current_parameters.update(
            {
                "input_file": file_path,
                "input_folder": folder,
                "showing_value": str(index or ""),
            }
        )

        self._set_line_edit("gisaxsPredictChooseGisaxsFileValue", base_name)
        self._set_line_edit("gisaxsPredictChooseFolderValue", folder)
        self._set_line_edit("gisaxsImageShowingValue", self.current_parameters.get("showing_value", ""))

        self._scan_directory_for_cbf(folder)
        if index is not None:
            self._current_file_index = index
        elif self._available_indices:
            self._current_file_index = self._available_indices[0]

        mode = self.current_parameters.get("mode", "single_file")
        if mode == "multi_files":
            typed_range = self._get_line_edit_text("gisaxsPredictStackValue")
            if typed_range.strip():
                self.current_parameters["range_value"] = typed_range
                self._set_line_edit("gisaxsPredictStackValue", typed_range)
            else:
                default_range = f"{self._current_file_index}-{self._current_file_index}"
                self.current_parameters["range_value"] = default_range
                self._set_line_edit("gisaxsPredictStackValue", default_range)
        else:
            stack_text = self._get_line_edit_text("gisaxsPredictStackValue") or self.current_parameters.get("stack_value", "1")
            self.current_parameters["stack_value"] = stack_text or "1"
            self._set_line_edit("gisaxsPredictStackValue", stack_text or "1")

        self._update_range_tooltip()
        self._persist_parameters()
        self._trigger_data_reload()
        self._refresh_predict_readiness()

    def _scan_directory_for_cbf(self, folder: str) -> None:
        folder = normalize_path(folder)
        entries: List[Tuple[str, int]] = []
        index_to_file: Dict[int, str] = {}

        try:
            for name in sorted(os.listdir(folder)):
                if not name.lower().endswith(".cbf"):
                    continue
                idx = self._extract_index(name)
                if idx is None:
                    continue
                full = os.path.join(folder, name)
                entries.append((name, idx))
                index_to_file[idx] = full

            if not entries:
                self._append_status_message("No numbered CBF files detected in the current folder", level="WARN")

            self._folder_entries = entries
            self._index_to_file = index_to_file
            self._available_indices = sorted(index_to_file.keys())
            self._update_range_tooltip()
        except Exception as exc:
            self._append_status_message(f"Failed to scan folder: {exc}", level="ERROR")

    def _extract_index(self, file_name: str) -> Optional[int]:
        match = re.search(r"_(\d+)(?=\.cbf$)", file_name, re.IGNORECASE)
        if not match:
            match = re.search(r"(\d+)(?=\.cbf$)", file_name, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        if file_name.lower().endswith(".cbf"):
            return 1
        return None

    def _update_range_tooltip(self) -> None:
        if not self._available_indices:
            tooltip = "No valid indices detected yet"
        else:
            tooltip = f"Available index range: {self._available_indices[0]} - {self._available_indices[-1]}"

        label = getattr(self.ui, "gisaxsPredictStackLabel", None)
        line_edit = getattr(self.ui, "gisaxsPredictStackValue", None)
        if label:
            label.setToolTip(tooltip)
        if line_edit:
            line_edit.setToolTip(tooltip)

    def _on_mode_changed(self) -> None:
        if self._ui_updating:
            return
        single_btn = getattr(self.ui, "gisaxsPredictSingleFileRadioButton", None)
        if single_btn is not None and single_btn.isChecked():
            self.current_parameters["mode"] = "single_file"
        else:
            self.current_parameters["mode"] = "multi_files"

        self._update_mode_controls(self.current_parameters["mode"])
        self._persist_parameters()
        self._refresh_predict_readiness()

    def _update_mode_controls(self, mode: str) -> None:
        label = getattr(self.ui, "gisaxsPredictStackLabel", None)
        stack_edit = getattr(self.ui, "gisaxsPredictStackValue", None)
        showing = getattr(self.ui, "gisaxsImageShowingValue", None)
        every_label = getattr(self.ui, "gisaxsPredictEveryLabel", None)
        every_value = getattr(self.ui, "gisaxsPredictEveryValue", None)

        if label:
            label.setText("Range:" if mode == "multi_files" else "Stack:")
        if stack_edit:
            text = (
                self.current_parameters.get("range_value", "")
                if mode == "multi_files"
                else self.current_parameters.get("stack_value", "1")
            )
            self._set_line_edit("gisaxsPredictStackValue", text or ("1" if mode == "single_file" else ""))
        if showing:
            showing.setEnabled(mode == "multi_files")
        # Only show the "Every" controls in multi-file mode
        if every_label:
            every_label.setVisible(mode == "multi_files")
        if every_value:
            every_value.setVisible(mode == "multi_files")

        # 显示/隐藏多文件结果列表
        if self._multifile_results_widget:
            self._multifile_results_widget.setVisible(mode == "multi_files")
        
        # 在多文件模式下调整布局
        self._adjust_predict_layout_for_mode(mode)

    def _sync_pending_text_fields(self) -> None:
        """Apply user-typed values without triggering loads."""
        mode = self.current_parameters.get("mode", "single_file")
        stack_text = self._get_line_edit_text("gisaxsPredictStackValue")
        if mode == "multi_files":
            if stack_text.strip():
                self.current_parameters["range_value"] = stack_text
        else:
            if stack_text.strip():
                self.current_parameters["stack_value"] = stack_text

        if mode == "multi_files":
            showing_text = self._get_line_edit_text("gisaxsImageShowingValue")
            if showing_text.strip():
                self.current_parameters["showing_value"] = showing_text

    def _on_stack_field_committed(self) -> None:
        if self._ui_updating:
            return
        mode = self.current_parameters.get("mode", "single_file")
        text = self._get_line_edit_text("gisaxsPredictStackValue")
        if mode == "multi_files":
            self.current_parameters["range_value"] = text
            self._persist_parameters()
            self._trigger_data_reload()
            return

        try:
            count = max(1, int(text or "1"))
        except ValueError:
            count = 1
        self.current_parameters["stack_value"] = str(count)
        self._set_line_edit("gisaxsPredictStackValue", str(count))
        self._persist_parameters()
        self._trigger_data_reload()

    def _on_showing_value_committed(self) -> None:
        if self._ui_updating:
            return
        mode = self.current_parameters.get("mode", "single_file")
        if mode != "multi_files":
            return
        text = self._get_line_edit_text("gisaxsImageShowingValue")
        try:
            index = int(text)
        except ValueError:
            self._append_status_message("Showing Value must be a valid index", level="WARN")
            return
        if index not in self._index_to_file:
            self._append_status_message("Index is outside the available range", level="WARN")
            return
        self.current_parameters["showing_value"] = str(index)
        self._persist_parameters()
        self._start_image_loading(self._index_to_file[index], 1, {"mode": "multi_files", "index": index})

    def _parse_range_text(self, text: str) -> List[int]:
        match = re.match(r"\s*(\d+)\s*(?:-\s*(\d+))?\s*", text)
        if not match:
            return []
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
        if end < start:
            start, end = end, start
        return list(range(start, end + 1))

    def _trigger_data_reload(self) -> None:
        if self.current_parameters.get("mode", "single_file") == "single_file":
            self._load_single_stack()
        else:
            self._load_multi_sequence()

    def _load_single_stack(self) -> None:
        file_path = self.current_parameters.get("input_file")
        if not file_path:
            return
        try:
            stack = max(1, int(self.current_parameters.get("stack_value", "1")))
        except ValueError:
            stack = 1
            self.current_parameters["stack_value"] = "1"
        self._start_image_loading(file_path, stack, {"mode": "single_file", "stack": stack})

    def _load_multi_sequence(self) -> None:
        if not self._index_to_file:
            if self.current_parameters.get("input_folder"):
                self._scan_directory_for_cbf(self.current_parameters["input_folder"])
            if not self._index_to_file:
                return

        range_text = self.current_parameters.get("range_value") or self._get_line_edit_text("gisaxsPredictStackValue")
        indices = [idx for idx in self._parse_range_text(range_text) if idx in self._index_to_file]
        if not indices:
            if self._available_indices:
                indices = [self._available_indices[0]]
            else:
                self._append_status_message("No Multi File indices available", level="WARN")
                return

        self._sequence_indices = indices
        first = indices[0]
        self.current_parameters["range_value"] = range_text
        self.current_parameters["showing_value"] = str(first)
        self._set_line_edit("gisaxsImageShowingValue", str(first))
        self._persist_parameters()
        self._start_image_loading(self._index_to_file[first], 1, {"mode": "multi_files", "index": first})
        self._refresh_predict_readiness()

    def _input_ready(self) -> bool:
        mode = self.current_parameters.get("mode", "single_file")
        if mode == "single_file":
            file_path = self.current_parameters.get("input_file")
            return bool(file_path and os.path.exists(file_path))
        folder = self.current_parameters.get("input_folder")
        if not folder or not os.path.isdir(folder):
            return False
        range_text = self.current_parameters.get("range_value") or self._get_line_edit_text("gisaxsPredictStackValue")
        return bool(range_text.strip() or self._available_indices or self._folder_entries)

    def _model_ready(self) -> bool:
        return self._current_model is not None and not self._model_loading

    def _refresh_predict_readiness(self) -> None:
        if not hasattr(self, "ui"):
            return
        input_ready = self._input_ready()
        model_ready = self._model_ready()
        framework_ready = self._framework_ready()
        mode = self.current_parameters.get("mode", "single_file")

        labels = {
            "gisaxsPredictInputReadyLabel": ("Input: Ready" if input_ready else "Input: Missing", input_ready),
            "gisaxsPredictModelReadyLabel": ("Model: Loaded" if model_ready else "Model: Not loaded", model_ready),
            "gisaxsPredictFrameworkReadyLabel": ("Framework: OK" if framework_ready else "Framework: Missing/Incompatible", framework_ready),
            "gisaxsPredictModeLabel": (f"Mode: {'Multi Files' if mode == 'multi_files' else 'Single File'}", True),
        }
        for name, (text, ok) in labels.items():
            label = getattr(self.ui, name, None)
            if label is not None:
                label.setText(text)
                label.setStyleSheet("color: #166534;" if ok else "color: #b91c1c;")

        btn = getattr(self.ui, "gisaxsPredictPredictButton", None)
        if btn is not None and not self._multifile_prediction_active:
            btn.setEnabled(input_ready and model_ready and framework_ready)
        stop_btn = getattr(self.ui, "gisaxsPredictStopButton", None)
        if stop_btn is not None:
            stop_btn.setEnabled(bool(self._multifile_prediction_active))

        for export_name in ("gisaxsImageExportButton", "predict2dExportButton"):
            export_btn = getattr(self.ui, export_name, None)
            if export_btn is not None:
                export_btn.setEnabled(bool(self.prediction_results))

    # ------------------------------------------------------------------
    # 图像加载与显示
    # ------------------------------------------------------------------
    def _start_image_loading(self, file_path: str, stack_count: int, context: Dict[str, object]) -> None:
        if not os.path.exists(file_path):
            QMessageBox.warning(self.main_window, "File Not Found", file_path)
            return
        if not is_fabio_available():
            QMessageBox.warning(
                self.main_window,
                "Missing Dependency",
                "Install the fabio package to read CBF files (pip install fabio)",
            )
            return

        self._load_request_seq += 1
        request_id = self._load_request_seq
        loader = AsyncImageLoader()
        self._active_loaders[request_id] = loader

        # Precompute stack file names for logging so we can show per-file progress
        stack_files: List[str] = []
        if stack_count > 1:
            try:
                directory = os.path.dirname(file_path)
                start_name = os.path.basename(file_path)
                cbf_files = sorted([f for f in os.listdir(directory) if f.lower().endswith(".cbf")])
                start_idx = cbf_files.index(start_name)
                stack_files = cbf_files[start_idx : start_idx + stack_count]
            except Exception:
                stack_files = []

        self._pending_contexts[request_id] = {
            **context,
            "file": file_path,
            "stack": stack_count,
            "stack_files": stack_files,
            "_last_progress_file": None,
        }

        loader.image_loaded.connect(lambda data, path, rid=request_id: self._on_image_loaded(rid, data, path))
        loader.progress_updated.connect(lambda progress, msg, rid=request_id: self._on_loader_progress(rid, progress, msg))
        loader.error_occurred.connect(lambda err, rid=request_id: self._on_loader_error(rid, err))
        loader.finished.connect(lambda rid=request_id: self._cleanup_loader(rid))

        loader.load_image(file_path, stack_count)
        self._latest_display_request = request_id
        self._append_status_message(f"Loading {os.path.basename(file_path)} (Stack={stack_count}) ...")

    def _on_loader_progress(self, request_id: int, progress: int, message: str) -> None:
        if request_id != self._latest_display_request:
            return

        context = self._pending_contexts.get(request_id, {})
        stack_files = context.get("stack_files") or []
        if stack_files and "Processing file" in message:
            # Example message: "Processing file 2/5: foo.cbf"
            parts = message.split(":", 1)
            fname = parts[1].strip() if len(parts) == 2 else ""
            last_file = context.get("_last_progress_file")
            if fname and fname != last_file:
                self._append_status_message(f"Loading {fname} ...")
                context["_last_progress_file"] = fname
                self._pending_contexts[request_id] = context
        self.status_updated.emit(f"Image loading {progress}%: {message}")
        self.progress_updated.emit(progress)

    def _on_loader_error(self, request_id: int, error: str) -> None:
        if request_id == self._latest_display_request:
            QMessageBox.critical(self.main_window, "Image Load Failed", error)
            self.status_updated.emit(error)
        self._cleanup_loader(request_id)

    def _cleanup_loader(self, request_id: int) -> None:
        loader = self._active_loaders.pop(request_id, None)
        if loader:
            loader.deleteLater()
        self._pending_contexts.pop(request_id, None)

    def _on_image_loaded(self, request_id: int, image_data: np.ndarray, file_path: str) -> None:
        context = self._pending_contexts.get(request_id)
        if context is None:
            return
        if request_id != self._latest_display_request:
            return

        self._current_image = image_data.astype(np.float32, copy=False)

        stack_files = context.get("stack_files") or []
        if context.get("stack", 1) and context.get("stack", 1) > 1 and stack_files:
            first = stack_files[0]
            last = stack_files[-1]
            self._append_status_message(f"Image loaded: {first} - {last}")
        else:
            self._append_status_message(f"Image loaded: {os.path.basename(file_path)}")

        if context.get("mode") == "multi_files" and context.get("index") is not None:
            self.current_parameters["showing_value"] = str(context["index"])
            self._set_line_edit("gisaxsImageShowingValue", str(context["index"]))

        self._update_image_display()

    def _maybe_log_scale(self, image: np.ndarray, enabled: bool) -> np.ndarray:
        if not enabled:
            return image
        img = np.array(image, dtype=np.float32, copy=False)
        finite = np.isfinite(img)
        if not finite.any():
            return img
        positives = img[finite & (img > 0)]
        floor = float(np.min(positives)) if positives.size else 1e-6
        floor = max(floor, 1e-6)
        return np.log10(np.maximum(img, floor))

    def _on_gisaxs_log_scale_toggled(self, checked: bool) -> None:
        if self._ui_updating:
            return
        self.current_parameters["gisaxs_log_scale"] = bool(checked)
        self._persist_parameters()
        self._update_image_display()

    def _export_gisaxs_image(self) -> None:
        if not self.prediction_results:
            QMessageBox.information(self.main_window, "Export", "Run a prediction before exporting the current result.")
            self._append_status_message("No prediction result to export", level="WARN")
            return
        if self._current_pixmap is None:
            self._append_status_message("No GISAXS image to export", level="WARN")
            return
        export_path = self._prompt_export_folder("Save GISAXS Image To")
        if not export_path:
            return
        if not os.path.isdir(export_path):
            QMessageBox.warning(self.main_window, "Export Path", f"Export folder not found: {export_path}")
            return
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(export_path, f"gisaxs_{timestamp}.jpg")
        try:
            if not self._current_pixmap.save(file_path, "JPG"):
                raise IOError("Save returned False")
            self._append_status_message(f"GISAXS image exported: {file_path}")
        except Exception as exc:
            self._append_status_message(f"Export failed: {exc}", level="ERROR")

    def _update_image_display(self) -> None:
        if self._current_image is None or self._graphics_scene is None:
            return

        display_img = self._maybe_log_scale(self._current_image, bool(self.current_parameters.get("gisaxs_log_scale", False)))

        auto_scale = bool(self.current_parameters.get("auto_scale", True))
        vmin = self.current_parameters.get("vmin")
        vmax = self.current_parameters.get("vmax")

        if auto_scale or vmin is None or vmax is None:
            vmin, vmax = self._auto_scale_percentiles(display_img, 0.5, 99.5)
            self.current_parameters["vmin"] = vmin
            self.current_parameters["vmax"] = vmax
            self._set_double_spin("gisaxsImageVminValue", vmin)
            self._set_double_spin("gisaxsImageVmaxValue", vmax)
        self._set_checkbox("gisaxsImageAutoScaleCheckBox", auto_scale)

        pixmap = self._create_pixmap_from_array(
            display_img,
            vmin,
            vmax,
            self.current_parameters.get("colormap", self._DEFAULT_COLORMAPS[0]),
        )
        if pixmap is None:
            return

        self._graphics_scene.clear()
        self._graphics_scene.addPixmap(pixmap)
        self._graphics_scene.setSceneRect(QRectF(pixmap.rect()))
        self._current_pixmap = pixmap
        self._zoom_reset()

        cmap_name = self.current_parameters.get("colormap", "")
        self.status_updated.emit(f"Display complete (vmin={vmin:.3f}, vmax={vmax:.3f}, cmap={cmap_name})")
        self._persist_parameters()

    def _auto_scale_values(self, image: np.ndarray) -> Tuple[float, float]:
        finite = np.isfinite(image)
        if not np.any(finite):
            return 0.0, 1.0
        data = image[finite]
        vmin = float(np.min(data))
        vmax = float(np.max(data))
        if vmin == vmax:
            vmax = vmin + 1.0
        return vmin, vmax

    def _auto_scale_percentiles(self, image: np.ndarray, low: float, high: float) -> Tuple[float, float]:
        finite = np.isfinite(image)
        if not np.any(finite):
            return 0.0, 1.0
        data = image[finite]
        vmin = float(np.percentile(data, low))
        vmax = float(np.percentile(data, high))
        if vmin == vmax:
            vmax = vmin + 1.0
        return vmin, vmax

    def _create_pixmap_from_array(self, image: np.ndarray, vmin: float, vmax: float, cmap_name: str) -> Optional[QPixmap]:
        data = np.clip(image, vmin, vmax)
        norm = (data - vmin) / max(vmax - vmin, 1e-9)
        norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)

        mpl_cm = self._get_mpl_cm()
        if mpl_cm is None:
            gray = (norm * 255).astype(np.uint8)
            rgba = np.dstack([gray, gray, gray, np.full_like(gray, 255)])
        else:
            cmap = mpl_cm.get_cmap(cmap_name or self._DEFAULT_COLORMAPS[0])
            rgba = (cmap(norm) * 255).astype(np.uint8)

        height, width = rgba.shape[:2]
        bytes_per_line = rgba.strides[0]
        image_q = QImage(rgba.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
        return QPixmap.fromImage(image_q.copy())

    def _get_mpl_cm(self):
        if not is_matplotlib_available():
            return None
        if self.__class__._mpl_cm is None:
            try:
                from matplotlib import cm as mpl_cm  # type: ignore
                self.__class__._mpl_cm = mpl_cm
            except Exception:
                self.__class__._mpl_cm = False
        return None if self.__class__._mpl_cm is False else self.__class__._mpl_cm

    # ------------------------------------------------------------------
    # Preprocessing & Prediction
    # ------------------------------------------------------------------
    def _preprocess_for_module(self, image: np.ndarray) -> Optional[np.ndarray]:
        # Ensure a module is selected; fall back to saved name or first available
        if not self._current_module:
            try:
                name = self.current_parameters.get("module_name", "") if isinstance(self.current_parameters, dict) else ""
                if not name and self._modules_by_name:
                    name = sorted(self._modules_by_name.keys())[0]
                if name and name in self._modules_by_name:
                    self._current_module = self._modules_by_name.get(name)
            except Exception:
                pass
        if image is None:
            return None
        img = image.astype(np.float32, copy=True)
        try:
            self._append_status_message(f"Preprocess input shape {img.shape}")
        except Exception:
            pass

        # If module provides a preprocess entry, call it with ordered steps/params
        try:
            spec = self._current_module or {}
            entry = spec.get("preprocess_entry") if isinstance(spec, dict) else ""
            pre_cfg = spec.get("preprocess_raw") if isinstance(spec, dict) else None
            # Ensure preprocess config is populated (params/steps) so module entry honors YAML
            try:
                needs_refresh = not isinstance(pre_cfg, dict)
                if not needs_refresh:
                    p_params = pre_cfg.get("params") if isinstance(pre_cfg.get("params"), dict) else None
                    p_steps = pre_cfg.get("steps") if isinstance(pre_cfg.get("steps"), list) else None
                    needs_refresh = (p_params is None) or (p_steps is None)
                if needs_refresh and isinstance(spec, dict):
                    yaml_path = spec.get("yaml_path")
                    if isinstance(yaml_path, str) and os.path.isfile(yaml_path):
                        parsed_spec = self._parse_module_yaml(yaml_path)
                        if isinstance(parsed_spec, dict):
                            refreshed = parsed_spec.get("preprocess_raw") if isinstance(parsed_spec.get("preprocess_raw"), dict) else None
                            if isinstance(refreshed, dict):
                                pre_cfg = refreshed
                                # keep current module in sync for future calls
                                if isinstance(self._current_module, dict):
                                    self._current_module["preprocess_raw"] = refreshed
                                    if isinstance(parsed_spec.get("preprocess_entry"), str):
                                        self._current_module["preprocess_entry"] = parsed_spec.get("preprocess_entry")
                                    if isinstance(parsed_spec.get("preprocess_steps"), list):
                                        self._current_module["preprocess_steps"] = parsed_spec.get("preprocess_steps")
                                    if isinstance(parsed_spec.get("preprocess_params"), dict):
                                        self._current_module["preprocess_params"] = parsed_spec.get("preprocess_params")
            except Exception:
                pass
            if isinstance(entry, str) and entry:
                # entry format: "module:function", module is a .py in the module folder
                module_name, _, func_name = entry.partition(":")
                folder = spec.get("folder") if isinstance(spec, dict) else None
                if module_name and func_name and isinstance(folder, str) and os.path.isdir(folder):
                    py_path = os.path.join(folder, f"{module_name}.py")
                    if os.path.isfile(py_path):
                        import importlib.util
                        spec_imp = importlib.util.spec_from_file_location(f"mod_{module_name}", py_path)
                        if spec_imp and spec_imp.loader:
                            mod = importlib.util.module_from_spec(spec_imp)
                            spec_imp.loader.exec_module(mod)
                            fn = getattr(mod, func_name, None)
                            if callable(fn):
                                try:
                                    # Ask module to return step snapshots so we can verify order
                                    out = None
                                    try:
                                        out = fn(img, pre_cfg, module_folder=folder, return_steps=True)
                                    except TypeError:
                                        # Fallback to signature without return_steps
                                        out = fn(img, pre_cfg, module_folder=folder)
                                    arr = None
                                    steps_payload = None
                                    if isinstance(out, tuple):
                                        arr = out[0]
                                        steps_payload = out[1] if len(out) > 1 else None
                                    elif isinstance(out, dict):
                                        arr = out.get("image") or out.get("result")
                                        steps_payload = out.get("steps")
                                    elif isinstance(out, np.ndarray):
                                        arr = out
                                    else:
                                        try:
                                            self._append_status_message(f"Module preprocess returned {type(out)}")
                                        except Exception:
                                            pass
                                    # Log step order & shapes for confirmation
                                    if isinstance(steps_payload, list):
                                        for entry_step in steps_payload:
                                            try:
                                                lbl = entry_step.get("label") or entry_step.get("step") or "Step"
                                                snap = entry_step.get("image")
                                                if isinstance(snap, np.ndarray):
                                                    self._append_status_message(f"Preprocess step: {lbl} – shape {snap.shape[0]}x{snap.shape[1]}")
                                                else:
                                                    self._append_status_message(f"Preprocess step: {lbl} (no snapshot)")
                                            except Exception:
                                                pass
                                    else:
                                        try:
                                            self._append_status_message("Module preprocess returned no steps payload")
                                        except Exception:
                                            pass
                                    if isinstance(arr, np.ndarray):
                                        try:
                                            self._append_status_message(
                                                f"Module preprocess output shape {arr.shape}")
                                        except Exception:
                                            pass
                                        return self._prepare_model_input(arr)
                                except Exception as exc:
                                    self._append_status_message(f"Module preprocess failed: {exc}", level="ERROR")
                # If anything fails, fall back to built-in pipeline below
        except Exception:
            pass

        # Crop
        spec = self._current_module or {}
        crop_cfg = None
        try:
            crop_cfg = spec.get("preprocess_crop") or spec.get("crop")
        except Exception:
            crop_cfg = None
        # If module preprocess cfg exists, try to pull crop from it as fallback
        try:
            if crop_cfg is None and isinstance(spec.get("preprocess_raw"), dict):
                pcfg = spec.get("preprocess_raw")
                if isinstance(pcfg.get("params"), dict) and isinstance(pcfg["params"].get("crop"), dict):
                    crop_cfg = pcfg["params"]["crop"]
        except Exception:
            pass
        try:
            self._append_status_message(f"Fallback path crop cfg: {crop_cfg}")
        except Exception:
            pass
        if isinstance(crop_cfg, dict):
            if all(k in crop_cfg for k in ("y0","y1","x0","x1")):
                y0 = int(crop_cfg.get("y0", 0))
                y1 = int(crop_cfg.get("y1", img.shape[0]))
                x0 = int(crop_cfg.get("x0", 0))
                x1 = int(crop_cfg.get("x1", img.shape[1]))
            else:
                # left/up/down/right style
                left = int(crop_cfg.get("left", 0))
                up = int(crop_cfg.get("up", 0))
                down = int(crop_cfg.get("down", 0))
                right = int(crop_cfg.get("right", 0))
                y0 = up
                y1 = max(up, img.shape[0] - down)
                x0 = left
                x1 = max(left, img.shape[1] - right)
            y0 = max(0, min(y0, img.shape[0]))
            y1 = max(y0, min(y1, img.shape[0]))
            x0 = max(0, min(x0, img.shape[1]))
            x1 = max(x0, min(x1, img.shape[1]))
            img = img[y0:y1, x0:x1]
            try:
                self._append_status_message(
                    f"Fallback crop applied y:{y0}-{y1} x:{x0}-{x1} -> {img.shape}")
            except Exception:
                pass

        # Log and normalize
        try:
            from utils.tools.Preprocessing import Preprocessing  # type: ignore
            img = Preprocessing(img).log_and_normalize()
        except Exception as exc:
            self._append_status_message(f"Preprocessing error: log_and_normalize not available or failed: {exc}", level="ERROR")
            return None

        # Resize
        resize_cfg = None
        try:
            resize_cfg = spec.get("preprocess_resize") or spec.get("resize")
        except Exception:
            resize_cfg = None
        if isinstance(resize_cfg, dict):
            target_h = int(resize_cfg.get("height", img.shape[0]))
            target_w = int(resize_cfg.get("width", img.shape[1]))
            try:
                import tensorflow as tf  # type: ignore
                t = tf.convert_to_tensor(img[None, ..., None], dtype=tf.float32)
                r = tf.image.resize(t, [target_h, target_w], method='bilinear')
                img = r.numpy()[0, ..., 0]
            except Exception:
                # Simple nearest-neighbor numpy resize
                ys = np.linspace(0, img.shape[0] - 1, target_h).astype(np.int32)
                xs = np.linspace(0, img.shape[1] - 1, target_w).astype(np.int32)
                img = img[np.ix_(ys, xs)]
            try:
                self._append_status_message(f"Fallback resize -> {img.shape}")
            except Exception:
                pass
        elif isinstance(resize_cfg, (list, tuple)) and len(resize_cfg) == 2:
            target_h = int(resize_cfg[0])
            target_w = int(resize_cfg[1])
            try:
                import tensorflow as tf  # type: ignore
                t = tf.convert_to_tensor(img[None, ..., None], dtype=tf.float32)
                r = tf.image.resize(t, [target_h, target_w], method='bilinear')
                img = r.numpy()[0, ..., 0]
            except Exception:
                ys = np.linspace(0, img.shape[0] - 1, target_h).astype(np.int32)
                xs = np.linspace(0, img.shape[1] - 1, target_w).astype(np.int32)
                img = img[np.ix_(ys, xs)]
            try:
                self._append_status_message(f"Fallback resize -> {img.shape}")
            except Exception:
                pass

        # Load/apply mask after normalization, set to mask_value instead of 0
        if self._current_mask is not None:
            try:
                mask = np.array(self._current_mask)
                # Crop mask if specified
                m_crop = spec.get("mask_crop") if isinstance(spec, dict) else None
                if isinstance(m_crop, dict):
                    left = int(m_crop.get("left", 0))
                    up = int(m_crop.get("up", 0))
                    down = int(m_crop.get("down", 0))
                    right = int(m_crop.get("right", 0))
                    y0 = up
                    y1 = max(up, mask.shape[0] - down)
                    x0 = left
                    x1 = max(left, mask.shape[1] - right)
                    y0 = max(0, min(y0, mask.shape[0]))
                    y1 = max(y0, min(y1, mask.shape[0]))
                    x0 = max(0, min(x0, mask.shape[1]))
                    x1 = max(x0, min(x1, mask.shape[1]))
                    mask = mask[y0:y1, x0:x1]

                # Resize mask to match img size if requested or needed
                target_h, target_w = img.shape
                m_resize = spec.get("mask_resize") if isinstance(spec, dict) else None
                if isinstance(m_resize, dict):
                    target_h = int(m_resize.get("height", target_h))
                    target_w = int(m_resize.get("width", target_w))
                if mask.shape != (target_h, target_w):
                    try:
                        import tensorflow as tf  # type: ignore
                        mt = tf.convert_to_tensor(mask[None, ..., None], dtype=tf.float32)
                        mr = tf.image.resize(mt, [target_h, target_w], method='nearest')
                        mask = mr.numpy()[0, ..., 0]
                    except Exception:
                        ys = np.linspace(0, mask.shape[0] - 1, target_h).astype(np.int32)
                        xs = np.linspace(0, mask.shape[1] - 1, target_w).astype(np.int32)
                        mask = mask[np.ix_(ys, xs)]

                # Apply mask value
                mask_value = spec.get("mask_value") if isinstance(spec, dict) else None
                mv = float(mask_value) if isinstance(mask_value, (int, float)) else -1.0
                bad = mask != 0
                if bad.shape == img.shape:
                    img[bad] = mv
            except Exception as exc:
                self._append_status_message(f"Mask application failed: {exc}", level="ERROR")
                return None

        return self._prepare_model_input(img)

    def _prepare_model_input(self, image: np.ndarray) -> Optional[np.ndarray]:
        inp = self._normalize_input_rank(image)
        io_shape = None
        try:
            io_shape = (self._current_module or {}).get("io_input_shape")
        except Exception:
            io_shape = None
        if isinstance(io_shape, tuple):
            inp = self._coerce_array_to_shape(inp, io_shape)
            if len(io_shape) == inp.ndim and tuple(inp.shape) != tuple(io_shape):
                self._append_status_message(f"Preprocessing output shape {inp.shape} does not match io.input_shape {io_shape}", level="ERROR")
                return None
        return inp

    def _normalize_parameter_prediction(self, pred: object) -> Optional[Dict[str, object]]:
        spec = self._current_module if isinstance(self._current_module, dict) else {}
        output_type = str(spec.get("output_type") or "").lower()
        is_sf = output_type in {"sf_4_parameters", "sf_parameters", "parameters"}
        if not is_sf and isinstance(pred, dict):
            is_sf = "branch_thickness" in pred and "branch_size" in pred
        if not is_sf:
            return None

        normalized = None
        if isinstance(pred, dict):
            if "branch_thickness" in pred and "branch_size" in pred:
                thickness = np.asarray(pred["branch_thickness"], dtype=np.float32)
                size = np.asarray(pred["branch_size"], dtype=np.float32)
                normalized = np.concatenate([thickness, size], axis=-1)
            elif "parameters" in pred:
                normalized = np.asarray(pred["parameters"], dtype=np.float32)
            elif pred:
                values = [np.asarray(value, dtype=np.float32) for value in pred.values()]
                if values:
                    normalized = np.concatenate(values, axis=-1)
        else:
            arr = np.asarray(pred, dtype=np.float32)
            if arr.size:
                normalized = arr

        if normalized is None:
            return None
        arr = np.asarray(normalized, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.reshape((-1, arr.shape[-1]))[0]
        arr = arr.reshape(-1)

        names = spec.get("parameter_names") if isinstance(spec.get("parameter_names"), list) else []
        names = [str(name) for name in names] if names else ["t_Cu", "t_polymer", "D", "sigma"]
        target_min = np.asarray(spec.get("target_min") or [0.0, 10.0, 4.0, 0.2], dtype=np.float32)
        target_max = np.asarray(spec.get("target_max") or [25.0, 50.0, 20.0, 4.0], dtype=np.float32)

        count = min(arr.size, len(names), target_min.size, target_max.size)
        if count <= 0:
            return None
        arr = arr[:count]
        values = arr * (target_max[:count] - target_min[:count]) + target_min[:count]
        return {
            "parameters": values.astype(np.float32),
            "parameters_normalized": arr.astype(np.float32),
            "parameter_names": names[:count],
        }

    def _predict_with_current_model(self, inp: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        if self._current_model is None or inp is None:
            return None
        out_image: Optional[np.ndarray] = None
        scalar_out: Optional[np.ndarray] = None
        # Coerce input to model's expected input shape if available
        exp_shape = None
        try:
            exp_shape = self._model_input_shape(self._current_model)
            if isinstance(exp_shape, tuple):
                inp = self._coerce_array_to_shape(inp, exp_shape)
            # Log effective shapes
            try:
                self._append_status_message(f"Model expected input_shape={exp_shape}, sending {tuple(inp.shape)}")
            except Exception:
                pass
        except Exception:
            # If anything goes wrong, proceed with original inp
            pass
        predict_error = None
        try:
            # Keras/tf.keras model path. Do not require top-level ``keras`` here:
            # TensorFlow 2.13 installations often expose only ``tensorflow.keras``.
            if hasattr(self._current_model, 'predict'):
                pred = self._current_model.predict(inp, verbose=0)
                parameter_out = self._normalize_parameter_prediction(pred)
                if parameter_out is not None:
                    return parameter_out  # type: ignore[return-value]
                if isinstance(pred, (list, tuple)) and len(pred) > 0:
                    # Prefer 2D-like output if present, else capture scalar
                    cand0 = np.array(pred[0])
                    if cand0.squeeze().ndim >= 2:
                        out_image = cand0
                    else:
                        scalar_out = cand0.squeeze()
                elif isinstance(pred, dict):
                    # Try common keys
                    val = pred.get('hr', None)
                    if val is None:
                        val = pred.get('output', None)
                    if val is None and len(pred) > 0:
                        val = list(pred.values())[0]
                    if val is not None:
                        val_arr = np.array(val)
                        if val_arr.squeeze().ndim >= 2:
                            out_image = val_arr
                        else:
                            scalar_out = val_arr.squeeze()
                else:
                    arr = np.array(pred)
                    if arr.squeeze().ndim >= 2:
                        out_image = arr
                    else:
                        scalar_out = arr.squeeze()
            else:
                raise Exception("Model has no predict method")
        except Exception as exc:
            predict_error = exc
            # Try TF SavedModel signature
            try:
                import tensorflow as tf  # type: ignore
                if hasattr(self._current_model, 'signatures'):
                    fn = self._current_model.signatures.get('serving_default') or next(iter(self._current_model.signatures.values()))
                    # Build input dict from structured input signature
                    sig = fn.structured_input_signature
                    # sig is (args, kwargs); use kwargs keys
                    input_kwargs = sig[1] if isinstance(sig, tuple) and len(sig) > 1 and isinstance(sig[1], dict) else {}
                    input_keys = list(input_kwargs.keys())
                    input_spec = input_kwargs.get(input_keys[0]) if input_keys else None
                    x = self._coerce_array_to_tensor_spec(inp, input_spec)
                    t = tf.convert_to_tensor(x, dtype=getattr(input_spec, "dtype", tf.float32) if input_spec is not None else tf.float32)
                    if input_keys:
                        res = fn(**{input_keys[0]: t})
                    else:
                        res = fn(t)
                    # res is a dict of tensors
                    vals = list(res.values())
                    if vals:
                        pred_dict = {str(key): value.numpy() for key, value in res.items()}
                        parameter_out = self._normalize_parameter_prediction(pred_dict)
                        if parameter_out is not None:
                            return parameter_out  # type: ignore[return-value]
                        tmp = vals[0].numpy()
                        if np.squeeze(tmp).ndim >= 2:
                            out_image = tmp
                        else:
                            scalar_out = np.squeeze(tmp)
            except Exception as exc:
                if predict_error is not None:
                    self._append_status_message(f"Keras predict failed: {predict_error}", level="ERROR")
                self._append_status_message(f"SavedModel prediction failed: {exc}", level="ERROR")
                return None

        # If scalar output (e.g., two numbers), return as 'scalars'
        if out_image is None and scalar_out is not None:
            scal = np.array(scalar_out).reshape(-1)
            return {"scalars": scal}
        if out_image is None:
            self._append_status_message("Prediction produced no output", level="WARN")
            return None

        # Ensure 2D image from batch
        img2d = out_image.squeeze()
        if img2d.ndim == 3:
            img2d = img2d[..., 0]
        if img2d.ndim != 2:
            self._append_status_message(f"Unexpected prediction shape: {out_image.shape}", level="WARN")
            return None

        # Derive h and r distributions
        h_sum = np.sum(img2d, axis=0)
        r_sum = np.sum(img2d, axis=1)
        return {"hr": img2d, "h": h_sum, "r": r_sum}

    def _coerce_array_to_tensor_spec(self, arr: np.ndarray, tensor_spec: object) -> np.ndarray:
        """Best-effort NHWC reshape/resize for TensorFlow SavedModel signatures."""
        shape = getattr(tensor_spec, "shape", None)
        dims = None
        try:
            dims = shape.as_list() if shape is not None and hasattr(shape, "as_list") else list(shape)
        except Exception:
            dims = None
        return self._coerce_array_to_shape(arr, tuple(dims)) if isinstance(dims, list) else np.asarray(arr, dtype=np.float32)

    def _model_input_shape(self, model: object) -> Optional[Tuple[object, ...]]:
        exp_shape = getattr(model, 'input_shape', None)
        if isinstance(exp_shape, list) and exp_shape:
            exp_shape = exp_shape[0]
        if not isinstance(exp_shape, tuple):
            try:
                inputs = getattr(model, "inputs", None)
                if inputs:
                    shape = getattr(inputs[0], "shape", None)
                    exp_shape = tuple(shape.as_list()) if hasattr(shape, "as_list") else tuple(shape)
            except Exception:
                exp_shape = None
        return tuple(exp_shape) if isinstance(exp_shape, (list, tuple)) else None

    def _normalize_input_rank(self, arr: np.ndarray) -> np.ndarray:
        x = np.asarray(arr, dtype=np.float32)
        while x.ndim > 4 and 1 in x.shape:
            x = np.squeeze(x, axis=x.shape.index(1))
        if x.ndim == 2:
            return x[None, ..., None].astype(np.float32, copy=False)
        if x.ndim == 3:
            if x.shape[0] == 1:
                return x[..., None].astype(np.float32, copy=False)
            if x.shape[-1] in (1, 2, 3, 4):
                return x[None, ...].astype(np.float32, copy=False)
            return x[..., None].astype(np.float32, copy=False)
        return x.astype(np.float32, copy=False)

    def _coerce_array_to_shape(self, arr: np.ndarray, shape: Tuple[object, ...]) -> np.ndarray:
        x = self._normalize_input_rank(arr)
        if len(shape) == 4:
            _, h, w, c = shape
            target_h = int(h) if isinstance(h, (int, np.integer)) else (x.shape[1] if x.ndim >= 3 else x.shape[0])
            target_w = int(w) if isinstance(w, (int, np.integer)) else (x.shape[2] if x.ndim >= 4 else x.shape[1])
            target_c = int(c) if isinstance(c, (int, np.integer)) else (x.shape[-1] if x.ndim == 4 else 1)
            if x.ndim != 4:
                x = self._normalize_input_rank(x)
            if x.ndim == 4:
                x = self._resize_nhwc(x, target_h, target_w)
                if x.shape[-1] != target_c:
                    if target_c == 1:
                        x = x[..., :1]
                    elif x.shape[-1] == 1:
                        x = np.repeat(x, target_c, axis=-1)
                    else:
                        x = x[..., :target_c]
            return x.astype(np.float32, copy=False)

        if len(shape) == 3:
            _, h, w = shape
            target_h = int(h) if isinstance(h, (int, np.integer)) else (x.shape[1] if x.ndim >= 3 else x.shape[0])
            target_w = int(w) if isinstance(w, (int, np.integer)) else (x.shape[2] if x.ndim >= 3 else x.shape[1])
            if x.ndim == 4 and x.shape[-1] == 1:
                x = x[..., 0]
            elif x.ndim == 2:
                x = x[None, ...]
            if x.ndim == 3:
                x4 = self._resize_nhwc(x[..., None], target_h, target_w)
                return x4[..., 0].astype(np.float32, copy=False)
        return x.astype(np.float32, copy=False)

    def _resize_nhwc(self, x: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        if x.ndim != 4 or (x.shape[1], x.shape[2]) == (target_h, target_w):
            return x
        try:
            import tensorflow as tf  # type: ignore
            t = tf.convert_to_tensor(x, dtype=tf.float32)
            return tf.image.resize(t, [target_h, target_w], method='bilinear').numpy()
        except Exception:
            ys = np.linspace(0, x.shape[1] - 1, target_h).astype(np.int32)
            xs = np.linspace(0, x.shape[2] - 1, target_w).astype(np.int32)
            return x[:, ys][:, :, xs]

    def _get_or_create_predict2d_tabs(self) -> Optional[QTabWidget]:
        # Embed inner tabs inside the existing Predict-2D tab of the main tab widget
        main_tabs = getattr(self.ui, "gisaxsPredictImageShowTabWidget", None)
        if main_tabs is None:
            return None
        pred_index = -1
        try:
            for i in range(main_tabs.count()):
                try:
                    label = main_tabs.tabText(i)
                    if isinstance(label, str) and label.lower().strip() in ("predict-2d", "predict 2d", "predict"):
                        pred_index = i
                        break
                except Exception:
                    pass
        except Exception:
            pass
        if pred_index < 0:
            # fallback to current tab
            try:
                pred_index = main_tabs.currentIndex()
            except Exception:
                pred_index = 0
        pred_page = main_tabs.widget(pred_index)
        if pred_page is None:
            return None
        layout = pred_page.layout()
        if layout is None:
            layout = QVBoxLayout(pred_page)
        # Reuse existing inner tabs if present
        try:
            inner_tabs = next(iter(pred_page.findChildren(QTabWidget)), None)
        except Exception:
            inner_tabs = None
        if inner_tabs is None:
            inner_tabs = QTabWidget(pred_page)
            # 允许横向扩展，不限制最大宽度，避免挤压父容器
            try:
                inner_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            except Exception:
                pass
            layout.addWidget(inner_tabs)
        self._predict_tabs = inner_tabs
        return inner_tabs

    def _rebuild_predict_tabs(self, tabs: QTabWidget) -> None:
        blocker = QSignalBlocker(tabs)
        try:
            while tabs.count() > 0:
                w = tabs.widget(0)
                tabs.removeTab(0)
                if w:
                    w.deleteLater()
            for spec in self._predict_tab_specs:
                page = QWidget()
                # 不要将页面最大高度设为0，保持可扩展的尺寸策略
                try:
                    page.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
                except Exception:
                    pass
                tabs.addTab(page, str(spec.get("title", "Panel")))
        finally:
            del blocker
        try:
            tabs.currentChanged.disconnect(self._on_predict_tab_changed)
        except Exception:
            pass
        tabs.currentChanged.connect(self._on_predict_tab_changed)
        if self._predict_tab_specs:
            tabs.setCurrentIndex(0)

    def _on_predict_tab_changed(self, index: int) -> None:
        self._render_predict_tab_by_index(index)

    def _render_predict_tab_by_index(self, index: int) -> None:
        if index < 0 or index >= len(self._predict_tab_specs):
            return
        spec = self._predict_tab_specs[index]
        self._render_predict_panel(spec)

    def _render_predict_panel(self, spec: Dict[str, object]) -> None:
        # Clear any step buttons when switching kinds
        if getattr(self, "_step_buttons", None):
            try:
                for b in self._step_buttons:
                    if b and hasattr(b, "deleteLater"):
                        b.deleteLater()
            except Exception:
                pass
        self._step_buttons = []

        kind = spec.get("kind") if isinstance(spec, dict) else None
        data = spec.get("data") if isinstance(spec, dict) else None
        self._predict_current_kind = kind if isinstance(kind, str) else None
        self._predict_current_curve = None
        if kind == "hr" and isinstance(data, np.ndarray):
            self._render_predict2d_into_view(data)
            self._refresh_predict_controls("hr")
            return
        if kind == "array" and isinstance(data, np.ndarray):
            self._predict_current_image = data
            disp, vmin, vmax = self._prepare_predict_image(data)
            cmap = spec.get("colormap") if isinstance(spec.get("colormap"), str) else self.current_parameters.get("colormap", self._DEFAULT_COLORMAPS[0])
            pix = self._create_pixmap_from_array(disp, vmin, vmax, cmap)
            self._show_pixmap_in_predict_view(pix)
            self._refresh_predict_controls("array")
            return
        if kind == "curve" and isinstance(data, np.ndarray):
            title = spec.get("title", "Curve")
            xlabel = spec.get("xlabel", "Index")
            self._predict_current_curve = data
            pix = self._render_curve_figure(
                data,
                x_label=str(xlabel),
                title=str(title),
                log_x=bool(self.current_parameters.get("predict_curve_logx", False)),
                log_y=bool(self.current_parameters.get("predict_curve_logy", False)),
                xlim=self._get_curve_xlim(),
                ylim=self._get_curve_ylim(),
            )
            self._show_pixmap_in_predict_view(pix)
            self._refresh_predict_controls("curve")
            return
        if kind == "parameters" and isinstance(data, np.ndarray):
            names = spec.get("names") if isinstance(spec.get("names"), list) else None
            pix = self._render_parameters_figure(data, [str(name) for name in names] if names else None)
            if pix is not None:
                self._show_pixmap_in_predict_view(pix)
            self._refresh_predict_controls("array")
            return
        if kind == "steps":
            steps = spec.get("steps") if isinstance(spec.get("steps"), list) else []
            if not steps:
                return
            self._step_snapshots = steps
            # Show the final model input by default when the preprocess panel provides it.
            default_idx = spec.get("default_index") if isinstance(spec, dict) else None
            if isinstance(default_idx, int) and 0 <= default_idx < len(steps):
                start_idx = default_idx
            else:
                start_idx = self._current_step_index if 0 <= self._current_step_index < len(steps) else 0
            self._render_step_snapshot(start_idx)
            self._refresh_predict_controls("steps")
            # Build buttons under the tabs page to switch steps
            tabs = getattr(self, "_predict_tabs", None)
            page = tabs.currentWidget() if tabs else None
            if page is None:
                return
            layout = page.layout()
            if layout is None:
                layout = QVBoxLayout(page)
            # Clear existing items in page layout
            while layout.count():
                item = layout.takeAt(0)
                w = item.widget()
                if w:
                    w.deleteLater()
            # Estimate columns based on viewport width to avoid stretching right side
            cols = 4
            try:
                pview = getattr(self.ui, "predict2dGraphicsView", None)
                if pview is not None:
                    vw = max(1, pview.viewport().size().width())
                    cols = max(1, vw // 120)
            except Exception:
                pass
            grid = QGridLayout()
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setSpacing(6)
            btns = []
            for idx, st in enumerate(steps):
                lbl = st.get("label") or st.get("step") or f"Step {idx+1}"
                btn = QPushButton(str(lbl))
                btn.setCheckable(True)
                btn.setChecked(idx == start_idx)
                btn.clicked.connect(lambda checked, i=idx: self._render_step_snapshot(i))
                r, c = divmod(idx, cols)
                grid.addWidget(btn, r, c)
                btns.append(btn)
            layout.addLayout(grid)
            try:
                row_count = (len(btns) + cols - 1) // cols
                row_h = btns[0].sizeHint().height() if btns else 24
                # 仅设置最小高度，允许父布局根据可用空间扩展
                page.setMinimumHeight(row_count * (row_h + 6) + 4)
                try:
                    page.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
                except Exception:
                    pass
            except Exception:
                pass
            self._step_buttons = btns
            return

    def _predict_viewport_pixels(self) -> Optional[Tuple[int, int]]:
        pview = getattr(self.ui, "predict2dGraphicsView", None)
        if pview is None:
            return None
        viewport = pview.viewport().size()
        return (max(400, viewport.width()), max(320, viewport.height()))

    def _render_step_snapshot(self, idx: int) -> None:
        steps = getattr(self, "_step_snapshots", None)
        if not isinstance(steps, list) or idx < 0 or idx >= len(steps):
            return
        snap = steps[idx].get("image") if isinstance(steps[idx], dict) else None
        if not isinstance(snap, np.ndarray):
            return
        self._current_step_index = idx
        self._predict_current_image = snap
        # update buttons state
        for i, b in enumerate(getattr(self, "_step_buttons", []) or []):
            try:
                b.setChecked(i == idx)
            except Exception:
                pass
        display, vmin, vmax = self._prepare_predict_image(snap)
        cmap = self.current_parameters.get("colormap", self._DEFAULT_COLORMAPS[0])
        pix = self._create_pixmap_from_array(display, vmin, vmax, cmap)
        self._show_pixmap_in_predict_view(pix)

    def _render_parameters_figure(self, values: np.ndarray, names: Optional[List[str]] = None) -> Optional[QPixmap]:
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            vals = np.asarray(values, dtype=np.float32).reshape(-1)
            labels = names if names and len(names) >= vals.size else [f"p{i + 1}" for i in range(vals.size)]
            fig = Figure(figsize=(7.2, 3.8), dpi=120)
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
            x = np.arange(vals.size)
            bars = ax.bar(x, vals, color=["#2563eb", "#16a34a", "#f59e0b", "#dc2626"][: vals.size])
            ax.set_xticks(x)
            ax.set_xticklabels(labels[: vals.size])
            ax.set_ylabel("Predicted value")
            ax.set_title("SF Predicted Parameters")
            ax.grid(axis="y", alpha=0.25)
            for bar, value in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height(),
                    f"{float(value):.5g}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            fig.tight_layout()
            canvas.draw()
            buf = np.asarray(canvas.buffer_rgba())
            height, width = buf.shape[:2]
            qimg = QImage(buf.data, width, height, buf.strides[0], QImage.Format_RGBA8888)
            return QPixmap.fromImage(qimg.copy())
        except Exception as exc:
            self._append_status_message(f"Parameter plot failed: {exc}", level="ERROR")
            return None

    def _refresh_predict_controls(self, kind: str) -> None:
        param_widget = getattr(self.ui, "predict2dParameterWidget", None)
        if param_widget is None:
            return
        two_d_widgets = [
            "predict2dColorScaleLabel",
            "predict2dAutoScaleCheckBox",
            "predict2dAutoScaleResetButton",
            "predict2dVminLabel",
            "predict2dVmaxLabel",
            "predict2dVminValue",
            "predict2dVmaxValue",
            "predict2dColormapLabel",
            "predict2dLabelCombox",
            "predict2dLogScaleCheckBox",
            "predict2dCountourCheckBox",
            "predict2dCountourLevelsLabel",
            "predict2dCountourLevelsValue",
        ]
        is_curve = kind == "curve"
        for name in two_d_widgets:
            w = getattr(self.ui, name, None)
            if w is not None:
                w.setVisible(not is_curve)

        controls = self._ensure_predict_curve_controls()
        if not controls:
            return

        # 显示/隐藏整个1D参数部分
        curve_widget = getattr(self.ui, "predict2dParameter1dpartWidget", None)
        if curve_widget is not None:
            curve_widget.setVisible(is_curve)

        if not is_curve:
            return

        self._ui_updating = True
        try:
            controls["logx"].setChecked(bool(self.current_parameters.get("predict_curve_logx", False)))
            controls["logy"].setChecked(bool(self.current_parameters.get("predict_curve_logy", False)))
            autoscale = bool(self.current_parameters.get("predict_curve_autoscale", True))
            controls["autoscale"].setChecked(autoscale)
            for key in ("xmin", "xmax", "ymin", "ymax"):
                val = self.current_parameters.get(f"predict_curve_{key}")
                box = controls.get(key)
                if isinstance(box, QDoubleSpinBox):
                    if val is None:
                        box.setValue(0.0)
                    else:
                        box.setValue(float(val))
                    box.setEnabled(not autoscale)
        finally:
            self._ui_updating = False

    def _ensure_predict_curve_controls(self) -> Dict[str, object]:
        if self._predict_curve_controls:
            return self._predict_curve_controls
        parent = getattr(self.ui, "predict2dParameter1dpartWidget", None)
        if parent is None:
            return {}

        # 检查是否已经有布局，如果没有则创建一个
        grid = parent.layout()
        if grid is None:
            grid = QGridLayout(parent)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setSpacing(6)

        logx = QCheckBox("Log X")
        logy = QCheckBox("Log Y")
        autoscale = QCheckBox("AutoScale")

        xmin = QDoubleSpinBox()
        xmax = QDoubleSpinBox()
        ymin = QDoubleSpinBox()
        ymax = QDoubleSpinBox()
        for box in (xmin, xmax, ymin, ymax):
            box.setRange(-1e9, 1e9)
            box.setDecimals(6)
            box.setSingleStep(0.1)

        grid.addWidget(logx, 0, 0)
        grid.addWidget(logy, 0, 1)
        grid.addWidget(autoscale, 0, 2)
        grid.addWidget(QLabel("X min"), 1, 0)
        grid.addWidget(xmin, 1, 1)
        grid.addWidget(QLabel("X max"), 1, 2)
        grid.addWidget(xmax, 1, 3)
        grid.addWidget(QLabel("Y min"), 2, 0)
        grid.addWidget(ymin, 2, 1)
        grid.addWidget(QLabel("Y max"), 2, 2)
        grid.addWidget(ymax, 2, 3)

        logx.toggled.connect(self._on_predict_curve_control_changed)
        logy.toggled.connect(self._on_predict_curve_control_changed)
        autoscale.toggled.connect(self._on_predict_curve_control_changed)
        for box in (xmin, xmax, ymin, ymax):
            box.editingFinished.connect(self._on_predict_curve_control_changed)

        self._predict_curve_controls = {
            "logx": logx,
            "logy": logy,
            "autoscale": autoscale,
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
        }
        # Hide initially until a curve is shown
        parent.setVisible(False)
        return self._predict_curve_controls

    def _on_predict_curve_control_changed(self) -> None:
        if self._ui_updating:
            return
        controls = self._ensure_predict_curve_controls()
        if not controls:
            return
        self.current_parameters["predict_curve_logx"] = bool(controls.get("logx").isChecked()) if controls.get("logx") else False
        self.current_parameters["predict_curve_logy"] = bool(controls.get("logy").isChecked()) if controls.get("logy") else False
        autoscale = bool(controls.get("autoscale").isChecked()) if controls.get("autoscale") else True
        self.current_parameters["predict_curve_autoscale"] = autoscale
        for key in ("xmin", "xmax", "ymin", "ymax"):
            box = controls.get(key)
            if isinstance(box, QDoubleSpinBox):
                box.setEnabled(not autoscale)
                if not autoscale:
                    self.current_parameters[f"predict_curve_{key}"] = float(box.value())
                else:
                    self.current_parameters[f"predict_curve_{key}"] = None
        self._persist_parameters()
        if self._predict_current_kind == "curve":
            self._rerender_predict_view()

    def _get_curve_xlim(self) -> Optional[Tuple[float, float]]:
        if self.current_parameters.get("predict_curve_autoscale", True):
            return None
        xmin = self.current_parameters.get("predict_curve_xmin")
        xmax = self.current_parameters.get("predict_curve_xmax")
        if xmin is None or xmax is None:
            return None
        return float(xmin), float(xmax)

    def _get_curve_ylim(self) -> Optional[Tuple[float, float]]:
        if self.current_parameters.get("predict_curve_autoscale", True):
            return None
        ymin = self.current_parameters.get("predict_curve_ymin")
        ymax = self.current_parameters.get("predict_curve_ymax")
        if ymin is None or ymax is None:
            return None
        return float(ymin), float(ymax)

    def _show_pixmap_in_predict_view(self, pix: Optional[QPixmap]) -> None:
        if pix is None:
            return
        pview = getattr(self.ui, "predict2dGraphicsView", None)
        if pview is None:
            return
        if self._predict_scene is None:
            self._predict_scene = QGraphicsScene(pview)
            pview.setScene(self._predict_scene)
            pview.setTransformationAnchor(pview.AnchorUnderMouse)
            pview.setDragMode(pview.ScrollHandDrag)
        self._predict_scene.clear()
        self._predict_scene.addPixmap(pix)
        self._predict_scene.setSceneRect(QRectF(pix.rect()))
        self._predict_pixmap = pix
        self._predict_zoom_steps = 0
        self._apply_predict_zoom(reset=True)

    def _rerender_predict_view(self) -> None:
        tabs = getattr(self, "_predict_tabs", None)
        idx = 0
        try:
            if tabs is not None:
                idx = max(0, tabs.currentIndex())
        except Exception:
            idx = 0
        self._render_predict_tab_by_index(idx)

    def _on_predict_log_scale_toggled(self, checked: bool) -> None:
        if self._ui_updating:
            return
        self.current_parameters["predict_log_scale"] = bool(checked)
        self._persist_parameters()
        self._rerender_predict_view()

    def _on_predict_export_clicked(self) -> None:
        """Export prediction results for single-file or multi-file mode."""

        # 检查当前模式
        mode = self.current_parameters.get("mode", "single_file")
        
        if mode == "multi_files" and self._multifile_results_widget:
            # 多文件模式：触发多文件导出界面
            self._multifile_results_widget.onExportClicked()
            return

        if not self.prediction_results:
            QMessageBox.information(self.main_window, "Export", "Run a prediction before exporting the current result.")
            self._append_status_message("No prediction result to export", level="WARN")
            return

        # 单文件模式：使用原有逻辑
        spec = None
        tabs = getattr(self, "_predict_tabs", None)
        try:
            if tabs is not None and 0 <= tabs.currentIndex() < len(self._predict_tab_specs):
                spec = self._predict_tab_specs[tabs.currentIndex()]
        except Exception:
            spec = None
        if spec is None and self._predict_tab_specs:
            spec = self._predict_tab_specs[0]
        if spec is None:
            self._append_status_message("No prediction output to export", level="WARN")
            return

        kind = self._predict_current_kind
        if kind is None and isinstance(spec, dict):
            kind = spec.get("kind")

        dialog = QMessageBox(self.main_window)
        dialog.setWindowTitle("Export Predict-2D")
        dialog.setText("Select what to export")
        btn_img = dialog.addButton("Image (JPG)", QMessageBox.AcceptRole)
        btn_data = dialog.addButton("Data (ASCII)", QMessageBox.AcceptRole)
        btn_both = dialog.addButton("Both", QMessageBox.AcceptRole)
        dialog.addButton(QMessageBox.Cancel)
        dialog.exec_()
        clicked = dialog.clickedButton()
        if clicked is None or clicked == dialog.button(QMessageBox.Cancel):
            return
        export_image = clicked in (btn_img, btn_both)
        export_data = clicked in (btn_data, btn_both)

        export_path = self._prompt_export_folder("Save Prediction Output To")
        if not export_path:
            return
        if not os.path.isdir(export_path):
            QMessageBox.warning(self.main_window, "Export Path", f"Export folder not found: {export_path}")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        if export_image:
            if self._predict_pixmap is None:
                self._append_status_message("No predict view image to export", level="WARN")
            else:
                img_path = os.path.join(export_path, f"predict_{kind or 'view'}_{timestamp}.jpg")
                try:
                    if not self._predict_pixmap.save(img_path, "JPG"):
                        raise IOError("Save returned False")
                    self._append_status_message(f"Predict image exported: {img_path}")
                except Exception as exc:
                    self._append_status_message(f"Predict image export failed: {exc}", level="ERROR")

        if export_data:
            try:
                if kind == "curve" and isinstance(self._predict_current_curve, np.ndarray):
                    curve = np.array(self._predict_current_curve, dtype=np.float32)
                    x = np.arange(len(curve), dtype=np.float32)
                    data = np.column_stack([x, curve])
                    data_path = os.path.join(export_path, f"predict_curve_{timestamp}.txt")
                    np.savetxt(data_path, data, fmt="%.6g", header="x y", comments="")
                    self._append_status_message(f"Curve data exported: {data_path}")
                elif kind in ("hr", "array", "steps") and isinstance(self._predict_current_image, np.ndarray):
                    arr = np.array(self._predict_current_image, dtype=np.float32)
                    step_suffix = ""
                    if kind == "steps" and isinstance(getattr(self, "_step_snapshots", None), list):
                        try:
                            lbl = self._step_snapshots[self._current_step_index].get("label")
                            if lbl:
                                step_suffix = f"_{str(lbl)}"
                        except Exception:
                            step_suffix = ""
                    data_path = os.path.join(export_path, f"predict_{kind}{step_suffix}_{timestamp}.txt")
                    np.savetxt(data_path, arr, fmt="%.6g")
                    self._append_status_message(f"Matrix data exported: {data_path}")
                else:
                    self._append_status_message("No data available to export", level="WARN")
            except Exception as exc:
                self._append_status_message(f"Predict data export failed: {exc}", level="ERROR")

    # ------------------------------------------------------------------
    # Preprocess steps collection
    # ------------------------------------------------------------------
    def _collect_preprocess_steps(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[List[Dict[str, object]]]]:
        if image is None:
            return None, None
        try:
            spec = self._current_module or {}
            entry = spec.get("preprocess_entry") if isinstance(spec, dict) else ""
            pre_cfg = spec.get("preprocess_raw") if isinstance(spec, dict) else None
            folder = spec.get("folder") if isinstance(spec, dict) else None
            if isinstance(entry, str) and entry and isinstance(folder, str) and os.path.isdir(folder):
                module_name, _, func_name = entry.partition(":")
                if module_name and func_name:
                    py_path = os.path.join(folder, f"{module_name}.py")
                    if os.path.isfile(py_path):
                        import importlib.util
                        spec_imp = importlib.util.spec_from_file_location(f"mod_{module_name}_steps", py_path)
                        if spec_imp and spec_imp.loader:
                            mod = importlib.util.module_from_spec(spec_imp)
                            spec_imp.loader.exec_module(mod)
                            fn = getattr(mod, func_name, None)
                            if callable(fn):
                                try:
                                    out = fn(image.astype(np.float32, copy=True), pre_cfg, module_folder=folder, return_steps=True)
                                except TypeError:
                                    out = fn(image.astype(np.float32, copy=True), pre_cfg, module_folder=folder)
                                arr = None
                                steps_payload = None
                                if isinstance(out, tuple):
                                    arr = out[0]
                                    steps_payload = out[1] if len(out) > 1 else None
                                elif isinstance(out, dict):
                                    arr = out.get("image") or out.get("result")
                                    steps_payload = out.get("steps")
                                elif isinstance(out, np.ndarray):
                                    arr = out
                                steps_list = steps_payload if isinstance(steps_payload, list) else None
                                return arr, steps_list
            # Fallback: just return final preprocessed image without steps
            arr = self._preprocess_for_module(image)
            return arr, None
        except Exception:
            return None, None

    def _display_prediction(self, outputs: Dict[str, np.ndarray]) -> None:
        if not outputs:
            return
        self.prediction_results = outputs
        self._refresh_predict_readiness()
        # 1) If only scalar outputs, print to status and return
        scal = outputs.get("scalars") if isinstance(outputs, dict) else None
        if isinstance(scal, np.ndarray):
            vals = ", ".join(f"{float(x):.4g}" for x in scal.reshape(-1))
            self._append_status_message(f"Predicted scalars: [{vals}]", level="INFO")
            return
        panels: List[Dict[str, object]] = []
        params = outputs.get("parameters") if isinstance(outputs, dict) else None
        param_names = outputs.get("parameter_names") if isinstance(outputs, dict) else None
        if isinstance(params, np.ndarray):
            names = [str(name) for name in param_names] if isinstance(param_names, list) else [f"p{i + 1}" for i in range(params.size)]
            text = ", ".join(
                f"{name}={float(value):.6g}"
                for name, value in zip(names, np.asarray(params).reshape(-1))
            )
            self._append_status_message(f"Predicted parameters: {text}", level="INFO")
            panels.append({
                "kind": "parameters",
                "title": "Parameters",
                "data": np.asarray(params, dtype=np.float32).reshape(-1),
                "names": names,
            })

        # Optional: Preprocessed steps panel with buttons following YAML order
        try:
            if self._current_image is not None:
                pre_img, pre_steps = self._collect_preprocess_steps(self._current_image)
                if pre_steps:
                    display_steps = list(pre_steps)
                    if isinstance(pre_img, np.ndarray):
                        final_input = np.squeeze(pre_img)
                        if isinstance(final_input, np.ndarray) and final_input.ndim == 3 and final_input.shape[-1] >= 2:
                            display_steps = [{
                                "step": "Final Input: intensity",
                                "label": "Final Input: intensity",
                                "image": final_input[..., 0],
                            }, {
                                "step": "Final Input: mask channel",
                                "label": "Final Input: mask channel",
                                "image": final_input[..., 1],
                            }] + display_steps
                        elif isinstance(final_input, np.ndarray) and final_input.ndim == 2:
                            display_steps = [{
                                "step": "Final Input",
                                "label": "Final Input",
                                "image": final_input,
                            }] + display_steps
                    panels.append({
                        "kind": "steps",
                        "title": "Preprocessed",
                        "steps": display_steps,
                        "default_index": 0,
                    })
                elif isinstance(pre_img, np.ndarray):
                    pre_img2d = np.squeeze(pre_img)
                    if isinstance(pre_img2d, np.ndarray) and pre_img2d.ndim == 2:
                        panels.append({
                            "kind": "array",
                            "title": "Preprocessed",
                            "data": pre_img2d,
                            "colormap": self.current_parameters.get("colormap", self._DEFAULT_COLORMAPS[0]),
                        })
        except Exception as exc:
            self._append_status_message(f"Preprocessed panel failed: {exc}", level="ERROR")

        # HR panel
        hr = outputs.get("hr") if isinstance(outputs, dict) else None
        if isinstance(hr, np.ndarray) and hr.ndim == 2:
            panels.append({"kind": "hr", "title": "hr distribution", "data": hr})

        # 1D curves
        h = outputs.get("h") if isinstance(outputs, dict) else None
        if isinstance(h, np.ndarray):
            panels.append({"kind": "curve", "title": "h distribution (nm)", "xlabel": "h (nm)", "data": h})
        r = outputs.get("r") if isinstance(outputs, dict) else None
        if isinstance(r, np.ndarray):
            panels.append({"kind": "curve", "title": "R distribution (nm)", "xlabel": "R (nm)", "data": r})

        if not panels:
            self._append_status_message("No plottable prediction outputs", level="WARN")
            return

        self._predict_tab_specs = panels
        tabs = self._get_or_create_predict2d_tabs()
        if tabs is not None:
            self._rebuild_predict_tabs(tabs)
            if hasattr(tabs, "setTabBarAutoHide"):
                tabs.setTabBarAutoHide(len(panels) <= 1)
            hr_index = next((idx for idx, spec in enumerate(self._predict_tab_specs) if spec.get("kind") == "hr"), None)
            target_index = hr_index if hr_index is not None else (tabs.currentIndex() if tabs.currentIndex() >= 0 else 0)
            if target_index is not None:
                blocker = QSignalBlocker(tabs)
                tabs.setCurrentIndex(target_index)
                del blocker
            self._render_predict_tab_by_index(target_index if target_index is not None else 0)
            # Ensure the outer tab switches to Predict-2D when results are ready
            self._set_predict_main_tab("Predict-2D")
        else:
            self._render_predict_tab_by_index(0)

    def _render_predict2d_into_view(self, image2d: np.ndarray) -> None:
        try:
            self._predict_current_image = image2d
            disp, vmin, vmax = self._prepare_predict_image(image2d)
            target_pixels = self._predict_viewport_pixels()
            pix = self._render_hr_figure(disp, vmin=vmin, vmax=vmax, target_pixels=target_pixels)
            if pix is None:
                pix = self._create_pixmap_from_array(disp, vmin, vmax, self.current_parameters.get("colormap", self._DEFAULT_COLORMAPS[0]))
            if pix is None:
                return
            self._show_pixmap_in_predict_view(pix)
            self._append_status_message("Predict-2D image updated.")
        except Exception as exc:
            self._append_status_message(f"Predict-2D draw failed: {exc}", level="ERROR")

    def _prepare_predict_image(self, image: np.ndarray) -> Tuple[np.ndarray, float, float]:
        disp = self._maybe_log_scale(np.array(image, dtype=np.float32), bool(self.current_parameters.get("predict_log_scale", False)))
        auto = bool(self.current_parameters.get("predict_auto_scale", True))
        vmin = self.current_parameters.get("predict_vmin")
        vmax = self.current_parameters.get("predict_vmax")

        if auto or vmin is None or vmax is None:
            vmin, vmax = self._auto_scale_percentiles(disp, 0, 100)
            self.current_parameters["predict_vmin"] = vmin
            self.current_parameters["predict_vmax"] = vmax

        self._ui_updating = True
        try:
            self._set_checkbox("predict2dAutoScaleCheckBox", auto)
            self._set_double_spin("predict2dVminValue", vmin)
            self._set_double_spin("predict2dVmaxValue", vmax)
        finally:
            self._ui_updating = False
        self._persist_parameters()
        return disp, float(vmin), float(vmax)

    def _render_hr_figure(self, image: np.ndarray, vmin: Optional[float] = None, vmax: Optional[float] = None, target_pixels: Optional[Tuple[int, int]] = None) -> Optional[QPixmap]:
        try:
            import matplotlib.pyplot as plt  # type: ignore
            import numpy as np
            from matplotlib.backends.backend_agg import FigureCanvasAgg  # type: ignore

            img = np.array(image, dtype=np.float32)
            vertical_sum = np.sum(img, axis=0)
            horizontal_sum = np.sum(img, axis=1)
            vmin_calc, vmax_calc = self._auto_scale_values(img)
            cmin = vmin if vmin is not None else vmin_calc
            cmax = vmax if vmax is not None else vmax_calc

            R_bins = np.linspace(0.05, 15, img.shape[0] + 1)
            h_bins = np.linspace(0.05, 15, img.shape[1] + 1)
            R_centers = (R_bins[:-1] + R_bins[1:]) / 2
            h_centers = (h_bins[:-1] + h_bins[1:]) / 2

            dpi = 120.0
            if target_pixels:
                fig_w = max(4.0, target_pixels[0] / dpi)
                fig_h = max(4.0, target_pixels[1] / dpi)
            else:
                fig_w = fig_h = 10.0
            fig, ax = plt.subplots(2, 2, figsize=(fig_w, fig_h), dpi=dpi,
                                   gridspec_kw={'width_ratios': [4, 1], 'height_ratios': [1, 4]})
            scale = max(0.6, min(2.0, fig_w / 10.0))
            title_size = 14 * scale
            tick_size = 12 * scale
            cbar_label_size = 14 * scale
            cbar_tick_size = 12 * scale

            cmap_name = self.current_parameters.get("colormap", self._DEFAULT_COLORMAPS[0])
            im = ax[1, 0].imshow(img, cmap=cmap_name, vmin=cmin, vmax=cmax)
            ax[1, 0].axis('off')

            ax[0, 0].plot(h_centers, vertical_sum, color='red', linewidth=2)
            ax[0, 0].set_title('h distribution (nm)', fontsize=title_size, fontweight='bold')
            ax[0, 0].set_facecolor('#f0f0f0')
            ax[0, 0].grid(True, which='both', linestyle='--', linewidth=0.5)
            ax[0, 0].tick_params(axis='both', which='major', labelsize=tick_size)

            ax[1, 1].plot(horizontal_sum, R_centers, color='red', linewidth=2)
            ax[1, 1].set_title('R distribution (nm)', fontsize=title_size, fontweight='bold')
            ax[1, 1].set_facecolor('#f0f0f0')
            ax[1, 1].grid(True, which='both', linestyle='--', linewidth=0.5)
            ax[1, 1].tick_params(axis='both', which='major', labelsize=tick_size)
            ax[1, 1].invert_yaxis()

            ax[0, 1].axis('off')

            cax = fig.add_axes([0.95, 0.11, 0.02, 0.56])
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label('Intensity', fontsize=cbar_label_size, fontweight='bold')
            cbar.ax.tick_params(labelsize=cbar_tick_size)

            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            buf = canvas.buffer_rgba()
            img_rgba = np.asarray(buf)
            plt.close(fig)

            height, width = img_rgba.shape[:2]
            bytes_per_line = img_rgba.strides[0]
            image_q = QImage(img_rgba.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
            return QPixmap.fromImage(image_q.copy())
        except Exception as exc:
            self._append_status_message(f"HR figure render error: {exc}", level="ERROR")
            return None

    def _render_curve_figure(
        self,
        curve: np.ndarray,
        x_label: str,
        title: str,
        log_x: bool = False,
        log_y: bool = False,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
    ) -> Optional[QPixmap]:
        try:
            import matplotlib.pyplot as plt  # type: ignore
            from matplotlib.backends.backend_agg import FigureCanvasAgg  # type: ignore

            y = np.array(curve, dtype=np.float32)
            x = np.arange(len(y), dtype=np.float32)
            if log_x:
                x = np.arange(1, len(y) + 1, dtype=np.float32)

            y_plot = y.copy()
            if log_y:
                y_plot = np.where(y_plot > 0, y_plot, np.nan)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(x, y_plot, color='red', linewidth=2)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(x_label)
            ax.set_facecolor('#f0f0f0')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.tick_params(axis='both', which='major', labelsize=12)

            if log_x:
                ax.set_xscale('log')
            if log_y:
                ax.set_yscale('log')

            if xlim:
                low, high = xlim
                if log_x and low <= 0:
                    low = max(low, 1e-6)
                if log_x and high <= 0:
                    high = max(high, low + 1e-6)
                ax.set_xlim(low, high)
            if ylim:
                low, high = ylim
                if log_y and low <= 0:
                    low = max(low, 1e-6)
                if log_y and high <= 0:
                    high = max(high, low + 1e-6)
                ax.set_ylim(low, high)

            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            buf = canvas.buffer_rgba()
            img_rgba = np.asarray(buf)
            plt.close(fig)

            height, width = img_rgba.shape[:2]
            bytes_per_line = img_rgba.strides[0]
            image_q = QImage(img_rgba.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
            return QPixmap.fromImage(image_q.copy())
        except Exception as exc:
            self._append_status_message(f"Curve figure render error: {exc}", level="ERROR")
            return None

    # ------------------------------------------------------------------
    # 显示控制
    # ------------------------------------------------------------------
    def _zoom_in(self) -> None:
        self._view_zoom_steps += 1
        self._apply_zoom()

    def _zoom_out(self) -> None:
        self._view_zoom_steps -= 1
        self._apply_zoom()

    def _zoom_reset(self) -> None:
        self._view_zoom_steps = 0
        self._apply_zoom(reset=True)

    def _predict_zoom_in(self) -> None:
        self._predict_zoom_steps += 1
        self._apply_predict_zoom()

    def _predict_zoom_out(self) -> None:
        self._predict_zoom_steps -= 1
        self._apply_predict_zoom()

    def _predict_zoom_reset(self) -> None:
        self._predict_zoom_steps = 0
        self._apply_predict_zoom(reset=True)

    def _apply_zoom(self, reset: bool = False) -> None:
        view = getattr(self.ui, "gisaxsImageGraphicsView", None)
        if view is None or self._current_pixmap is None:
            return
        view.resetTransform()
        if reset:
            view.fitInView(QRectF(self._current_pixmap.rect()), Qt.KeepAspectRatio)
            return
        factor = 1.15 ** self._view_zoom_steps
        view.scale(factor, factor)

    def _apply_predict_zoom(self, reset: bool = False) -> None:
        view = getattr(self.ui, "predict2dGraphicsView", None)
        if view is None or self._predict_pixmap is None:
            return
        view.resetTransform()
        if reset:
            view.fitInView(QRectF(self._predict_pixmap.rect()), Qt.KeepAspectRatio)
            return
        factor = 1.15 ** self._predict_zoom_steps
        view.scale(factor, factor)

    def _on_auto_scale_toggled(self) -> None:
        if self._ui_updating:
            return
        auto = getattr(self.ui, "gisaxsImageAutoScaleCheckBox", None)
        checked = bool(auto.isChecked()) if auto else True
        self.current_parameters["auto_scale"] = checked
        self._persist_parameters()
        if checked:
            self._update_image_display()

    def _on_auto_scale_reset(self) -> None:
        self.current_parameters["auto_scale"] = True
        self._set_checkbox("gisaxsImageAutoScaleCheckBox", True)
        self._persist_parameters()
        self._update_image_display()

    def _on_vmin_changed(self) -> None:
        if self._ui_updating:
            return
        value = self._get_double_spin_value("gisaxsImageVminValue")
        if value is None:
            return
        self.current_parameters["auto_scale"] = False
        self._set_checkbox("gisaxsImageAutoScaleCheckBox", False)
        self.current_parameters["vmin"] = value
        self._persist_parameters()
        self._update_image_display()

    def _on_vmax_changed(self) -> None:
        if self._ui_updating:
            return
        value = self._get_double_spin_value("gisaxsImageVmaxValue")
        if value is None:
            return
        self.current_parameters["auto_scale"] = False
        self._set_checkbox("gisaxsImageAutoScaleCheckBox", False)
        self.current_parameters["vmax"] = value
        self._persist_parameters()
        self._update_image_display()

    def _on_colormap_changed(self, text: str) -> None:
        if self._ui_updating:
            return
        self.current_parameters["colormap"] = text or self._DEFAULT_COLORMAPS[0]
        self._update_image_display()
        self._rerender_predict_view()

    def _on_predict_auto_scale_toggled(self) -> None:
        if self._ui_updating:
            return
        cb = getattr(self.ui, "predict2dAutoScaleCheckBox", None)
        checked = bool(cb.isChecked()) if cb else True
        self.current_parameters["predict_auto_scale"] = checked
        self._persist_parameters()
        self._rerender_predict_view()

    def _on_predict_auto_scale_reset(self) -> None:
        self.current_parameters["predict_auto_scale"] = True
        self._set_checkbox("predict2dAutoScaleCheckBox", True)
        self._persist_parameters()
        self._rerender_predict_view()

    def _on_predict_vmin_changed(self) -> None:
        if self._ui_updating:
            return
        value = self._get_double_spin_value("predict2dVminValue")
        if value is None:
            return
        self.current_parameters["predict_auto_scale"] = False
        self._set_checkbox("predict2dAutoScaleCheckBox", False)
        self.current_parameters["predict_vmin"] = value
        self._persist_parameters()
        self._rerender_predict_view()

    def _on_predict_vmax_changed(self) -> None:
        if self._ui_updating:
            return
        value = self._get_double_spin_value("predict2dVmaxValue")
        if value is None:
            return
        self.current_parameters["predict_auto_scale"] = False
        self._set_checkbox("predict2dAutoScaleCheckBox", False)
        self.current_parameters["predict_vmax"] = value
        self._persist_parameters()
        self._rerender_predict_view()

    # ------------------------------------------------------------------
    # Module selection (scan modules/, parse module.yaml, select & load)
    # ------------------------------------------------------------------
    def eventFilter(self, obj, event):  # noqa: N802 - Qt signature
        try:
            combo = getattr(self.ui, "gisaxsPredictModuleSelectCombox", None)
            if combo and obj is combo and event is not None:
                if event.type() in (QEvent.MouseButtonPress, QEvent.FocusIn):
                    # Refresh module list when user is about to open/select
                    self._refresh_modules()
                
        except Exception:
            pass
        return super().eventFilter(obj, event)

    def _populate_framework_combo(self, combo) -> None:
        available = self.detect_available_frameworks()
        options = [label for label in available.values() if self.is_framework_compatible(self._current_module, label)]
        if not options:
            options = ["No compatible framework installed"]
        try:
            if options and not options[0].startswith("No compatible"):
                self.current_parameters["framework"] = options[0]
        except Exception:
            pass
        blocker = QSignalBlocker(combo)
        combo.clear()
        combo.addItems(options)
        combo.setEnabled(bool(options) and not options[0].startswith("No compatible"))
        del blocker
        self._refresh_framework_status()

    def detect_available_frameworks(self) -> Dict[str, str]:
        frameworks: Dict[str, str] = {}
        try:
            from importlib.metadata import version

            try:
                frameworks["tensorflow"] = f"tensorflow {version('tensorflow')}"
            except Exception:
                pass
            try:
                frameworks["torch"] = f"torch {version('torch')}"
            except Exception:
                pass
        except Exception:
            pass
        return frameworks

    def is_framework_compatible(self, module: Optional[Dict[str, object]], framework_text: str) -> bool:
        framework = (framework_text or "").lower()
        if not framework or framework.startswith("no compatible"):
            return False
        spec = module or {}
        model_format = str(spec.get("model_format") or "").lower() if isinstance(spec, dict) else ""
        model_path = str(spec.get("model_path") or "").lower() if isinstance(spec, dict) else ""

        if any(token in model_format for token in ("torch", "pytorch")) or model_path.endswith((".pt", ".pth")):
            return "torch" in framework
        if any(token in model_format for token in ("tensorflow", "keras", "savedmodel", "h5")) or model_path.endswith((".keras", ".h5")) or os.path.isdir(model_path):
            return "tensorflow" in framework
        return "tensorflow" in framework or "torch" in framework

    def refresh_framework_options_for_current_module(self) -> None:
        combo = getattr(self.ui, "gisaxsPredictFrameworkCombox", None)
        if combo is None:
            return
        current = combo.currentText()
        self._populate_framework_combo(combo)
        if current and self.is_framework_compatible(self._current_module, current):
            idx = combo.findText(current)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        self._refresh_framework_status()
        self._refresh_predict_readiness()

    def _framework_ready(self) -> bool:
        combo = getattr(self.ui, "gisaxsPredictFrameworkCombox", None)
        if combo is None:
            return False
        return combo.isEnabled() and self.is_framework_compatible(self._current_module, combo.currentText())

    def _refresh_framework_status(self) -> None:
        label = getattr(self.ui, "gisaxsPredictFrameworkStatusLabel", None)
        if label is None:
            return
        combo = getattr(self.ui, "gisaxsPredictFrameworkCombox", None)
        text = combo.currentText() if combo is not None else ""
        if self._framework_ready():
            label.setText(f"Framework OK: {text}")
            label.setStyleSheet("color: #166534;")
        elif text.startswith("No compatible"):
            label.setText("Framework missing or incompatible")
            label.setStyleSheet("color: #b91c1c;")
        else:
            label.setText("Framework incompatible")
            label.setStyleSheet("color: #b91c1c;")

    def _initialize_modules_ui(self) -> None:
        self._refresh_modules()
        # Restore last selected module if available
        module_name = self.current_parameters.get("module_name") or ""
        self._set_combobox_text("gisaxsPredictModuleSelectCombox", module_name)
        if module_name:
            self._on_module_selected(module_name)

    def _refresh_modules(self) -> None:
        modules = self._scan_modules()
        new_names = sorted(modules.keys())
        old_names = sorted(self._modules_by_name.keys())
        current_name = self.current_parameters.get("module_name", "")
        self._modules_by_name = modules
        self._modules_by_id = {m.get("id", name): m for name, m in modules.items()}
        if new_names != old_names:
            self._populate_module_combo()
        elif current_name and current_name in modules:
            self._current_module = modules[current_name]

    def _modules_root(self) -> str:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        return os.path.join(base_dir, "modules")

    def _scan_modules(self) -> Dict[str, Dict[str, object]]:
        root = self._modules_root()
        result: Dict[str, Dict[str, object]] = {}
        if not os.path.isdir(root):
            return result
        try:
            for entry in sorted(os.listdir(root)):
                folder = os.path.join(root, entry)
                if not os.path.isdir(folder):
                    continue
                yaml_path = os.path.join(folder, "module.yaml")
                if not os.path.isfile(yaml_path):
                    continue
                spec = self._parse_module_yaml(yaml_path)
                if not spec:
                    continue
                # Enrich with paths
                spec["folder"] = folder
                spec["yaml_path"] = yaml_path
                # Normalize model path to absolute if set
                model_path = spec.get("model_path") or ""
                if model_path and not os.path.isabs(model_path):
                    spec["model_path"] = os.path.abspath(os.path.join(folder, model_path))
                # Normalize mask path
                mask_path = spec.get("mask_path") or ""
                if mask_path and not os.path.isabs(mask_path):
                    spec["mask_path"] = os.path.abspath(os.path.join(folder, mask_path))
                name = spec.get("name") or entry
                result[name] = spec
        except Exception as exc:
            self._append_status_message(f"Module scan failed: {exc}", level="ERROR")
        return result

    def _parse_module_yaml(self, yaml_path: str) -> Optional[Dict[str, object]]:
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception:
            return None

        # Best effort: try PyYAML if available
        data = None
        try:
            import yaml  # type: ignore

            try:
                data = yaml.safe_load(text)
            except Exception:
                # Try parsing only head (before 'outputs:') to avoid YAML errors later in file
                head_lines = []
                for line in text.splitlines():
                    if line.strip().startswith("outputs:"):
                        break
                    head_lines.append(line)
                data = yaml.safe_load("\n".join(head_lines))
        except Exception:
            data = None

        if isinstance(data, dict):
            return self._extract_spec_from_dict(data)

        # Fallback: manual lightweight parse for key fields
        return self._extract_spec_fallback(text)

    def _extract_spec_from_dict(self, data: Dict[str, object]) -> Dict[str, object]:
        spec: Dict[str, object] = {}
        spec["id"] = (data.get("id") or "") if isinstance(data.get("id"), str) else ""
        spec["name"] = (data.get("name") or "") if isinstance(data.get("name"), str) else ""
        model = data.get("model") if isinstance(data.get("model"), dict) else {}
        if isinstance(model, dict):
            spec["model_format"] = model.get("format") if isinstance(model.get("format"), str) else ""
            spec["model_path"] = model.get("model_path") if isinstance(model.get("model_path"), str) else ""
        preprocess = data.get("preprocess") if isinstance(data.get("preprocess"), dict) else {}
        if isinstance(preprocess, dict):
            # Keep raw for module-driven preprocessing
            spec["preprocess_raw"] = preprocess
            spec["preprocess_entry"] = preprocess.get("entry") if isinstance(preprocess.get("entry"), str) else ""
            params = preprocess.get("params") if isinstance(preprocess.get("params"), dict) else {}
            steps = preprocess.get("steps") if isinstance(preprocess.get("steps"), list) else []
            spec["preprocess_params"] = params
            spec["preprocess_steps"] = steps
            mask_cfg = params.get("mask") if isinstance(params, dict) else None
            if isinstance(mask_cfg, dict):
                path = mask_cfg.get("path")
                if isinstance(path, str):
                    spec["mask_path"] = path
                # mask options
                if isinstance(mask_cfg.get("mask_value"), (int, float)):
                    spec["mask_value"] = float(mask_cfg.get("mask_value"))
                resize = mask_cfg.get("resize")
                if isinstance(resize, (list, tuple)) and len(resize) == 2:
                    spec["mask_resize"] = {"height": int(resize[0]), "width": int(resize[1])}
                crop_m = mask_cfg.get("crop_mask")
                if isinstance(crop_m, dict):
                    spec["mask_crop"] = {
                        "left": int(crop_m.get("left", 0)),
                        "up": int(crop_m.get("up", 0)),
                        "down": int(crop_m.get("down", 0)),
                        "right": int(crop_m.get("right", 0)),
                    }
            # Optional crop/resize
            crop_cfg = params.get("crop") if isinstance(params, dict) else None
            if isinstance(crop_cfg, dict):
                # Support either y0/y1/x0/x1 or left/up/down/right
                if all(k in crop_cfg for k in ("y0","y1","x0","x1")):
                    spec["preprocess_crop"] = {
                        "y0": int(crop_cfg.get("y0")),
                        "y1": int(crop_cfg.get("y1")),
                        "x0": int(crop_cfg.get("x0")),
                        "x1": int(crop_cfg.get("x1")),
                    }
                else:
                    spec["preprocess_crop"] = {
                        "left": int(crop_cfg.get("left", 0)),
                        "up": int(crop_cfg.get("up", 0)),
                        "down": int(crop_cfg.get("down", 0)),
                        "right": int(crop_cfg.get("right", 0)),
                    }
            resize_cfg = params.get("resize") if isinstance(params, dict) else None
            if isinstance(resize_cfg, dict):
                spec["preprocess_resize"] = {
                    "height": resize_cfg.get("height"),
                    "width": resize_cfg.get("width"),
                }
            elif isinstance(resize_cfg, (list, tuple)) and len(resize_cfg) == 2:
                spec["preprocess_resize"] = {
                    "height": int(resize_cfg[0]),
                    "width": int(resize_cfg[1]),
                }
        # io settings
        io_cfg = data.get("io") if isinstance(data.get("io"), dict) else {}
        if isinstance(io_cfg, dict):
            shp = io_cfg.get("input_shape")
            if isinstance(shp, (list, tuple)) and len(shp) == 4:
                try:
                    spec["io_input_shape"] = (int(shp[0]), int(shp[1]), int(shp[2]), int(shp[3]))
                except Exception:
                    pass
        outputs = data.get("outputs")
        if isinstance(outputs, dict):
            out_type = outputs.get("type")
            if isinstance(out_type, str):
                spec["output_type"] = out_type
            names = outputs.get("parameter_names")
            if isinstance(names, list):
                spec["parameter_names"] = [str(name) for name in names]
            target_min = outputs.get("target_min")
            target_max = outputs.get("target_max")
            if isinstance(target_min, list):
                try:
                    spec["target_min"] = [float(v) for v in target_min]
                except Exception:
                    pass
            if isinstance(target_max, list):
                try:
                    spec["target_max"] = [float(v) for v in target_max]
                except Exception:
                    pass
        return spec

    def _extract_spec_fallback(self, text: str) -> Optional[Dict[str, object]]:
        spec: Dict[str, object] = {"id": "", "name": "", "model_path": "", "model_format": "", "preprocess_entry": "", "mask_path": ""}
        lines = text.splitlines()
        # top-level fields
        for line in lines:
            m = re.match(r"\s*id\s*:\s*['\"]?(.*?)['\"]?\s*$", line)
            if m:
                spec["id"] = m.group(1)
            m = re.match(r"\s*name\s*:\s*['\"]?(.*?)['\"]?\s*$", line)
            if m:
                spec["name"] = m.group(1)
        # nested: model.model_path and model.format
        in_model = False
        model_indent = 0
        for line in lines:
            if not in_model:
                m = re.match(r"(\s*)model\s*:\s*$", line)
                if m:
                    in_model = True
                    model_indent = len(m.group(1))
                    continue
            else:
                # break if indentation dedent
                leading = len(line) - len(line.lstrip(" "))
                if leading <= model_indent:
                    in_model = False
                    continue
                m1 = re.match(r"\s*format\s*:\s*['\"]?(.*?)['\"]?\s*$", line)
                if m1:
                    spec["model_format"] = m1.group(1)
                m2 = re.match(r"\s*model_path\s*:\s*['\"]?(.*?)['\"]?\s*$", line)
                if m2:
                    spec["model_path"] = m2.group(1)
        # nested: preprocess.entry and preprocess.params.mask.path
        in_pre = False
        pre_indent = 0
        in_params = False
        params_indent = 0
        in_mask = False
        mask_indent = 0
        for line in lines:
            if not in_pre:
                m = re.match(r"(\s*)preprocess\s*:\s*$", line)
                if m:
                    in_pre = True
                    pre_indent = len(m.group(1))
                    continue
            else:
                leading = len(line) - len(line.lstrip(" "))
                if leading <= pre_indent:
                    in_pre = in_params = in_mask = False
                    continue
                m_entry = re.match(r"\s*entry\s*:\s*['\"]?(.*?)['\"]?\s*$", line)
                if m_entry:
                    spec["preprocess_entry"] = m_entry.group(1)
                if not in_params:
                    m_params = re.match(r"(\s*)params\s*:\s*$", line)
                    if m_params:
                        in_params = True
                        params_indent = len(m_params.group(1))
                        continue
                else:
                    leading2 = len(line) - len(line.lstrip(" "))
                    if leading2 <= params_indent:
                        in_params = in_mask = False
                        continue
                    if not in_mask:
                        m_mask = re.match(r"(\s*)mask\s*:\s*$", line)
                        if m_mask:
                            in_mask = True
                            mask_indent = len(m_mask.group(1))
                            continue
                    else:
                        leading3 = len(line) - len(line.lstrip(" "))
                        if leading3 <= mask_indent:
                            in_mask = False
                            continue
                        m_path = re.match(r"\s*path\s*:\s*['\"]?(.*?)['\"]?\s*$", line)
                        if m_path:
                            spec["mask_path"] = m_path.group(1)

        # Must have at least name
        return spec if (spec.get("name") or spec.get("id")) else None

    def _populate_module_combo(self) -> None:
        combo = getattr(self.ui, "gisaxsPredictModuleSelectCombox", None)
        if combo is None:
            return
        current = combo.currentText()
        names = sorted(self._modules_by_name.keys())
        blocker = QSignalBlocker(combo)
        combo.clear()
        combo.addItems(names)
        # Try restore
        idx = combo.findText(self.current_parameters.get("module_name", ""))
        if idx >= 0:
            combo.setCurrentIndex(idx)
        elif current:
            idx2 = combo.findText(current)
            if idx2 >= 0:
                combo.setCurrentIndex(idx2)
        del blocker

    def _select_model_folder(self, start_dir: str = "") -> str:
        folder = QFileDialog.getExistingDirectory(
            self.main_window,
            "Select TensorFlow SavedModel Folder",
            start_dir or "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        return os.path.abspath(normalize_path(folder)) if folder else ""

    def _install_legacy_keras_load_shims(self) -> None:
        """Allow older Keras 2.x .keras files to load in newer Keras 3.x runtimes."""
        try:
            import types
            import sys as _sys
            import keras  # type: ignore
            from keras.src.models.functional import Functional  # type: ignore
            from keras.src.layers.core.lambda_layer import Lambda  # type: ignore
            from keras.src.layers.normalization.batch_normalization import BatchNormalization  # type: ignore
            from keras.src.utils import python_utils  # type: ignore

            engine_pkg = types.ModuleType("keras.src.engine")
            functional_mod = types.ModuleType("keras.src.engine.functional")
            functional_mod.Functional = Functional
            _sys.modules.setdefault("keras.src.engine", engine_pkg)
            _sys.modules.setdefault("keras.src.engine.functional", functional_mod)

            if not getattr(Lambda, "_gimap_legacy_from_config", False):
                original_from_config = Lambda.from_config

                @classmethod
                def _legacy_lambda_from_config(cls, config, custom_objects=None, safe_mode=None):
                    if isinstance(config, dict):
                        config = dict(config)
                        for key in ("function_type", "module", "output_shape_type", "output_shape_module"):
                            config.pop(key, None)
                        fn = config.get("function")
                        if isinstance(fn, (list, tuple)) and fn:
                            try:
                                defaults = fn[1] if len(fn) > 1 else None
                                closure = fn[2] if len(fn) > 2 else None
                                config["function"] = python_utils.func_load(fn[0], defaults=defaults, closure=closure)
                            except Exception:
                                pass
                        if callable(config.get("function")):
                            return cls(**config)
                    try:
                        return original_from_config(config, custom_objects=custom_objects, safe_mode=safe_mode)
                    except TypeError:
                        return original_from_config(config)

                Lambda.from_config = _legacy_lambda_from_config  # type: ignore[method-assign]
                Lambda._gimap_legacy_from_config = True  # type: ignore[attr-defined]

            if not getattr(BatchNormalization, "_gimap_legacy_from_config", False):
                original_bn_from_config = BatchNormalization.from_config

                @classmethod
                def _legacy_bn_from_config(cls, config):
                    if isinstance(config, dict):
                        config = dict(config)
                        axis = config.get("axis")
                        if isinstance(axis, list) and len(axis) == 1:
                            config["axis"] = axis[0]
                    return original_bn_from_config(config)

                BatchNormalization.from_config = _legacy_bn_from_config  # type: ignore[method-assign]
                BatchNormalization._gimap_legacy_from_config = True  # type: ignore[attr-defined]

            if "keras.src.layers.core.tf_op_layer" not in _sys.modules:
                tf_op_mod = types.ModuleType("keras.src.layers.core.tf_op_layer")

                @keras.saving.register_keras_serializable(package="keras.src.layers.core.tf_op_layer")
                class SlicingOpLambda(keras.layers.Layer):  # type: ignore[misc]
                    def __init__(self, function=None, **kwargs):
                        super().__init__(**kwargs)
                        self.function = function

                    def call(self, inputs, slice_spec=None, **kwargs):
                        if slice_spec is None:
                            return inputs
                        slices = []
                        for spec in slice_spec:
                            if isinstance(spec, dict):
                                slices.append(slice(spec.get("start"), spec.get("stop"), spec.get("step")))
                            else:
                                slices.append(spec)
                        return inputs[tuple(slices)]

                    def get_config(self):
                        config = super().get_config()
                        config["function"] = self.function
                        return config

                tf_op_mod.SlicingOpLambda = SlicingOpLambda
                _sys.modules["keras.src.layers.core.tf_op_layer"] = tf_op_mod
        except Exception as exc:
            self._append_status_message(f"Legacy Keras compatibility shim unavailable: {exc}", level="WARN")

    def _on_module_selected(self, name: str) -> None:
        if not name:
            return
        spec = self._modules_by_name.get(name)
        if not spec:
            return
        self._current_module = spec
        self.current_parameters["module_name"] = spec.get("name", name)
        self.current_parameters["module_model_path"] = ""
        self._current_model = None
        self._set_model_status_color("gray", "Not loaded")
        self.refresh_framework_options_for_current_module()

        # The selected module owns its model path. Do not inherit a previous
        # module's model path here, or the wrong model can be silently loaded.
        model_path = spec.get("model_path") or ""
        if not model_path or not os.path.exists(model_path):
            self._load_module_mask(self._current_module)
            self._persist_parameters()
            self._append_status_message("Module selected. Use Import Model to choose and load a model.", level="INFO")
            self._refresh_predict_readiness()
            return

        # Persist chosen model path in session parameters (not writing back to YAML)
        abs_model = os.path.abspath(model_path)
        self.current_parameters["module_model_path"] = abs_model
        self._current_module["model_path"] = abs_model

        # Load mask if available
        self._load_module_mask(self._current_module)

        self._persist_parameters()
        self._append_status_message(f"Module selected: {self.current_parameters['module_name']}")

    def _load_module_mask(self, spec: Dict[str, object]) -> None:
        self._current_mask = None
        mask_path = spec.get("mask_path") if isinstance(spec, dict) else None
        if not isinstance(mask_path, str) or not mask_path:
            return
        mask_path = normalize_path(mask_path)
        try:
            if os.path.isfile(mask_path) and mask_path.lower().endswith(".npy"):
                self._current_mask = np.load(mask_path)
                self._append_status_message(f"Mask loaded: {os.path.basename(mask_path)}")
            else:
                # Only .npy supported for now
                self._append_status_message("Mask file found but unsupported format (only .npy)", level="WARN")
        except Exception as exc:
            self._append_status_message(f"Failed to load mask: {exc}", level="ERROR")

    # ------------------------------------------------------------------
    # Module actions: Edit and Model Import
    # ------------------------------------------------------------------
    def _on_edit_module_clicked(self) -> None:
        combo = getattr(self.ui, "gisaxsPredictModuleSelectCombox", None)
        name = combo.currentText().strip() if combo else ""
        spec = self._modules_by_name.get(name) if name else None
        yaml_path = spec.get("yaml_path") if isinstance(spec, dict) else None
        if not isinstance(yaml_path, str) or not os.path.isfile(yaml_path):
            QMessageBox.information(self.main_window, "File Missing", "module.yaml not found for this module.")
            return
        try:
            if sys.platform.startswith("win"):
                subprocess.Popen(["notepad.exe", yaml_path])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", yaml_path])
            else:
                subprocess.Popen(["xdg-open", yaml_path])
            self._start_module_edit_watch(yaml_path)
        except Exception as exc:
            QMessageBox.warning(self.main_window, "Open Failed", f"Cannot open file:\n{yaml_path}\n\n{exc}")

    def _start_module_edit_watch(self, yaml_path: str) -> None:
        try:
            self._module_edit_watch_mtime = os.path.getmtime(yaml_path)
        except OSError:
            self._module_edit_watch_mtime = None
        self._module_edit_watch_path = yaml_path
        self._module_edit_watch_ticks = 0
        if self._module_edit_watch_timer is None:
            self._module_edit_watch_timer = QTimer(self)
            self._module_edit_watch_timer.timeout.connect(self._check_module_edit_watch)
        self._module_edit_watch_timer.start(1000)
        self._append_status_message("Watching module.yaml for saved edits...")

    def _check_module_edit_watch(self) -> None:
        path = self._module_edit_watch_path
        if not path:
            return
        self._module_edit_watch_ticks += 1
        if self._module_edit_watch_ticks > 300:
            if self._module_edit_watch_timer:
                self._module_edit_watch_timer.stop()
            self._module_edit_watch_path = None
            return
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return
        if self._module_edit_watch_mtime is not None and mtime == self._module_edit_watch_mtime:
            return

        if self._module_edit_watch_timer:
            self._module_edit_watch_timer.stop()
        self._module_edit_watch_path = None
        self._module_edit_watch_mtime = mtime
        selected_name = self.current_parameters.get("module_name", "")
        self._refresh_modules()
        if selected_name and selected_name in self._modules_by_name:
            self._current_module = self._modules_by_name[selected_name]
            self._load_module_mask(self._current_module)
        self._append_status_message("module.yaml saved; module settings reloaded.")

    def _on_reload_module_config_clicked(self) -> None:
        combo = getattr(self.ui, "gisaxsPredictModuleSelectCombox", None)
        selected_name = combo.currentText().strip() if combo else ""
        old_spec = self._modules_by_name.get(selected_name) if selected_name else self._current_module
        old_yaml_path = old_spec.get("yaml_path") if isinstance(old_spec, dict) else None
        old_model_path = ""
        if isinstance(self._current_module, dict):
            old_model_path = str(self._current_module.get("model_path") or "")

        self._refresh_modules()

        refreshed_spec = None
        if isinstance(old_yaml_path, str) and old_yaml_path:
            old_yaml_abs = os.path.normcase(os.path.abspath(old_yaml_path))
            for spec in self._modules_by_name.values():
                yaml_path = spec.get("yaml_path") if isinstance(spec, dict) else None
                if isinstance(yaml_path, str) and os.path.normcase(os.path.abspath(yaml_path)) == old_yaml_abs:
                    refreshed_spec = spec
                    break

        if refreshed_spec is None and selected_name:
            refreshed_spec = self._modules_by_name.get(selected_name)

        if not isinstance(refreshed_spec, dict):
            QMessageBox.warning(
                self.main_window,
                "Reload Config",
                "Could not reload the selected module. Please check module.yaml."
            )
            self._append_status_message("Module config reload failed: selected module not found after scan.", level="ERROR")
            return

        new_name = str(refreshed_spec.get("name") or selected_name)
        new_model_path = str(refreshed_spec.get("model_path") or "")
        self._current_module = refreshed_spec
        self.current_parameters["module_name"] = new_name

        if combo is not None and combo.findText(new_name) >= 0:
            blocker = QSignalBlocker(combo)
            combo.setCurrentText(new_name)
            del blocker

        if new_model_path:
            self.current_parameters["module_model_path"] = os.path.abspath(new_model_path)
        else:
            self.current_parameters["module_model_path"] = ""
            self._current_model = None
            self._set_model_status_color("gray", "Not loaded")
        if old_model_path and new_model_path and os.path.abspath(old_model_path) != os.path.abspath(new_model_path):
            self._current_model = None
            self._set_model_status_color("gray", "Not loaded")

        self.refresh_framework_options_for_current_module()
        self._load_module_mask(self._current_module)
        self._persist_parameters()
        self._refresh_predict_readiness()

        steps = refreshed_spec.get("preprocess_steps")
        step_text = ", ".join(str(s) for s in steps) if isinstance(steps, list) and steps else "default"
        self._append_status_message(f"Module config reloaded: {new_name}; preprocess steps: {step_text}")

    def _on_model_import_clicked(self) -> None:
        # Ensure a module is selected
        combo = getattr(self.ui, "gisaxsPredictModuleSelectCombox", None)
        name = combo.currentText().strip() if combo else ""
        spec = self._modules_by_name.get(name) if name else None
        if not spec:
            QMessageBox.information(self.main_window, "No Module", "Please select a module first.")
            return
        model_path = (spec.get("model_path") or "") if isinstance(spec, dict) else ""
        if not model_path or not os.path.exists(model_path):
            model_path = self._select_model_folder(spec.get("folder", "") if isinstance(spec, dict) else "")
            if not model_path:
                return
            self.current_parameters["module_model_path"] = model_path
            self._current_module = spec
            self._current_module["model_path"] = model_path
            self._write_model_path_to_yaml(self._current_module, model_path)
            self.refresh_framework_options_for_current_module()
        else:
            model_path = os.path.abspath(model_path)
            self.current_parameters["module_model_path"] = model_path

        # Run load in background
        self._append_status_message("Loading model (this may take a while)...")
        self.progress_updated.emit(5)
        self._model_loading = True
        self._model_cancel_requested = False
        self._set_model_status_color("red", "Loading...")
        btn_import = getattr(self.ui, "gisaxsPredictModelImportButton", None)
        if btn_import:
            btn_import.setEnabled(False)
        def _load():
            try:
                # Inform UI about what is being loaded
                self._append_status_message(f"Loading model from: {model_path}")
                # Set mixed precision policy based on GPU availability
                try:
                    import tensorflow as tf  # type: ignore
                    from tensorflow.keras import mixed_precision as tf_mixed_precision  # type: ignore
                    gpus = tf.config.list_physical_devices('GPU')
                    if not gpus:
                        tf_mixed_precision.set_global_policy('float32')
                        self._append_status_message("No GPU detected: using float32 precision.")
                    else:
                        tf_mixed_precision.set_global_policy('mixed_float16')
                        self._append_status_message("GPU detected: using mixed_float16 precision.")
                except Exception:
                    # If TensorFlow is not available or policy cannot be set, continue without setting
                    pass
                model = None
                # Support both Keras .keras files and TensorFlow SavedModel directories
                import os
                if os.path.isdir(model_path):
                    try:
                        import tensorflow as tf  # type: ignore
                        try:
                            # Try TF-Keras loader on SavedModel directory
                            model = tf.keras.models.load_model(model_path, compile=False)
                        except Exception:
                            # Fallback to raw TF SavedModel loader (returns a trackable object)
                            model = tf.saved_model.load(model_path)
                        self._append_status_message(f"Model successfully loaded from: {model_path}")
                        return model, None
                    except Exception as exc:
                        self._append_status_message(f"Failed to load model from: {model_path} | {exc}", level="ERROR")
                        return None, str(exc)
                try:
                    import keras  # type: ignore
                    self._install_legacy_keras_load_shims()
                    try:
                        model = keras.models.load_model(model_path, safe_mode=False, compile=False)  # type: ignore[call-arg]
                    except TypeError:
                        model = keras.models.load_model(model_path, compile=False)
                except Exception:
                    import tensorflow as tf  # type: ignore
                    self._install_legacy_keras_load_shims()
                    # Fallback to TF Keras, also disable compile to speed import
                    model = tf.keras.models.load_model(model_path, compile=False)
                self._append_status_message(f"Model successfully loaded from: {model_path}")
                return model, None
            except Exception as exc:
                self._append_status_message(f"Unexpected error loading model from: {model_path} | {exc}", level="ERROR")
                return None, str(exc)

        import threading
        def _run():
            model, err = _load()
            self.model_load_finished.emit(model, err or "", model_path)

        import threading as _threading
        self._model_loader_thread = _threading.Thread(target=_run, daemon=True)
        self._model_loader_thread.start()

    def _on_model_load_finished(self, model: object, err: str, model_path: str) -> None:
        """Finalize model loading on the Qt UI thread."""
        expected_model_path = str(self.current_parameters.get("module_model_path") or "")
        if expected_model_path and os.path.abspath(model_path) != os.path.abspath(expected_model_path):
            self._append_status_message(
                f"Ignored stale model load result from: {model_path}",
                level="WARN",
            )
            self._model_loading = False
            btn = getattr(self.ui, "gisaxsPredictModelImportButton", None)
            if btn:
                btn.setEnabled(True)
            self._refresh_predict_readiness()
            return
        if not expected_model_path:
            self._append_status_message(
                f"Ignored model load result because no module model is selected: {model_path}",
                level="WARN",
            )
            self._model_loading = False
            btn = getattr(self.ui, "gisaxsPredictModelImportButton", None)
            if btn:
                btn.setEnabled(True)
            self._refresh_predict_readiness()
            return
        if err:
            self._append_status_message(f"Model load failed: {err}", level="ERROR")
            self.progress_updated.emit(0)
            self._current_model = None
            self._model_loading = False
            self._set_model_status_color("gray", "Not loaded")
        else:
            self._current_model = model
            self.current_parameters["module_model_path"] = model_path
            if self._current_module is not None:
                self._current_module["model_path"] = model_path
            if self._model_cancel_requested:
                self._current_model = None
                self._append_status_message("Model load canceled.")
                self.progress_updated.emit(0)
                self._set_model_status_color("gray", "Canceled")
            else:
                self._append_status_message("Model loaded successfully.")
                self.progress_updated.emit(100)
                self._set_model_status_color("green", "Loaded")
            self._model_loading = False

        btn = getattr(self.ui, "gisaxsPredictModelImportButton", None)
        if btn:
            btn.setEnabled(True)
        self._persist_parameters()
        self._refresh_predict_readiness()

    def _write_model_path_to_yaml(self, spec: Dict[str, object], model_path: str) -> None:
        yaml_path = spec.get("yaml_path") if isinstance(spec, dict) else None
        if not isinstance(yaml_path, str) or not os.path.isfile(yaml_path):
            return
        yaml_model_path = "'" + str(model_path).replace("'", "''") + "'"
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                text = f.read()

            lines = text.splitlines()
            out: List[str] = []
            in_model = False
            model_indent = 0
            wrote = False
            for i, line in enumerate(lines):
                if not in_model:
                    m = re.match(r"(\s*)model\s*:\s*$", line)
                    if m:
                        in_model = True
                        model_indent = len(m.group(1))
                        out.append(line)
                        continue
                    out.append(line)
                else:
                    leading = len(line) - len(line.lstrip(" "))
                    # leaving model block
                    if leading <= model_indent:
                        if not wrote:
                            out.append(" " * (model_indent + 2) + f"model_path: {yaml_model_path}")
                            wrote = True
                        in_model = False
                        out.append(line)
                        continue
                    # inside model block: replace model_path
                    m2 = re.match(r"(\s*)model_path\s*:\s*(.*)$", line)
                    if m2:
                        indent = m2.group(1)
                        out.append(f"{indent}model_path: {yaml_model_path}")
                        wrote = True
                    else:
                        out.append(line)

            # If file ended while still in model block
            if in_model and not wrote:
                out.append(" " * (model_indent + 2) + f"model_path: {yaml_model_path}")
                wrote = True

            if wrote:
                new_text = "\n".join(out) + ("\n" if text.endswith("\n") else "")
                with open(yaml_path, "w", encoding="utf-8") as f:
                    f.write(new_text)
                self._append_status_message(f"Updated model_path in {os.path.basename(yaml_path)}")
        except Exception as exc:
            self._append_status_message(f"Failed to update module.yaml: {exc}", level="ERROR")

    # ------------------------------------------------------------------
    # 预测逻辑（当前为占位实现，无TensorFlow依赖）
    # ------------------------------------------------------------------
    def _run_gisaxs_predict(self) -> None:
        self._execute_prediction()

    def _execute_prediction(self) -> None:
        self._update_parameters_from_ui()
        if not self._validate_parameters():
            return
        try:
            self.status_updated.emit("Starting GISAXS prediction...")
            self.progress_updated.emit(0)
            mode = self.current_parameters.get("mode", "single_file")
            if mode == "single_file":
                # Predict for the currently loaded image
                if self._current_image is None:
                    self._append_status_message("No image loaded for prediction", level="WARN")
                    return
                self.progress_updated.emit(10)
                inp = self._preprocess_for_module(self._current_image)
                if inp is None:
                    self._append_status_message("Preprocessing failed", level="ERROR")
                    return
                self.progress_updated.emit(40)
                outs = self._predict_with_current_model(inp)
                if not outs:
                    self._append_status_message("Prediction failed", level="ERROR")
                    self.progress_updated.emit(0)
                    return
                self.progress_updated.emit(70)
                self._display_prediction(outs)
                self.progress_updated.emit(100)
                self.status_updated.emit("GISAXS prediction finished!")
            else:
                # Multi-files: use new queue-based processing
                results = self._predict_multi_files()
                if results and results.get("processing_started"):
                    # 不需要等待完成，处理在后台进行
                    # progress和completion信号会由multifile_manager发出
                    pass
                else:
                    self.progress_updated.emit(0)
                    self.status_updated.emit("Failed to start multi-file prediction")
        except Exception as exc:  # pragma: no cover - runtime safety
            QMessageBox.critical(self.main_window, "Prediction Error", str(exc))
            self.status_updated.emit(f"GISAXS prediction error: {exc}")
            # 重置多文件预测状态
            if self._multifile_prediction_active:
                self._on_multifile_prediction_completed()

    def _update_parameters_from_ui(self) -> None:
        combo = getattr(self.ui, "gisaxsPredictFrameworkCombox", None)
        if combo is not None:
            self.current_parameters["framework"] = combo.currentText()
        export_edit = getattr(self.ui, "gisaxsPredictExportFolderValue", None)
        if export_edit is not None:
            text = export_edit.text().strip()
            if text:
                self.current_parameters["export_path"] = text

    def _validate_parameters(self) -> bool:
        mode = self.current_parameters.get("mode", "single_file")
        if mode == "single_file":
            file_path = self.current_parameters.get("input_file")
            if not file_path or not os.path.exists(file_path):
                QMessageBox.warning(self.main_window, "Invalid Parameters", "Please select a valid input file")
                return False
        else:
            folder = self.current_parameters.get("input_folder")
            if not folder or not os.path.exists(folder):
                QMessageBox.warning(self.main_window, "Invalid Parameters", "Please select a valid folder")
                return False
        if not self._framework_ready():
            QMessageBox.warning(self.main_window, "Framework", "The selected model requires a compatible installed framework.")
            return False
        if not self._model_ready():
            QMessageBox.warning(self.main_window, "Model", "Please import a model before running prediction.")
            return False
        return True

    def _predict_single_file(self) -> Optional[Dict[str, object]]:
        file_path = self.current_parameters.get("input_file")
        if not file_path:
            return None
        self.status_updated.emit(f"Processing file: {os.path.basename(file_path)}")
        self.progress_updated.emit(25)
        results = {
            "file": file_path,
            "predictions": [],
            "confidence": 0.95,
            "processing_time": 1.5,
        }
        self.progress_updated.emit(75)
        return results

    def _predict_multi_files(self) -> Optional[Dict[str, object]]:
        """多文件预测 - 使用新的队列处理系统"""
        folder = self.current_parameters.get("input_folder")
        if not folder:
            self._append_status_message("No input folder selected", level="WARN")
            return None
            
        # 获取文件列表
        try:
            files = []
            for f in sorted(os.listdir(folder)):
                if f.lower().endswith(('.cbf', '.tif', '.tiff')):
                    files.append(os.path.join(folder, f))
        except Exception as e:
            self._append_status_message(f"Error scanning folder: {e}", level="ERROR")
            return None
            
        if not files:
            self._append_status_message("No compatible image files found in folder", level="WARN")
            return None

        # 应用范围过滤
        range_text = self.current_parameters.get("range_value", "")
        if range_text:
            try:
                indices = self._parse_range_text(range_text)
                if indices:
                    self._scan_directory_for_cbf(folder)
                    missing = [idx for idx in indices if idx not in self._index_to_file]
                    files = [self._index_to_file[idx] for idx in indices if idx in self._index_to_file]
                    if missing:
                        missing_text = ", ".join(f"{idx:05d}" for idx in missing[:10])
                        if len(missing) > 10:
                            missing_text += ", ..."
                        self._append_status_message(f"Range skipped missing CBF indices: {missing_text}", level="WARN")
            except Exception as e:
                self._append_status_message(f"Error parsing range: {e}", level="WARN")

        if not files:
            self._append_status_message("No files selected by range", level="WARN") 
            return None

        try:
            every = max(1, int(self._get_line_edit_text("gisaxsPredictEveryValue") or "1"))
        except ValueError:
            every = 1
            self._set_line_edit("gisaxsPredictEveryValue", "1")
            self._append_status_message("Every must be a positive integer; using 1.", level="WARN")

        if every > 1:
            batches = [files[i : i + every] for i in range(0, len(files), every) if len(files[i : i + every]) == every]
            skipped = len(files) - (len(batches) * every)
            if skipped:
                self._append_status_message(
                    f"Skipped {skipped} trailing file(s) that do not make a full Every={every} stack.",
                    level="WARN",
                )
        else:
            batches = [[file_path] for file_path in files]
        self._multifile_batch_map = {batch[0]: batch for batch in batches if batch}
        files_to_process = list(self._multifile_batch_map.keys())
        if not files_to_process:
            self._append_status_message("No complete multi-file stacks selected by range/every.", level="WARN")
            return None
        if every > 1:
            self._append_status_message(
                f"Multi-file range grouped into {len(files_to_process)} batch(es), Every={every}.",
                level="INFO",
            )

        # 清空现有结果并添加新的待处理项目
        if self._multifile_results_widget:
            self._multifile_results_widget.clearResults()
            
            # 添加所有文件到结果列表
            for file_path in files_to_process:
                row = self._multifile_results_widget.addPredictResult(file_path)
                batch = self._multifile_batch_map.get(file_path, [])
                if len(batch) > 1:
                    result = self._multifile_results_widget.table_model.getResult(row)
                    if result is not None:
                        result.file_name = f"{os.path.basename(batch[0])} - {os.path.basename(batch[-1])}"
                        result.file_path = "\n".join(batch)
                        result.stack_count = len(batch)
                        self._multifile_results_widget.table_model.updateResult(row, result)
                        self._append_status_message(
                            f"Queued stack: {os.path.basename(batch[0])} - {os.path.basename(batch[-1])} ({len(batch)} files)",
                            level="INFO",
                        )
                elif batch:
                    result = self._multifile_results_widget.table_model.getResult(row)
                    if result is not None:
                        result.stack_count = 1
                        self._multifile_results_widget.table_model.updateResult(row, result)

        # 开始批量预测
        if self._multifile_manager:
            self._multifile_prediction_active = True
            self._show_multifile_results_window()
            self._multifile_manager.start_batch_prediction(files_to_process, self._predict_single_file_for_batch)
            
        # 立即返回，实际处理将在后台进行
        return {
            "folder": folder,
            "total_files": len(files_to_process),
            "processing_started": True
        }

    def _predict_single_file_for_batch(self, file_path: str) -> Dict[str, object]:
        """为批量处理执行单文件预测"""
        try:
            # 临时设置当前文件用于预测
            old_file = self.current_parameters.get("input_file", "")
            self.current_parameters["input_file"] = file_path
            batch = self._multifile_batch_map.get(file_path) or [file_path]
            if len(batch) > 1:
                self.status_updated.emit(
                    f"Predicting stack ({len(batch)} files): {os.path.basename(batch[0])} - {os.path.basename(batch[-1])}"
                )
            else:
                self.status_updated.emit(f"Predicting file: {os.path.basename(file_path)}")
            
            # 执行实际预测逻辑（这里需要调用真正的预测代码）
            result = self._execute_single_file_prediction(file_path, batch)
            
            # 恢复原来的文件设置
            self.current_parameters["input_file"] = old_file
            
            return result
            
        except Exception as e:
            # 恢复原来的文件设置
            if 'old_file' in locals():
                self.current_parameters["input_file"] = old_file
            raise e

    def _execute_single_file_prediction(self, file_path: str, stack_files: Optional[List[str]] = None) -> Dict[str, object]:
        """执行单个文件的预测逻辑 - 真正调用预测流程"""
        try:
            # 保存原有参数和状态
            old_input_file = self.current_parameters.get("input_file", "")
            old_mode = self.current_parameters.get("mode", "single_file")
            old_current_image = self._current_image
            
            # 临时设置为单文件模式
            self.current_parameters["input_file"] = file_path
            self.current_parameters["mode"] = "single_file"
            
            # 加载图像（使用同步方法）
            image_data = self._load_cbf_stack_sync(stack_files) if stack_files and len(stack_files) > 1 else self._load_cbf_file_sync(file_path)
            
            if image_data is None:
                raise Exception(f"Failed to load image: {file_path}")
            
            # 设置当前图像
            self._current_image = image_data
            
            # 执行真正的预测流程（与单文件相同）
            # 1. 预处理
            inp = self._preprocess_for_module(self._current_image)
            if inp is None:
                raise Exception("Preprocessing failed")
                
            # 2. 模型预测
            outs = self._predict_with_current_model(inp)
            if not outs:
                raise Exception("Prediction failed")
                
            # 恢复原有参数和状态
            self.current_parameters["input_file"] = old_input_file
            self.current_parameters["mode"] = old_mode
            self._current_image = old_current_image
            
            # 返回结果（只包含预测数据，预处理步骤按需计算）
            return {
                "file": file_path,
                "stack_count": len(stack_files) if stack_files else 1,
                "stack_files": list(stack_files) if stack_files else [file_path],
                "prediction_data": outs  # 真正的预测结果
            }
            
        except Exception as e:
            # 确保恢复原有参数和状态
            if 'old_input_file' in locals():
                self.current_parameters["input_file"] = old_input_file
            if 'old_mode' in locals():
                self.current_parameters["mode"] = old_mode
            if 'old_current_image' in locals():
                self._current_image = old_current_image
            raise e

    def _on_multifile_result_selected(self, result: PredictResult) -> None:
        """多文件结果选中处理 - 双击显示单文件结果"""
        if result.status != PredictStatus.COMPLETED or not result.prediction_data:
            # 如果结果还未完成，只更新当前文件显示
            self._update_current_file_display(result.file_path.splitlines()[0], getattr(result, "stack_count", 1))
            return
            
        try:
            # 更新当前文件显示
            self._update_current_file_display(result.file_path.splitlines()[0], getattr(result, "stack_count", 1))
            
            # 获取预测结果数据
            prediction_data = result.prediction_data.get("prediction_data", {})
            
            if prediction_data:
                # 临时加载当前图像以支持预处理显示（按需计算）
                try:
                    temp_image = self._load_cbf_stack_sync(result.file_path.splitlines()) if "\n" in result.file_path else self._load_cbf_file_sync(result.file_path)
                    if temp_image is not None:
                        # 临时设置当前图像用于预处理显示
                        old_current_image = self._current_image
                        self._current_image = temp_image
                        
                        # 使用标准显示方法（会实时计算预处理步骤）
                        self._display_prediction(prediction_data)
                        
                        # 恢复原来的图像
                        self._current_image = old_current_image
                    else:
                        # 如果无法加载图像，仅显示预测结果（无预处理tab）
                        self._current_image = None
                        self._display_prediction(prediction_data)
                except Exception as e:
                    # 如果图像加载失败，仍然显示预测结果
                    self._append_status_message(f"Could not load image for preprocessing display: {e}", level="WARN")
                    self._current_image = None
                    self._display_prediction(prediction_data)
                
                # 设置当前参数
                self.current_parameters["input_file"] = result.file_path.splitlines()[0]
                
                # 切换到Predict-2D tab
                self._set_predict_main_tab("Predict-2D")
                
                # 更新状态
                self._append_status_message(f"Displaying results for: {os.path.basename(result.file_path.splitlines()[0])}", level="INFO")
            else:
                self._append_status_message(f"No prediction data available for: {os.path.basename(result.file_path.splitlines()[0])}", level="WARN")
                
        except Exception as e:
            self._append_status_message(f"Error displaying result: {e}", level="ERROR")

    def _on_multifile_export_requested(self, config: dict, results: List[PredictResult]) -> None:
        """多文件导出请求处理"""
        if not results:
            QMessageBox.information(self.main_window, "Export", "No results to export.")
            return
            
        export_path = self._prompt_export_folder("Save Multi-File Prediction Output To")
        if not export_path:
            return
            
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 过滤只导出已完成的结果
            completed_results = [r for r in results if r.status == PredictStatus.COMPLETED]
            if not completed_results:
                QMessageBox.information(self.main_window, "Export", "No completed results to export.")
                return
            
            # 导出JSONL格式
            if config.get('jsonl', False):
                self._export_results_jsonl(completed_results, export_path, timestamp)
                
            # 导出JPG图像
            if config.get('jpg', False):
                self._export_results_jpg(completed_results, export_path, timestamp)
                
            # 导出ASCII 1D曲线
            if config.get('ascii', False):
                self._export_results_ascii(completed_results, export_path, timestamp)
                
            self._append_status_message(f"Export completed to {export_path}", level="INFO")
            
        except Exception as e:
            QMessageBox.critical(self.main_window, "Export Error", f"Export failed: {e}")
            self._append_status_message(f"Export error: {e}", level="ERROR")

    def _on_multifile_prediction_started(self) -> None:
        """多文件预测开始"""
        self._multifile_prediction_active = True
        # 禁用Predict按钮
        btn = getattr(self.ui, "gisaxsPredictPredictButton", None)
        if btn:
            btn.setEnabled(False)
            btn.setText("Predicting...")
        stop_btn = getattr(self.ui, "gisaxsPredictStopButton", None)
        if stop_btn:
            stop_btn.setEnabled(True)

    def _on_multifile_prediction_completed(self) -> None:
        """多文件预测完成"""
        self._multifile_prediction_active = False
        # 重新启用Predict按钮
        btn = getattr(self.ui, "gisaxsPredictPredictButton", None)
        if btn:
            btn.setEnabled(True)
            btn.setText("Predict")
        stop_btn = getattr(self.ui, "gisaxsPredictStopButton", None)
        if stop_btn:
            stop_btn.setEnabled(False)
            
        self._append_status_message("Multi-file prediction completed!", level="INFO")

    def _on_multifile_result_updated(self, index: int, update_data: dict) -> None:
        """多文件预测结果更新"""
        if self._multifile_results_widget:
            self._multifile_results_widget.updatePredictResult(index, **update_data)
            if update_data.get("status") == PredictStatus.RUNNING:
                result = self._multifile_results_widget.table_model.getResult(index)
                if result is not None:
                    first = result.file_path.splitlines()[0] if result.file_path else result.file_name
                    stack_count = max(1, int(getattr(result, "stack_count", 1)))
                    self._append_status_message(
                        f"Running stack ({stack_count} file{'s' if stack_count != 1 else ''}): {os.path.basename(first)}",
                        level="INFO",
                    )

    def _on_multifile_progress_updated(self, completed: int, total: int) -> None:
        """多文件预测进度更新"""
        if self._multifile_results_widget:
            self._multifile_results_widget.updateProgress(completed, total)
        
        # 更新主进度条
        if total > 0:
            progress = int((completed / total) * 100)
            self.progress_updated.emit(progress)

    def _export_results_jsonl(self, results: List[PredictResult], export_path: str, timestamp: str) -> None:
        """导出JSONL格式结果"""
        jsonl_path = os.path.join(export_path, f"prediction_results_{timestamp}.jsonl")
        
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for i, result in enumerate(results):
                record = {
                    "index": i,
                    "filename": result.file_name,
                    "filepath": result.file_path,
                    "stack_count": max(1, int(getattr(result, "stack_count", 1))),
                    "timestamp": result.start_time.isoformat() if result.start_time else None,
                    "processing_time": result.processing_time,
                    "confidence": self._result_confidence(result),
                    "prediction_data": self._serialize_prediction_data(result.prediction_data)
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def _result_confidence(self, result: PredictResult) -> Optional[float]:
        """Return confidence when older/newer prediction payloads provide it."""
        value = getattr(result, "confidence", None)
        if isinstance(value, (int, float)):
            return float(value)
        payload = result.prediction_data if isinstance(result.prediction_data, dict) else {}
        value = payload.get("confidence")
        if isinstance(value, (int, float)):
            return float(value)
        inner = payload.get("prediction_data")
        if isinstance(inner, dict):
            value = inner.get("confidence")
            if isinstance(value, (int, float)):
                return float(value)
        return None

    def _export_results_jpg(self, results: List[PredictResult], export_path: str, timestamp: str) -> None:
        """导出JPG图像到文件夹"""
        jpg_folder = os.path.join(export_path, f"prediction_images_{timestamp}")
        os.makedirs(jpg_folder, exist_ok=True)
        
        for i, result in enumerate(results):
            if not result.prediction_data:
                continue
                
            # 导出2D结果图像
            prediction_data = result.prediction_data.get("prediction_data", {})
            hr_data = prediction_data.get("hr")
            
            if isinstance(hr_data, np.ndarray):
                # 创建图像
                image_path = os.path.join(jpg_folder, f"{result.file_name}_{i:04d}_hr.jpg")
                self._save_array_as_image(hr_data, image_path)

    def _export_results_ascii(self, results: List[PredictResult], export_path: str, timestamp: str) -> None:
        """导出ASCII 1D曲线数据"""
        ascii_path = os.path.join(export_path, f"prediction_curves_{timestamp}.txt")
        parameter_rows = []
        parameter_names: List[str] = []
        
        # 收集所有1D数据
        all_h_data = []
        all_r_data = []
        headers = []
        
        for i, result in enumerate(results):
            if not result.prediction_data:
                continue
                
            prediction_data = result.prediction_data.get("prediction_data", {})
            h_data = prediction_data.get("h")
            r_data = prediction_data.get("r")
            p_data = prediction_data.get("parameters")
            p_names = prediction_data.get("parameter_names")
            if isinstance(p_data, np.ndarray):
                arr = np.asarray(p_data, dtype=np.float32).reshape(-1)
                if isinstance(p_names, list) and len(p_names) >= arr.size:
                    names = [str(name) for name in p_names[: arr.size]]
                else:
                    names = [f"p{idx + 1}" for idx in range(arr.size)]
                if not parameter_names:
                    parameter_names = names
                parameter_rows.append((result.file_name, arr))
            
            if isinstance(h_data, np.ndarray):
                all_h_data.append(h_data)
                headers.append(f"{result.file_name}_h")
                
            if isinstance(r_data, np.ndarray):
                all_r_data.append(r_data)
                headers.append(f"{result.file_name}_r")
        
        # 写入文件
        if all_h_data or all_r_data:
            with open(ascii_path, 'w', encoding='utf-8') as f:
                # 写入头部
                f.write("# Prediction 1D Curves Export\n")
                f.write(f"# Generated: {timestamp}\n")
                f.write("# Columns: " + " | ".join(headers) + "\n")
                f.write("# Index\t" + "\t".join(headers) + "\n")
                
                # 确定最大长度
                max_len = 0
                all_data = all_h_data + all_r_data
                if all_data:
                    max_len = max(len(data) for data in all_data)
                
                # 写入数据
                for i in range(max_len):
                    line = [str(i)]
                    for data in all_data:
                        if i < len(data):
                            line.append(f"{data[i]:.6g}")
                        else:
                            line.append("NaN")
                    f.write("\t".join(line) + "\n")
        elif parameter_rows:
            param_path = os.path.join(export_path, f"prediction_parameters_{timestamp}.txt")
            with open(param_path, 'w', encoding='utf-8') as f:
                f.write("# Prediction Parameters Export\n")
                f.write(f"# Generated: {timestamp}\n")
                f.write("filename\t" + "\t".join(parameter_names) + "\n")
                for file_name, values in parameter_rows:
                    f.write(str(file_name) + "\t" + "\t".join(f"{float(v):.8g}" for v in values) + "\n")

    def _serialize_prediction_data(self, prediction_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """序列化预测数据为JSON兼容格式"""
        if not prediction_data:
            return None
            
        try:
            serialized = {}
            inner_data = prediction_data.get("prediction_data", {})
            
            for key, value in inner_data.items():
                if isinstance(value, np.ndarray):
                    # 2D数组只保存形状信息
                    if value.ndim > 1:
                        serialized[key] = {
                            "type": "array_2d",
                            "shape": list(value.shape),
                            "dtype": str(value.dtype)
                        }
                    else:
                        # 1D数组保存完整数据
                        serialized[key] = value.tolist()
                else:
                    serialized[key] = value
                    
            return serialized
        except Exception:
            return {"error": "Failed to serialize prediction data"}

    def _load_cbf_stack_sync(self, file_paths: Optional[List[str]]) -> Optional[np.ndarray]:
        if not file_paths:
            return None
        summed: Optional[np.ndarray] = None
        for file_path in file_paths:
            data = self._load_cbf_file_sync(file_path)
            if data is None:
                continue
            if summed is None:
                summed = data.astype(np.float32, copy=True)
            else:
                summed += data.astype(np.float32, copy=False)
        if summed is None:
            self._append_status_message("Failed to load any file in this stack.", level="ERROR")
        return summed

    def _load_cbf_file_sync(self, file_path: str) -> Optional[np.ndarray]:
        """同步加载CBF文件"""
        try:
            import fabio
            cbf_image = fabio.open(file_path)
            data = cbf_image.data
            
            if data.dtype != np.float32:
                data = data.astype(np.float32, copy=False)
            
            return data
        except Exception as e:
            self._append_status_message(f"Failed to load CBF file {file_path}: {e}", level="ERROR")
            return None

    def _save_array_as_image(self, array: np.ndarray, image_path: str) -> None:
        """将数组保存为图像文件"""
        try:
            # 标准化数组到0-255范围
            if array.dtype != np.uint8:
                array_norm = (array - array.min()) / (array.max() - array.min()) * 255
                array = array_norm.astype(np.uint8)
                
            # 创建QImage并保存
            height, width = array.shape
            qimage = QImage(array.data, width, height, width, QImage.Format_Grayscale8)
            qimage.save(image_path, "JPEG", 90)
            
        except Exception as e:
            self._append_status_message(f"Failed to save image {image_path}: {e}", level="WARN")

    def _save_results(self, results: Dict[str, object]) -> None:
        export_path = self.current_parameters.get("export_path")
        if not export_path:
            return
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(export_path, f"gisaxs_prediction_results_{timestamp}.json")
        with open(file_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        self._append_status_message(f"Results saved: {file_path}")

    # ------------------------------------------------------------------
    # UI 辅助方法
    # ------------------------------------------------------------------
    def _set_line_edit(self, name: str, text: Optional[str]) -> None:
        widget = getattr(self.ui, name, None)
        if widget is None:
            return
        blocker = QSignalBlocker(widget)
        widget.setText(text or "")
        del blocker

    def _get_line_edit_text(self, name: str) -> str:
        widget = getattr(self.ui, name, None)
        return widget.text().strip() if widget else ""

    def _set_checkbox(self, name: str, checked: bool) -> None:
        widget = getattr(self.ui, name, None)
        if widget is None:
            return
        blocker = QSignalBlocker(widget)
        widget.setChecked(bool(checked))
        del blocker

    def _set_double_spin(self, name: str, value: Optional[float]) -> None:
        widget = getattr(self.ui, name, None)
        if widget is None or value is None:
            return
        blocker = QSignalBlocker(widget)
        widget.setValue(float(value))
        del blocker

    def _configure_color_spin(self, name: str) -> None:
        widget = getattr(self.ui, name, None)
        if not isinstance(widget, QDoubleSpinBox):
            return
        widget.setDecimals(6)
        widget.setRange(-1e12, 1e12)
        widget.setSingleStep(0.1)

    def _set_combobox_text(self, name: str, text: str) -> None:
        widget = getattr(self.ui, name, None)
        if widget is None or text is None:
            return
        blocker = QSignalBlocker(widget)
        index = widget.findText(text)
        widget.setCurrentIndex(index if index >= 0 else 0)
        del blocker

    def _get_double_spin_value(self, name: str) -> Optional[float]:
        widget = getattr(self.ui, name, None)
        return float(widget.value()) if widget is not None else None

    def _append_status_message(self, message: str, level: str = "INFO") -> None:
        self.status_updated.emit(message)
        browser = getattr(self.ui, "predictStatusTextBrowser", None)
        line = f"[{level}] {message}"
        if browser is not None:
            browser.append(line)
        if self._status_text_window_browser is not None:
            self._status_text_window_browser.append(line)
