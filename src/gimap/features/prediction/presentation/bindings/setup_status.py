"""Setup Status coordination for Prediction."""

from __future__ import annotations

import os


from PyQt5.QtCore import Qt, QSignalBlocker

from PyQt5.QtGui import QKeySequence

from PyQt5.QtWidgets import (
    QGraphicsScene,
    QLabel,
    QShortcut,
    QTabWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QDialog,
    QTextBrowser,
)

from src.gimap.app.presentation.responsive_layout import (
    apply_density_profile,
    install_adaptive_window_profile,
    move_window_to_cursor_screen,
)


class SetupStatusMixin:
    """Own setup status presentation behavior."""

    def _load_saved_parameters(self) -> None:
        try:
            saved = self.prediction_view_model.load_settings()
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
            lambda profile, screen, window=win: self._apply_floating_screen_profile(
                window, profile
            ),
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

            self._set_line_edit(
                "gisaxsPredictChooseGisaxsFileValue",
                os.path.basename(self.current_parameters.get("input_file", "")),
            )
            self._set_line_edit(
                "gisaxsPredictChooseFolderValue", self.current_parameters.get("input_folder", "")
            )
            self._set_line_edit(
                "gisaxsPredictExportFolderValue", self.current_parameters.get("export_path", "")
            )

            stack_text = self.current_parameters.get("stack_value", "1")
            if mode == "multi_files":
                stack_text = self.current_parameters.get("range_value", "") or stack_text
            self._set_line_edit("gisaxsPredictStackValue", stack_text)
            self._set_line_edit(
                "gisaxsImageShowingValue", self.current_parameters.get("showing_value", "")
            )

            auto_scale = bool(self.current_parameters.get("auto_scale", True))
            self._configure_color_spin("gisaxsImageVminValue")
            self._configure_color_spin("gisaxsImageVmaxValue")
            self._configure_color_spin("predict2dVminValue")
            self._configure_color_spin("predict2dVmaxValue")
            self._set_checkbox("gisaxsImageAutoScaleCheckBox", auto_scale)
            self._set_checkbox(
                "gisaxsImageLogScaleCheckBox",
                bool(self.current_parameters.get("gisaxs_log_scale", False)),
            )
            self._set_double_spin("gisaxsImageVminValue", self.current_parameters.get("vmin"))
            self._set_double_spin("gisaxsImageVmaxValue", self.current_parameters.get("vmax"))

            predict_auto_scale = bool(self.current_parameters.get("predict_auto_scale", True))
            self._set_checkbox("predict2dAutoScaleCheckBox", predict_auto_scale)
            self._set_double_spin("predict2dVminValue", self.current_parameters.get("predict_vmin"))
            self._set_double_spin("predict2dVmaxValue", self.current_parameters.get("predict_vmax"))

            colormap = self.current_parameters.get("colormap") or self._DEFAULT_COLORMAPS[0]
            self._set_combobox_text("gisaxsImageColormapCombox", colormap)
            self._set_combobox_text("predict2dLabelCombox", colormap)

            self._set_checkbox(
                "predict2dLogScaleCheckBox",
                bool(self.current_parameters.get("predict_log_scale", False)),
            )

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
            framework_combo.currentTextChanged.connect(
                lambda _=None: (self._refresh_framework_status(), self._refresh_predict_readiness())
            )

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
        parent_widget = (
            getattr(self.ui, "widget_4", None)
            or getattr(self.ui, "centralwidget", None)
            or self.main_window
        )
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
        status_text = (
            tooltip
            if tooltip in ("Loaded", "Not loaded", "Canceled")
            else text_map.get(color, tooltip or "Not loaded")
        )
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
