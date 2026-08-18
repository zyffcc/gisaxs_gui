"""Insitu Dialog for fitting presentation."""

from __future__ import annotations


from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import (
    QBoxLayout,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QTextBrowser,
    QSizePolicy,
    QDialog,
    QComboBox,
    QGridLayout,
    QLabel,
    QDoubleSpinBox,
    QCheckBox,
    QPushButton,
    QLineEdit,
    QSpinBox,
)


from ..binding_primitives import (
    _ai_catalog,
    is_matplotlib_available,
)


class InsituDialogMixin:
    """Own insitu dialog behavior."""

    def _setup_insitu_workflow_button(self):
        """Add the In-situ Workflow launcher under the Load Mode controls."""
        try:
            if self._insitu_workflow_button is not None:
                return
            parent = (
                getattr(self.ui, "gisaxsInputBox", None)
                or getattr(self.ui, "centralwidget", None)
                or self.ui
            )
            button = QPushButton("In-situ Workflow", parent)
            button.setObjectName("gisaxsInputInsituWorkflowButton")
            button.setToolTip("Open the In-situ workflow automation panel")
            button.clicked.connect(self._open_insitu_workflow_dialog)
            button.setText("In-situ Workflow")
            button.setMinimumHeight(32)
            button.setMaximumHeight(36)
            button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            self._insitu_workflow_button = button

            inserted = False
            combo = getattr(self.ui, "gisaxsInputModelCombox", None)
            combo_layout = self._find_layout_containing_widget(combo) if combo is not None else None
            if combo_layout is not None:
                try:
                    if isinstance(combo_layout, QGridLayout):
                        for index in range(combo_layout.count()):
                            item = combo_layout.itemAt(index)
                            if item is not None and item.widget() is combo:
                                row, col, _row_span, _col_span = combo_layout.getItemPosition(index)
                                holder = self._make_insitu_workflow_mode_box(combo, button, parent)
                                combo_layout.removeWidget(combo)
                                combo_layout.addWidget(
                                    holder, row, col, 1, 1, Qt.AlignTop | Qt.AlignLeft
                                )
                                inserted = True
                                break
                    elif isinstance(combo_layout, QBoxLayout):
                        index = combo_layout.indexOf(combo)
                        if index >= 0:
                            holder = self._make_insitu_workflow_mode_box(combo, button, parent)
                            combo_layout.removeWidget(combo)
                            combo_layout.insertWidget(index, holder, 0, Qt.AlignLeft)
                            inserted = True
                except Exception:
                    inserted = False
            if not inserted:
                grid = getattr(self.ui, "gridLayout_23", None)
                if grid is not None:
                    try:
                        grid.addWidget(button, 1, 1, 1, 1)
                        inserted = True
                    except Exception:
                        inserted = False
            if not inserted:
                button.setParent(parent)
            self._update_insitu_workflow_button_visibility()
        except Exception as exc:
            self.status_updated.emit(f"Failed to create In-situ Workflow button: {exc}")

    def _make_insitu_workflow_mode_box(self, combo, button, parent):
        holder = QWidget(parent)
        holder.setObjectName("gisaxsInputInsituWorkflowModeBox")
        holder_layout = QVBoxLayout(holder)
        holder_layout.setContentsMargins(0, 0, 0, 5)
        holder_layout.setSpacing(4)
        holder_layout.addWidget(combo, 0, Qt.AlignLeft)
        holder_layout.addWidget(button, 0, Qt.AlignLeft)
        holder.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum)
        return holder

    def _find_layout_containing_widget(self, widget):
        if widget is None:
            return None

        # 函数说明：实现 walk 相关逻辑。
        def walk(layout):
            if layout is None:
                return None
            for index in range(layout.count()):
                item = layout.itemAt(index)
                if item is None:
                    continue
                if item.widget() is widget:
                    return layout
                nested = item.layout()
                found = walk(nested)
                if found is not None:
                    return found
            return None

        parent = widget.parentWidget()
        while parent is not None:
            found = walk(parent.layout())
            if found is not None:
                return found
            parent = parent.parentWidget()
        return None

    def _update_insitu_workflow_button_visibility(self):
        try:
            button = getattr(self, "_insitu_workflow_button", None)
            if button is not None:
                button.setVisible(getattr(self, "load_mode", "Single") == "In-situ")
                for widget in (
                    button.parentWidget(),
                    getattr(self.ui, "gisaxsInputBox", None),
                    getattr(self.ui, "gisaxsInputStackWidget", None),
                ):
                    if widget is None:
                        continue
                    layout = widget.layout()
                    if layout is not None:
                        layout.invalidate()
                    widget.updateGeometry()
        except Exception:
            pass

    def _insitu_workflow_parent_widget(self):
        for candidate in (
            getattr(self, "main_window", None),
            getattr(self, "parent", None),
            getattr(self.ui, "centralwidget", None),
        ):
            if isinstance(candidate, QWidget):
                return candidate
        return None

    def _open_insitu_workflow_dialog(self):
        try:
            if (
                getattr(self, "_insitu_workflow_dialog", None) is not None
                and self._insitu_workflow_dialog.isVisible()
            ):
                self._insitu_workflow_dialog.raise_()
                self._insitu_workflow_dialog.activateWindow()
                return
            dialog = QDialog(self._insitu_workflow_parent_widget())
            dialog.setWindowTitle("In-situ Workflow")
            dialog.resize(1180, 760)
            dialog.setModal(False)
            dialog.setAttribute(Qt.WA_DeleteOnClose, True)
            root = QHBoxLayout(dialog)

            left = QWidget(dialog)
            left_layout = QVBoxLayout(left)
            left_layout.setContentsMargins(0, 0, 8, 0)
            root.addWidget(left, 0)

            mode_grid = QGridLayout()
            run_mode = QComboBox(dialog)
            run_mode.addItems(["Process Existing Sequence", "Live Watch"])
            mode_grid.addWidget(QLabel("Run Mode:", dialog), 0, 0)
            mode_grid.addWidget(run_mode, 0, 1)
            left_layout.addLayout(mode_grid)

            workflow_grid = QGridLayout()
            auto_show = QCheckBox("Auto Show latest image", dialog)
            auto_cut = QCheckBox("Auto Cut using current Cut Line settings", dialog)
            auto_fit = QCheckBox("Auto Fit using current fitting model", dialog)
            auto_show.setObjectName("insituWorkflowAutoShowCheckBox")
            auto_cut.setObjectName("insituWorkflowAutoCutCheckBox")
            auto_fit.setObjectName("insituWorkflowAutoFitCheckBox")
            workflow_grid.addWidget(auto_show, 0, 0, 1, 2)
            workflow_grid.addWidget(auto_cut, 1, 0, 1, 2)
            workflow_grid.addWidget(auto_fit, 2, 0, 1, 2)

            use_previous = QCheckBox("Use previous fit result as next initial guess", dialog)
            use_previous.setChecked(True)
            full_auto_fit = QCheckBox("Full Auto Fit", dialog)
            auto_refine = QCheckBox("Auto Refine", dialog)
            workflow_grid.addWidget(use_previous, 3, 0, 1, 2)
            workflow_grid.addWidget(full_auto_fit, 4, 0)
            workflow_grid.addWidget(auto_refine, 4, 1)
            insitu_profile = QComboBox(dialog)
            insitu_profile.addItems(list(_ai_catalog(self).profile_names()))
            insitu_profile.setCurrentText(
                str(self._ai_run_settings().get("profile", _ai_catalog(self).default_profile_name))
            )
            workflow_grid.addWidget(QLabel("AI profile:", dialog), 5, 0)
            workflow_grid.addWidget(insitu_profile, 5, 1)
            left_layout.addLayout(workflow_grid)

            watch_grid = QGridLayout()
            live_settings_widget = QWidget(dialog)
            live_settings_layout = QGridLayout(live_settings_widget)
            live_settings_layout.setContentsMargins(0, 0, 0, 0)
            poll = QDoubleSpinBox(dialog)
            poll.setRange(0.2, 3600.0)
            poll.setDecimals(1)
            poll.setSingleStep(0.5)
            poll.setValue(2.0)
            fit_every = QSpinBox(dialog)
            fit_every.setRange(1, 100000)
            fit_every.setValue(1)
            ui_every = QSpinBox(dialog)
            ui_every.setRange(1, 100000)
            ui_every.setValue(5)
            ui_every.setToolTip("Refresh heavy plots every N processed files during workflow runs.")
            stable = QCheckBox("Wait until file size is stable", dialog)
            stable.setChecked(True)
            live_settings_layout.addWidget(QLabel("Poll interval:", dialog), 0, 0)
            live_settings_layout.addWidget(poll, 0, 1)
            live_settings_layout.addWidget(QLabel("s", dialog), 0, 2)
            live_settings_layout.addWidget(stable, 1, 0, 1, 3)
            watch_grid.addWidget(live_settings_widget, 0, 0, 1, 3)
            watch_grid.addWidget(QLabel("Stack N files per fit:", dialog), 1, 0)
            watch_grid.addWidget(fit_every, 1, 1)
            watch_grid.addWidget(QLabel("UI update every N files:", dialog), 1, 2)
            watch_grid.addWidget(ui_every, 1, 3)

            sequence_settings_widget = QWidget(dialog)
            sequence_settings_layout = QGridLayout(sequence_settings_widget)
            sequence_settings_layout.setContentsMargins(0, 0, 0, 0)
            sequence_folder = QLineEdit(dialog)
            sequence_folder.setPlaceholderText("Current image folder")
            sequence_browse = QPushButton("Browse", dialog)
            sequence_pattern = QLineEdit("*.cbf", dialog)
            sequence_start = QSpinBox(dialog)
            sequence_end = QSpinBox(dialog)
            sequence_step = QSpinBox(dialog)
            for spin in (sequence_start, sequence_end):
                spin.setRange(0, 100000000)
                spin.setSpecialValueText("Auto")
                spin.setValue(0)
            sequence_step.setRange(1, 1000000)
            sequence_step.setValue(1)
            sequence_settings_layout.addWidget(QLabel("Folder:", dialog), 0, 0)
            sequence_settings_layout.addWidget(sequence_folder, 0, 1)
            sequence_settings_layout.addWidget(sequence_browse, 0, 2)
            sequence_settings_layout.addWidget(QLabel("Pattern:", dialog), 1, 0)
            sequence_settings_layout.addWidget(sequence_pattern, 1, 1, 1, 2)
            sequence_settings_layout.addWidget(QLabel("Start index:", dialog), 2, 0)
            sequence_settings_layout.addWidget(sequence_start, 2, 1)
            sequence_settings_layout.addWidget(QLabel("End index:", dialog), 3, 0)
            sequence_settings_layout.addWidget(sequence_end, 3, 1)
            sequence_settings_layout.addWidget(QLabel("Step:", dialog), 4, 0)
            sequence_settings_layout.addWidget(sequence_step, 4, 1)
            watch_grid.addWidget(sequence_settings_widget, 2, 0, 1, 3)
            left_layout.addLayout(watch_grid)

            button_row = QHBoxLayout()
            start_btn = QPushButton("Start Watch", dialog)
            process_btn = QPushButton("Start Process", dialog)
            pause_btn = QPushButton("Pause", dialog)
            stop_btn = QPushButton("Stop", dialog)
            trend_btn = QPushButton("Open Trend Monitor", dialog)
            heatmap_btn = QPushButton("Open Cut Heatmap", dialog)
            heatmap_btn.setToolTip("Show each completed auto-cut as one heatmap column.")
            export_btn = QPushButton("Export Results...", dialog)
            clear_cache_btn = QPushButton("Clear Session Cache", dialog)
            open_cache_btn = QPushButton("Open Cache Folder", dialog)
            pause_btn.setEnabled(False)
            stop_btn.setEnabled(False)
            button_row.addWidget(start_btn)
            button_row.addWidget(process_btn)
            button_row.addWidget(pause_btn)
            button_row.addWidget(stop_btn)
            left_layout.addLayout(button_row)
            output_row = QHBoxLayout()
            output_row.addWidget(heatmap_btn)
            output_row.addWidget(trend_btn)
            output_row.addWidget(export_btn)
            output_row.addWidget(clear_cache_btn)
            output_row.addWidget(open_cache_btn)
            left_layout.addLayout(output_row)

            status_grid = QGridLayout()
            status_names = [
                ("run_mode", "Run mode"),
                ("status", "Watch status"),
                ("file", "Current file"),
                ("processed", "Processed count"),
                ("failed", "Failed count"),
                ("queue", "Queue count"),
                ("fit", "Last fit status"),
                ("chi", "Last chi-square"),
                ("cache", "Cache path"),
            ]
            status_labels = {}
            for row, (key, label_text) in enumerate(status_names):
                status_grid.addWidget(QLabel(f"{label_text}:", dialog), row, 0)
                value = QLabel("Idle" if key == "status" else "-", dialog)
                value.setWordWrap(True)
                status_grid.addWidget(value, row, 1)
                status_labels[key] = value
            left_layout.addLayout(status_grid)

            log_browser = QTextBrowser(dialog)
            log_browser.setMinimumHeight(180)
            left_layout.addWidget(log_browser, 1)

            right = QWidget(dialog)
            right_layout = QVBoxLayout(right)
            right_layout.setContentsMargins(8, 0, 0, 0)
            root.addWidget(right, 1)

            image_label = QLabel("Current image: -", dialog)
            image_label.setWordWrap(True)
            right_layout.addWidget(image_label)
            image_canvas = self._make_insitu_workflow_canvas(right)
            curve_canvas = self._make_insitu_workflow_canvas(right)
            right_layout.addWidget(image_canvas, 2)
            right_layout.addWidget(curve_canvas, 2)

            self._insitu_workflow_dialog = dialog
            self._insitu_workflow_widgets = {
                "run_mode": run_mode,
                "auto_show": auto_show,
                "auto_cut": auto_cut,
                "auto_fit": auto_fit,
                "use_previous": use_previous,
                "full_auto_fit": full_auto_fit,
                "auto_refine": auto_refine,
                "profile": insitu_profile,
                "live_settings": live_settings_widget,
                "sequence_settings": sequence_settings_widget,
                "sequence_folder": sequence_folder,
                "sequence_browse": sequence_browse,
                "sequence_pattern": sequence_pattern,
                "sequence_start": sequence_start,
                "sequence_end": sequence_end,
                "sequence_step": sequence_step,
                "poll": poll,
                "fit_every": fit_every,
                "ui_every": ui_every,
                "stable": stable,
                "start": start_btn,
                "process": process_btn,
                "pause": pause_btn,
                "stop": stop_btn,
                "trend": trend_btn,
                "heatmap": heatmap_btn,
                "export": export_btn,
                "clear_cache": clear_cache_btn,
                "open_cache": open_cache_btn,
                "status_labels": status_labels,
                "log": log_browser,
                "image_label": image_label,
            }
            self._insitu_workflow_canvas_image = image_canvas
            self._insitu_workflow_canvas_curve = curve_canvas

            for checkbox in (
                auto_show,
                auto_cut,
                auto_fit,
                use_previous,
                full_auto_fit,
                auto_refine,
            ):
                checkbox.toggled.connect(self._refresh_insitu_workflow_step_styles)
            run_mode.currentTextChanged.connect(lambda _text: self._update_insitu_run_mode_ui())
            insitu_profile.currentTextChanged.connect(self._set_ai_profile)
            sequence_browse.clicked.connect(self._browse_insitu_sequence_folder)
            start_btn.clicked.connect(self._start_insitu_workflow)
            process_btn.clicked.connect(self._start_insitu_sequence_processing)
            pause_btn.clicked.connect(self._pause_insitu_workflow)
            stop_btn.clicked.connect(self._stop_insitu_workflow)
            trend_btn.clicked.connect(self._open_insitu_trend_monitor)
            heatmap_btn.clicked.connect(self._open_insitu_heatmap)
            export_btn.clicked.connect(self._export_insitu_workflow_results)
            clear_cache_btn.clicked.connect(self._clear_insitu_session_cache)
            open_cache_btn.clicked.connect(self._open_insitu_cache_folder)
            dialog.finished.connect(lambda _result: setattr(self, "_insitu_workflow_dialog", None))

            self._populate_insitu_sequence_folder_default()
            self._update_insitu_run_mode_ui()
            self._refresh_insitu_workflow_step_styles()
            self._refresh_insitu_workflow_status()
            dialog.show()
        except Exception as exc:
            self._add_fitting_error(f"Failed to open In-situ Workflow: {exc}")

    def _make_insitu_workflow_canvas(self, parent):
        holder = QWidget(parent)
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        if is_matplotlib_available():
            try:
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

                fig = Figure(figsize=(5.5, 3.1), dpi=80)
                canvas = FigureCanvas(fig)
                holder._insitu_figure = fig
                holder._insitu_canvas = canvas
                layout.addWidget(canvas)
                return holder
            except Exception:
                pass
        label = QLabel("Matplotlib preview unavailable", holder)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        return holder

    def _update_stack_controls_visibility(self):
        """No description."""
        try:
            base_widget = getattr(self.ui, "gisaxsInputStackValue", None)
            editor_widget = getattr(self.ui, "gisaxsInputStackEditorWidget", None)
            editor_layout = getattr(self.ui, "gisaxsInputStackEditorLayout", None)
            stack_widget = getattr(self.ui, "gisaxsInputStackWidget", None)
            stack_layout = stack_widget.layout() if stack_widget is not None else None
            try:
                from PyQt5.QtGui import QIntValidator, QRegularExpressionValidator
                from PyQt5.QtCore import QRegularExpression

                if self.load_mode == "In-situ":
                    if base_widget is not None:
                        base_widget.setVisible(False)
                    if editor_widget is not None:
                        editor_widget.setVisible(True)
                    if not hasattr(self, "_insitu_lineedit") or self._insitu_lineedit is None:
                        parent = None
                        try:
                            parent = (
                                editor_widget
                                if editor_widget is not None
                                else (
                                    base_widget.parent()
                                    if base_widget is not None
                                    else self.ui.gisaxsInputStackDisplayLabel.parent()
                                )
                            )
                        except Exception:
                            parent = None
                        from PyQt5.QtWidgets import QLineEdit as _QLE

                        self._insitu_lineedit = _QLE(parent)
                        try:
                            layout = (
                                editor_layout
                                if editor_layout is not None
                                else (parent.layout() if parent is not None else None)
                            )
                            if layout is not None:
                                try:
                                    disp_label = getattr(
                                        self.ui, "gisaxsInputStackDisplayLabel", None
                                    )
                                    if disp_label is not None:
                                        idx = layout.indexOf(disp_label)
                                        if idx >= 0:
                                            layout.insertWidget(idx, self._insitu_lineedit)
                                        else:
                                            layout.addWidget(self._insitu_lineedit)
                                    else:
                                        layout.addWidget(self._insitu_lineedit)
                                except Exception:
                                    layout.addWidget(self._insitu_lineedit)
                        except Exception:
                            pass
                        regex = QRegularExpression(r"^\s*(?:\d+|\d+\s*-\s*\d+|\d+\s*-)\s*$")
                        self._insitu_lineedit.setValidator(
                            QRegularExpressionValidator(regex, self._insitu_lineedit)
                        )
                        self._insitu_lineedit.setText("1-")
                        self._insitu_lineedit.setPlaceholderText("e.g. 1-, 1-10, 5")
                        try:
                            self._insitu_lineedit.returnPressed.connect(
                                self._on_stack_value_changed
                            )
                            self._insitu_lineedit.editingFinished.connect(
                                self._on_stack_value_changed
                            )
                        except Exception:
                            pass
                    self._insitu_lineedit.setVisible(True)
                else:
                    if hasattr(self, "_insitu_lineedit") and self._insitu_lineedit is not None:
                        self._insitu_lineedit.setVisible(False)
                    if base_widget is not None:
                        base_widget.setVisible(self.load_mode != "Single")
                        if editor_widget is not None:
                            editor_widget.setVisible(self.load_mode != "Single")
                        if self.load_mode == "Stack":
                            base_widget.setValidator(QIntValidator(1, 9999, base_widget))
                            base_widget.setPlaceholderText("e.g. 5")
                        else:
                            base_widget.setValidator(None)
                            base_widget.setPlaceholderText("")
            except Exception:
                if base_widget is not None:
                    base_widget.setVisible(self.load_mode != "Single")
                if editor_widget is not None:
                    editor_widget.setVisible(self.load_mode != "Single")
            if stack_layout is not None:
                top_margin = 0
                if self.load_mode == "Single":
                    combo = getattr(self.ui, "gisaxsInputModelCombox", None)
                    label = getattr(self.ui, "gisaxsInputStackDisplayLabel", None)
                    combo_height = combo.sizeHint().height() if combo is not None else 0
                    text_height = label.fontMetrics().height() if label is not None else 0
                    top_margin = max(0, (combo_height - text_height) // 2)
                left, _top, right, bottom = stack_layout.getContentsMargins()
                stack_layout.setContentsMargins(left, top_margin, right, bottom)
            if hasattr(self.ui, "gisaxsInputStackDisplayLabel"):
                self.ui.gisaxsInputStackDisplayLabel.setVisible(True)
        except Exception:
            pass

    def _enforce_insitu_visibility_once(self):
        """No description."""
        try:
            mode = getattr(self, "load_mode", "Single")
            base_widget = getattr(self.ui, "gisaxsInputStackValue", None)
            editor_widget = getattr(self.ui, "gisaxsInputStackEditorWidget", None)
            insitu_edit = getattr(self, "_insitu_lineedit", None)
            if mode == "In-situ":
                if base_widget is not None:
                    base_widget.setVisible(False)
                if editor_widget is not None:
                    editor_widget.setVisible(True)
                if insitu_edit is not None:
                    insitu_edit.setVisible(True)
            elif mode == "Stack":
                if insitu_edit is not None:
                    insitu_edit.setVisible(False)
                if base_widget is not None:
                    base_widget.setVisible(True)
                if editor_widget is not None:
                    editor_widget.setVisible(True)
            else:
                # Single
                if insitu_edit is not None:
                    insitu_edit.setVisible(False)
                if base_widget is not None:
                    base_widget.setVisible(False)
                if editor_widget is not None:
                    editor_widget.setVisible(False)
        except Exception:
            pass
