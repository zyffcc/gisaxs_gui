"""Shell Layout section for the Trainset page."""

from __future__ import annotations

from typing import Dict, Optional


from PyQt5.QtCore import QTimer, Qt


from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QWidget,
)


class ShellLayoutMixin:
    """Own the shell layout section."""

    def _bind_shell(self) -> None:
        """Install dynamic workflow pages into the Python-owned shell."""
        self.validation_badge = self.validationBadge
        self.step_list = self.trainsetStepList
        self.stack = self.trainsetWorkflowStack

        for index in range(self.step_list.count()):
            self.step_list.item(index).setData(Qt.UserRole, index)
        self.step_list.currentRowChanged.connect(self._step_selected)

        for layout, page in (
            (self.datasetPageHostLayout, self._dataset_page()),
            (self.previewPageHostLayout, self._preview_page()),
            (self.modelPageHostLayout, self._model_page()),
            (self.runPageHostLayout, self._hpc_page()),
            (self.monitorPageHostLayout, self._monitor_page()),
        ):
            layout.addWidget(page)

        self.trainsetContentSplitter.setStretchFactor(1, 1)
        self._polish_workflow_shell()
        self.back_button.clicked.connect(
            lambda: self.step_list.setCurrentRow(max(0, self.step_list.currentRow() - 1))
        )
        self.step_list.setCurrentRow(0)
        QTimer.singleShot(0, self._apply_responsive_layout)
        QTimer.singleShot(80, self._apply_responsive_layout)

    def _polish_workflow_shell(self) -> None:
        """Clarify project actions without changing their connected commands."""
        self.pageTitle.setText("Trainset builder")
        self.pageSubtitle.setText(
            "Design a simulated GISAXS dataset, validate it locally, then prepare training jobs."
        )
        self.validate_button.setText("Validate design")
        self.preview_button.setText("Open local preview")
        self.prepare_button.setText("Prepare job package")
        self.submit_button.setText("Maxwell (unavailable)")

        self.trainset_action_hint = QLabel("Start by validating the dataset design.", self)
        self.trainset_action_hint.setObjectName("trainsetActionHint")
        self.trainset_action_hint.setWordWrap(True)
        self.back_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        while self.trainsetActionsLayout.count():
            self.trainsetActionsLayout.takeAt(0)
        self.trainsetActionsLayout.addWidget(self.back_button)
        self.trainsetActionsLayout.addWidget(self.trainset_action_hint)
        self.trainsetActionsLayout.addStretch(1)
        self.trainsetActionsLayout.addLayout(self.trainsetActionGrid)

        while self.trainsetActionGrid.count():
            self.trainsetActionGrid.takeAt(0)
        for column, button in enumerate(
            (
                self.validate_button,
                self.preview_button,
                self.prepare_button,
                self.submit_button,
                self.load_button,
                self.save_button,
            )
        ):
            self.trainsetActionGrid.addWidget(button, 0, column)

    def set_step_state(self, index: int, state: str) -> None:
        if not 0 <= index < len(self.STEPS):
            return
        self._step_states[index] = state
        item = self.step_list.item(index)
        item.setText(f"{index + 1}.  {self.STEPS[index]}\n{state}")
        item.setToolTip(state)

    def set_design_stage_ready(self, index: int, ready: bool = True) -> None:
        if not 0 <= index < len(self._design_stage_ready):
            return
        self._design_stage_ready[index] = ready
        labels = ("Full detector", "ROI", "Masked image", "Mask only")
        self.design_tabs.setTabText(index, f"{'✓ ' if ready else ''}{labels[index]}")

    def _scroll(self, content: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.NoFrame)
        area.setWidget(content)
        return area

    def _spin(self, path: str, value: int, minimum: int = 0, maximum: int = 100000000) -> QSpinBox:
        widget = QSpinBox()
        widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        widget.setMinimumWidth(72)
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        self.fields[path] = widget
        return widget

    def _double(
        self,
        path: str,
        value: float,
        minimum: float = -1e12,
        maximum: float = 1e12,
        decimals: int = 6,
    ) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        widget.setMinimumWidth(82)
        widget.setRange(minimum, maximum)
        widget.setDecimals(decimals)
        widget.setValue(value)
        self.fields[path] = widget
        return widget

    def _line(self, path: str, value: str = "") -> QLineEdit:
        widget = QLineEdit(value)
        widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        widget.setMinimumWidth(90)
        self.fields[path] = widget
        return widget

    def _combo(self, path: str, values, current: Optional[str] = None) -> QComboBox:
        widget = QComboBox()
        widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        widget.setMinimumWidth(90)
        widget.addItems(list(values))
        if current and widget.findText(current) >= 0:
            widget.setCurrentText(current)
        self.fields[path] = widget
        return widget

    def _check(self, path: str, checked: bool = False, text: str = "") -> QCheckBox:
        widget = QCheckBox(text)
        widget.setChecked(checked)
        self.fields[path] = widget
        return widget

    def _make_display_bar(self, key: str) -> QWidget:
        bar = QWidget()
        bar.setProperty("displayBar", True)
        layout = QGridLayout(bar)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setHorizontalSpacing(7)
        layout.setVerticalSpacing(5)

        colormap = QComboBox()
        colormap.addItems(("gray", "viridis", "magma", "inferno", "plasma", "cividis", "turbo"))
        log_scale = QCheckBox("Log")
        auto_scale = QCheckBox("Auto")
        auto_scale.setChecked(True)
        vmin = QDoubleSpinBox()
        vmax = QDoubleSpinBox()
        for control, value in ((vmin, 0.0), (vmax, 1.0)):
            control.setRange(-1e30, 1e30)
            control.setDecimals(6)
            control.setValue(value)
            control.setMinimumWidth(92)
            control.setEnabled(False)

        # Two compact rows remain legible in the 340 px preview pane used at
        # 1280×720; a single row truncates labels and makes Vmin/Vmax overlap.
        layout.addWidget(QLabel("Colormap"), 0, 0)
        layout.addWidget(colormap, 0, 1)
        layout.addWidget(log_scale, 0, 2)
        layout.addWidget(auto_scale, 0, 3)
        layout.addWidget(QLabel("Vmin"), 1, 0)
        layout.addWidget(vmin, 1, 1)
        layout.addWidget(QLabel("Vmax"), 1, 2)
        layout.addWidget(vmax, 1, 3)
        layout.setColumnStretch(4, 1)

        controls: Dict[str, QWidget] = {
            "colormap": colormap,
            "log": log_scale,
            "auto": auto_scale,
            "vmin": vmin,
            "vmax": vmax,
        }
        self._display_controls[key] = controls
        setattr(self, f"{key}_display_colormap", colormap)
        setattr(self, f"{key}_display_log", log_scale)
        setattr(self, f"{key}_display_auto", auto_scale)
        setattr(self, f"{key}_display_vmin", vmin)
        setattr(self, f"{key}_display_vmax", vmax)

        def apply_display(*_args) -> None:
            automatic = auto_scale.isChecked()
            vmin.setEnabled(not automatic)
            vmax.setEnabled(not automatic)
            self._apply_display_settings(key)

        colormap.currentTextChanged.connect(apply_display)
        log_scale.toggled.connect(apply_display)
        auto_scale.toggled.connect(apply_display)
        vmin.valueChanged.connect(apply_display)
        vmax.valueChanged.connect(apply_display)
        self._apply_display_settings(key)
        return bar

    def _apply_display_settings(self, key: str) -> None:
        controls = self._display_controls.get(key)
        if not controls:
            return
        if key == "design":
            canvases = [
                self.full_detector_canvas,
                self.roi_design_canvas,
                self.masked_design_canvas,
                self.mask_only_canvas,
            ]
        elif key == "manual" and hasattr(self, "_what_if_canvas"):
            canvases = [self._what_if_canvas]
        else:
            canvases = list(self.preview_canvases.values())
            for copies in getattr(self, "impact_canvases", {}).values():
                canvases.extend(copies)
        for canvas in canvases:
            canvas.set_display_options(
                controls["colormap"].currentText(),
                controls["log"].isChecked(),
                controls["auto"].isChecked(),
                controls["vmin"].value(),
                controls["vmax"].value(),
            )
