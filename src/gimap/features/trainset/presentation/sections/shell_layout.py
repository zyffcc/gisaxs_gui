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

from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.app.presentation.theme import set_state

from ..value_combos import fill_values

# Numbers and choices keep a natural width; only text fields (paths, hosts) stretch with the form.
NUMBER_FIELD_MAX_WIDTH = 240


class ShellLayoutMixin:
    """Own the shell layout section."""

    def _bind_shell(self) -> None:
        """Install dynamic workflow pages into the Python-owned shell."""
        self.validation_badge = self.validationBadge
        set_state(self.validation_badge, "state", "pending")  # pending / ok / warn / error
        # The steps: the shared StepRail (row API: setCurrentRow, currentRow, currentRowChanged) in a card.
        self.step_list = self.trainsetStepRail
        self.step_panel = self.trainsetStepList
        self.stack = self.trainsetWorkflowStack
        for index in range(len(self.STEPS)):
            self._show_step(index)
        self.step_list.currentRowChanged.connect(self._step_selected)

        for layout, page in (
            (self.datasetPageHostLayout, self._dataset_page()),
            (self.previewPageHostLayout, self._preview_page()),
            (self.modelPageHostLayout, self._model_page()),
            (self.runPageHostLayout, self._hpc_page()),
            (self.monitorPageHostLayout, self._monitor_page()),
        ):
            layout.addWidget(page)
        self._keep_fields_beside_labels()

        self.trainsetContentSplitter.setStretchFactor(1, 1)
        info = getattr(self, "design_info", None)
        if info is not None:
            # A long reference file name must not widen the preview pane (and so force the
            # stacked layout); the full path stays in the Reference file field.
            info.setMinimumWidth(120)
        for name in ("full_detector_canvas", "roi_design_canvas", "masked_design_canvas", "mask_only_canvas"):
            canvas = getattr(self, name, None)
            if canvas is not None:
                # The design preview shares a 1280 × 800 window with its hint, display bar and
                # file info: a smaller canvas scales the whole image down instead of being cut.
                canvas.setMinimumHeight(150)
        self._polish_workflow_shell()
        self.back_button.clicked.connect(
            lambda: self.step_list.setCurrentRow(max(0, self.step_list.currentRow() - 1))
        )
        self.step_list.setCurrentRow(0)
        QTimer.singleShot(0, self._apply_responsive_layout)
        QTimer.singleShot(80, self._apply_responsive_layout)

    def _keep_fields_beside_labels(self) -> None:
        """A capped number or choice field sits at the left of its cell, right beside its label: aligned, its
        column may still grow (the free width stays empty instead of widening the label column)."""
        for widget in self.fields.values():
            if isinstance(widget, (QSpinBox, QDoubleSpinBox, QComboBox)):
                parent = widget.parentWidget()
                layout = parent.layout() if parent is not None else None
                if layout is not None:
                    _align_left(layout, widget)

    def _polish_workflow_shell(self) -> None:
        """Clarify project actions without changing their connected commands."""
        self.pageTitle.setText("Trainset builder")
        self.pageSubtitle.setText(
            "Design a simulated GISAXS dataset, validate it locally, then prepare training jobs."
        )
        self.validate_button.setText("Validate design")
        self.validate_button.setToolTip("Check the detector, ROI, particles and sampling before generating anything")
        self.preview_button.setText("Open local preview")
        self.preview_button.setToolTip("Simulate a few images with these settings to see what the training set looks like")
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

    def set_step_state(self, index: int, state: str, **values) -> None:
        """The state line of step ``index`` in English ("Reference loaded"; a template with ``values``, as
        "Job {job}"): shown in the interface language, its colour from the state (step_rail.rail_state)."""
        if not 0 <= index < len(self.STEPS):
            return
        self._step_states[index] = str(state).format(**values) if values else str(state)
        self._step_values[index] = (str(state), dict(values))
        self._show_step(index)

    def _show_step(self, index: int) -> None:
        template, values = self._step_values[index] or (self._step_states[index], {})
        shown = trf(template, **values) if values else tr(template)
        self.step_list.set_row_state(index, self._step_states[index], shown)

    def set_validation_state(self, text: str, state: str) -> None:
        """Badge text (English, shown translated) and its colour state: pending, ok, warn or error."""
        self._validation_text = str(text)
        self.validation_badge.setText(tr(text))
        set_state(self.validation_badge, "state", state)

    def validation_state(self) -> str:
        return str(self.validation_badge.property("state") or "pending")

    def validation_text(self) -> str:
        """The badge text in English (what set_validation_state was given)."""
        return self._validation_text

    def set_design_stage_ready(self, index: int, ready: bool = True) -> None:
        if not 0 <= index < len(self._design_stage_ready):
            return
        self._design_stage_ready[index] = ready
        labels = ("Full detector", "ROI", "Masked image", "Mask only")
        self.design_tabs.setTabText(index, ("✓ " if ready else "") + tr(labels[index]))

    def step_states(self) -> list:
        """The state line of each workflow step (as given to set_step_state)."""
        return list(self._step_states)

    def step_entries(self) -> list:
        """Each step's (English template, values), so ``set_step_state(i, template, **values)`` shows it again
        in either language ("Job {job}" stays a template, not the composed "Job 4711")."""
        return [
            (entry[0], dict(entry[1])) if entry is not None else (state, {})
            for entry, state in zip(self._step_values, self._step_states)
        ]

    def design_stages_ready(self) -> list:
        return list(self._design_stage_ready)

    def _scroll(self, content: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.NoFrame)
        area.setWidget(content)
        return area

    def _spin(self, path: str, value: int, minimum: int = 0, maximum: int = 100000000) -> QSpinBox:
        widget = QSpinBox()
        widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        widget.setMaximumWidth(NUMBER_FIELD_MAX_WIDTH)
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
        widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        widget.setMaximumWidth(NUMBER_FIELD_MAX_WIDTH)
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
        widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        widget.setMaximumWidth(NUMBER_FIELD_MAX_WIDTH)
        widget.setMinimumWidth(90)
        fill_values(widget, values, current)  # the item data is the configuration value
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


def _align_left(layout, widget) -> bool:
    """Align ``widget`` left in ``layout`` or in one of its nested layouts; whether it was found."""
    if layout.setAlignment(widget, Qt.AlignLeft | Qt.AlignVCenter):
        widget.updateGeometry()  # the layout item caches its maximum size: drop it, the aligned item may grow
        return True
    for index in range(layout.count()):
        child = layout.itemAt(index).layout()
        if child is not None and _align_left(child, widget):
            return True
    return False
