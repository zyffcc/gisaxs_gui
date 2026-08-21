"""Detector preview toolbar and display inspector widgets."""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.presentation.responsive_layout import current_profile, scale_value

from .layout_primitives import detach_from_parent_layout


class DetectorDisplayInspector(QFrame):
    """Compact home for detector display controls already owned by the UI."""

    def __init__(self, ui, parent: QWidget | None = None, profile=None):
        super().__init__(parent)
        profile = profile or current_profile(parent or ui.centralwidget)
        self.setObjectName("fittingDetectorDisplayInspector")
        self.setProperty("displayInspector", True)
        self.setMinimumWidth(scale_value(230, profile, 210))
        self.setMaximumWidth(scale_value(315, profile, 275))
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)
        title = QLabel("Image display", self)
        title.setObjectName("fittingDetectorDisplayInspectorTitle")
        title.setProperty("sectionTitle", True)
        layout.addWidget(title)

        for widget in (
            ui.gisaxsInputAutoScaleCheckBox,
            ui.gisaxsInputIntLogCheckBox,
            ui.gisaxsInputShowCutRegionCheckBox,
            ui.gisaxsInputShowCenterCheckBox,
            ui.gisaxsInputFlipUdCheckBox,
            ui.gisaxsInputVminLabel,
            ui.gisaxsInputVminValue,
            ui.gisaxsInputVmaxLabel,
            ui.gisaxsInputVmaxValue,
            ui.gisaxsInputColormapCombo,
            ui.gisaxsInputMirrorGapFillCheckBox,
            ui.gisaxsInputMirrorGapMarginLabel,
            ui.gisaxsInputMirrorGapMarginSpinBox,
            ui.gisaxsInputMirrorGapMarginUnitLabel,
            ui.gisaxsInputThresholdMaskCheckBox,
            ui.gisaxsInputThresholdMinLabel,
            ui.gisaxsInputThresholdMinSpinBox,
            ui.gisaxsInputThresholdMaxLabel,
            ui.gisaxsInputThresholdMaxSpinBox,
        ):
            detach_from_parent_layout(widget)

        quick_grid = QGridLayout()
        quick_grid.setContentsMargins(0, 0, 0, 0)
        quick_grid.setHorizontalSpacing(8)
        quick_grid.setVerticalSpacing(8)
        quick_grid.addWidget(ui.gisaxsInputAutoScaleCheckBox, 0, 0)
        quick_grid.addWidget(ui.gisaxsInputIntLogCheckBox, 0, 1)
        quick_grid.addWidget(ui.gisaxsInputShowCenterCheckBox, 1, 0)
        quick_grid.addWidget(ui.gisaxsInputShowCutRegionCheckBox, 1, 1)
        layout.addLayout(quick_grid)

        range_form = QFormLayout()
        range_form.setContentsMargins(0, 0, 0, 0)
        range_form.setHorizontalSpacing(8)
        range_form.setVerticalSpacing(7)
        range_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        range_form.addRow(ui.gisaxsInputVminLabel, ui.gisaxsInputVminValue)
        range_form.addRow(ui.gisaxsInputVmaxLabel, ui.gisaxsInputVmaxValue)
        range_form.addRow("Color map", ui.gisaxsInputColormapCombo)
        layout.addLayout(range_form)
        preprocessing_title = QLabel("Preprocessing", self)
        preprocessing_title.setObjectName("fittingDetectorPreprocessingTitle")
        preprocessing_title.setProperty("sectionTitle", True)
        layout.addWidget(preprocessing_title)
        layout.addWidget(self._build_preprocessing(ui))
        layout.addStretch(1)

    def _build_preprocessing(self, ui) -> QWidget:
        preprocessing = QWidget(self)
        preprocessing.setObjectName("fittingDetectorPreprocessing")
        grid = QGridLayout(preprocessing)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        grid.addWidget(ui.gisaxsInputFlipUdCheckBox, 0, 0, 1, 2)
        grid.addWidget(ui.gisaxsInputMirrorGapFillCheckBox, 1, 0, 1, 2)
        grid.addWidget(ui.gisaxsInputMirrorGapMarginLabel, 2, 0)
        margin_row = QWidget(preprocessing)
        margin_layout = QHBoxLayout(margin_row)
        margin_layout.setContentsMargins(0, 0, 0, 0)
        margin_layout.setSpacing(4)
        margin_layout.addWidget(ui.gisaxsInputMirrorGapMarginSpinBox)
        margin_layout.addWidget(ui.gisaxsInputMirrorGapMarginUnitLabel)
        grid.addWidget(margin_row, 2, 1)
        grid.addWidget(ui.gisaxsInputThresholdMaskCheckBox, 3, 0, 1, 2)
        grid.addWidget(ui.gisaxsInputThresholdMinLabel, 4, 0)
        grid.addWidget(ui.gisaxsInputThresholdMinSpinBox, 4, 1)
        grid.addWidget(ui.gisaxsInputThresholdMaxLabel, 5, 0)
        grid.addWidget(ui.gisaxsInputThresholdMaxSpinBox, 5, 1)
        ui.fittingDetectorPreprocessing = preprocessing
        return preprocessing


class DetectorToolBar(QFrame):
    """Discoverable detector interactions; commands remain in the binding."""

    def __init__(self, ui, inspector: DetectorDisplayInspector, parent=None):
        super().__init__(parent)
        self.setObjectName("fittingDetectorToolBar")
        self.setProperty("previewToolbar", True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        ui.fittingPickCenterButton = self._tool("Center", "fittingPickCenterButton")
        ui.fittingPickCenterButton.setToolTip(
            "Click one point in the preview to set the detector center"
        )
        ui.fittingSelectRegionButton = self._tool(
            "Region", "fittingSelectRegionButton"
        )
        ui.fittingSelectRegionButton.setToolTip(
            "Drag directly in the preview to define the cut region"
        )
        ui.fittingResetDetectorViewButton = QPushButton("Reset", self)
        ui.fittingResetDetectorViewButton.setObjectName("fittingResetDetectorViewButton")
        ui.fittingOpenDetectorWindowButton = QPushButton("Open", self)
        ui.fittingOpenDetectorWindowButton.setObjectName("fittingOpenDetectorWindowButton")
        ui.fittingOpenDetectorWindowButton.setToolTip(
            "Open the larger pan/zoom detector viewer"
        )
        ui.fittingDisplayInspectorButton = self._tool(
            "Display", "fittingDisplayInspectorButton"
        )
        ui.fittingDisplayInspectorButton.setChecked(not inspector.isHidden())
        ui.fittingDisplayInspectorButton.toggled.connect(inspector.setVisible)

        for widget in (
            ui.fittingPickCenterButton,
            ui.fittingSelectRegionButton,
            ui.fittingResetDetectorViewButton,
            ui.fittingOpenDetectorWindowButton,
        ):
            layout.addWidget(widget)
        layout.addStretch(1)
        ui.fittingDetectorToolHint = QLabel("Esc cancels a selection tool", self)
        ui.fittingDetectorToolHint.setObjectName("fittingDetectorToolHint")
        ui.fittingDetectorToolHint.setProperty("cardMeta", True)
        ui.fittingDetectorToolHint.setVisible(False)
        layout.addWidget(ui.fittingDetectorToolHint)
        layout.addWidget(ui.fittingDisplayInspectorButton)

    def _tool(self, text: str, name: str) -> QToolButton:
        button = QToolButton(self)
        button.setObjectName(name)
        button.setText(text)
        button.setCheckable(True)
        return button


__all__ = ["DetectorDisplayInspector", "DetectorToolBar"]
