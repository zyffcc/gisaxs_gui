"""Feature-owned layout for Fitting detector input controls。"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QBoxLayout,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.presentation.layout_primitives import CARD_SPACING, FORM_ROW_SPACING, normalize_checkbox, normalize_input
from src.gimap.app.presentation.responsive_layout import current_profile, scale_value

from .layout_primitives import CardFrame, DisclosurePanel
from .layout_primitives import detach_from_parent_layout as _detach_from_parent_layout
from .layout_primitives import take_widget as _take_widget


class GisaxsInputCard(CardFrame):
    def __init__(self, ui, profile=None):
        super().__init__("Data source", "GisaxsInputCard")
        self.ui = ui
        content = ui.gisaxsInputBox
        profile = profile or current_profile(ui.centralwidget)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        content.setTitle("")
        if hasattr(content, "setFlat"):
            content.setFlat(True)
        content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self._rebuild_layout(content, profile)
        self.ui.gisaxsInputAutoShowCheckBox.setChecked(True)
        self.add_content(content)
        self.lock_to_natural_height()

    def _rebuild_layout(self, content: QWidget, profile) -> None:
        layout = content.layout()
        if layout is None:
            layout = QGridLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        if hasattr(layout, "setHorizontalSpacing"):
            layout.setHorizontalSpacing(CARD_SPACING)
            layout.setVerticalSpacing(FORM_ROW_SPACING)
        else:
            layout.setSpacing(CARD_SPACING)

        self._detach_input_widgets()
        self._rebuild_stack_widget(profile)

        file_section = self._create_section_widget("Detector data", content)
        file_row = QGridLayout()
        file_row.setContentsMargins(0, 0, 0, 0)
        file_row.setHorizontalSpacing(CARD_SPACING)
        file_row.setVerticalSpacing(max(6, FORM_ROW_SPACING - 2))
        self._configure_file_controls(profile)
        self.ui.gisaxsInputImportButton.setProperty("gimapPrimaryAction", True)
        self.ui.gisaxsInputImportButtonValue.setPlaceholderText(
            "Choose a CBF, NXS, TIFF, or supported detector file"
        )
        self.ui.gisaxsInputFileNavigationWidget = QWidget(file_section)
        self.ui.gisaxsInputFileNavigationWidget.setObjectName("gisaxsInputFileNavigationWidget")
        self.ui.gisaxsInputFileNavigationLayout = QHBoxLayout(
            self.ui.gisaxsInputFileNavigationWidget
        )
        self.ui.gisaxsInputFileNavigationLayout.setContentsMargins(0, 0, 0, 0)
        self.ui.gisaxsInputFileNavigationLayout.setSpacing(max(6, CARD_SPACING - 2))
        self.ui.gisaxsInputFileNavigationWidget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.ui.gisaxsInputShowButton.setToolTip(
            "Reload the current detector file and refresh the preview"
        )
        self.ui.gisaxsInputShowButton.setText("Show")
        self.ui.gisaxsInputAutoShowCheckBox.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Fixed
        )
        self.ui.gisaxsInputFileNavigationLayout.addWidget(
            self.ui.gisaxsInputAutoShowCheckBox
        )
        self.ui.gisaxsInputFileNavigationLayout.addWidget(self.ui.gisaxsInputShowButton)
        file_row.addWidget(self.ui.gisaxsInputImportButton, 0, 0)
        file_row.addWidget(self.ui.gisaxsInputImportButtonValue, 0, 1)
        file_row.addWidget(
            self.ui.gisaxsInputFileNavigationWidget,
            1,
            0,
            1,
            2,
            Qt.AlignRight,
        )
        file_row.setColumnStretch(0, 0)
        file_row.setColumnStretch(1, 1)
        file_section.layout().addLayout(file_row)

        mode_section = self._create_section_widget("Series mode", content)
        mode_grid = QGridLayout()
        mode_grid.setContentsMargins(0, 0, 0, 0)
        mode_grid.setHorizontalSpacing(CARD_SPACING)
        mode_grid.setVerticalSpacing(max(4, FORM_ROW_SPACING - 2))
        normalize_input(self.ui.gisaxsInputModelCombox)
        self.ui.gisaxsInputModelCombox.setMinimumWidth(scale_value(110, profile, 92))
        self.ui.gisaxsInputModelCombox.setMaximumWidth(scale_value(150, profile, 132))
        self.ui.gisaxsInputModelCombox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        mode_grid.addWidget(self.ui.gisaxsInputModelCombox, 0, 0, Qt.AlignTop)
        mode_grid.addWidget(self.ui.gisaxsInputStackWidget, 0, 1, Qt.AlignTop)
        mode_grid.setColumnStretch(0, 0)
        mode_grid.setColumnStretch(1, 1)
        mode_section.layout().addLayout(mode_grid)

        if not hasattr(self.ui, "gisaxsInputFlipUdCheckBox"):
            self.ui.gisaxsInputFlipUdCheckBox = QCheckBox("Flip UD", content)
            self.ui.gisaxsInputFlipUdCheckBox.setObjectName("gisaxsInputFlipUdCheckBox")
            self.ui.gisaxsInputFlipUdCheckBox.setToolTip(
                "Flip the loaded detector input vertically for display and all downstream processing."
            )
        normalize_checkbox(self.ui.gisaxsInputFlipUdCheckBox)
        self.ui.gisaxsInputFlipUdCheckBox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        scale_section = self._create_section_widget("Display Range", content)
        scale_grid = QGridLayout()
        scale_grid.setContentsMargins(0, 0, 0, 0)
        scale_grid.setHorizontalSpacing(CARD_SPACING)
        scale_grid.setVerticalSpacing(FORM_ROW_SPACING)
        self.ui.gisaxsInputAutoScaleCheckBox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.ui.gisaxsInputIntLogCheckBox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        normalize_input(self.ui.gisaxsInputVminValue)
        normalize_input(self.ui.gisaxsInputVmaxValue)
        if not hasattr(self.ui, "gisaxsInputShowCutRegionCheckBox"):
            self.ui.gisaxsInputShowCutRegionCheckBox = QCheckBox("Cut Region", scale_section)
            self.ui.gisaxsInputShowCutRegionCheckBox.setObjectName(
                "gisaxsInputShowCutRegionCheckBox"
            )
        if not hasattr(self.ui, "gisaxsInputShowCenterCheckBox"):
            self.ui.gisaxsInputShowCenterCheckBox = QCheckBox("Center", scale_section)
            self.ui.gisaxsInputShowCenterCheckBox.setObjectName("gisaxsInputShowCenterCheckBox")
        if not hasattr(self.ui, "gisaxsInputColormapCombo"):
            self.ui.gisaxsInputColormapCombo = QComboBox(scale_section)
            self.ui.gisaxsInputColormapCombo.setObjectName("gisaxsInputColormapCombo")
        if not hasattr(self.ui, "gisaxsInputMirrorGapFillCheckBox"):
            self.ui.gisaxsInputMirrorGapFillCheckBox = QCheckBox(
                "Mirror-fill detector gaps", scale_section
            )
            self.ui.gisaxsInputMirrorGapFillCheckBox.setObjectName(
                "gisaxsInputMirrorGapFillCheckBox"
            )
            self.ui.gisaxsInputMirrorGapFillCheckBox.setToolTip(
                "Fill detector gap pixels using left-right symmetry around the beam center. Raw data is preserved."
            )
        if not hasattr(self.ui, "gisaxsInputMirrorGapMarginLabel"):
            self.ui.gisaxsInputMirrorGapMarginLabel = QLabel("Gap margin:", scale_section)
            self.ui.gisaxsInputMirrorGapMarginLabel.setObjectName("gisaxsInputMirrorGapMarginLabel")
        if not hasattr(self.ui, "gisaxsInputMirrorGapMarginSpinBox"):
            self.ui.gisaxsInputMirrorGapMarginSpinBox = QSpinBox(scale_section)
            self.ui.gisaxsInputMirrorGapMarginSpinBox.setObjectName(
                "gisaxsInputMirrorGapMarginSpinBox"
            )
            self.ui.gisaxsInputMirrorGapMarginSpinBox.setRange(0, 20)
            self.ui.gisaxsInputMirrorGapMarginSpinBox.setSingleStep(1)
            self.ui.gisaxsInputMirrorGapMarginSpinBox.setValue(0)
            self.ui.gisaxsInputMirrorGapMarginSpinBox.setToolTip(
                "Extra pixels on each horizontal side of detector gaps to mirror-fill."
            )
        if not hasattr(self.ui, "gisaxsInputMirrorGapMarginUnitLabel"):
            self.ui.gisaxsInputMirrorGapMarginUnitLabel = QLabel("px", scale_section)
            self.ui.gisaxsInputMirrorGapMarginUnitLabel.setObjectName(
                "gisaxsInputMirrorGapMarginUnitLabel"
            )
        if not hasattr(self.ui, "gisaxsInputThresholdMaskCheckBox"):
            self.ui.gisaxsInputThresholdMaskCheckBox = QCheckBox("Threshold Mask", scale_section)
            self.ui.gisaxsInputThresholdMaskCheckBox.setObjectName(
                "gisaxsInputThresholdMaskCheckBox"
            )
            self.ui.gisaxsInputThresholdMaskCheckBox.setToolTip(
                "Exclude NaN values and intensities outside the lower/upper thresholds from all processing."
            )
        if not hasattr(self.ui, "gisaxsInputThresholdMinLabel"):
            self.ui.gisaxsInputThresholdMinLabel = QLabel("Mask lower:", scale_section)
            self.ui.gisaxsInputThresholdMinLabel.setObjectName("gisaxsInputThresholdMinLabel")
        if not hasattr(self.ui, "gisaxsInputThresholdMinSpinBox"):
            self.ui.gisaxsInputThresholdMinSpinBox = QDoubleSpinBox(scale_section)
            self.ui.gisaxsInputThresholdMinSpinBox.setObjectName("gisaxsInputThresholdMinSpinBox")
            self.ui.gisaxsInputThresholdMinSpinBox.setRange(-1e12, 1e12)
            self.ui.gisaxsInputThresholdMinSpinBox.setDecimals(6)
            self.ui.gisaxsInputThresholdMinSpinBox.setValue(-1e12)
            self.ui.gisaxsInputThresholdMinSpinBox.setToolTip(
                "Values below this threshold are excluded."
            )
        if not hasattr(self.ui, "gisaxsInputThresholdMaxLabel"):
            self.ui.gisaxsInputThresholdMaxLabel = QLabel("Mask upper:", scale_section)
            self.ui.gisaxsInputThresholdMaxLabel.setObjectName("gisaxsInputThresholdMaxLabel")
        if not hasattr(self.ui, "gisaxsInputThresholdMaxSpinBox"):
            self.ui.gisaxsInputThresholdMaxSpinBox = QDoubleSpinBox(scale_section)
            self.ui.gisaxsInputThresholdMaxSpinBox.setObjectName("gisaxsInputThresholdMaxSpinBox")
            self.ui.gisaxsInputThresholdMaxSpinBox.setRange(-1e12, 1e12)
            self.ui.gisaxsInputThresholdMaxSpinBox.setDecimals(6)
            self.ui.gisaxsInputThresholdMaxSpinBox.setValue(1e12)
            self.ui.gisaxsInputThresholdMaxSpinBox.setToolTip(
                "Values above this threshold are excluded."
            )
        self.ui.gisaxsInputShowCutRegionCheckBox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.ui.gisaxsInputShowCenterCheckBox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.ui.gisaxsInputMirrorGapFillCheckBox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.ui.gisaxsInputMirrorGapMarginLabel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.ui.gisaxsInputMirrorGapMarginSpinBox.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Fixed
        )
        self.ui.gisaxsInputMirrorGapMarginUnitLabel.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Fixed
        )
        self.ui.gisaxsInputThresholdMaskCheckBox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.ui.gisaxsInputThresholdMinLabel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.ui.gisaxsInputThresholdMinSpinBox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.ui.gisaxsInputThresholdMaxLabel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.ui.gisaxsInputThresholdMaxSpinBox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        normalize_checkbox(self.ui.gisaxsInputShowCutRegionCheckBox)
        normalize_checkbox(self.ui.gisaxsInputShowCenterCheckBox)
        normalize_checkbox(self.ui.gisaxsInputMirrorGapFillCheckBox)
        normalize_checkbox(self.ui.gisaxsInputThresholdMaskCheckBox)
        normalize_input(self.ui.gisaxsInputColormapCombo)
        normalize_input(self.ui.gisaxsInputMirrorGapMarginSpinBox)
        normalize_input(self.ui.gisaxsInputThresholdMinSpinBox)
        normalize_input(self.ui.gisaxsInputThresholdMaxSpinBox)
        self.ui.gisaxsInputVminLabel.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.ui.gisaxsInputVmaxLabel.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.ui.gisaxsInputMirrorGapMarginLabel.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.ui.gisaxsInputMirrorGapMarginUnitLabel.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.ui.gisaxsInputThresholdMinLabel.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.ui.gisaxsInputThresholdMaxLabel.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        scale_grid.addWidget(self.ui.gisaxsInputAutoScaleCheckBox, 0, 0)
        scale_grid.addWidget(self.ui.gisaxsInputVminLabel, 0, 1)
        scale_grid.addWidget(self.ui.gisaxsInputVminValue, 0, 2)
        scale_grid.addWidget(self.ui.gisaxsInputVmaxLabel, 0, 3)
        scale_grid.addWidget(self.ui.gisaxsInputVmaxValue, 0, 4)
        scale_grid.addWidget(self.ui.gisaxsInputIntLogCheckBox, 0, 5)
        scale_grid.addWidget(self.ui.gisaxsInputShowCutRegionCheckBox, 1, 0)
        scale_grid.addWidget(self.ui.gisaxsInputShowCenterCheckBox, 1, 1)
        scale_grid.addWidget(QLabel("Color Map:", scale_section), 1, 2)
        scale_grid.addWidget(self.ui.gisaxsInputColormapCombo, 1, 3, 1, 2)
        scale_grid.addWidget(self.ui.gisaxsInputMirrorGapFillCheckBox, 2, 0, 1, 2)
        scale_grid.addWidget(self.ui.gisaxsInputMirrorGapMarginLabel, 2, 2)
        scale_grid.addWidget(self.ui.gisaxsInputMirrorGapMarginSpinBox, 2, 3)
        scale_grid.addWidget(self.ui.gisaxsInputMirrorGapMarginUnitLabel, 2, 4)
        scale_grid.addWidget(self.ui.gisaxsInputThresholdMaskCheckBox, 3, 0, 1, 2)
        scale_grid.addWidget(self.ui.gisaxsInputThresholdMinLabel, 3, 2)
        scale_grid.addWidget(self.ui.gisaxsInputThresholdMinSpinBox, 3, 3)
        scale_grid.addWidget(self.ui.gisaxsInputThresholdMaxLabel, 3, 4)
        scale_grid.addWidget(self.ui.gisaxsInputThresholdMaxSpinBox, 3, 5)
        scale_grid.setColumnStretch(6, 1)
        scale_section.layout().addLayout(scale_grid)

        # Display and preprocessing controls live beside the detector preview.
        # This hidden host only survives until DetectorDisplayInspector reparents
        # the legacy controls; it is never exposed as an empty disclosure.
        scale_section.hide()
        cache_section = self._create_section_widget("", content)
        self.ui.gisaxsRemoteCacheControlsHost = cache_section
        self.ui.gisaxsRemoteCacheControlsLayout = cache_section.layout()
        cache_disclosure = DisclosurePanel(
            "Remote file cache",
            "fittingRemoteCacheDisclosure",
            content,
        )
        cache_disclosure.add_widget(cache_section)
        self.ui.fittingRemoteCacheDisclosure = cache_disclosure

        layout.addWidget(file_section, 0, 0, 1, 4)
        layout.addWidget(mode_section, 1, 0, 1, 4)
        layout.addWidget(cache_disclosure, 2, 0, 1, 4)
        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 0)
        layout.setColumnStretch(3, 0)

    def _detach_input_widgets(self) -> None:
        widgets = (
            self.ui.gisaxsInputImportButton,
            self.ui.gisaxsInputImportButtonValue,
            self.ui.gisaxsInputModelCombox,
            self.ui.gisaxsInputStackWidget,
            self.ui.gisaxsInputColorScaleLabel,
            self.ui.gisaxsInputAutoScaleCheckBox,
            self.ui.gisaxsInputIntLogCheckBox,
            self.ui.gisaxsInputAutoShowCheckBox,
            self.ui.gisaxsInputShowButton,
            self.ui.gisaxsInputVminLabel,
            self.ui.gisaxsInputVminValue,
            self.ui.gisaxsInputVmaxLabel,
            self.ui.gisaxsInputVmaxValue,
        )
        for widget in widgets:
            _detach_from_parent_layout(widget)
        self.ui.gisaxsInputColorScaleLabel.hide()

    def _rebuild_stack_widget(self, profile) -> None:
        stack_layout = self.ui.gisaxsInputStackWidget.layout()
        if isinstance(stack_layout, QBoxLayout):
            _take_widget(stack_layout, self.ui.gisaxsInputStackValue)
            _take_widget(stack_layout, self.ui.gisaxsInputStackDisplayLabel)
            stack_layout.setDirection(QBoxLayout.TopToBottom)
            stack_layout.setContentsMargins(0, 0, 0, 0)
            stack_layout.setSpacing(max(4, CARD_SPACING - 2))

        editor_widget = getattr(self.ui, "gisaxsInputStackEditorWidget", None)
        if editor_widget is None:
            editor_widget = QWidget(self.ui.gisaxsInputStackWidget)
            editor_widget.setObjectName("gisaxsInputStackEditorWidget")
            editor_layout = QHBoxLayout(editor_widget)
            editor_layout.setContentsMargins(0, 0, 0, 0)
            editor_layout.setSpacing(max(4, CARD_SPACING - 2))
            self.ui.gisaxsInputStackEditorWidget = editor_widget
            self.ui.gisaxsInputStackEditorLayout = editor_layout
        else:
            editor_layout = getattr(self.ui, "gisaxsInputStackEditorLayout", None)
            if editor_layout is None:
                editor_layout = QHBoxLayout(editor_widget)
                editor_layout.setContentsMargins(0, 0, 0, 0)
                editor_layout.setSpacing(max(4, CARD_SPACING - 2))
                self.ui.gisaxsInputStackEditorLayout = editor_layout

        if isinstance(editor_layout, QBoxLayout):
            _take_widget(editor_layout, self.ui.gisaxsInputStackValue)

        normalize_input(self.ui.gisaxsInputStackValue)
        self.ui.gisaxsInputStackValue.setMinimumWidth(scale_value(120, profile, 96))
        self.ui.gisaxsInputStackValue.setMaximumWidth(scale_value(156, profile, 136))
        self.ui.gisaxsInputStackValue.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.ui.gisaxsInputStackDisplayLabel.setWordWrap(True)
        self.ui.gisaxsInputStackDisplayLabel.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.ui.gisaxsInputStackDisplayLabel.setMinimumHeight(scale_value(36, profile, 30))
        self.ui.gisaxsInputStackDisplayLabel.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Minimum
        )
        self.ui.gisaxsInputStackDisplayLabel.setStyleSheet("color: #64748b;")
        self.ui.gisaxsInputStackWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        if isinstance(stack_layout, QBoxLayout):
            if isinstance(editor_layout, QBoxLayout):
                editor_layout.addWidget(self.ui.gisaxsInputStackValue, 0, Qt.AlignLeft)
            stack_layout.addWidget(editor_widget, 0)
            stack_layout.addWidget(self.ui.gisaxsInputStackDisplayLabel, 0, Qt.AlignTop)
            stack_layout.addStretch(1)

    def _configure_file_controls(self, profile) -> None:
        normalize_input(self.ui.gisaxsInputImportButtonValue)
        self.ui.gisaxsInputImportButtonValue.setMinimumWidth(scale_value(260, profile, 220))
        self.ui.gisaxsInputImportButtonValue.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    @staticmethod
    def _create_section_widget(title: str, parent: QWidget) -> QWidget:
        section = QWidget(parent)
        section_layout = QVBoxLayout(section)
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.setSpacing(max(6, CARD_SPACING - 2))
        if title:
            label = QLabel(title, section)
            label.setStyleSheet("font-size: 11px; font-weight: 600; color: #64748b;")
            section_layout.addWidget(label)
        return section


__all__ = ["GisaxsInputCard"]
