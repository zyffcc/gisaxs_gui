"""Layout-only ownership of the Prediction GISAXS and Predict-2D previews。"""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.presentation.layout_primitives import normalize_button
from src.gimap.app.presentation.responsive_layout import scale_value


class PredictionPreviewLayout:
    """Reorganize generated preview controls without replacing them。"""

    def __init__(self, ui, profile) -> None:
        self.ui = ui
        self.profile = profile
        self.gisaxs_output_section: QFrame | None = None
        self.predict2d_output_section: QFrame | None = None

    def rebuild(self) -> None:
        self._rebuild_gisaxs_tab()
        self._rebuild_predict2d_tab()

    @staticmethod
    def _clear_layout(layout) -> None:
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            child_layout = item.layout()
            if child_layout is not None:
                PredictionPreviewLayout._clear_layout(child_layout)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

    @staticmethod
    def _make_section(title: str, parent: QWidget) -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame(parent)
        frame.setObjectName("predictPreviewSection")
        frame.setStyleSheet(
            """
            QFrame#predictPreviewSection {
                background: #f8fafc;
                border: 1px solid #dde5ef;
                border-radius: 8px;
            }
            """
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)
        label = QLabel(title, frame)
        label.setProperty("sectionTitle", True)
        layout.addWidget(label)
        return frame, layout

    def _rebuild_gisaxs_tab(self) -> None:
        tab = self.ui.gisaxsImageTab
        page_layout = tab.layout()
        if page_layout is None:
            page_layout = QGridLayout(tab)
        self._clear_layout(page_layout)

        view = self.ui.gisaxsImageGraphicsView
        panel = self.ui.gisaxsImageParametersWidget
        panel_layout = panel.layout()
        if panel_layout is None:
            panel_layout = QGridLayout(panel)
        self._clear_layout(panel_layout)
        panel.setMinimumWidth(scale_value(300, self.profile, 270))
        panel.setMaximumWidth(scale_value(360, self.profile, 330))
        panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        current_section, current_layout = self._make_section("Current", panel)
        current_row = QHBoxLayout()
        current_row.setContentsMargins(0, 0, 0, 0)
        current_row.setSpacing(6)
        current_row.addWidget(self.ui.gisaxsImageShowingLabel)
        current_row.addWidget(self.ui.gisaxsImageShowingValue, 1)
        current_layout.addLayout(current_row)

        scale_section, scale_layout = self._make_section("Display", panel)
        limits = QGridLayout()
        limits.setContentsMargins(0, 0, 0, 0)
        limits.setHorizontalSpacing(6)
        limits.setVerticalSpacing(6)
        limits.addWidget(self.ui.gisaxsImageVminLabel, 0, 0)
        limits.addWidget(self.ui.gisaxsImageVminValue, 0, 1)
        limits.addWidget(self.ui.gisaxsImageVmaxLabel, 1, 0)
        limits.addWidget(self.ui.gisaxsImageVmaxValue, 1, 1)
        scale_layout.addWidget(self.ui.gisaxsImageColorScaleLabel)
        scale_layout.addLayout(limits)
        checks = QHBoxLayout()
        checks.addWidget(self.ui.gisaxsImageAutoScaleCheckBox)
        checks.addWidget(self.ui.gisaxsImageLogScaleCheckBox)
        scale_layout.addLayout(checks)
        scale_layout.addWidget(self.ui.gisaxsImageAutoScaleResetButton)
        cmap_row = QHBoxLayout()
        cmap_row.addWidget(self.ui.gisaxsImageColormapLabel)
        cmap_row.addWidget(self.ui.gisaxsImageColormapCombox, 1)
        scale_layout.addLayout(cmap_row)

        zoom_section, zoom_layout = self._make_section("Zoom", panel)
        zoom_row = QHBoxLayout()
        zoom_row.setSpacing(6)
        for button in (
            self.ui.gisaxsImageZoomInButton,
            self.ui.gisaxsImageZoomOutButton,
            self.ui.gisaxsImageZoomResetButton,
        ):
            normalize_button(button)
            button.setMinimumWidth(scale_value(76, self.profile, 68))
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            zoom_row.addWidget(button)
        zoom_layout.addLayout(zoom_row)

        output_section, output_layout = self._make_section("Output", panel)
        self.gisaxs_output_section = output_section
        normalize_button(self.ui.gisaxsImageExportButton, wide=True)
        self.ui.gisaxsImageExportButton.setMinimumWidth(scale_value(180, self.profile, 150))
        output_layout.addWidget(self.ui.gisaxsImageExportButton)

        panel_layout.addWidget(current_section, 0, 0)
        panel_layout.addWidget(scale_section, 1, 0)
        panel_layout.addWidget(zoom_section, 2, 0)
        panel_layout.addWidget(output_section, 3, 0)
        panel_layout.setRowStretch(4, 1)

        page_layout.addWidget(view, 0, 0)
        page_layout.addWidget(panel, 0, 1)
        page_layout.setColumnStretch(0, 1)
        page_layout.setColumnStretch(1, 0)

    def _rebuild_predict2d_tab(self) -> None:
        tab = self.ui.predict2dImageTab
        page_layout = tab.layout()
        if page_layout is None:
            page_layout = QGridLayout(tab)
        self._clear_layout(page_layout)

        view = self.ui.predict2dGraphicsView
        panel = self.ui.predict2dParameterWidget
        panel_layout = panel.layout()
        if panel_layout is None:
            panel_layout = QGridLayout(panel)
        self._clear_layout(panel_layout)
        panel.setMinimumWidth(scale_value(300, self.profile, 270))
        panel.setMaximumWidth(scale_value(380, self.profile, 340))
        panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        scale_section, scale_layout = self._make_section("Display", panel)
        limits = QGridLayout()
        limits.setContentsMargins(0, 0, 0, 0)
        limits.setHorizontalSpacing(6)
        limits.setVerticalSpacing(6)
        limits.addWidget(self.ui.predict2dVminLabel, 0, 0)
        limits.addWidget(self.ui.predict2dVminValue, 0, 1)
        limits.addWidget(self.ui.predict2dVmaxLabel, 1, 0)
        limits.addWidget(self.ui.predict2dVmaxValue, 1, 1)
        scale_layout.addWidget(self.ui.predict2dColorScaleLabel)
        scale_layout.addLayout(limits)
        checks = QHBoxLayout()
        checks.addWidget(self.ui.predict2dAutoScaleCheckBox)
        checks.addWidget(self.ui.predict2dLogScaleCheckBox)
        scale_layout.addLayout(checks)
        scale_layout.addWidget(self.ui.predict2dAutoScaleResetButton)
        cmap_row = QHBoxLayout()
        cmap_row.addWidget(self.ui.predict2dColormapLabel)
        cmap_row.addWidget(self.ui.predict2dLabelCombox, 1)
        scale_layout.addLayout(cmap_row)

        zoom_section, zoom_layout = self._make_section("Zoom", panel)
        zoom_row = QHBoxLayout()
        zoom_row.setSpacing(6)
        for button in (
            self.ui.predict2dZoomInButton,
            self.ui.predict2dZoomOutButton,
            self.ui.predict2dZoomResetButton,
        ):
            normalize_button(button)
            button.setMinimumWidth(scale_value(76, self.profile, 68))
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            zoom_row.addWidget(button)
        zoom_layout.addLayout(zoom_row)

        output_section, output_layout = self._make_section("Output", panel)
        self.predict2d_output_section = output_section
        normalize_button(self.ui.predict2dExportButton, wide=True)
        self.ui.predict2dExportButton.setMinimumWidth(scale_value(180, self.profile, 150))
        output_layout.addWidget(self.ui.predict2dExportButton)

        curve_section, curve_layout = self._make_section("Curve", panel)
        curve_layout.addWidget(self.ui.predict2dParameter1dpartWidget)

        panel_layout.addWidget(scale_section, 0, 0)
        panel_layout.addWidget(zoom_section, 1, 0)
        panel_layout.addWidget(output_section, 2, 0)
        panel_layout.addWidget(curve_section, 3, 0)
        panel_layout.setRowStretch(4, 1)

        page_layout.addWidget(view, 0, 0)
        page_layout.addWidget(panel, 0, 1)
        page_layout.setColumnStretch(0, 1)
        page_layout.setColumnStretch(1, 0)


__all__ = ["PredictionPreviewLayout"]
