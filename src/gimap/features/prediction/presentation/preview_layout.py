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

from .workflow_components import PredictionDisclosure


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

    def _build_display_section(
        self,
        panel: QWidget,
        *,
        auto_scale,
        log_scale,
        color_label,
        colormap,
        vmin_label,
        vmin,
        vmax_label,
        vmax,
        reset,
        disclosure_name: str,
    ) -> QFrame:
        section, layout = self._make_section("Display", panel)
        quick = QHBoxLayout()
        quick.setContentsMargins(0, 0, 0, 0)
        quick.setSpacing(8)
        quick.addWidget(auto_scale)
        quick.addWidget(log_scale)
        quick.addStretch(1)
        layout.addLayout(quick)

        cmap_row = QHBoxLayout()
        cmap_row.setContentsMargins(0, 0, 0, 0)
        cmap_row.addWidget(color_label)
        cmap_row.addWidget(colormap, 1)
        layout.addLayout(cmap_row)

        manual = PredictionDisclosure(
            "Manual color range",
            disclosure_name,
            section,
        )
        form = QWidget(manual.content)
        limits = QGridLayout(form)
        limits.setContentsMargins(0, 0, 0, 0)
        limits.setHorizontalSpacing(6)
        limits.setVerticalSpacing(6)
        limits.addWidget(vmin_label, 0, 0)
        limits.addWidget(vmin, 0, 1)
        limits.addWidget(vmax_label, 1, 0)
        limits.addWidget(vmax, 1, 1)
        limits.addWidget(reset, 2, 0, 1, 2)
        limits.setColumnStretch(1, 1)
        manual.add_widget(form)
        layout.addWidget(manual)
        return section

    def _build_zoom_section(self, panel: QWidget, buttons) -> QFrame:
        section, layout = self._make_section("Zoom", panel)
        row = QHBoxLayout()
        row.setSpacing(6)
        for button in buttons:
            normalize_button(button)
            button.setMinimumWidth(scale_value(76, self.profile, 68))
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            row.addWidget(button)
        layout.addLayout(row)
        return section

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
        self.ui.gisaxsImageColorScaleLabel.setParent(panel)
        self.ui.gisaxsImageColorScaleLabel.hide()
        panel.setMinimumWidth(scale_value(270, self.profile, 240))
        panel.setMaximumWidth(scale_value(340, self.profile, 310))
        panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        current_section, current_layout = self._make_section("Current", panel)
        current_row = QHBoxLayout()
        current_row.setContentsMargins(0, 0, 0, 0)
        current_row.setSpacing(6)
        current_row.addWidget(self.ui.gisaxsImageShowingLabel)
        current_row.addWidget(self.ui.gisaxsImageShowingValue, 1)
        current_layout.addLayout(current_row)

        scale_section = self._build_display_section(
            panel,
            auto_scale=self.ui.gisaxsImageAutoScaleCheckBox,
            log_scale=self.ui.gisaxsImageLogScaleCheckBox,
            color_label=self.ui.gisaxsImageColormapLabel,
            colormap=self.ui.gisaxsImageColormapCombox,
            vmin_label=self.ui.gisaxsImageVminLabel,
            vmin=self.ui.gisaxsImageVminValue,
            vmax_label=self.ui.gisaxsImageVmaxLabel,
            vmax=self.ui.gisaxsImageVmaxValue,
            reset=self.ui.gisaxsImageAutoScaleResetButton,
            disclosure_name="predictionInputManualRange",
        )
        zoom_section = self._build_zoom_section(
            panel,
            (
                self.ui.gisaxsImageZoomInButton,
                self.ui.gisaxsImageZoomOutButton,
                self.ui.gisaxsImageZoomResetButton,
            ),
        )

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
        self.ui.gisaxsPreviewCurrentSection = current_section
        self.ui.gisaxsPreviewDisplaySection = scale_section
        self.ui.gisaxsPreviewZoomSection = zoom_section

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
        self.ui.predict2dColorScaleLabel.setParent(panel)
        self.ui.predict2dColorScaleLabel.hide()
        panel.setMinimumWidth(scale_value(270, self.profile, 240))
        panel.setMaximumWidth(scale_value(340, self.profile, 310))
        panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        scale_section = self._build_display_section(
            panel,
            auto_scale=self.ui.predict2dAutoScaleCheckBox,
            log_scale=self.ui.predict2dLogScaleCheckBox,
            color_label=self.ui.predict2dColormapLabel,
            colormap=self.ui.predict2dLabelCombox,
            vmin_label=self.ui.predict2dVminLabel,
            vmin=self.ui.predict2dVminValue,
            vmax_label=self.ui.predict2dVmaxLabel,
            vmax=self.ui.predict2dVmaxValue,
            reset=self.ui.predict2dAutoScaleResetButton,
            disclosure_name="predictionResultManualRange",
        )
        zoom_section = self._build_zoom_section(
            panel,
            (
                self.ui.predict2dZoomInButton,
                self.ui.predict2dZoomOutButton,
                self.ui.predict2dZoomResetButton,
            ),
        )

        output_section, output_layout = self._make_section("Output", panel)
        self.predict2d_output_section = output_section
        normalize_button(self.ui.predict2dExportButton, wide=True)
        self.ui.predict2dExportButton.setMinimumWidth(scale_value(180, self.profile, 150))
        output_layout.addWidget(self.ui.predict2dExportButton)

        curve_section, curve_layout = self._make_section("Curve", panel)
        curve_layout.addWidget(self.ui.predict2dParameter1dpartWidget)
        curve_section.setVisible(False)

        panel_layout.addWidget(scale_section, 0, 0)
        panel_layout.addWidget(zoom_section, 1, 0)
        panel_layout.addWidget(output_section, 2, 0)
        panel_layout.addWidget(curve_section, 3, 0)
        panel_layout.setRowStretch(4, 1)
        self.ui.predict2dPreviewDisplaySection = scale_section
        self.ui.predict2dPreviewZoomSection = zoom_section
        self.ui.predict2dPreviewCurveSection = curve_section

        page_layout.addWidget(view, 0, 0)
        page_layout.addWidget(panel, 0, 1)
        page_layout.setColumnStretch(0, 1)
        page_layout.setColumnStretch(1, 0)


__all__ = ["PredictionPreviewLayout"]
