"""Feature-owned detector, plot, plot-control and log cards。"""

from __future__ import annotations

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QFrame,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.presentation import CollapsibleCardFrame
from src.gimap.app.presentation.layout_primitives import (
    CARD_MARGIN,
    CARD_SPACING,
    FORM_ROW_SPACING,
    SECTION_MIN_WIDTH,
    normalize_checkbox,
    normalize_input,
)
from src.gimap.app.presentation.responsive_layout import current_profile, scale_value

from .layout_primitives import detach_from_parent_layout as _detach_from_parent_layout
from .layout_primitives import take_widget as _take_widget


class DetectorPreviewCard(CollapsibleCardFrame):
    def __init__(self, graphics_view: QGraphicsView, profile=None):
        super().__init__("Detector Preview", "DetectorPreviewCard", default_expanded=True)
        profile = profile or current_profile(graphics_view)
        self.setMinimumWidth(SECTION_MIN_WIDTH)
        self.setMinimumHeight(scale_value(260, profile, 210))
        hint = QLabel(
            "Drop a CBF, NXS, or TIFF file here to load it. Double-click to open a larger independent image window.",
            self,
        )
        hint.setObjectName("DetectorPreviewDoubleClickHint")
        hint.setProperty("cardMeta", True)
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #64748b;")
        self.add_content(hint)
        graphics_view.setToolTip(
            "Drop a CBF, NXS, or TIFF file here to load it. Double-click to open a larger independent image window."
        )
        graphics_view.setMinimumSize(scale_value(320, profile, 260), scale_value(240, profile, 190))
        graphics_view.setMaximumSize(16777215, 16777215)
        graphics_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.add_content(graphics_view, 1)
        base_height = scale_value(260, profile, 210)
        self.enable_content_resize(base_height, base_height * 2)


class PlotCanvasArea(QFrame):
    def __init__(self, graphics_view: QGraphicsView, parent: QWidget | None = None, profile=None):
        super().__init__(parent)
        profile = profile or current_profile(parent or graphics_view)
        self.setObjectName("plotCanvasContainer")
        self.setProperty("previewSection", True)
        self.setMinimumHeight(scale_value(260, profile, 200))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        graphics_view.setMinimumSize(scale_value(320, profile, 260), scale_value(260, profile, 200))
        graphics_view.setMaximumSize(16777215, 16777215)
        graphics_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(graphics_view, 1)


class SectionCard(QFrame):
    """Small card-style section with a QLabel title instead of QGroupBox title."""

    def __init__(
        self,
        title: str,
        object_name: str,
        parent: QWidget | None = None,
        fixed_height: int | None = None,
        profile=None,
    ):
        super().__init__(parent)
        profile = profile or current_profile(parent or self)
        self.setObjectName(object_name)
        self.setProperty("sectionCard", True)
        self.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed if fixed_height else QSizePolicy.Preferred
        )
        if fixed_height is not None:
            height = scale_value(fixed_height, profile, int(fixed_height * 0.78))
            self.setMinimumHeight(height)
            self.setMaximumHeight(height)

        self.section_layout = QVBoxLayout(self)
        self.section_layout.setContentsMargins(12, 10, 12, 12)
        self.section_layout.setSpacing(CARD_SPACING)

        self.title_label = QLabel(title, self)
        self.title_label.setObjectName(f"{object_name}Title")
        self.title_label.setProperty("sectionTitle", True)
        self.title_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.section_layout.addWidget(self.title_label)


class FittingRegionControl(SectionCard):
    def __init__(self, ui, parent: QWidget | None = None, profile=None):
        profile = profile or current_profile(parent or ui.centralwidget)
        super().__init__("Fitting Region", "FittingRegionControl", parent, profile=profile)

        for widget in (
            ui.fitFittingRegionLabel,
            ui.fitFittingRegionSlider,
            ui.fitFittingRegionMinValue,
            ui.fitFittingRegionMaxValue,
        ):
            _detach_from_parent_layout(widget)

        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(CARD_SPACING)
        layout.setVerticalSpacing(FORM_ROW_SPACING)
        self.section_layout.addLayout(layout)

        ui.fitFittingRegionLabel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        ui.fitFittingRegionSlider.setMinimumHeight(scale_value(28, profile, 24))
        ui.fitFittingRegionSlider.setMaximumHeight(scale_value(36, profile, 30))
        ui.fitFittingRegionSlider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        normalize_input(ui.fitFittingRegionMinValue)
        normalize_input(ui.fitFittingRegionMaxValue)

        filter_label = QLabel("Data Filter:", self)
        filter_label.setObjectName("fitRegionDataFilterLabel")
        filter_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        filter_widget = QWidget(self)
        filter_layout = QHBoxLayout(filter_widget)
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.setSpacing(CARD_SPACING)
        ui.fitRegionPositiveOnlyCheckBox = QCheckBox("Positive Only", filter_widget)
        ui.fitRegionPositiveOnlyCheckBox.setObjectName("fitRegionPositiveOnlyCheckBox")
        ui.fitRegionNegativeOnlyCheckBox = QCheckBox("Negative Only", filter_widget)
        ui.fitRegionNegativeOnlyCheckBox.setObjectName("fitRegionNegativeOnlyCheckBox")
        normalize_checkbox(ui.fitRegionPositiveOnlyCheckBox)
        normalize_checkbox(ui.fitRegionNegativeOnlyCheckBox)
        filter_layout.addWidget(ui.fitRegionPositiveOnlyCheckBox)
        filter_layout.addWidget(ui.fitRegionNegativeOnlyCheckBox)
        filter_layout.addStretch(1)

        hint_label = QLabel(
            "Select Positive Only or Negative Only to edit Fitting Region.",
            self,
        )
        hint_label.setObjectName("fitRegionEditHintLabel")
        hint_label.setWordWrap(True)
        hint_label.setStyleSheet(
            "QLabel {"
            "background: #eff6ff;"
            "border: 1px solid #bfdbfe;"
            "border-radius: 6px;"
            "color: #1d4ed8;"
            "padding: 6px 8px;"
            "line-height: 135%;"
            "}"
        )
        hint_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        hint_label.setMinimumHeight(scale_value(42, profile, 34))
        hint_label.setVisible(False)
        ui.fitRegionEditHintLabel = hint_label

        layout.addWidget(filter_label, 0, 0)
        layout.addWidget(filter_widget, 0, 1)
        layout.addWidget(hint_label, 1, 0, 1, 2)
        layout.addWidget(ui.fitFittingRegionLabel, 2, 0, 1, 2)
        layout.addWidget(ui.fitFittingRegionSlider, 3, 0, 1, 2)
        layout.addWidget(ui.fitFittingRegionMinValue, 4, 0)
        layout.addWidget(ui.fitFittingRegionMaxValue, 4, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        self.setMinimumHeight(
            max(self.minimumHeight(), self.sizeHint().height() + scale_value(10, profile, 8))
        )
        self.setMaximumHeight(16777215)


class PlotSamplingControl(SectionCard):
    def __init__(self, ui, parent: QWidget | None = None, profile=None):
        profile = profile or current_profile(parent or ui.centralwidget)
        super().__init__("Sampling", "PlotSamplingControl", parent, profile=profile)

        for widget in (
            ui.fitDataPointsNumLabel,
            ui.fitDataPointsNumValue,
            ui.fitInterpolationMethodLabel,
            ui.fitInterpolationMethodValue,
        ):
            _detach_from_parent_layout(widget)

        layout = QFormLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(CARD_SPACING)
        layout.setVerticalSpacing(FORM_ROW_SPACING)
        layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        layout.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.section_layout.addLayout(layout)

        for label in (ui.fitDataPointsNumLabel, ui.fitInterpolationMethodLabel):
            label.setMinimumWidth(130)
            label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        normalize_input(ui.fitDataPointsNumValue)
        normalize_input(ui.fitInterpolationMethodValue)

        layout.addRow(ui.fitDataPointsNumLabel, ui.fitDataPointsNumValue)
        layout.addRow(ui.fitInterpolationMethodLabel, ui.fitInterpolationMethodValue)


class ParticleOptionsLayout(QVBoxLayout):
    """Dynamic particle checkbox column used by the fitting controller."""

    def addWidget(
        self, widget: QWidget, stretch: int = 0, alignment: Qt.Alignment = Qt.Alignment()
    ) -> None:
        super().addWidget(widget, stretch, alignment)
        self._refresh_after_change(widget)

    def insertWidget(
        self,
        index: int,
        widget: QWidget,
        stretch: int = 0,
        alignment: Qt.Alignment = Qt.Alignment(),
    ) -> None:
        super().insertWidget(index, widget, stretch, alignment)
        self._refresh_after_change(widget)

    def _refresh_after_change(self, widget: QWidget | None = None) -> None:
        if isinstance(widget, QCheckBox):
            normalize_checkbox(widget)
            if not widget.property("plotOptionGeometryHooked"):
                widget.destroyed.connect(
                    lambda _=None: QTimer.singleShot(0, self._refresh_after_change)
                )
                widget.setProperty("plotOptionGeometryHooked", True)

        parent = self.parentWidget()
        while parent is not None:
            parent.updateGeometry()
            parent.adjustSize()
            if parent.objectName() in ("PlotPreviewCard", "FittingPlotControlsCard"):
                base_min_height = getattr(parent, "_base_min_height", parent.minimumHeight())
                parent.setMinimumHeight(max(base_min_height, parent.sizeHint().height()))
                parent.updateGeometry()
                break
            parent = parent.parentWidget()


class PlotOptionsControl(SectionCard):
    def __init__(self, ui, parent: QWidget | None = None, profile=None):
        super().__init__("Display Options", "PlotOptionsControl", parent, profile=profile)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        checkboxes = (
            ui.fitBGShowCheckBox,
            ui.fitParticle1ShowCheckBox,
            ui.fitResShowCheckBox,
            ui.fitParticle2ShowCheckBox,
            ui.fitParticle3ShowCheckBox,
        )
        for widget in (ui.fitDisplayOptionsLabel, *checkboxes):
            _detach_from_parent_layout(widget)
        ui.fitDisplayOptionsLabel.hide()

        for checkbox in checkboxes:
            normalize_checkbox(checkbox)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(32)
        grid.setVerticalSpacing(FORM_ROW_SPACING)
        self.section_layout.addLayout(grid)

        static_column = QWidget(self)
        static_column.setObjectName("plotStaticOptionsColumn")
        static_layout = QVBoxLayout(static_column)
        static_layout.setContentsMargins(0, 0, 0, 0)
        static_layout.setSpacing(FORM_ROW_SPACING)
        static_layout.addWidget(ui.fitBGShowCheckBox)
        static_layout.addWidget(ui.fitResShowCheckBox)
        static_layout.addStretch(1)

        particle_column = QWidget(self)
        particle_column.setObjectName("fitParticlesNumWidget")
        particle_column.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        particle_layout = ParticleOptionsLayout(particle_column)
        particle_layout.setContentsMargins(0, 0, 0, 0)
        particle_layout.setSpacing(FORM_ROW_SPACING)
        ui.fitParticlesNumWidget = particle_column

        particle_layout.addWidget(ui.fitParticle1ShowCheckBox)
        particle_layout.addWidget(ui.fitParticle2ShowCheckBox)
        particle_layout.addWidget(ui.fitParticle3ShowCheckBox)

        grid.addWidget(static_column, 0, 0)
        grid.addWidget(particle_column, 0, 1)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)


class PlotPreviewCard(CollapsibleCardFrame):
    def __init__(self, ui, content: QWidget, graphics_view: QGraphicsView, profile=None):
        super().__init__("Fitting Plot", "PlotPreviewCard", default_expanded=True)
        profile = profile or current_profile(content)
        self._base_min_height = scale_value(360, profile, 280)
        hint = QLabel(
            "Double-click the fitting plot to open a larger independent fit window.", self
        )
        hint.setObjectName("FittingPlotDoubleClickHint")
        hint.setProperty("cardMeta", True)
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #64748b;")
        self.add_content(hint)
        graphics_view.setToolTip("Double-click to open a larger independent fit window.")
        self._build_plot_layout(content, graphics_view, profile)

        self.setMinimumWidth(SECTION_MIN_WIDTH)
        self.setMinimumHeight(self._base_min_height)
        self.add_content(content, 1)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.enable_content_resize(self._base_min_height, self._base_min_height * 2)

    @staticmethod
    def _build_plot_layout(content: QWidget, graphics_view: QGraphicsView, profile) -> None:
        """Build only the plot canvas; controls live in FittingPlotControlsCard."""
        root_layout = content.layout()
        if root_layout is None:
            root_layout = QGridLayout(content)

        _take_widget(root_layout, graphics_view)
        plot_area = PlotCanvasArea(graphics_view, content, profile)
        plot_area.setMinimumHeight(scale_value(320, profile, 260))
        content.setMinimumSize(scale_value(300, profile, 260), scale_value(320, profile, 260))
        content.setMaximumSize(16777215, 16777215)
        content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(plot_area, 0, 0)


class FittingPlotControlsCard(CollapsibleCardFrame):
    def __init__(self, ui, content: QWidget, profile=None):
        super().__init__("Fitting Controls", "FittingPlotControlsCard", default_expanded=True)
        profile = profile or current_profile(content)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setMinimumWidth(SECTION_MIN_WIDTH)

        root_layout = content.layout()
        if root_layout is None:
            root_layout = QGridLayout(content)

        for widget in (
            content.findChild(QWidget, "fitFittingRegionwidget"),
            content.findChild(QWidget, "fitDataPointsNumWidget"),
            content.findChild(QWidget, "fitFittingShowWidget"),
        ):
            if widget is not None:
                _take_widget(root_layout, widget)

        controls_container = QWidget(self.content_widget)
        controls_container.setObjectName("plotControlsContainer")
        controls_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(CARD_SPACING)
        controls_layout.addWidget(FittingRegionControl(ui, controls_container, profile))
        controls_layout.addWidget(PlotSamplingControl(ui, controls_container, profile))
        controls_layout.addWidget(PlotOptionsControl(ui, controls_container, profile))
        controls_container.setMinimumHeight(controls_container.sizeHint().height())
        controls_container.setMaximumHeight(16777215)
        self.add_content(controls_container)
        self.setMinimumHeight(max(self.minimumHeight(), self.sizeHint().height()))


class StatusCard(CollapsibleCardFrame):
    def __init__(self, browser: QWidget, profile=None):
        super().__init__("Run Log", "FittingStatusCard", default_expanded=True)
        profile = profile or current_profile(browser)
        self.body_layout.setContentsMargins(CARD_MARGIN, 6, CARD_MARGIN, 8)
        self.content_layout.setSpacing(4)
        browser_min_height = scale_value(180, profile, 140)
        self.setMinimumHeight(scale_value(230, profile, 176))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        browser.setMinimumHeight(browser_min_height)
        browser.setMaximumHeight(16777215)
        browser.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.add_content(browser, 1)
        base_height = scale_value(230, profile, 176)
        self.enable_content_resize(base_height, base_height * 2)


__all__ = [
    "DetectorPreviewCard",
    "PlotCanvasArea",
    "SectionCard",
    "FittingRegionControl",
    "PlotSamplingControl",
    "ParticleOptionsLayout",
    "PlotOptionsControl",
    "PlotPreviewCard",
    "FittingPlotControlsCard",
    "StatusCard",
]
