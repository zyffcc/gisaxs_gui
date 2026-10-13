"""Python-owned Trainset workflow page."""

from __future__ import annotations

from typing import Any, Dict, Optional


from PyQt5.QtCore import pyqtSignal


from PyQt5.QtWidgets import (
    QDialog,
    QWidget,
)


from src.gimap.app.presentation.i18n import tr
from src.gimap.features.trainset.application import TrainsetUiCatalog

from .views import (
    TrainsetPageView,
)

from .visualization_widgets import ArrayCanvas, HistogramWidget, ParameterCoverageWidget
from .composed_text import ComposedTexts
from .step_rail import NOT_STARTED, STEP_TITLES

from .sections.shell_layout import ShellLayoutMixin
from .sections.dataset import DatasetMixin
from .sections.preview import PreviewMixin
from .sections.run_monitor import RunMonitorMixin
from .sections.design_state import DesignStateMixin
from .sections.comparison import ComparisonMixin
from .sections.responsive_style import ResponsiveStyleMixin

__all__ = [
    "ArrayCanvas",
    "HistogramWidget",
    "ParameterCoverageWidget",
    "TrainsetBuildPage",
]


class TrainsetBuildPage(
    ShellLayoutMixin,
    DatasetMixin,
    PreviewMixin,
    RunMonitorMixin,
    DesignStateMixin,
    ComparisonMixin,
    ResponsiveStyleMixin,
    QWidget,
    TrainsetPageView,
):
    step_changed = pyqtSignal(int)

    mask_region_created = pyqtSignal(str, dict)

    configuration_edited = pyqtSignal()

    what_if_requested = pyqtSignal(dict)

    # A file dropped on the design preview: load it as the reference (the binding does).
    reference_dropped = pyqtSignal(str)

    STEPS = STEP_TITLES

    def __init__(self, parent: Optional[QWidget] = None, *, catalog=None):
        super().__init__(parent)
        self.catalog = catalog or TrainsetUiCatalog()
        self.fields: Dict[str, QWidget] = {}
        self.preview_canvases: Dict[str, ArrayCanvas] = {}
        self._display_controls: Dict[str, Dict[str, QWidget]] = {}
        self._comparison_details: Dict[str, Any] = {}
        self._comparison_parameter_specs: Dict[str, Any] = {}
        self._comparison_config: Dict[str, Any] = {}
        self._parameter_dialog: Optional[QDialog] = None
        self._step_states = [NOT_STARTED] * len(self.STEPS)
        self._step_values: list = [None] * len(self.STEPS)  # (template, values) per step
        self._validation_text = "Not validated"
        # Run-time texts made from a template (summaries, file information): composed again after a
        # switch of the interface language (refresh_language).
        self.texts = ComposedTexts()
        self._design_stage_ready = [False, False, False, False]
        self._apply_style()  # before the widgets exist: they are polished once
        self.setupUi(self)
        self._bind_shell()

    def refresh_language(self) -> None:
        """After a switch of the interface language: compose the step lines, the badge, the hint, the
        design tabs and every run-time text again in the new language (the walker knows exact keys only)."""
        for index in range(len(self.STEPS)):
            self._show_step(index)
        self.validation_badge.setText(tr(self._validation_text))
        for index, ready in enumerate(self._design_stage_ready):
            self.set_design_stage_ready(index, ready)
        self._show_action_hint(max(0, self.stack.currentIndex()))
        self._show_pipeline_tab()
        self.texts.refresh()
