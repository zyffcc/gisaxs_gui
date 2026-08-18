"""Python-owned Trainset workflow page."""

from __future__ import annotations

from typing import Any, Dict, Optional


from PyQt5.QtCore import pyqtSignal


from PyQt5.QtWidgets import (
    QDialog,
    QWidget,
)


from src.gimap.features.trainset.application import TrainsetUiCatalog

from .views import (
    TrainsetPageView,
)

from .visualization_widgets import ArrayCanvas, HistogramWidget, ParameterCoverageWidget

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

    STEPS = ("Dataset Design", "Local Preview", "Model Design", "Local Run", "Monitor & Results")

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
        self._step_states = ["Not started"] * len(self.STEPS)
        self._design_stage_ready = [False, False, False, False]
        self.setupUi(self)
        self._bind_shell()
        self._apply_style()
