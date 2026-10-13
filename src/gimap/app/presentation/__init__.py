"""Application-owned shared PyQt presentation building blocks."""

from .components import (
    AdvancedSection,
    CurvePlot,
    DetectorView,
    EmptyState,
    ErrorBanner,
    FilePicker,
    JobStatus,
    ParameterSection,
    PlotPanel,
    ScientificImageViewer,
    ResultTable,
    SafeWheelComboBox,
    SafeWheelDoubleSpinBox,
    SafeWheelInputFilter,
    SafeWheelSpinBox,
    install_safe_wheel_behavior,
)
from .collapsible_card import CardContentResizeHandle, CollapsibleCardFrame
from .navigation import NAVIGATION_ITEMS, NavigationItem, NavigationSidebar
from .parameter_commit import ParameterCommitCoordinator, ParameterUpdatePolicy
from .task_runner import TaskRunner
from .theme import apply_theme, set_role, set_state, style_widget, theme_color, theme_manager

__all__ = [
    "AdvancedSection",
    "CardContentResizeHandle",
    "CollapsibleCardFrame",
    "CurvePlot",
    "DetectorView",
    "EmptyState",
    "ErrorBanner",
    "FilePicker",
    "JobStatus",
    "NAVIGATION_ITEMS",
    "NavigationItem",
    "NavigationSidebar",
    "ParameterCommitCoordinator",
    "ParameterUpdatePolicy",
    "ParameterSection",
    "PlotPanel",
    "ScientificImageViewer",
    "ResultTable",
    "SafeWheelComboBox",
    "SafeWheelDoubleSpinBox",
    "SafeWheelInputFilter",
    "SafeWheelSpinBox",
    "TaskRunner",
    "apply_theme",
    "install_safe_wheel_behavior",
    "set_role",
    "set_state",
    "style_widget",
    "theme_color",
    "theme_manager",
]
