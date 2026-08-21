"""Application-owned shared PyQt presentation building blocks。"""

from .components import (
    AdvancedSection,
    EmptyState,
    ErrorBanner,
    FilePicker,
    JobStatus,
    ParameterSection,
    PlotPanel,
    ResultTable,
    SafeWheelComboBox,
    SafeWheelDoubleSpinBox,
    SafeWheelInputFilter,
    SafeWheelSpinBox,
    install_safe_wheel_behavior,
)
from .collapsible_card import CardContentResizeHandle, CollapsibleCardFrame
from .navigation import NavigationSidebar
from .parameter_commit import ParameterCommitCoordinator, ParameterUpdatePolicy
from .shell import ContentStack, MainShell, PageDefinition
from .styles import apply_design_system

__all__ = [
    "AdvancedSection",
    "CardContentResizeHandle",
    "CollapsibleCardFrame",
    "ContentStack",
    "EmptyState",
    "ErrorBanner",
    "FilePicker",
    "JobStatus",
    "MainShell",
    "NavigationSidebar",
    "ParameterCommitCoordinator",
    "ParameterUpdatePolicy",
    "PageDefinition",
    "ParameterSection",
    "PlotPanel",
    "ResultTable",
    "SafeWheelComboBox",
    "SafeWheelDoubleSpinBox",
    "SafeWheelInputFilter",
    "SafeWheelSpinBox",
    "apply_design_system",
    "install_safe_wheel_behavior",
]
