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
)
from .collapsible_card import CardContentResizeHandle, CollapsibleCardFrame
from .navigation import NavigationSidebar
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
    "PageDefinition",
    "ParameterSection",
    "PlotPanel",
    "ResultTable",
    "apply_design_system",
]
