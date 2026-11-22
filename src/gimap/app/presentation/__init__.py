"""Application-owned shared PyQt presentation building blocks.

The names load on first use (PEP 562): the splash screen (``splash.py``) and the interface language are
imported before the heavy imports of the start-up, and must not pull in the plots and their NumPy.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    **{name: ".components" for name in (
        "AdvancedSection", "CurvePlot", "DetectorView", "EmptyState", "ErrorBanner", "FilePicker", "JobStatus",
        "ParameterSection", "PlotPanel", "ScientificImageViewer", "ResultTable", "SafeWheelComboBox",
        "SafeWheelDoubleSpinBox", "SafeWheelInputFilter", "SafeWheelSpinBox", "install_safe_wheel_behavior")},
    "CardContentResizeHandle": ".collapsible_card",
    "CollapsibleCardFrame": ".collapsible_card",
    "NAVIGATION_ITEMS": ".navigation",
    "NavigationItem": ".navigation",
    "NavigationSidebar": ".navigation",
    "ParameterCommitCoordinator": ".parameter_commit",
    "ParameterUpdatePolicy": ".parameter_commit",
    "TaskRunner": ".task_runner",
    **{name: ".theme" for name in ("apply_theme", "set_role", "set_state", "style_widget", "theme_color",
                                    "theme_manager")},
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    owner = _EXPORTS.get(name)
    if owner is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(owner, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
