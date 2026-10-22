"""GIMaP 应用装配、显式 context 与顶层 state。

The names below load on first use (PEP 562). A job worker imports ``src.gimap.app.jobs``, which
imports this package first; importing ``runtime`` here would load PyQt5 into every worker, and
PyQt5's own msvcp140.dll (older than BornAgain's) then serves that process: BornAgain 24.1, built
with a newer MSVC, crashes in ``simulate()`` (exit 0xC0000005).
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .context import AppContext
    from .jobs import JobRunner
    from .ports import ProjectParametersRepository, SessionRepository, SettingsRepository
    from .project_parameters import LoadProjectParameters, SaveProjectParameters
    from .runtime import ApplicationRuntime
    from .state import FeatureState, ProjectState

_EXPORTS = {
    "AppContext": ".context",
    "ApplicationRuntime": ".runtime",
    "FeatureState": ".state",
    "JobRunner": ".jobs",
    "ProjectState": ".state",
    "ProjectParametersRepository": ".ports",
    "LoadProjectParameters": ".project_parameters",
    "SaveProjectParameters": ".project_parameters",
    "SessionRepository": ".ports",
    "SettingsRepository": ".ports",
}

__all__ = [
    "AppContext",
    "ApplicationRuntime",
    "FeatureState",
    "JobRunner",
    "ProjectState",
    "ProjectParametersRepository",
    "LoadProjectParameters",
    "SaveProjectParameters",
    "SessionRepository",
    "SettingsRepository",
]


def __getattr__(name: str) -> Any:
    owner = _EXPORTS.get(name)
    if owner is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(owner, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
