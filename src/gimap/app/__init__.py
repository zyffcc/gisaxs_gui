"""GIMaP 应用装配、显式 context 与顶层 state。"""

from .context import AppContext
from .jobs import JobRunner
from .ports import SessionRepository, SettingsRepository
from .state import FeatureState, ProjectState

__all__ = [
    "AppContext",
    "FeatureState",
    "JobRunner",
    "ProjectState",
    "SessionRepository",
    "SettingsRepository",
]
