"""GIMaP 顶层依赖与状态的显式容器。"""

from __future__ import annotations

from dataclasses import dataclass, field

from .jobs import JobRunner
from .ports import SessionRepository, SettingsRepository
from .state import ProjectState


@dataclass
class AppContext:
    """由 composition root 创建一次，并通过构造函数向下传递。"""

    settings: SettingsRepository
    session: SessionRepository
    jobs: JobRunner | None = None
    project_state: ProjectState = field(default_factory=ProjectState)

    def restore_session(self) -> bool:
        state = self.session.load()
        if state is None:
            return False
        self.project_state.restore(state)
        return True

    def save_session(self) -> None:
        self.session.save(self.project_state.snapshot())
