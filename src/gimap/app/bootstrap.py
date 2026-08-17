"""Legacy application 与新 AppContext 的 composition root。"""

from __future__ import annotations

from pathlib import Path

from .context import AppContext
from .ports import SessionRepository
from ..integrations.state import (
    GlobalParamsSettingsRepository,
    InMemorySessionRepository,
    JsonSessionRepository,
)
from ..integrations.jobs import LocalProcessJobRunner


def create_app_context(
    *,
    session: SessionRepository | None = None,
    restore_session: bool = True,
) -> AppContext:
    """包装现有 global_params；本函数不缓存或创建新的全局 context。"""
    from core.global_params import global_params

    context = AppContext(
        settings=GlobalParamsSettingsRepository(global_params),
        session=session or JsonSessionRepository(Path(".gimap_cache") / "session.json"),
        jobs=LocalProcessJobRunner(),
    )
    if restore_session:
        context.restore_session()
    return context


def create_standalone_legacy_context() -> AppContext:
    """旧 dialog 无 host 时的兼容装配，不写 session 文件。"""
    return create_app_context(
        session=InMemorySessionRepository(),
        restore_session=False,
    )
