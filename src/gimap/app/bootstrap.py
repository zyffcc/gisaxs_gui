"""Composition root of the application context (settings, session, profiles, jobs)."""

from __future__ import annotations

from pathlib import Path

from .context import AppContext
from .ports import SessionRepository
from ..integrations.jobs import LocalProcessJobRunner
from ..integrations.state import (
    InMemoryInstrumentProfileRepository,
    InMemorySessionRepository,
    JsonInstrumentProfileRepository,
    JsonProjectParametersRepository,
    JsonSessionRepository,
    StorePreferencesRepository,
    StoreSettingsRepository,
    UserStore,
    migrate_legacy_files,
    user_data_dir,
)
from ..integrations.state.user_store import PROFILES_FILE, SESSION_FILE, SETTINGS_FILE


def create_app_context(
    *,
    session: SessionRepository | None = None,
    restore_session: bool = True,
    data_dir: str | Path | None = None,
) -> AppContext:
    """One store in the user data folder backs settings and preferences.

    Files from before the store existed are imported once (see
    ``integrations.state.user_store``); this function caches nothing.
    """
    folder = Path(data_dir) if data_dir is not None else user_data_dir()
    migrate_legacy_files(folder)
    store = UserStore(folder / SETTINGS_FILE)
    context = AppContext(
        settings=StoreSettingsRepository(store),
        preferences=StorePreferencesRepository(store),
        session=session or JsonSessionRepository(folder / SESSION_FILE),
        jobs=LocalProcessJobRunner(),
        project_parameters=JsonProjectParametersRepository(),
        instrument_profiles=JsonInstrumentProfileRepository(folder / PROFILES_FILE),
        data_dir=folder,
    )
    if restore_session:
        context.restore_session()
    return context


def create_standalone_legacy_context() -> AppContext:
    """Context for a dialog opened without the main window: no session or profile writes."""
    context = create_app_context(
        session=InMemorySessionRepository(),
        restore_session=False,
    )
    context.instrument_profiles = InMemoryInstrumentProfileRepository()
    return context
