"""Composition root of the application context (settings, session, profiles, jobs)."""

from __future__ import annotations

from pathlib import Path

from .context import AppContext
from .ports import SessionRepository
from ..integrations.jobs import KeptProcessJobRunner, LocalProcessJobRunner
from ..integrations.state import (
    InMemoryInstrumentProfileRepository,
    InMemorySessionRepository,
    JsonInstrumentProfileRepository,
    JsonProjectParametersRepository,
    JsonSessionRepository,
    StorePreferencesRepository,
    StoreSettingsRepository,
    UserStore,
    explicit_home,
    migrate_once,
    user_data_dir,
)
from ..integrations.state.user_store import PROFILES_FILE, SESSION_FILE, SETTINGS_FILE


def create_app_context(
    *,
    session: SessionRepository | None = None,
    restore_session: bool = True,
    data_dir: str | Path | None = None,
    read_only: bool = False,
) -> AppContext:
    """One store in the user data folder backs settings and preferences.

    Settings files from before the store existed are imported once, into the default user folder only
    (``integrations.state.user_store.migrate_once``); this function caches nothing. ``read_only``: a second
    GIMaP started while one runs reads the folder and keeps its own changes in memory.
    """
    folder = Path(data_dir) if data_dir is not None else user_data_dir()
    if not read_only:
        migrate_once(folder, explicit=data_dir is not None or explicit_home())
    store = UserStore(folder / SETTINGS_FILE)
    store.read_only = read_only
    profiles = JsonInstrumentProfileRepository(folder / PROFILES_FILE)
    if read_only:
        profiles = InMemoryInstrumentProfileRepository(profiles.load_all())
        session = session or InMemorySessionRepository(JsonSessionRepository(folder / SESSION_FILE).load_quietly())
    context = AppContext(
        settings=StoreSettingsRepository(store),
        preferences=StorePreferencesRepository(store),
        session=session or JsonSessionRepository(folder / SESSION_FILE),
        jobs=LocalProcessJobRunner(),
        model_jobs=KeptProcessJobRunner(),
        project_parameters=JsonProjectParametersRepository(),
        instrument_profiles=profiles,
        data_dir=folder,
        read_only=read_only,
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
