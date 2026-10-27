"""Qt's platform plugin is found before the QApplication starts (no "no Qt platform plugin" stop)."""

import os
from pathlib import Path

from src.gimap.app.qt_plugins import (
    PLATFORM_ENV,
    PLUGIN_ENV,
    candidate_plugin_dirs,
    ensure_qt_platform_plugin_path,
    has_platform_plugin,
)


def _plugins(root: Path, name: str = "qwindows.dll") -> Path:
    platforms = root / "plugins" / "platforms"
    platforms.mkdir(parents=True)
    (platforms / name).write_bytes(b"")
    return root / "plugins"


def test_missing_path_is_filled_in(tmp_path):
    plugins = _plugins(tmp_path)
    env = {}

    chosen = ensure_qt_platform_plugin_path(env, [tmp_path / "none", plugins], platform="win32")

    assert chosen == plugins / "platforms"
    assert env == {PLATFORM_ENV: str(plugins / "platforms")}


def test_a_working_user_setting_is_kept(tmp_path):
    mine = _plugins(tmp_path / "mine")
    other = _plugins(tmp_path / "other")
    env = {PLATFORM_ENV: str(mine / "platforms")}

    assert ensure_qt_platform_plugin_path(env, [other], platform="win32") is None
    assert env == {PLATFORM_ENV: str(mine / "platforms")}


def test_a_broken_setting_and_another_programs_qt_are_replaced(tmp_path):
    ours = _plugins(tmp_path / "ours")
    foreign = _plugins(tmp_path / "foreign")
    env = {PLATFORM_ENV: str(tmp_path / "gone"), PLUGIN_ENV: str(foreign)}

    ensure_qt_platform_plugin_path(env, [ours], platform="win32")

    assert env[PLATFORM_ENV] == str(ours / "platforms")
    assert env[PLUGIN_ENV] == str(ours)


def test_nothing_found_changes_nothing(tmp_path):
    env = {}
    assert ensure_qt_platform_plugin_path(env, [tmp_path], platform="win32") is None
    assert env == {}


def test_plugin_name_follows_the_system(tmp_path):
    linux = _plugins(tmp_path / "linux", "libqxcb.so") / "platforms"
    assert has_platform_plugin(linux, "linux")
    assert not has_platform_plugin(linux, "win32")
    assert not has_platform_plugin(None, "win32")


def test_conda_and_pip_locations_are_candidates(tmp_path):
    dirs = candidate_plugin_dirs(prefix=tmp_path)
    assert tmp_path / "Library" / "plugins" in dirs
    assert tmp_path / "plugins" in dirs


def test_this_pyqt5_has_a_platform_plugin_among_the_candidates():
    """The Python running the tests (as GIMaP runs) finds its own qwindows/xcb plugin."""
    import sys

    assert any(has_platform_plugin(d / "platforms", sys.platform) for d in candidate_plugin_dirs())


def test_real_environment_is_untouched_when_it_already_works(monkeypatch, tmp_path):
    plugins = _plugins(tmp_path)
    monkeypatch.setenv(PLATFORM_ENV, str(plugins / "platforms"))
    before = dict(os.environ)

    assert ensure_qt_platform_plugin_path(platform="win32") is None
    assert dict(os.environ) == before
