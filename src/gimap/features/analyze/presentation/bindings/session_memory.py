"""What Analyze remembers between sessions for the person: recently opened data and the last set-up.

* **Recent** — every file or folder opened (dialog, drop, Start page, Batch Export) is remembered,
  newest first, at most ``MAX_RECENT`` (settings ``analyze.recent``); File ▸ Open Recent and the Start
  page list them. Paths that no longer exist are left out.
* **Last set-up** — when Analyze closes after a frame was analysed, the set-up (geometry profile,
  masks, corrections, cut regions … as a settings file) is written to ``last_analyze_setup.json`` in
  the user folder. The next session offers it once, when its first frame is shown — never applied
  without asking, since a new beamtime may need other settings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import tr

from ...application import settings_from_record, settings_record

MAX_RECENT = 8
RECENT_KEY = "recent"
SECTION = "analyze"


class SessionMemoryMixin:
    """Needs ``view_model`` (with ``settings`` and ``last_setup_path``), ``settings_summary``, ``_read_settings``."""

    _setup_offered = False

    # -- recent data ---------------------------------------------------------------------

    def recent_paths(self) -> list[Path]:
        settings = self.view_model.settings
        if settings is None:
            return []
        try:
            values = settings.get(SECTION, RECENT_KEY, []) or []
        except Exception:
            return []
        return [Path(value) for value in values if isinstance(value, str) and Path(value).exists()][:MAX_RECENT]

    def remember_recent(self, paths) -> None:
        settings = self.view_model.settings
        if settings is None or not paths:
            return
        try:
            known = [str(value) for value in (settings.get(SECTION, RECENT_KEY, []) or []) if isinstance(value, str)]
        except Exception:
            known = []
        new = [str(Path(path)) for path in paths]
        folded = {item.casefold() for item in new}
        merged = new + [item for item in known if item.casefold() not in folded]
        try:
            settings.set(SECTION, RECENT_KEY, merged[:MAX_RECENT])
            settings.save()
        except Exception:
            pass

    def clear_recent(self) -> None:
        settings = self.view_model.settings
        if settings is not None:
            try:
                settings.set(SECTION, RECENT_KEY, [])
                settings.save()
            except Exception:
                pass

    # -- the last set-up -----------------------------------------------------------------

    def save_last_setup(self) -> Optional[Path]:
        """At the end of a session in which a frame was analysed: keep its set-up for the next one."""
        path = getattr(self.view_model, "last_setup_path", None)
        if path is None or self.view_model.state.analysis is None:
            return None
        try:
            record = settings_record(self.view_model.current_settings())
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        except (OSError, ValueError, TypeError):
            return None
        return Path(path)

    def _offer_last_setup(self) -> None:
        """Once per session, with the first frame: “use the set-up of your last session?”."""
        if self._setup_offered:
            return
        self._setup_offered = True
        path = getattr(self.view_model, "last_setup_path", None)
        if path is None or not Path(path).exists():
            return
        try:
            last = settings_from_record(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return
        current = self.view_model.current_settings()
        if settings_record(last) == settings_record(current):
            return  # nothing would change
        summary = _setup_summary(last)
        show_toast(
            self.window(), tr("Use the set-up of your last session? {summary}").format(summary=summary), level="info",
            action=(tr("Use It"), lambda: self._read_settings(path)), timeout_ms=15000,
        )


def _setup_summary(settings) -> str:
    parts = [settings.mode.upper() if settings.mode != "auto" else tr("GISAXS or GIWAXS by angle")]
    if settings.profile_name:
        parts.append(tr("profile “{name}”").format(name=settings.profile_name))
    if settings.giwaxs.regions:
        parts.append(tr("{n} cut regions").format(n=len(settings.giwaxs.regions)))
    shapes = len(settings.corrections.mask_shapes) + (1 if settings.corrections.mask_path else 0)
    if shapes:
        parts.append(tr("{n} masks").format(n=shapes))
    if settings.corrections.mirror_fill:
        parts.append(tr("mirror filling"))
    return " · ".join(parts)


__all__ = ["MAX_RECENT", "SessionMemoryMixin"]
