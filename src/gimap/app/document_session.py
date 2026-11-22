"""Unsaved changes, autosave and crash recovery of the work in the main window.

The work is what a project holds (``project.collect``): Analyze's frames and set-up, Fitting's curve, model and
series, Compare's series and the Sputter form. ``fingerprint`` digests the pages' own ``project_state``; the
work is *changed* while its digest differs from the one of the last save, open or restore. The digest is
looked at every ``CHECK_MS`` and at once when a page says something changed (``document_changes``); a few
seconds after an open (``SETTLE_S``: the frames are still being read) only move the reference along.

* **Title and question.** Changed work puts ``*`` in the title (Qt's ``[*]``). Closing the window or opening
  another project asks Save / Don't Save / Cancel first (``app_dialogs.ask_save_changes``).
* **Autosave.** Changed work is written every ``AUTOSAVE_MS`` and after each long job ends (Batch Export, the
  automatic analysis, a fit, a series …), as a project in ``<user data>/autosave/session.gimap`` with a record
  of when and of which project (``autosave.json``). Nothing is written while a job runs.
* **Recovery.** ``running.json`` there says a session is open; a clean close removes it and the autosave. When
  the next start finds it, GIMaP was not closed properly: ``pending_recovery`` offers the autosave, if any.

Only the launcher attaches a ``DocumentSession`` (``window.document``): windows built by tests and scripts
close without questions and write nothing here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from . import project
from .presentation import app_dialogs
from .presentation.components import show_toast
from .presentation.document_changes import document_changes
from .presentation.i18n import tr, trf
from src.gimap.shared.version import __version__

logger = logging.getLogger(__name__)

AUTOSAVE_FOLDER = "autosave"
AUTOSAVE_FILE = "session.gimap"
AUTOSAVE_RECORD = "autosave.json"
RUNNING_MARKER = "running.json"
AUTOSAVE_MS = 120_000
CHECK_MS = 1500
SETTLE_S = 4.0


def fingerprint(components) -> Optional[str]:
    """A digest of the work in the window, or ``None`` while it cannot be read (Compare still importing)."""
    built = getattr(components, "built_page", lambda key: getattr(components, f"{key}_page", None))
    try:
        analyze = dict(components.analyze_page.project_state())
        for key in ("current", "frame"):  # the frame looked at is no change of the work
            analyze.pop(key, None)
        series = analyze.get("series_map")
        if isinstance(series, dict) and series.get("rows"):
            # The Series map's rows by their frame and the hash of their encoded values (the same text objects while a
            # row stays, so the hash is cheap): not the whole of its arrays at every look.
            analyze["series_map"] = {**series, "rows": [[row.get("ref"), hash(row["y"]["f8"])] for row in series["rows"]]}
        workspace = components.fitting_workspace
        parts = {"analyze": analyze, "single": workspace.fit_page.project_state(),
                 "series": workspace.series_page.project_state()}
        compare = built("compare")
        if compare is not None:
            parts["compare"] = compare.project_state(None)
            parts["compare_names"] = [getattr(item, "name", "") for item in getattr(compare, "series", ())]
        sputter = built("sputter")
        if sputter is not None:
            parts["sputter"] = sputter.project_state()
    except (ValueError, RuntimeError, AttributeError):  # busy, or a page being torn down
        return None
    text = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def pending_recovery(data_dir: Path | str | None) -> Optional[dict]:
    """What the session before left when it was not closed properly: ``{"path", "saved", "project"}`` of its
    autosave, or ``None`` (closed properly, or nothing autosaved). Call before a new session starts."""
    if data_dir is None:
        return None
    folder = Path(data_dir) / AUTOSAVE_FOLDER
    if not (folder / RUNNING_MARKER).is_file():
        return None
    autosave, record = folder / AUTOSAVE_FILE, _read_json(folder / AUTOSAVE_RECORD)
    if not autosave.is_file():
        logger.warning("The last session was not closed properly; nothing had been autosaved")
        return None
    return {"path": str(autosave), "saved": str(record.get("saved") or ""), "project": str(record.get("project") or "")}


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


class DocumentSession(QObject):
    """Watches the work of ``window`` (its ``components`` and ``menus``); ``changed(dirty)`` on each change."""

    changed = pyqtSignal(bool)

    def __init__(self, window, data_dir: Path | str | None = None, *, autosave: bool = True, parent=None):
        super().__init__(parent or window)
        self.window = window
        self.folder = Path(data_dir) / AUTOSAVE_FOLDER if (autosave and data_dir is not None) else None
        self.dirty = False
        self.holding = False
        """While the recovery question waits: no autosave (it would replace the work offered)."""
        self._reference: Optional[str] = None
        self._settle_until = 0.0
        self._restored = False
        self._autosaved: Optional[str] = None
        self._jobs_were_running = False
        self._timer = QTimer(self)
        self._timer.setInterval(CHECK_MS)
        self._timer.timeout.connect(self.check)
        self._timer.start()
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setInterval(AUTOSAVE_MS)
        self._autosave_timer.timeout.connect(self.autosave)
        self._autosave_timer.start()
        self._soon = QTimer(self)  # a change a page reports: looked at once the events of the edit are done
        self._soon.setSingleShot(True)
        self._soon.setInterval(200)
        self._soon.timeout.connect(self.check)
        document_changes().changed.connect(self._soon.start)
        self.mark_clean()
        if self.folder is not None:
            self._write_marker()

    # -- the reference ---------------------------------------------------------------------

    def mark_clean(self, settle: float = SETTLE_S) -> None:
        """The work as it is now is saved (or just opened): no ``*``; ``settle`` seconds of reading follow."""
        self._reference = fingerprint(self.window.components)
        self._settle_until = time.monotonic() + max(0.0, settle)
        self._restored = False
        self._set_dirty(False)

    def mark_restored(self) -> None:
        """Work brought back from an autosave: in no saved project, so changed until it is saved."""
        self.mark_clean()
        self._restored = True
        self._set_dirty(True)

    def check(self) -> None:
        """Compare the work with the reference; autosave after a job that just ended."""
        digest = fingerprint(self.window.components)
        if digest is not None:
            if time.monotonic() < self._settle_until and not self.dirty:
                self._reference = digest
            else:
                self._set_dirty(self._restored or digest != self._reference)
        running = self._jobs_running()
        if self._jobs_were_running and not running:
            self.autosave()
        self._jobs_were_running = running

    def _set_dirty(self, dirty: bool) -> None:
        dirty = bool(dirty)
        if dirty == self.dirty:
            return
        self.dirty = dirty
        context = getattr(self.window, "app_context", None)
        if context is not None:
            context.project_state.mark_dirty(dirty)
        self.changed.emit(dirty)

    def _jobs_running(self) -> bool:
        try:
            return bool(self.window.components.running_jobs())
        except Exception:  # noqa: BLE001 - a check never stops the watch
            return False

    # -- questions -------------------------------------------------------------------------

    def may_close(self) -> bool:
        """Before the window closes or another project replaces the work: ``True`` to go on (saved or
        discarded), ``False`` to stay (Cancel, or the save did not happen)."""
        self.check()
        if not self.dirty:
            return True
        menus = getattr(self.window, "menus", None)
        name = Path(menus.project_path).name if menus is not None and menus.project_path else ""
        answer = app_dialogs.ask_save_changes(self.window, name)
        if answer == "save":
            return bool(menus is not None and menus.save_project())
        return answer == "discard"

    # -- autosave and recovery -------------------------------------------------------------

    def autosave(self) -> Optional[Path]:
        """Write the changed work to the autosave folder (not while a job runs or recovery waits)."""
        if self.folder is None or self.holding or not self.dirty or self._jobs_running():
            return None
        digest = fingerprint(self.window.components)
        if digest is None or digest == self._autosaved:
            return None
        components, menus = self.window.components, getattr(self.window, "menus", None)
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            written = project.save(components, self.folder / AUTOSAVE_FILE, components.current_page_key() or "")
            record = {"saved": datetime.now().isoformat(timespec="seconds"), "version": __version__,
                      "project": getattr(menus, "project_path", "") or ""}
            (self.folder / AUTOSAVE_RECORD).write_text(json.dumps(record, indent=2), encoding="utf-8")
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("Autosave failed: %s", exc)
            return None
        self._autosaved = digest
        logger.info("Autosaved the session to %s", written)
        return written

    def restore(self, recovery: dict) -> bool:
        """Open the autosave ``recovery`` (``pending_recovery``) in the window; the project it came from stays
        the one Save writes to."""
        path = Path(recovery["path"])
        try:
            data = project.read(path)
        except (OSError, ValueError) as exc:
            logger.warning("The autosave could not be read: %s", exc)
            show_toast(self.window, trf("The autosaved session could not be read: {error}", error=exc), level="warning")
            return False
        components, menus = self.window.components, getattr(self.window, "menus", None)
        project.apply(components, data, path)
        if menus is not None:
            original = recovery.get("project") or ""
            menus.project_path = original if original and Path(original).is_file() else ""
            menus._sync_title()
        page = data.get("page") if data.get("page") in components.pages else "analyze"
        show = getattr(menus, "show_workspace", None)
        (show or components.show_page)(page)
        self.mark_restored()
        show_toast(self.window, tr("Restored the autosaved session. Save it to keep it."), level="ok")
        return True

    def _write_marker(self) -> None:
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            (self.folder / RUNNING_MARKER).write_text(json.dumps({
                "pid": os.getpid(), "started": datetime.now().isoformat(timespec="seconds"), "version": __version__}),
                encoding="utf-8")
        except OSError as exc:
            logger.warning("The session marker could not be written: %s", exc)

    def discard_autosave(self) -> None:
        """The autosave and its record go (the work was saved, discarded or not wanted back)."""
        if self.folder is None:
            return
        for path in list(self.folder.glob("session*.gimap")) + list(self.folder.glob("session*.compare.npz")) + \
                [self.folder / AUTOSAVE_RECORD]:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        self._autosaved = None

    def closed_cleanly(self) -> None:
        """The window closes properly: no recovery next time."""
        self._timer.stop()
        self._autosave_timer.stop()
        try:
            document_changes().changed.disconnect(self._soon.start)
        except (TypeError, RuntimeError):
            pass
        if self.folder is not None:
            self.discard_autosave()
            try:
                (self.folder / RUNNING_MARKER).unlink(missing_ok=True)
            except OSError:
                pass


__all__ = ["AUTOSAVE_FILE", "AUTOSAVE_FOLDER", "AUTOSAVE_MS", "CHECK_MS", "DocumentSession", "RUNNING_MARKER",
           "fingerprint", "pending_recovery"]
