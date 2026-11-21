"""The online mode of a watched folder (the Watch button's arrow): pause and resume, how long a new file must rest
before it is read, the Series map growing while watching (with or without Export New Frames While Watching), the
log of the files the watch could not take, the results database, and the hand-over to Fitting for a fit of every
new frame.

* **Pause Watching** stops the looks at the folder; what was written meanwhile is listed at the first look after
  Resume (nothing is lost: the watch knows what it had seen).
* **Settle time**: a new file is read once it has stayed the same for that long (and at two looks), so a frame a
  detector is still writing is never read half-written (``FolderWatch.settle_s``).
* **Add New Frames to the Series Map** (on by default): every complete new frame is reduced with the set-up on
  screen and its curve appended to the map, the traces and the timeline — also when no map was built and nothing
  is exported. Only curves are kept, never the frames.
* **Skipped Files…**: every file the watch could not take, when and why (held by its writer, a failed reduction, a
  frame without the curve of the map); also written to the results database while it records.
* **Record Every Frame in the Results Database**: the curve of every frame appended to the map goes into
  ``gimap_analysis/gimap_series.sqlite`` next to the data (``application.store_rows``), once a second; Compare reads
  it (Add Series ▸ Results Database…). Export ▾ ▸ Save Results Database… writes the whole map at any time.
* **Fit New Frames in Fitting…**: exports the frame on screen, switches Export New Frames While Watching on and
  opens the folder of the exports in Fitting ▸ In-situ series, where Start with Watch for new curves fits every new
  frame from the result of the one before.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QSignalBlocker
from PyQt5.QtWidgets import QDialog, QFileDialog, QPlainTextEdit, QPushButton, QVBoxLayout

from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.shared.series_store import STORE_NAME, SeriesStore

from ...application import series_info, store_fit_previews, store_rows
from ..texts import message_text
from .series_display import row_files
from .series_stages import series_name

SETTLE_TIMES = (0.0, 5.0, 10.0, 30.0)
"""The settle times offered (s); 0: two looks the same (2 s apart) are enough."""
WATCH_PAUSED = "Watching paused: frames written meanwhile are listed when you resume."
WATCH_RESUMED = "Watching {folder} again …"
SKIPPED_TITLE = "Skipped Files"
NO_SKIPPED = "No file was skipped while watching."
SKIPPED_LINE = "{time}  {file}  {what}: {reason}"
WHAT_WAITING, WHAT_FAILED, WHAT_NO_CURVE = "waiting", "failed", "no curve"
RECORD_FAILED = "Could not write the results database: {error}"
RECORDED = "Saved {n} frames to {name}"
FIT_NEEDS_WATCH = "Watch a folder first: Fit New Frames in Fitting fits the frames written into it."
FIT_HANDED = ("Fitting ▸ In-situ series lists the curves written so far: tick Watch for new curves and press Start — "
              "every new frame is then fitted from the result of the one before.")
MOST_SKIPPED = 2000
"""The log keeps the latest this many entries (a folder of unreadable files does not grow it without end)."""


class WatchOnlineMixin:
    """Needs the watch menu (``views/analyze_steps_view.py``), ``view_model``, ``_series_rows``, ``_series_built``,
    ``_series_key``, ``_status`` and the watch of ``batch_watch.py``."""

    def _connect_watch_online(self) -> None:
        self._watch_paused = False
        self._watch_skipped: list[tuple[str, str, str, str]] = []
        self._watch_logged: set = set()
        self._recorded: set = set()
        self._records_name: Optional[str] = None
        self._skipped_dialog = None
        read = getattr(self.view_model, "_read_setting", lambda _section, _key, default: default)
        on = lambda key, default: str(read("analyze", key, default)).strip().lower() not in ("false", "0", "no", "off")  # noqa: E731
        with QSignalBlocker(self.watch_map_action), QSignalBlocker(self.watch_record_action):
            self.watch_map_action.setChecked(on("watch_map", True))
            self.watch_record_action.setChecked(on("watch_record", False))
        try:
            settle = float(read("analyze", "watch_settle_s", 0.0) or 0.0)
        except (TypeError, ValueError):
            settle = 0.0
        for action in self.watch_settle_group.actions():
            action.setChecked(abs(float(action.data()) - settle) < 1e-9)
        self.view_model.watch.settle_s = settle
        self.watch_pause_action.toggled.connect(self._watch_pause_toggled)
        self.watch_map_action.toggled.connect(lambda on: self._watch_option("watch_map", bool(on)))
        self.watch_record_action.toggled.connect(self._watch_record_toggled)
        self.watch_settle_group.triggered.connect(self._watch_settle_chosen)
        self.watch_log_action.triggered.connect(lambda: self.show_skipped_files())
        self.watch_fitting_action.triggered.connect(lambda: self.fit_new_frames_in_fitting())
        self.series_export_store_action.triggered.connect(lambda: self.save_results_database())

    def _watch_option(self, key: str, value) -> None:
        settings = getattr(self.view_model, "settings", None)
        if settings is not None:
            settings.set("analyze", key, value)

    # -- pause, settle -------------------------------------------------------------------------------

    def watch_paused(self) -> bool:
        return bool(getattr(self, "_watch_paused", False)) and self.view_model.watch.active

    def _watch_pause_toggled(self, paused: bool) -> None:
        self._watch_paused = bool(paused)
        if not self.view_model.watch.active:
            return
        if paused:
            self._status(tr(WATCH_PAUSED), "info")
        else:
            self._status(trf(WATCH_RESUMED, folder=self.view_model.watch.folder), "info")
            self._watch_tick()

    def _watch_settle_chosen(self, action) -> None:
        settle = float(action.data() or 0.0)
        self.view_model.watch.settle_s = settle
        self._watch_option("watch_settle_s", settle)

    def _watch_map_wanted(self) -> bool:
        """Add New Frames to the Series Map: a map grows while watching even when none was built or exported."""
        return bool(getattr(self, "watch_map_action", None) is not None and self.watch_map_action.isChecked())

    def _watch_online_stopped(self) -> None:
        """The watch ended: not paused for the next one; the frames recorded are forgotten with the map's name."""
        self._watch_paused = False
        with QSignalBlocker(self.watch_pause_action):
            self.watch_pause_action.setChecked(False)

    # -- skipped files -------------------------------------------------------------------------------

    def _log_skipped(self, file, what: str, reason: str) -> None:
        """A file the watch could not take: in the log (once per file, kind and reason), and in the database."""
        key = (str(file).casefold(), what, str(reason))
        if key in self._watch_logged:
            return
        self._watch_logged.add(key)
        moment = datetime.now().astimezone().isoformat(timespec="seconds")
        self._watch_skipped.append((moment, str(file), what, str(reason)))
        del self._watch_skipped[:-MOST_SKIPPED]
        if self.watch_record_action.isChecked():
            folder = self.view_model.default_export_dir()
            if folder is not None:
                try:
                    with SeriesStore(folder / STORE_NAME) as store:
                        store.log_skipped(self._store_name(), file, str(reason), what=what, time=moment)
                except (OSError, ValueError):
                    pass  # (the log in the window keeps it; the next write says what is wrong with the folder)
        if self._skipped_dialog is not None and self._skipped_dialog.isVisible():
            self._fill_skipped()

    def _watch_note_waiting(self) -> None:
        for path, reason in getattr(self.view_model, "watch_waiting", ()) or ():
            self._log_skipped(path, WHAT_WAITING, message_text(reason))

    def skipped_files(self) -> list[tuple[str, str, str, str]]:
        """``(time, file, what, reason)`` of every file skipped while watching (the latest ``MOST_SKIPPED``)."""
        return list(self._watch_skipped)

    def show_skipped_files(self) -> QDialog:
        """A window with the log of skipped files (it stays open, and fills as files are skipped)."""
        if self._skipped_dialog is None:
            dialog = QDialog(self)
            dialog.setObjectName("analyzeSkippedFiles")
            dialog.setWindowTitle(tr(SKIPPED_TITLE))
            layout = QVBoxLayout(dialog)
            dialog.text = QPlainTextEdit(dialog)
            dialog.text.setReadOnly(True)
            dialog.text.setMinimumSize(560, 240)
            layout.addWidget(dialog.text)
            close = QPushButton(tr("Close"), dialog)
            close.clicked.connect(dialog.close)
            layout.addWidget(close)
            self._skipped_dialog = dialog
        self._fill_skipped()
        self._skipped_dialog.show()
        return self._skipped_dialog

    def _fill_skipped(self) -> None:
        lines = [trf(SKIPPED_LINE, time=time, file=file, what=tr(what), reason=reason)
                 for time, file, what, reason in self._watch_skipped]
        self._skipped_dialog.text.setPlainText("\n".join(lines) if lines else tr(NO_SKIPPED))

    # -- the results database ------------------------------------------------------------------------

    def _store_name(self) -> str:
        """The name the map's frames are recorded under: fixed at the first write of a map (its files' name)."""
        if self._records_name:
            return self._records_name
        rows = getattr(self, "_series_rows", [])
        watched = self.view_model.watch.folder
        name = series_name(row_files(rows)) if len(rows) >= 2 else (watched.name if watched is not None else "")
        self._records_name = name or getattr(self, "_series_name", None) or "series"
        return self._records_name

    def _watch_record_toggled(self, on: bool) -> None:
        self._watch_option("watch_record", bool(on))
        if on:
            self._record_rows()

    def _record_rows(self, path: Optional[Path] = None, *, every: bool = False) -> int:
        """Write the map's rows not written yet (``every``: all of them) into the results database next to the data
        (or at ``path``). Returns how many were written; says why when nothing could be."""
        rows = list(getattr(self, "_series_rows", []))
        key = getattr(self, "_series_key", None)
        if not rows or key is None or (path is None and not self.watch_record_action.isChecked()):
            return 0
        folder = self.view_model.default_export_dir()
        target = Path(path) if path is not None else (folder / STORE_NAME if folder is not None else None)
        if target is None:
            return 0
        written = set() if every else self._recorded
        new = {(str(ref[0]), int(ref[1])) for *_rest, ref in rows} - written
        if not new:
            return 0
        built = getattr(self, "_series_built", {}) or {}
        try:
            with SeriesStore(target) as store:
                count = store_rows(store, self._store_name(), rows, key, numbers=built.get("numbers"),
                                   group=int(built.get("group") or 1), only=new)
                store.set_info(self._store_name(), series_info(key, getattr(self, "_series_curve_title", ""), built,
                                                               built.get("kind")))
        except (OSError, ValueError) as exc:
            self._status(trf(RECORD_FAILED, error=message_text(exc)), "warning")
            return 0
        if path is None:
            self._recorded |= new
        return count

    def _forget_records(self) -> None:
        """A new map (Clear, another list): its frames are recorded afresh, under the name of its own files."""
        self._recorded = set()
        self._records_name = None

    def save_results_database(self, path: Optional[Path] = None) -> Optional[Path]:
        """Export ▾ ▸ Save Results Database…: every row of the map into a results database (added to one that is
        there; a frame recorded before is replaced)."""
        if self._series_map is None or not self._series_rows:
            return None
        if path is None:
            folder = self.view_model.default_export_dir() or Path(self._last_folder or ".")
            chosen, _ = QFileDialog.getSaveFileName(self, tr("Save Results Database"), str(folder / STORE_NAME),
                                                    tr("Results database") + " (*.sqlite)")
            if not chosen:
                return None
            path = Path(chosen)
        count = self._record_rows(Path(path), every=True)
        run, entries = getattr(self, "_series_fit_run", None), list(getattr(self, "_series_fit_entries", []))
        if count and run is not None and entries:  # the fits of a Batch Export shown under the map, with them
            try:
                with SeriesStore(Path(path)) as store:
                    store_fit_previews(store, self._store_name(), str(run.choices.fit), entries)
            except (OSError, ValueError) as exc:
                self._status(trf(RECORD_FAILED, error=message_text(exc)), "warning")
        if count:
            self.notify_written(trf(RECORDED, n=count, name=Path(path).name), Path(path).parent)
            return Path(path)
        return None

    # -- a fit of every new frame --------------------------------------------------------------------

    def fit_new_frames_in_fitting(self) -> bool:
        """Export the frame on screen, export every new frame from now on, and open the folder in Fitting."""
        model = self.view_model
        send = getattr(self, "_send_series_to_fitting", None)
        if not model.watch.active or send is None or model.state.analysis is None:
            self._status(tr(FIT_NEEDS_WATCH), "warning")
            return False
        destination = model.default_export_dir()
        folder = self._auto_export_folder(destination) if destination is not None else None
        if folder is None:
            return False
        try:
            model.export(folder)
        except (ValueError, OSError) as exc:
            self._status(trf("Export failed: {error}", error=message_text(exc)), "error")
            return False
        if not self.auto_export_check.isChecked():
            self.auto_export_check.setChecked(True)
        if not send(folder):
            return False
        self._status(tr(FIT_HANDED), "info")
        return True


__all__ = ["FIT_HANDED", "FIT_NEEDS_WATCH", "MOST_SKIPPED", "NO_SKIPPED", "RECORDED", "RECORD_FAILED", "SETTLE_TIMES",
           "SKIPPED_LINE", "SKIPPED_TITLE", "WATCH_PAUSED", "WATCH_RESUMED", "WatchOnlineMixin"]
