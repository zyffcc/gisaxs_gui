"""Send to Fitting when the folder of the data takes no file.

The curve for Fitting is written next to the data (``gimap_analysis/``). Raw data of a beamline are read-only
for the people who measured them, so that folder often cannot be written. ``send_to_fitting`` then asks for
another folder, as Export does — once: a folder a file was written into is kept for the session
(``session_export_dir``) and the next curve goes there without a question. The status line says where the
file is; so does Fitting's notice.

A folder is kept only after a file was written into it. One that takes no file either — another folder of the
same file system, or the session's folder once it is gone or locked — is forgotten, and the question comes
again with that folder named (``write_elsewhere``); a send asks ``QUESTIONS`` times at most, then it fails and
says why, and the next send asks anew.

``fit_current`` is what the page's two Fitting buttons are connected to (the command bar, the Export step):
the same Send to Fitting. (This mixin stands before ``BatchWatchMixin`` among the page's bases, which had a
``fit_current`` of its own that knew no other folder.)

``write_elsewhere(write, error)`` is for any file of the frame on screen that the data folder refuses: Export
can take the session's folder through it too.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, TypeVar

from PyQt5.QtWidgets import QFileDialog

from src.gimap.app.presentation.i18n import trf

from ..texts import message_text

SENT_ELSEWHERE = "Opened {name} in Fitting. The folder of the data cannot be written: the file is in {folder}"
DATA_REFUSED = "Cannot write next to the data ({error}); export to…"
"""The question when the folder of the data takes no file (Export asks the same) …"""
FOLDER_REFUSED = "Cannot write in {folder} ({error}); export to…"
"""… and when the folder answered, or the one kept for the session, takes none either."""
QUESTIONS = 2
"""How often one send asks for a folder before it gives up."""

Written = TypeVar("Written")


class FitHandoverMixin:
    """Needs ``view_model``, ``_send_to_fitting``, ``_status`` and ``_last_folder`` (the Analyze page)."""

    _session_export_dir: Optional[Path] = None
    """Where this session's files go when the folder of the data cannot be written: chosen once, and kept
    from the first file written into it."""

    def session_export_dir(self) -> Optional[Path]:
        """The folder kept for files that cannot be written next to the data (``None``: none yet — nothing
        failed, the question was cancelled, or the folder answered took no file)."""
        return self._session_export_dir

    def _ask_export_dir(self, error: OSError, refused: Optional[Path] = None) -> Optional[Path]:
        """Ask where to write instead: ``error`` is why the folder of the data — or ``refused``, the folder
        tried last — took no file. ``None`` when the question is cancelled."""
        reason = error.strerror or error
        title = trf(DATA_REFUSED, error=reason) if refused is None else trf(FOLDER_REFUSED, folder=refused, error=reason)
        folder = QFileDialog.getExistingDirectory(self, title, self._last_folder)
        return Path(folder) if folder else None

    def write_elsewhere(self, write: Callable[[Path], Written], error: OSError) -> tuple[Written, Path]:
        """``(write(folder), folder)`` for a file that could not be written next to the data (``error``): into
        the session's folder, else into one the person chooses now. A folder that takes the file is kept for
        the session; one that does not is forgotten and asked about again. Raises the ``OSError`` of the last
        try when the question is cancelled or ``QUESTIONS`` answers took no file."""
        folder, refused, asked = self._session_export_dir, None, 0
        while True:
            if folder is None:
                if asked >= QUESTIONS:
                    raise error
                folder = self._ask_export_dir(error, refused)
                asked += 1
                if folder is None:
                    raise error
            try:
                written = write(folder)
            except OSError as exc:
                self._session_export_dir = None
                error, refused, folder = exc, folder, None
                continue
            self._session_export_dir = folder
            return written, folder

    def send_to_fitting(self) -> None:
        """Send to Fitting: write the curve of the frame on screen and open it in Fitting ▸ Single analysis.
        Next to the data when that folder can be written, else in the session's folder (``write_elsewhere``)."""
        if self._send_to_fitting is None:
            return
        model = self.view_model
        elsewhere: Optional[Path] = None
        try:
            try:
                path = model.write_fit_input()
            except OSError as exc:
                # The view model writes next to the data only: the same export, with the folder given.
                path, elsewhere = self.write_elsewhere(
                    lambda folder: model.frame_tools().export.fit_input(model.state.analysis, folder, halves=model.fit_side),
                    exc)
            self._send_to_fitting(path, model.fit_side)
        except (ValueError, OSError, RuntimeError) as exc:
            self._status(trf("Could not open the curve in Fitting: {error}", error=message_text(exc)), "error")
            return
        if elsewhere is None:
            self._status(trf("Opened {name} in Fitting.", name=path.name), "ok")
        else:
            self._status(trf(SENT_ELSEWHERE, name=path.name, folder=path.parent), "ok")

    def fit_current(self) -> None:
        """The Fitting buttons of Analyze (module docstring): Send to Fitting."""
        self.send_to_fitting()


__all__ = ["DATA_REFUSED", "FOLDER_REFUSED", "FitHandoverMixin", "QUESTIONS", "SENT_ELSEWHERE"]
