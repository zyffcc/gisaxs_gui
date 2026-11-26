"""Help ▸ Check for Updates… for a source copy: what is new on GitHub, then Update and Restart.

The dialog checks as soon as it opens (in the background), lists the new commits and offers Update and Restart.
The work is ``app/source_update.py``'s, handed in as ``UpdateActions`` (this module imports no update code):
``check()`` and ``apply(status, progress, cancelled)`` run on a pool thread; ``restart()`` closes the window and
starts GIMaP again (``False`` when the window stays open). Messages arrive as English templates with values and
are translated here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QLabel, QPlainTextEdit, QVBoxLayout, QWidget

from .i18n import tr, trf
from .task_runner import TaskRunner

RESTART_DELAY_MS = 1200


@dataclass
class UpdateActions:
    kind: str  # "git" or "folder"
    root: str
    version: str
    check: Callable[[], object]
    apply: Callable[[object, Callable, Callable], object]
    restart: Callable[[], bool]
    open_releases: Callable[[], object]
    python: str = "python"
    recorded: str = ""  # folder: the commit a previous update put there


class _Relay(QObject):
    said = pyqtSignal(str, object)  # emitted on the pool thread, delivered on the GUI thread


class _Failure:
    """An exception of the pool thread, kept whole (its template and values) for the GUI thread to translate."""

    def __init__(self, exc: Exception):
        self.exc = exc


def _guarded(fn: Callable[[], object]) -> Callable[[], object]:
    def run():
        try:
            return fn()
        except Exception as exc:  # shown in the dialog, translated there
            return _Failure(exc)
    return run


def _size(n: int) -> str:
    return f"{n / 1e6:.1f} MB" if n >= 1e5 else f"{max(n, 0) / 1e3:.0f} kB"


def _when(iso: str) -> str:
    return iso.replace("T", " ")[:16] if iso else "?"


def _error_text(exc: Exception) -> str:
    template = getattr(exc, "template", None)
    if template:
        return trf(template, **getattr(exc, "values", {}))
    return trf("The update failed: {error}", error=str(exc) or type(exc).__name__)


class UpdateDialog(QDialog):
    def __init__(self, parent: Optional[QWidget], actions: UpdateActions):
        super().__init__(parent)
        self.setObjectName("updateDialog")
        self.setWindowTitle(tr("Update GIMaP"))
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setMinimumWidth(560)
        self.actions = actions
        self.status = None
        self.result = None
        self.busy = False
        self._stop = False
        self._runner = TaskRunner(self, max_threads=1)
        self._relay = _Relay(self)
        self._relay.said.connect(self._progress)
        layout = QVBoxLayout(self)
        self.where = QLabel(self)
        self.where.setObjectName("updateWhere")
        self.where.setWordWrap(True)
        self.where.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.where)
        self.state = QLabel(self)
        self.state.setObjectName("updateState")
        self.state.setWordWrap(True)
        self.state.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.state)
        self.commits = QPlainTextEdit(self)
        self.commits.setObjectName("updateCommits")
        self.commits.setReadOnly(True)
        self.commits.setMaximumHeight(180)
        self.commits.hide()
        layout.addWidget(self.commits)
        buttons = QDialogButtonBox(self)
        self.releases = buttons.addButton(tr("Open Releases Page"), QDialogButtonBox.ActionRole)
        self.releases.clicked.connect(lambda: actions.open_releases())
        self.close_button = buttons.addButton(tr("Close"), QDialogButtonBox.RejectRole)
        self.close_button.clicked.connect(self._close_or_stop)
        self.update_button = buttons.addButton(tr("Update and Restart"), QDialogButtonBox.AcceptRole)
        self.update_button.setObjectName("updateApply")
        self.update_button.setEnabled(False)
        self.update_button.clicked.connect(self._update_or_restart)
        layout.addWidget(buttons)
        self._describe()
        QTimer.singleShot(0, self.check)

    # -- texts ----------------------------------------------------------------------------------------------

    def _describe(self) -> None:
        a = self.actions
        if a.kind == "git":
            text = trf("GIMaP {version} runs from a Git clone: {folder}.", version=a.version, folder=a.root)
        else:
            text = trf("GIMaP {version} runs from a folder downloaded from GitHub: {folder}.", version=a.version,
                       folder=a.root)
            if a.recorded:
                text += " " + trf("Last updated here to {commit}.", commit=a.recorded[:7])
        text += " " + tr("Update fetches the newest version of the branch from GitHub, updates this folder and "
                         "starts GIMaP again. Your data, results and settings are not touched.")
        self.where.setText(text)

    def _show_status(self, status) -> None:
        self.status = status
        values = dict(getattr(status, "values", {}) or {})
        state = status.state
        if state == "up_to_date":
            text = trf("This is the newest version ({commit}, {date}).", commit=status.latest[:7],
                       date=_when(status.latest_date))
        elif state == "behind" and self.actions.kind == "git":
            text = trf("{n} new commits on GitHub ({upstream}, newest {commit} of {date}).", n=status.n_new,
                       upstream=status.upstream, commit=status.latest[:7], date=_when(status.latest_date))
        elif state == "behind":
            text = trf("{files} files differ from the newest version on GitHub ({commit} of {date}): {size} to "
                       "download.", files=len(status.files), commit=status.latest[:7],
                       date=_when(status.latest_date), size=_size(status.download_bytes))
        else:
            text = trf(status.message, **values) if status.message else tr("Nothing to update.")
        if state == "behind" and status.message:
            text += "\n" + trf(status.message, **values)
        self.state.setText(text)
        commits = list(getattr(status, "new_commits", []) or [])
        self.commits.setPlainText("\n".join(commits))
        self.commits.setVisible(bool(commits) and state in ("behind", "blocked"))
        self.update_button.setText(tr("Update and Restart"))
        self.update_button.setEnabled(state == "behind")
        self.update_button.setDefault(state == "behind")

    # -- work -----------------------------------------------------------------------------------------------

    def check(self) -> None:
        self._set_busy(True)
        self.state.setText(tr("Checking GitHub for the newest version…"))
        self._runner.submit("check", _guarded(self.actions.check), on_done=self._checked, on_error=self._failed)

    def _checked(self, status) -> None:
        if isinstance(status, _Failure):
            return self._failed(_error_text(status.exc))
        self._set_busy(False)
        self._show_status(status)

    def _update_or_restart(self) -> None:
        if self.result is not None:  # updated already: only the restart is left
            self._restart()
            return
        if self.status is None or self.status.state != "behind":
            return
        self._stop = False
        self._set_busy(True, stoppable=self.actions.kind == "folder")
        self.state.setText(tr("Updating…"))
        status = self.status
        work = _guarded(lambda: self.actions.apply(status, self._relay.said.emit, lambda: self._stop))
        self._runner.submit("apply", work, on_done=self._applied, on_error=self._failed)

    def _progress(self, template: str, values) -> None:
        if self.busy:
            self.state.setText(trf(template, **(values or {})))

    def _applied(self, result) -> None:
        if isinstance(result, _Failure):
            return self._failed(_error_text(result.exc))
        self._set_busy(False)
        self.result = result
        lines = [trf("Updated to {commit}: {files} files changed.", commit=str(result.new)[:7],
                     files=result.changed_files)]
        if result.backup:
            lines.append(trf("The files it replaced are kept in {folder}.", folder=result.backup))
        self.commits.hide()
        self.update_button.setText(tr("Restart"))
        self.update_button.setEnabled(True)
        if result.requirements:
            lines.append(trf("The update changed the list of Python packages ({files}). Start_GIMaP.bat adds them at "
                             "its next start; in an environment of your own, run this first:",
                             files=", ".join(result.requirements)))
            self.commits.setPlainText("\n".join(f'"{self.actions.python}" -m pip install -r {name}'
                                                for name in result.requirements))
            self.commits.show()
            lines.append(tr("Then restart GIMaP."))
            self.state.setText("\n".join(lines))
            return
        lines.append(tr("GIMaP starts again now…"))
        self.state.setText("\n".join(lines))
        QTimer.singleShot(RESTART_DELAY_MS, self._restart)

    def _restart(self) -> None:
        if not self.actions.restart():
            self.state.setText(self.state.text().replace(tr("GIMaP starts again now…"), "").rstrip() + "\n"
                               + tr("GIMaP was not closed. Restart it to use the new version."))

    def _failed(self, message: str, details: str = "") -> None:
        self._set_busy(False)
        self.state.setText(message)
        if self.status is not None:
            self.update_button.setEnabled(self.status.state == "behind" and self.result is None)

    def _set_busy(self, busy: bool, stoppable: bool = False) -> None:
        self.busy = busy
        self.update_button.setEnabled(False)
        self.releases.setEnabled(not busy)
        self.close_button.setText(tr("Stop") if busy and stoppable else tr("Close"))
        self.close_button.setEnabled(not busy or stoppable)

    def _close_or_stop(self) -> None:
        if self.busy:
            self._stop = True
            self.close_button.setEnabled(False)
            return
        self.close()

    def closeEvent(self, event) -> None:
        if self.busy:  # a running update finishes (or stops) before the window may go
            self._stop = True
            event.ignore()
            return
        self._runner.shutdown(2000)
        super().closeEvent(event)


def show_update_dialog(parent: Optional[QWidget], actions: UpdateActions) -> UpdateDialog:
    shown = parent.findChild(QDialog, "updateDialog", Qt.FindDirectChildrenOnly) if parent is not None else None
    if shown is not None:
        shown.raise_()
        shown.activateWindow()
        return shown
    dialog = UpdateDialog(parent, actions)
    dialog.show()
    return dialog
