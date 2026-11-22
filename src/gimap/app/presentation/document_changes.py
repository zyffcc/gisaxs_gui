"""The hook a page calls when the user changed its work: ``notify_changed()``.

The window's ``DocumentSession`` (``app/document_session.py``) looks at the work again a moment later and puts
``*`` in the title when it differs from what was saved. The session also looks every second and a half, so a
page that does not call the hook is still noticed; the hook makes it immediate. Without a session (tests, a
dialog on its own) nobody listens and a call costs nothing.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import QObject, pyqtSignal


class DocumentChanges(QObject):
    changed = pyqtSignal()


_CHANGES: Optional[DocumentChanges] = None


def document_changes() -> DocumentChanges:
    global _CHANGES
    if _CHANGES is None:
        _CHANGES = DocumentChanges()
    return _CHANGES


def notify_changed(*_args) -> None:
    """Something of the work may have changed (a frame opened, a mask drawn, a model edited)."""
    document_changes().changed.emit()


__all__ = ["DocumentChanges", "document_changes", "notify_changed"]
