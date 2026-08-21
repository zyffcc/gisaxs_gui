"""Reusable Qt-side coordination for parameter preview and commit signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PyQt5.QtCore import QObject, QTimer


@dataclass(frozen=True)
class ParameterUpdatePolicy:
    """Timing policy for one group of controls that forms a single command."""

    debounce_ms: int = 220
    preview_interval_ms: int = 60


class _ParameterGroup:
    def __init__(
        self,
        owner: QObject,
        commit: Callable[[], None],
        preview: Callable[[], None] | None,
        policy: ParameterUpdatePolicy,
    ) -> None:
        self.commit = commit
        self.preview = preview
        self.dirty = False
        self.preview_pending = False
        self.commit_timer = QTimer(owner)
        self.commit_timer.setSingleShot(True)
        self.commit_timer.setInterval(max(0, int(policy.debounce_ms)))
        self.commit_timer.timeout.connect(self.flush)
        self.preview_timer = QTimer(owner)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(max(0, int(policy.preview_interval_ms)))
        self.preview_timer.timeout.connect(self._flush_preview)

    def changed(self) -> None:
        self.dirty = True
        if self.preview is not None:
            self.preview_pending = True
            if not self.preview_timer.isActive():
                self._flush_preview()
        self.commit_timer.start()

    def flush(self, *, force: bool = False) -> None:
        self.commit_timer.stop()
        self._flush_preview()
        if not self.dirty and not force:
            return
        self.dirty = False
        self.commit()

    def _flush_preview(self) -> None:
        self.preview_timer.stop()
        if not self.preview_pending or self.preview is None:
            return
        self.preview_pending = False
        self.preview()
        self.preview_timer.start()


class ParameterCommitCoordinator(QObject):
    """Coalesce rapid edits while keeping Enter/focus-out commits immediate."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._groups: dict[str, _ParameterGroup] = {}

    def register_group(
        self,
        key: str,
        *,
        commit: Callable[[], None],
        preview: Callable[[], None] | None = None,
        policy: ParameterUpdatePolicy | None = None,
    ) -> None:
        if key in self._groups:
            raise ValueError(f"Parameter group already registered: {key}")
        self._groups[key] = _ParameterGroup(
            self,
            commit,
            preview,
            policy or ParameterUpdatePolicy(),
        )

    def bind_numeric(self, key: str, widget) -> None:
        group = self._group(key)
        widget.valueChanged.connect(lambda _value: group.changed())
        widget.editingFinished.connect(group.flush)

    def bind_toggle(self, key: str, widget) -> None:
        group = self._group(key)
        widget.toggled.connect(lambda _checked: group.flush(force=True))

    def flush(self, key: str) -> None:
        self._group(key).flush(force=True)

    def _group(self, key: str) -> _ParameterGroup:
        try:
            return self._groups[key]
        except KeyError as exc:
            raise KeyError(f"Unknown parameter group: {key}") from exc


__all__ = ["ParameterCommitCoordinator", "ParameterUpdatePolicy"]
