"""Reusable empty, error and job-state presentation components。"""

from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..styles import apply_design_system


class EmptyState(QFrame):
    actionRequested = pyqtSignal()

    def __init__(
        self,
        title: str = "Nothing here yet",
        message: str = "",
        parent: QWidget | None = None,
        *,
        action_text: str = "",
    ) -> None:
        super().__init__(parent)
        self.setProperty("gimapEmptyState", True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(7)
        self.title_label = QLabel(title, self)
        self.title_label.setProperty("gimapEmptyTitle", True)
        self.title_label.setWordWrap(True)
        self.message_label = QLabel(message, self)
        self.message_label.setProperty("gimapMeta", True)
        self.message_label.setWordWrap(True)
        self.action_button = QPushButton(action_text, self)
        self.action_button.setProperty("gimapPrimaryAction", True)
        self.action_button.setVisible(bool(action_text))
        layout.addStretch(1)
        layout.addWidget(self.title_label, 0)
        layout.addWidget(self.message_label, 0)
        layout.addWidget(self.action_button, 0)
        layout.addStretch(1)
        self.action_button.clicked.connect(self.actionRequested)
        apply_design_system(self)

    def set_content(self, title: str, message: str = "", action_text: str = "") -> None:
        self.title_label.setText(title)
        self.message_label.setText(message)
        self.action_button.setText(action_text)
        self.action_button.setVisible(bool(action_text))


class ErrorBanner(QFrame):
    dismissed = pyqtSignal()
    detailsRequested = pyqtSignal()

    LEVELS = {"error", "warning", "info", "success"}

    def __init__(
        self,
        title: str = "",
        message: str = "",
        parent: QWidget | None = None,
        *,
        level: str = "error",
        dismissible: bool = True,
        show_details: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setProperty("gimapBannerLevel", "error")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 8, 8)
        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(2)
        self.title_label = QLabel(title, self)
        self.title_label.setProperty("gimapBannerTitle", True)
        self.message_label = QLabel(message, self)
        self.message_label.setWordWrap(True)
        text_layout.addWidget(self.title_label)
        text_layout.addWidget(self.message_label)
        layout.addLayout(text_layout, 1)
        self.details_button = QPushButton("Details", self)
        self.details_button.setVisible(show_details)
        self.dismiss_button = QToolButton(self)
        self.dismiss_button.setText("×")
        self.dismiss_button.setVisible(dismissible)
        layout.addWidget(self.details_button)
        layout.addWidget(self.dismiss_button)
        self.details_button.clicked.connect(self.detailsRequested)
        self.dismiss_button.clicked.connect(self._dismiss)
        self.set_level(level)
        apply_design_system(self)

    def set_level(self, level: str) -> None:
        normalized = level if level in self.LEVELS else "error"
        self.setProperty("gimapBannerLevel", normalized)
        self.style().unpolish(self)
        self.style().polish(self)

    def set_message(self, title: str, message: str) -> None:
        self.title_label.setText(title)
        self.message_label.setText(message)

    def _dismiss(self) -> None:
        self.hide()
        self.dismissed.emit()


class JobStatus(QFrame):
    pauseRequested = pyqtSignal(bool)
    cancelRequested = pyqtSignal()
    detailsRequested = pyqtSignal()

    STATES = {
        "idle",
        "queued",
        "running",
        "paused",
        "succeeded",
        "failed",
        "cancelled",
        "timed_out",
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("gimapJobStatus", True)
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)
        top = QHBoxLayout()
        self.state_label = QLabel("IDLE", self)
        self.state_label.setProperty("gimapJobState", "idle")
        self.message_label = QLabel("Ready", self)
        self.message_label.setWordWrap(True)
        self.message_label.setProperty("gimapMeta", True)
        top.addWidget(self.state_label)
        top.addWidget(self.message_label, 1)
        root.addLayout(top)
        lower = QHBoxLayout()
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.pause_button = QPushButton("Pause", self)
        self.pause_button.setCheckable(True)
        self.cancel_button = QPushButton("Cancel", self)
        self.details_button = QPushButton("Details", self)
        lower.addWidget(self.progress_bar, 1)
        lower.addWidget(self.pause_button)
        lower.addWidget(self.cancel_button)
        lower.addWidget(self.details_button)
        root.addLayout(lower)
        self.pause_button.toggled.connect(self._pause_toggled)
        self.cancel_button.clicked.connect(self.cancelRequested)
        self.details_button.clicked.connect(self.detailsRequested)
        self.set_state("idle", "Ready", progress=0.0)
        apply_design_system(self)

    def set_state(
        self,
        state: str,
        message: str = "",
        *,
        progress: float | None = None,
    ) -> None:
        normalized = state if state in self.STATES else "idle"
        self.state_label.setText(normalized.replace("_", " ").upper())
        self.state_label.setProperty("gimapJobState", normalized)
        self.state_label.style().unpolish(self.state_label)
        self.state_label.style().polish(self.state_label)
        self.message_label.setText(message)
        if progress is None:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, 1000)
            self.progress_bar.setValue(round(max(0.0, min(1.0, progress)) * 1000))
        active = normalized in {"queued", "running", "paused"}
        self.pause_button.setEnabled(active)
        self.cancel_button.setEnabled(active)
        self.pause_button.blockSignals(True)
        self.pause_button.setChecked(normalized == "paused")
        self.pause_button.setText("Resume" if normalized == "paused" else "Pause")
        self.pause_button.blockSignals(False)

    def set_actions_visible(
        self,
        *,
        pause: bool = True,
        cancel: bool = True,
        details: bool = True,
    ) -> None:
        self.pause_button.setVisible(pause)
        self.cancel_button.setVisible(cancel)
        self.details_button.setVisible(details)

    def _pause_toggled(self, paused: bool) -> None:
        self.pause_button.setText("Resume" if paused else "Pause")
        self.pauseRequested.emit(paused)
