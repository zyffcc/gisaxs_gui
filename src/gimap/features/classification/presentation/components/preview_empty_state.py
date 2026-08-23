"""Empty-state overlay for the Classification sample preview."""

from __future__ import annotations

from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtWidgets import QGraphicsView, QLabel


class ClassificationEmptyState(QLabel):
    """Non-interactive guidance shown while the sample preview is empty."""

    def __init__(self, view: QGraphicsView) -> None:
        super().__init__(
            "Add files or folders to inspect a sample.\nLabels can be added later.",
            view.viewport(),
        )
        self.view = view
        self.setObjectName("classificationPreviewEmptyState")
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        view.viewport().installEventFilter(self)
        self.refresh()

    def eventFilter(self, watched, event):
        if watched is self.view.viewport() and event.type() in (
            QEvent.Resize,
            QEvent.Show,
            QEvent.Paint,
        ):
            self.refresh()
        return False

    def refresh(self) -> None:
        scene = self.view.scene()
        self.setVisible(scene is None or not scene.items())
        self.setGeometry(self.view.viewport().rect().adjusted(24, 24, -24, -24))
        self.raise_()


__all__ = ["ClassificationEmptyState"]
