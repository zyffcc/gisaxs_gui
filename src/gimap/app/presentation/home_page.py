"""The start page: open data, choose what you want to know, or ask the AI.

For a new user the first screen is a question, not a toolbar: drop detector
images (or open them), pick a task card — crystals and orientation (GIWAXS),
nanostructure (GISAXS), an in-situ or batch series, detector calibration — or
describe the data and the question for the AI.  The page only emits requests;
the main window decides which workspace opens.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .theme import set_role

TASKS = (
    ("giwaxs", "Crystals and orientation", "GIWAXS",
     "Peaks, in-plane / out-of-plane, ring orientation, crystallite size — a guided analysis, no AI needed."),
    ("gisaxs", "Nanostructure", "GISAXS",
     "Yoneda cut, symmetric halves, spacing and a model fit — automatic, no AI needed."),
    ("series", "In-situ or batch series", "Series",
     "A frame × q map of the run, peaks followed through it, and Batch Export of every frame to a folder."),
    ("calibrate", "Calibrate the detector", "Calibration",
     "Distance and beam centre from an image of a standard (AgBh, LaB6, CeO2, LaB6 + CeO2)."),
)
MAX_WIDTH = 1040


def _label(text: str, parent: QWidget, *, role: str = "", size: float = 0.0, bold: bool = False) -> QLabel:
    label = QLabel(text, parent)
    label.setWordWrap(True)
    if role:
        label.setProperty("gimapRole", role)
    font = label.font()
    if size:
        font.setPointSizeF(font.pointSizeF() * size)
    font.setBold(bold)
    label.setFont(font)
    return label


class TaskCard(QFrame):
    clicked = pyqtSignal(str)

    def __init__(self, key: str, title: str, tag: str, text: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.key = key
        self.setObjectName(f"homeTask_{key}")
        self.setFrameShape(QFrame.StyledPanel)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(128)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.addWidget(_label(tag, self, role="muted"))
        layout.addWidget(_label(title, self, size=1.15, bold=True))
        layout.addWidget(_label(text, self, role="muted"))
        layout.addStretch(1)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self.rect().contains(event.pos()):
            self.click()
        super().mouseReleaseEvent(event)

    def click(self) -> None:
        self.clicked.emit(self.key)


class DropZone(QFrame):
    filesDropped = pyqtSignal(list)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("homeDropZone")
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("#homeDropZone { border: 2px dashed palette(mid); border-radius: 10px; }")

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self.filesDropped.emit(paths)
            event.acceptProposedAction()


class HomePage(QWidget):
    openRequested = pyqtSignal()
    folderRequested = pyqtSignal()
    batchRequested = pyqtSignal()
    recentRequested = pyqtSignal(str)
    """A recently opened file or folder, to open again."""
    filesDropped = pyqtSignal(list)
    taskRequested = pyqtSignal(str)
    askRequested = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("homePage")
        outer = QHBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)
        column_host = QWidget(self)
        column_host.setMaximumWidth(MAX_WIDTH)
        outer.addStretch(1)
        outer.addWidget(column_host, 12)
        outer.addStretch(1)
        column = QVBoxLayout(column_host)
        column.setSpacing(16)
        column.addWidget(_label("GIMaP", column_host, size=2.0, bold=True))
        column.addWidget(_label(
            "Grazing-incidence scattering, step by step: open your data, say what you want to know, "
            "and every result comes with how it was found.", column_host, role="muted", size=1.1,
        ))

        self.drop_zone = DropZone(column_host)
        drop = QVBoxLayout(self.drop_zone)
        drop.setContentsMargins(20, 18, 20, 18)
        for text, role, size, bold in (
            ("Drag detector images or a folder here", "", 1.2, True),
            ("CBF, TIFF, EDF, NeXus (multi-module series are stitched automatically)", "muted", 0.0, False),
        ):
            label = _label(text, self.drop_zone, role=role, size=size, bold=bold)
            label.setAlignment(Qt.AlignHCenter)  # centred text in a full-width label: nothing is clipped
            drop.addWidget(label)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.open_button = QPushButton("Open Files…", self.drop_zone)
        self.open_button.setObjectName("homeOpenButton")
        self.folder_button = QPushButton("Open Folder…", self.drop_zone)
        self.batch_button = QPushButton("Batch Export…", self.drop_zone)
        self.batch_button.setObjectName("homeBatchButton")
        self.batch_button.setToolTip(
            "A folder of raw frames straight to curves in a folder of your choice, with the settings of last time"
        )
        buttons.addWidget(self.open_button)
        buttons.addWidget(self.folder_button)
        buttons.addWidget(self.batch_button)
        buttons.addStretch(1)
        drop.addLayout(buttons)
        column.addWidget(self.drop_zone)
        self.recent_box = QWidget(column_host)
        self.recent_box.setObjectName("homeRecent")
        recent = QVBoxLayout(self.recent_box)
        recent.setContentsMargins(0, 0, 0, 0)
        recent.setSpacing(2)
        recent.addWidget(_label("Recent", self.recent_box, size=1.05, bold=True))
        self.recent_rows = QVBoxLayout()
        self.recent_rows.setSpacing(0)
        recent.addLayout(self.recent_rows)
        self.recent_box.hide()
        column.addWidget(self.recent_box)
        self._recent_provider = None

        column.addWidget(_label("What do you want to know?", column_host, size=1.2, bold=True))
        grid = QGridLayout()
        grid.setSpacing(12)
        self.cards: dict[str, TaskCard] = {}
        for index, (key, title, tag, text) in enumerate(TASKS):
            card = TaskCard(key, title, tag, text, column_host)
            card.clicked.connect(self.taskRequested)
            grid.addWidget(card, index // 2, index % 2)
            self.cards[key] = card
        column.addLayout(grid)

        column.addWidget(_label("Or ask the AI", column_host, size=1.2, bold=True))
        ask = QHBoxLayout()
        self.ask_edit = QLineEdit(column_host)
        self.ask_edit.setObjectName("homeAskEdit")
        self.ask_edit.setPlaceholderText(
            "e.g. In-situ GIWAXS at P03, Cu sputtered onto PEO, αi 0.4° — how does the film crystallise?"
        )
        self.ask_button = QPushButton("Ask the AI…", column_host)
        self.ask_button.setObjectName("homeAskButton")
        ask.addWidget(self.ask_edit, 1)
        ask.addWidget(self.ask_button)
        column.addLayout(ask)
        column.addWidget(_label(
            "The AI (Claude, DeepSeek, Qwen, OpenAI, a local model …, chosen in Settings) uses the same "
            "tools; its changes come back as cards you can preview, apply or undo.", column_host, role="muted",
        ))
        column.addStretch(1)

        self.open_button.clicked.connect(self.openRequested)
        self.folder_button.clicked.connect(self.folderRequested)
        self.batch_button.clicked.connect(self.batchRequested)
        self.drop_zone.filesDropped.connect(self.filesDropped)
        self.ask_button.clicked.connect(self._ask)
        self.ask_edit.returnPressed.connect(self._ask)

    def set_recent_provider(self, provider) -> None:
        """``provider()`` → the recent paths, newest first (asked each time the page is shown)."""
        self._recent_provider = provider
        self.refresh_recent()

    def refresh_recent(self) -> None:
        from pathlib import Path

        while self.recent_rows.count():
            item = self.recent_rows.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        paths = list(self._recent_provider() or [])[:5] if self._recent_provider is not None else []
        for path in paths:
            path = Path(path)
            button = QPushButton(f"{path.name or path}   —   {path.parent}", self.recent_box)
            button.setObjectName("homeRecentItem")
            button.setFlat(True)
            button.setCursor(Qt.PointingHandCursor)
            button.setToolTip(str(path))
            set_role(button, "link")
            button.clicked.connect(lambda _checked=False, target=str(path): self.recentRequested.emit(target))
            self.recent_rows.addWidget(button)
        self.recent_box.setVisible(bool(paths))

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().showEvent(event)
        self.refresh_recent()

    def _ask(self) -> None:
        self.askRequested.emit(self.ask_edit.text().strip())


__all__ = ["HomePage", "TASKS", "TaskCard"]
