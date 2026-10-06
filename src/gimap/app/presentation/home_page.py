"""The start page: open data, choose what you want to know, or ask the AI.

For a new user the first screen is a question, not a toolbar: drop detector
images (or open them), pick a task card — crystals and orientation (GIWAXS),
nanostructure (GISAXS), an in-situ or batch series, detector calibration — or
describe the data and the question for the AI.  The page only emits requests;
the main window decides which workspace opens.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .recent_items import recent_label
from .theme import set_role, set_state

TASKS = (
    ("giwaxs", "Crystals and orientation", "GIWAXS",
     "Peaks, in-plane / out-of-plane, ring orientation, crystallite size — a guided analysis, no AI needed."),
    ("gisaxs", "Nanostructure", "GISAXS",
     "Yoneda cut, symmetric halves, spacing and a model fit — automatic, no AI needed."),
    ("series", "In-situ or batch series", "Series",
     "A frame × q map of the run with its stages and odd frames, peaks followed through it, Batch Export of every frame — and Compare for several runs."),
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
    """A clickable card (``card`` look of the theme, ``homeTask`` for its hover state)."""

    clicked = pyqtSignal(str)

    def __init__(self, key: str, title: str, tag: str, text: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.key = key
        self.setObjectName(f"homeTask_{key}")
        self.setProperty("card", True)
        self.setProperty("homeTask", True)
        self.setAttribute(Qt.WA_Hover, True)
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
    """The dashed box (``#homeDropZone``); ``dragActive`` is true while files are dragged over it."""

    filesDropped = pyqtSignal(list)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("homeDropZone")
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.StyledPanel)  # the theme's #homeDropZone rule draws the dashed border
        self.setProperty("dragActive", False)

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            set_state(self, "dragActive", True)

    def dragLeaveEvent(self, event) -> None:  # noqa: N802 - Qt API
        set_state(self, "dragActive", False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt API
        set_state(self, "dragActive", False)
        paths = [str(Path(url.toLocalFile())) for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self.filesDropped.emit(paths)
            event.acceptProposedAction()


class HomePage(QWidget):
    openRequested = pyqtSignal()
    folderRequested = pyqtSignal()
    batchRequested = pyqtSignal()
    projectRequested = pyqtSignal()
    recentRequested = pyqtSignal(str)
    """A recently opened file or folder, to open again."""
    filesDropped = pyqtSignal(list)
    taskRequested = pyqtSignal(str)
    askRequested = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("homePage")
        # The page scrolls (vertically only) instead of squeezing its sections into each other on a small screen.
        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setObjectName("homeScrollArea")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        page_layout.addWidget(self.scroll_area)
        content = QWidget()
        content.setObjectName("homeContent")
        outer = QHBoxLayout(content)
        outer.setContentsMargins(24, 24, 24, 24)
        column_host = QWidget(content)
        column_host.setMaximumWidth(MAX_WIDTH)
        outer.addStretch(1)
        outer.addWidget(column_host, 12)
        outer.addStretch(1)
        self.scroll_area.setWidget(content)
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
        self.open_button.setToolTip("Detector frames (CBF, NXS, TIFF, EDF), opened in Analyze — Ctrl+O")
        self.folder_button = QPushButton("Open Folder…", self.drop_zone)
        self.folder_button.setToolTip("Every detector frame of a folder, opened in Analyze")
        self.batch_button = QPushButton("Batch Export…", self.drop_zone)
        self.batch_button.setObjectName("homeBatchButton")
        self.batch_button.setToolTip(
            "A folder of raw frames straight to curves in a folder of your choice, with the settings of last time"
        )
        self.project_button = QPushButton("Open Project…", self.drop_zone)
        self.project_button.setObjectName("homeProjectButton")
        self.project_button.setToolTip("Reopen a sample as it was left (File ▸ Save Project)")
        buttons.addWidget(self.open_button)
        buttons.addWidget(self.folder_button)
        buttons.addWidget(self.batch_button)
        buttons.addWidget(self.project_button)
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
        self.ask_button.setToolTip("The AI works with the same tools as you, with what you wrote as its task; "
                                   "its changes come back as cards to preview, apply or undo")
        ask.addWidget(self.ask_edit, 1)
        ask.addWidget(self.ask_button)
        column.addLayout(ask)
        column.addWidget(_label(
            "The AI (Claude, DeepSeek, Qwen, OpenAI, a local model …, chosen in Settings) uses the same "
            "tools; its changes come back as cards you can preview, apply or undo.", column_host, role="muted",
        ))
        column.addStretch(1)

        self.open_button.clicked.connect(self.openRequested)
        self.project_button.clicked.connect(self.projectRequested)
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
        """``name — parent folder name`` (with a folder or project tag) per row, the full path as tooltip."""
        while self.recent_rows.count():
            item = self.recent_rows.takeAt(0)
            if item.widget() is not None:  # hidden now: deleted only once the event loop runs again
                item.widget().hide()
                item.widget().deleteLater()
        paths = list(self._recent_provider() or [])[:5] if self._recent_provider is not None else []
        for path in paths:
            path = Path(path)
            button = QPushButton(recent_label(path), self.recent_box)
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

    def refresh_language(self) -> None:
        """After a switch of the interface language (the shell calls it): the recent rows again, whose
        `` (folder)`` / `` (project)`` tags are composed with ``tr`` (the page may be the one shown)."""
        self.refresh_recent()

    def _ask(self) -> None:
        self.askRequested.emit(self.ask_edit.text().strip())


__all__ = ["HomePage", "TASKS", "TaskCard"]
