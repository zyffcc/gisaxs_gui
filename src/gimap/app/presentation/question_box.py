"""A question with buttons of its own (Replace / Add to List / Cancel, Save / Don't Save / Cancel …).

``QuestionBox`` is a plain dialog with the parts of ``QMessageBox`` that GIMaP uses (``addButton``,
``setDefaultButton``, ``setEscapeButton``, ``text``, ``informativeText``, ``buttons``, ``clickedButton``). It does
not use ``QMessageBox``: on Windows its ``showEvent`` asks the platform for the window's system menu, which the
offscreen platform (tests, scripted runs, a remote session without a display) does not provide, and the process
died with an access violation when the question was shown.

``ask_choice`` shows one and returns the index of the button chosen. The question is modal to its window while it
waits, and the event loop keeps running (background tasks finish, signals arrive): a caller checks again
afterwards whether what it asked about is still possible."""

from __future__ import annotations

from typing import Optional, Sequence

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractButton,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStyle,
    QVBoxLayout,
    QWidget,
)

ACCEPT, REJECT, DESTRUCTIVE, ACTION = (QDialogButtonBox.AcceptRole, QDialogButtonBox.RejectRole,
                                       QDialogButtonBox.DestructiveRole, QDialogButtonBox.ActionRole)
"""Button roles (the values of ``QMessageBox.ButtonRole`` too): where the platform puts each button."""


class QuestionBox(QDialog):
    """A question and its buttons; ``exec_`` returns when one is pressed (Esc and the window's close button press
    the escape button, or none)."""

    def __init__(self, parent: Optional[QWidget] = None, title: str = "", text: str = "", informative: str = ""):
        super().__init__(parent)
        self.setObjectName("questionBox")
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModal if parent is not None else Qt.ApplicationModal)
        self._clicked: Optional[QAbstractButton] = None
        self._escape: Optional[QAbstractButton] = None
        self._default: Optional[QPushButton] = None
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        icon = QLabel(self)
        icon.setObjectName("questionIcon")
        icon.setPixmap(self.style().standardIcon(QStyle.SP_MessageBoxQuestion).pixmap(32, 32))
        row.addWidget(icon, 0, Qt.AlignTop)
        texts = QVBoxLayout()
        self._text = QLabel(text, self)
        self._text.setObjectName("questionText")
        self._informative = QLabel(informative, self)
        self._informative.setObjectName("questionInformative")
        for label in (self._text, self._informative):
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            label.setMinimumWidth(320)
            label.setMaximumWidth(520)
            texts.addWidget(label)
        self._informative.setVisible(bool(informative))
        row.addLayout(texts, 1)
        layout.addLayout(row)
        self._buttons = QDialogButtonBox(self)
        self._buttons.clicked.connect(self._chosen)
        layout.addWidget(self._buttons)

    # -- the parts of QMessageBox that GIMaP uses -----------------------------------------------------------------
    def setText(self, text: str) -> None:
        self._text.setText(text)

    def text(self) -> str:
        return self._text.text()

    def setInformativeText(self, text: str) -> None:
        self._informative.setText(text)
        self._informative.setVisible(bool(text))

    def informativeText(self) -> str:
        return self._informative.text()

    def addButton(self, text: str, role=ACTION) -> QPushButton:
        button = self._buttons.addButton(text, QDialogButtonBox.ButtonRole(int(role)))
        button.setAutoDefault(False)
        return button

    def buttons(self) -> list:
        return list(self._buttons.buttons())

    def setDefaultButton(self, button: QPushButton) -> None:
        for each in self._buttons.buttons():
            each.setDefault(each is button)
        self._default = button
        button.setFocus()

    def defaultButton(self) -> Optional[QPushButton]:
        return self._default

    def setEscapeButton(self, button: QAbstractButton) -> None:
        self._escape = button

    def escapeButton(self) -> Optional[QAbstractButton]:
        return self._escape

    def clickedButton(self) -> Optional[QAbstractButton]:
        return self._clicked

    # -- answering ------------------------------------------------------------------------------------------------
    def _chosen(self, button: QAbstractButton) -> None:
        self._clicked = button
        self.done(QDialog.Accepted if self._buttons.buttonRole(button) != REJECT else QDialog.Rejected)

    def reject(self) -> None:  # Esc, the close button of the window
        if self._clicked is None:
            self._clicked = self._escape
        super().reject()


def ask_choice(parent: Optional[QWidget], title: str, text: str, buttons: Sequence[tuple[str, int]], *, default: int = 0,
               escape: Optional[int] = None, informative: str = "") -> int:
    """Ask ``text`` with the buttons ``(label, role)``; the index of the one chosen (``escape`` for Esc or the close
    button of the window; -1 when there is none). The texts come in the interface language already."""
    box = QuestionBox(parent, title, text, informative)
    made = [box.addButton(label, role) for label, role in buttons]
    if made:
        box.setDefaultButton(made[default])
        if escape is not None:
            box.setEscapeButton(made[escape])
    try:
        box.exec_()
        clicked = box.clickedButton()
        return made.index(clicked) if clicked in made else (-1 if escape is None else escape)
    finally:
        box.deleteLater()


__all__ = ["ACCEPT", "ACTION", "DESTRUCTIVE", "QuestionBox", "REJECT", "ask_choice"]
