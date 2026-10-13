"""“Claude asks”: pick one of the options Claude offers (e.g. calibration files with their times)."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
)

from src.gimap.app.presentation.i18n import tr

from .gui_bridge import GuiBridge


class ChoiceDialog(QDialog):
    def __init__(self, question: str, options: Sequence[dict], allow_text: bool, parent=None):
        super().__init__(parent)
        self.setObjectName("assistantChoiceDialog")
        self.setWindowTitle(tr("The AI asks"))
        self.setMinimumWidth(560)
        layout = QVBoxLayout(self)
        prompt = QLabel(question, self)
        prompt.setWordWrap(True)
        prompt.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(prompt)
        self.option_list = QListWidget(self)
        self.option_list.setObjectName("assistantChoiceList")
        self.option_list.setWordWrap(True)
        self.option_list.setAlternatingRowColors(True)
        for option in options:
            label = str(option.get("label", ""))
            detail = str(option.get("detail", "") or "")
            item = QListWidgetItem(f"{label}\n{detail}" if detail else label)
            item.setToolTip(f"{label}\n{detail}".strip())
            self.option_list.addItem(item)
        self.option_list.setVisible(bool(options))
        layout.addWidget(self.option_list, 1)
        self.text_edit = QLineEdit(self)
        self.text_edit.setObjectName("assistantChoiceText")
        self.text_edit.setPlaceholderText(tr("Or type the answer here…") if options else tr("Type the answer here…"))
        self.text_edit.setVisible(bool(allow_text))
        layout.addWidget(self.text_edit)
        buttons = QDialogButtonBox(self)
        self.use_button = buttons.addButton(tr("Use This"), QDialogButtonBox.AcceptRole)
        self.none_button = buttons.addButton(tr("None of These") if options else tr("Skip"), QDialogButtonBox.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.option_list.currentRowChanged.connect(self._sync)
        self.text_edit.textChanged.connect(self._sync)
        self.option_list.itemDoubleClicked.connect(lambda _item: self.accept())
        if options:
            self.option_list.setCurrentRow(0)
        self._sync()

    def _sync(self, *_args) -> None:
        chosen = self.option_list.count() > 0 and self.option_list.currentRow() >= 0
        self.use_button.setEnabled(chosen or bool(self.text_edit.text().strip()))

    def answer(self) -> dict:
        row = self.option_list.currentRow() if self.option_list.count() else -1
        return {"index": row if row >= 0 else None, "text": self.text_edit.text().strip()}


class GuiChooser:
    """The ``Chooser`` port: shows ``ChoiceDialog`` on the GUI thread and waits for the answer."""

    def __init__(self, bridge: GuiBridge, parent, *, cancelled: Callable[[], bool] = lambda: False):
        self._bridge = bridge
        self._parent = parent
        self._cancelled = cancelled

    def choose(self, question: str, options: Sequence[dict], allow_text: bool) -> Optional[dict]:
        def ask() -> Optional[dict]:
            dialog = ChoiceDialog(question, options, allow_text, self._parent)
            return dialog.answer() if dialog.exec_() == QDialog.Accepted else None

        return self._bridge.call(ask, cancelled=self._cancelled)


__all__ = ["ChoiceDialog", "GuiChooser"]
