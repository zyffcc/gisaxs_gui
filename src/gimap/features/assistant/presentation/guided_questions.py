"""The questions only a person can answer, in the Results step of Analyze: one field per value a run reads.

A run that misses αi, the energy, the pixel size, the calibration or its standard asks for it here; the points
that share a value share its field (and its tooltip, with every reason). The calibration file has **…** beside
it: a file dialog that starts in the frame's folder (images of a standard and .poni files). The standard is a
choice of the four GIMaP can fit. Enter in any field runs again. Answers are kept for this session only (until
Analyze ▸ Clear or a project opens), never saved: an old αi or energy would silently change qz of new data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PyQt5.QtCore import QDir, QEvent, QObject, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.presentation.i18n import tr, trf

from ..application import READABLE_IMAGE_SUFFIXES, SUPPORTED_STANDARDS
from .guided_text import OPTION_FIELDS, answerable, label, listed
from .guided_words import words

STANDARD_LABELS = {"agbh": "AgBH", "lab6": "LaB6", "ceo2": "CeO2", "lab6_ceo2": "LaB6 + CeO2"}
"""What the choice of the standard shows (formulas, never translated); the run reads the key (``agbh`` …)."""
CHOOSE_TEXT = "Choose…"
CALIBRATION_TITLE = "Calibration File"


def answer_text(field: QWidget) -> str:
    """What a field holds: the text typed, or the key of the choice made ("" when nothing is chosen)."""
    if isinstance(field, QComboBox):
        return str(field.currentData() or "")
    return field.text().strip()


def calibration_filters() -> str:
    """The file dialog's filters: images GIMaP can fit and .poni files first, then GIMaP calibrations, then all."""
    patterns = " ".join(f"*{suffix}" for suffix in (*READABLE_IMAGE_SUFFIXES, ".poni"))
    return ";;".join((f"{tr('Images of a standard and .poni files')} ({patterns})",
                      f"{tr('GIMaP calibration')} (*.json)", f"{tr('All files')} (*)"))


class _EnterRuns(QObject):
    """Enter on a closed drop-down runs again, as it does in a text field (an open list takes Enter itself)."""

    def __init__(self, run: Callable[[], None], parent: QObject):
        super().__init__(parent)
        self._run = run

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API
        if event.type() == QEvent.KeyPress and event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._run()
            return True
        return False


class AnswerForm(QFrame):
    """The fields for the questions of the last run, the answers given before, and Run Again."""

    runRequested = pyqtSignal()
    """Run Again, or Enter in a field."""

    def __init__(self, folder: Callable[[], str], parent: Optional[QWidget] = None):
        """``folder()``: the folder of the frame shown, where the calibration file dialog starts."""
        super().__init__(parent)
        self.setObjectName("guidedQuestions")
        self.setProperty("gimapInfoCard", True)
        self._folder = folder
        self.fields: dict[str, QWidget] = {}
        """The field of every question shown (a ``QLineEdit``, or a ``QComboBox`` for the standard)."""
        self.answers: dict[str, str] = {}
        """Answers of earlier rounds: kept when the next question replaces the fields."""
        self._attention: list = []
        layout = QVBoxLayout(self)
        self.title_label = label(tr("Only you can answer these — then run again:"), self, bold=True)
        layout.addWidget(self.title_label)
        self.form = QFormLayout()
        self.form.setRowWrapPolicy(QFormLayout.WrapAllRows)  # each title above its field: the narrow step panel's width for a path
        layout.addLayout(self.form)
        self.again_button = QPushButton(tr("Run Again with These Answers"), self)
        self.again_button.setObjectName("guidedAgainButton")
        self.again_button.setToolTip(tr("Or press Enter in any of the fields"))
        layout.addWidget(self.again_button, 0, Qt.AlignLeft)
        self.answers_label = label("", self, role="muted")
        layout.addWidget(self.answers_label)
        self.again_button.clicked.connect(self.runRequested)
        self.hide()

    # -- the answers --------------------------------------------------------------------------

    def remember(self) -> None:
        """Keep what the fields hold now (an empty field keeps the answer given before)."""
        for option, field in self.fields.items():
            text = answer_text(field)
            if text:
                self.answers[option] = text

    def clear(self) -> None:
        """No fields (the answers stay)."""
        while self.form.rowCount():
            self.form.removeRow(0)
        self.fields.clear()
        self.answers_label.setText("")
        self.hide()

    def forget(self) -> None:
        """Analyze ▸ Clear or a project: no fields and no answers (an old αi would override a project's own)."""
        self.answers.clear()
        self._attention = []
        self.clear()

    def show_questions(self, attention) -> None:
        """One field per value the run asks for (``answerable``), filled with the answer given before."""
        self.remember()
        self.clear()
        self._attention = list(attention or ())
        for item in answerable(self._attention):
            option = item["option"]
            title, placeholder = OPTION_FIELDS[option]
            field, shown = self._field(option, placeholder)
            field.setObjectName(f"guidedAnswer_{option}")
            field.setToolTip("\n\n".join(
                f"{words(point.get('why'))}\n{words(point.get('hint'))}".strip()
                for point in self._attention if point.get("option") == option))
            self.form.addRow(tr(title), shown)  # built after the window was translated
            self.fields[option] = field
        given = [f"{tr(OPTION_FIELDS[key][0])}: {self._shown_answer(key, value)}" for key, value in self.answers.items()
                 if key in OPTION_FIELDS and key not in self.fields]
        self.answers_label.setText(trf("Your earlier answers are kept: {answers}", answers=listed(given, "; ")) if given else "")
        self.setVisible(bool(self.fields))

    def refresh_language(self) -> None:
        """The titles, tooltips and choices again in the interface language; what is typed stays."""
        self.title_label.setText(tr("Only you can answer these — then run again:"))
        self.again_button.setText(tr("Run Again with These Answers"))
        self.again_button.setToolTip(tr("Or press Enter in any of the fields"))
        if self.fields or self.answers_label.text():
            hidden = self.isHidden()  # put away with the results of another file: it stays so
            typed = {option: answer_text(field) for option, field in self.fields.items()}
            self.show_questions(self._attention)
            for option, text in typed.items():  # a field emptied on purpose stays empty
                field = self.fields.get(option)
                if isinstance(field, QLineEdit):
                    field.setText(text)
            if hidden:
                self.hide()

    @staticmethod
    def _shown_answer(option: str, value: str) -> str:
        return STANDARD_LABELS.get(value, value) if option == "standard" else value

    # -- the fields ---------------------------------------------------------------------------

    def _field(self, option: str, placeholder: str) -> tuple[QWidget, QWidget]:
        """(the field that holds the answer, what the form shows: the field, or the field with its …)."""
        if option == "standard":
            choice = QComboBox(self)
            choice.addItem(tr(CHOOSE_TEXT), "")
            for key in SUPPORTED_STANDARDS:
                choice.addItem(STANDARD_LABELS.get(key, key), key)
            choice.setCurrentIndex(max(0, choice.findData(self.answers.get(option, ""))))
            choice.installEventFilter(_EnterRuns(self.runRequested.emit, choice))
            return choice, choice
        field = QLineEdit(self)
        field.setPlaceholderText(tr(placeholder))
        field.setText(self.answers.get(option, ""))
        field.returnPressed.connect(self.runRequested)
        if option != "calibration":
            return field, field
        row = QWidget(self)
        line = QHBoxLayout(row)
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(4)
        line.addWidget(field, 1)
        browse = QToolButton(row)
        browse.setObjectName("guidedBrowseCalibration")
        browse.setText("…")
        browse.setToolTip(tr("Choose the calibration file: an image of a standard or a .poni file "
                             "(the dialog starts in the frame's folder)"))
        browse.clicked.connect(lambda: self.browse_calibration(field))
        line.addWidget(browse)
        return field, row

    def browse_calibration(self, field: QLineEdit) -> str:
        """A file dialog for the calibration that starts where the typed path is, else in the frame's folder."""
        path, _chosen = QFileDialog.getOpenFileName(
            self, tr(CALIBRATION_TITLE), self._start_folder(field.text().strip()), calibration_filters())
        if path:
            field.setText(QDir.toNativeSeparators(path))
            field.setFocus()
        return path or ""

    def _start_folder(self, typed: str) -> str:
        if typed:
            folder = Path(typed) if Path(typed).is_dir() else Path(typed).parent
            if folder.is_dir():
                return str(folder)
        try:
            return str(self._folder() or "")
        except Exception:  # no frame shown: the dialog's own start
            return ""


__all__ = ["AnswerForm", "CHOOSE_TEXT", "STANDARD_LABELS", "answer_text", "calibration_filters"]
