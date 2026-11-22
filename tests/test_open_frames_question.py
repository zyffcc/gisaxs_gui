"""Opening frames while others are listed asks Replace / Add to List / Cancel, offscreen as on a desktop.

The question used to be a ``QMessageBox``, whose ``showEvent`` on Windows asks the platform for the window's system
menu: the offscreen platform has none and the process died (access violation) at the second frame opened. It is a
``QuestionBox`` now (``app/presentation/question_box.py``); these tests show the real question and press its
buttons from the event loop, as a person would."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

from src.gimap.app.presentation.question_box import ACCEPT, ACTION, REJECT, QuestionBox, ask_choice
from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
from tests.test_analyze_workspace import _context, _done, _page


def _frame(path: Path, value: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((48, 64), value, np.float32), mode="F").save(path)
    return path


def _answer(label: str, seen: list, *, then=None) -> None:
    """Press ``label`` on the question once it is shown (``esc``: the Esc key); ``then`` runs while it is open."""

    def press():
        box = QApplication.activeModalWidget()
        if not isinstance(box, QuestionBox):
            QTimer.singleShot(20, press)
            return
        seen.append((box.windowTitle(), box.text(), sorted(button.text() for button in box.buttons())))
        if then is not None:
            then()
        if label == "esc":
            box.reject()
            return
        next(button for button in box.buttons() if button.text() == label).click()

    QTimer.singleShot(0, press)


def test_two_frames_opened_in_a_row_ask_with_a_question_that_does_not_kill_the_process(tmp_path: Path) -> None:
    first = _frame(tmp_path / "one" / "film_0001.tif", 1.0)
    second = _frame(tmp_path / "two" / "film_0002.tif", 2.0)
    third = _frame(tmp_path / "three" / "film_0003.tif", 3.0)
    page = _page(_context(InMemoryInstrumentProfileRepository([])))
    page.show()
    seen: list = []
    try:
        assert page.import_paths([str(first)])  # nothing listed yet: no question
        _done(page)
        assert [path.name for path in page.view_model.state.files] == ["film_0001.tif"] and not seen

        _answer("Replace", seen)
        assert page.import_paths([str(second)])
        _done(page)
        assert seen[-1] == ("Open Frames", "1 frames are listed. Replace them with the frames opened now, or add these "
                            "to the list?", ["Add to List", "Cancel", "Replace"])
        assert [path.name for path in page.view_model.state.files] == ["film_0002.tif"]

        _answer("Add to List", seen)
        assert page.import_paths([str(third)])
        _done(page)
        assert [path.name for path in page.view_model.state.files] == ["film_0002.tif", "film_0003.tif"]

        _answer("esc", seen)  # Esc is Cancel: nothing changes
        assert not page.import_paths([str(first)])
        assert [path.name for path in page.view_model.state.files] == ["film_0002.tif", "film_0003.tif"]
        assert page.status_label.text() == "Frame import cancelled."
        assert len(seen) == 3
    finally:
        page.dispose()


def test_frames_dropped_while_the_question_is_open_are_refused(tmp_path: Path) -> None:
    first = _frame(tmp_path / "one" / "film_0001.tif", 1.0)
    second = _frame(tmp_path / "two" / "film_0002.tif", 2.0)
    page = _page(_context(InMemoryInstrumentProfileRepository([])))
    page.show()
    seen, again = [], []
    try:
        assert page.import_paths([str(first)])
        _done(page)
        _answer("Cancel", seen, then=lambda: again.append((page.import_paths([str(second)]), page.status_label.text())))
        assert not page.import_paths([str(second)])
        assert again == [(False, "Answer the open question about the frames listed first.")]
        assert [path.name for path in page.view_model.state.files] == ["film_0001.tif"]
    finally:
        page.dispose()


def test_a_question_returns_the_button_pressed_and_esc_its_escape_button() -> None:
    QApplication.instance() or QApplication([])
    buttons = (("Save", ACCEPT), ("Don't Save", ACTION), ("Cancel", REJECT))
    for label, expected in (("Don't Save", 1), ("Save", 0), ("esc", 2)):
        seen: list = []
        _answer(label, seen)
        assert ask_choice(None, "Unsaved Changes", "Save the changes?", buttons, default=0, escape=2) == expected
        assert seen[0][0] == "Unsaved Changes"
    box = QuestionBox(None, "Title", "Text", "More")
    default = box.addButton("Yes", ACCEPT)
    box.setDefaultButton(default)
    assert (box.text(), box.informativeText(), box.defaultButton()) == ("Text", "More", default)
    box.deleteLater()


def test_no_question_is_built_as_a_qmessagebox() -> None:
    """A ``QMessageBox`` shown on the offscreen platform of Windows ends the process; GIMaP's questions with their
    own buttons are ``QuestionBox`` (the static ``QMessageBox.question`` & co. stay: tests replace them)."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "gimap"
    built = [str(path.relative_to(root)) for path in root.rglob("*.py")
             if "QMessageBox(" in path.read_text(encoding="utf-8")]
    assert built == []
