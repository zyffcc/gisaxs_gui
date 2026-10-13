"""Release 0.1.0, Labs / Trainset Build: the preview hint stays inside the card's border, the footer hint
takes the free width (one line), and an "&" in a tab title is drawn."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from PyQt5.QtCore import QCoreApplication, QEvent, QPoint
from PyQt5.QtWidgets import QApplication, QTabBar, QTabWidget

from src.gimap.app.presentation.i18n import ZH, apply_language
from src.gimap.app.presentation.theme import apply_theme
from src.gimap.features.trainset.presentation.page import TrainsetBuildPage
from src.gimap.features.trainset.presentation.sections.design_state import ACTION_HINTS
from tests.test_labs_trainset import _settle

ROOT = Path(__file__).resolve().parents[1]
LABS = [ROOT / "src" / "gimap" / "features" / name / "presentation" for name in ("trainset", "prediction")]
LONE_AMPERSAND = re.compile(r"(?<!&)&(?!&)")


def _page() -> TrainsetBuildPage:
    """The page at the size it has in a 1280 x 800 window (beside the 184 px sidebar)."""
    QApplication.instance() or QApplication([])
    page = TrainsetBuildPage()
    page.resize(1096, 716)
    page.show()
    page.step_list.setCurrentRow(0)
    _settle(0.3)
    return page


def _clear_of_the_border(hint, card) -> bool:
    """The hint's text starts and ends clear of the card's rounded 1 px border, and all its lines are shown."""
    text = hint.contentsRect()
    origin = hint.mapTo(card, text.topLeft())
    return (
        origin.x() >= 8 and origin.y() >= 6
        and card.width() - hint.mapTo(card, QPoint(text.right(), 0)).x() >= 8
        and hint.height() >= hint.heightForWidth(hint.width())
    )


def test_the_preview_hint_is_inside_the_card_border_and_the_tabs_keep_their_width():
    page = _page()
    try:
        ui = page._dataset_page_ui
        hint, card = ui.designPreviewHint, ui.designPreviewCard
        assert hint.isVisible()  # form and preview are side by side at this size
        # The card draws a rounded 1 px border (trainset_theme.qss): the text starts clear of it.
        assert _clear_of_the_border(hint, card)
        # The distance is the label's own: a theme switch styles the label again and must keep it.
        try:
            for mode in ("dark", "light"):
                apply_theme(mode, 9)
                _settle(0.2)
                assert _clear_of_the_border(hint, card), mode
        finally:
            apply_theme("light", 9)
        # The card itself keeps no margins, so the four stage tabs are not shortened ("Full detec…").
        margins = ui.designPreviewContentLayout.contentsMargins()
        assert (margins.left(), margins.top(), margins.right(), margins.bottom()) == (0, 0, 0, 0)
        bar = page.design_tabs.tabBar()
        metrics = bar.fontMetrics()
        for index in range(bar.count()):
            assert bar.tabRect(index).width() >= metrics.horizontalAdvance(bar.tabText(index)) + 8, bar.tabText(index)
        assert bar.tabRect(bar.count() - 1).right() <= page.design_tabs.width()
    finally:
        page.close()


def _one_line(label) -> bool:
    """The label is as wide as its sentence and as high as one line of it (QSS padding: 4 px 8 px)."""
    metrics = label.fontMetrics()
    return (
        label.width() >= metrics.horizontalAdvance(label.text()) + 16
        and label.heightForWidth(label.width()) < 2 * metrics.lineSpacing() + 8
    )


def test_the_footer_hint_takes_the_free_width_instead_of_a_narrow_column():
    page = _page()
    try:
        hint = page.trainset_action_hint
        layout = page.trainsetActionsLayout
        # Back, the hint (stretching), the buttons: no empty stretch beside a wrapped sentence.
        assert layout.count() == 3 and layout.itemAt(1).widget() is hint
        assert layout.stretch(1) == 1 and layout.itemAt(2).layout() is page.trainsetActionGrid
        for language in ("en", "zh"):
            apply_language(language, [page])  # the buttons beside the hint change their width too
            page.refresh_language()
            for row, english in enumerate(ACTION_HINTS):
                page.step_list.setCurrentRow(row)
                _settle(0.15)
                assert hint.text() == (ZH[english] if language == "zh" else english)
                assert _one_line(hint), (language, row, hint.width(), hint.text())
                # The buttons of the step keep their place at the right and their whole text.
                buttons = [button for button in (page.validate_button, page.preview_button, page.prepare_button,
                                                 page.submit_button, page.load_button, page.save_button)
                           if button.isVisible()]
                assert buttons and all(button.width() >= button.sizeHint().width() for button in buttons)
                assert hint.geometry().right() < min(button.mapTo(page, QPoint(0, 0)).x() for button in buttons)
                assert max(button.mapTo(page, QPoint(button.width(), 0)).x() for button in buttons) <= page.width()
    finally:
        apply_language("en", [page])
        page.close()


def test_the_footer_hint_wraps_only_when_the_window_is_too_narrow_for_it():
    page = _page()
    try:
        hint = page.trainset_action_hint
        page.step_list.setCurrentRow(3)  # Local Run: the step with the most buttons beside the hint
        page.resize(1096, 716)
        _settle(0.2)
        assert _one_line(hint)
        page.resize(840, 716)  # the page in a 1024 px window
        _settle(0.2)
        assert hint.width() > 0 and hint.height() >= hint.heightForWidth(hint.width())  # wrapped, never cut
        assert all(button.width() >= button.sizeHint().width()
                   for button in (page.prepare_button, page.submit_button, page.load_button, page.save_button))
    finally:
        page.close()


def _close_parameter_window(page) -> None:
    """Close the manual-simulation window and let Qt delete it while its page is alive: the window tells
    the page when it is destroyed, which must not happen while Python is collecting the page."""
    dialog = page._parameter_dialog
    if dialog is None:
        return
    dialog.close()  # WA_DeleteOnClose: deleted with the next deferred-delete pass
    del dialog
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    QApplication.processEvents()
    assert page._parameter_dialog is None


def _drawn(text: str) -> str:
    """What a tab bar draws for ``text``: "&&" is one "&", a single "&" marks a mnemonic and is not drawn."""
    return LONE_AMPERSAND.sub("", text).replace("&&", "&")


def test_an_ampersand_in_a_tab_title_is_drawn_not_taken_as_a_mnemonic():
    page = _page()
    try:
        page.step_list.setCurrentRow(1)
        _settle(0.1)
        bar = page.preprocessing_tabs.tabBar()
        assert [_drawn(bar.tabText(index)) for index in range(bar.count())] == [
            "Physical background", "Noise", "Mask & transforms",
        ]
        # The manual-simulation window builds its tabs from group names ("Mask & preprocessing").
        page.set_comparison_details({"midpoint": {"note": "no physics values: nothing is simulated"}}, {},
                                    {"mask": {"enabled": True}, "sample": {"density": 0.5}, "beam": {"alpha": 0.4}})
        page.show_comparison_parameters()
        dialog = page._parameter_dialog
        assert dialog is not None
        _settle(0.1)
        titles = [tabs.tabText(index) for tabs in dialog.findChildren(QTabWidget) for index in range(tabs.count())]
        assert "Mask & preprocessing" in [_drawn(title) for title in titles]
        # Every tab of the page and of that window (a child of the page): no lone "&".
        every = [bar.tabText(index) for bar in page.findChildren(QTabBar) for index in range(bar.count())]
        assert len(every) > len(titles) and [title for title in every if LONE_AMPERSAND.search(title)] == []
    finally:
        _close_parameter_window(page)
        page.close()


def test_no_labs_tab_title_is_written_with_a_single_ampersand():
    """A literal given to ``addTab`` / ``setTabText`` in Trainset or 2D Prediction writes "&" as "&&"."""
    found = []
    for folder in LABS:
        for path in sorted(folder.rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr in ("addTab", "insertTab", "setTabText")):
                    continue
                for argument in node.args:
                    if (isinstance(argument, ast.Constant) and isinstance(argument.value, str)
                            and LONE_AMPERSAND.search(argument.value)):
                        found.append(f"{path.relative_to(ROOT)}:{node.lineno} {argument.value!r}")
    assert found == []
