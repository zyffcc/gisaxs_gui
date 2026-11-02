"""Release 0.1.0, Labs / Trainset Build: the preview card and the footer keep their layout (their hint lines went
into tooltips in the 2026-11 small-text review), and an "&" in a tab title is drawn."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from PyQt5.QtCore import QCoreApplication, QEvent, QPoint
from PyQt5.QtWidgets import QApplication, QTabBar, QTabWidget

from src.gimap.app.presentation.i18n import apply_language
from src.gimap.features.trainset.presentation.page import TrainsetBuildPage
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


def test_the_preview_card_has_no_hint_and_the_tabs_keep_their_width():
    """The how-to line of the card went into the tooltips of Pick beam center and Draw rectangle ROI (2026-11
    small-text review); the four stage tabs still get the card's whole width."""
    page = _page()
    try:
        ui = page._dataset_page_ui
        assert not hasattr(ui, "designPreviewHint")
        assert "x, y and intensity" in page.pick_beam_center_button.toolTip()
        assert "tensor shape" in page.draw_roi_button.toolTip()
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


def test_the_footer_has_back_and_the_step_buttons_without_a_hint():
    """Back at the left, the step's buttons at the right with their whole text, in both languages."""
    page = _page()
    try:
        layout = page.trainsetActionsLayout
        assert layout.count() == 3 and layout.itemAt(0).widget() is page.back_button
        assert layout.itemAt(1).spacerItem() is not None and layout.itemAt(2).layout() is page.trainsetActionGrid
        for language in ("en", "zh"):
            apply_language(language, [page])
            page.refresh_language()
            for row in range(page.step_list.count()):
                page.step_list.setCurrentRow(row)
                _settle(0.15)
                buttons = [button for button in (page.validate_button, page.preview_button, page.prepare_button,
                                                 page.submit_button, page.load_button, page.save_button)
                           if button.isVisible()]
                assert buttons and all(button.width() >= button.sizeHint().width() for button in buttons)
                assert max(button.mapTo(page, QPoint(button.width(), 0)).x() for button in buttons) <= page.width()
    finally:
        apply_language("en", [page])
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
