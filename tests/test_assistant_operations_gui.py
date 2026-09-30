"""Previewable operations on a real Analyze page: preview first, cards with pictures, apply and undo."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import QApplication

from src.gimap.features.assistant.application import (
    APPLIED,
    BACKEND_API,
    GOALS,
    PERMISSION_PREVIEW,
    PROPOSED,
    RUN_COMPLETED,
    UNDONE,
    AnalysisGoals,
)
from tests.assistant_fakes import ScriptedLlm, call, report, turn
from tests.test_assistant_gui import _controller, _services, _wait, analyze  # noqa: F401 - the fixture


def _widths(page) -> tuple:
    giwaxs = page.view_model.state.giwaxs
    return giwaxs.in_plane_half_width_deg, giwaxs.out_of_plane_half_width_deg


def test_preview_first_leaves_analyze_as_it_was_and_the_cards_apply_and_undo(analyze, tmp_path: Path) -> None:  # noqa: F811
    window, page, context = analyze
    start = _widths(page)
    llm = ScriptedLlm([
        turn(call("set_sector_widths", in_plane_half_width_deg=4.0, out_of_plane_half_width_deg=6.0)),
        turn(call("find_peaks", curve="in_plane")),
        turn(call("propose_operations", operations=[
            {"tool": "set_custom_sector", "arguments": {"enabled": True, "chi_min_deg": 15.0, "chi_max_deg": 55.0},
             "title": "Compare in the bright region", "why": "The in-plane sector is shadowed."},
        ])),
        turn(report(("peaks", "done"))),
    ])
    controller = _controller(window, page, context, _services(tmp_path, llm), backend=BACKEND_API)
    assert controller.run(AnalysisGoals(goals=tuple(GOALS), permission=PERMISSION_PREVIEW))
    _wait(lambda: controller.outcome is not None, 60)
    assert controller.outcome.state == RUN_COMPLETED

    # Analyze is as before; both changes wait as suggestions, each with a picture of its result.
    assert _widths(page) == start and page.view_model.state.giwaxs.sector is None
    widths, sector = controller.outcome.results.operations
    assert widths.state == sector.state == PROPOSED
    cards = controller.panel.operations.cards
    # The AI's own suggestion (with its reason) comes first, then what it changed during the analysis.
    assert [card.title_label.text() for card in cards] == ["Compare in the bright region", "Sector half widths: in-plane ±4°, out-of-plane ±6°"]
    assert all(not card.picture.pixmap().isNull() for card in cards)
    assert cards[0].why_label.text() == "Why: The in-plane sector is shadowed."
    assert controller.panel.operations.isVisibleTo(controller.panel)

    controller.panel.operations.card(widths.id).apply_button.click()
    _wait(lambda: _widths(page) == (4.0, 6.0), 30)
    assert widths.state == APPLIED
    applied = controller.panel.operations.card(widths.id)
    assert applied.undo_button.isVisibleTo(applied) and not applied.apply_button.isVisibleTo(applied)
    controller.panel.operations.card(sector.id).apply_button.click()
    _wait(lambda: page.view_model.state.giwaxs.sector is not None, 30)
    assert (page.view_model.state.giwaxs.sector.chi_min_deg, page.view_model.state.giwaxs.sector.chi_max_deg) == (15.0, 55.0)

    controller.panel.operations.undo_all_button.click()
    _wait(lambda: _widths(page) == start and page.view_model.state.giwaxs.sector is None, 30)
    QApplication.processEvents()
    assert [operation.state for operation in controller.outcome.results.operations] == [UNDONE, PROPOSED]
    assert controller.panel.operations.card(widths.id).apply_button.isVisibleTo(controller.panel.operations.card(widths.id))
    record = controller._record(controller.outcome)
    assert [item["title"] for item in record["results"]["operations"]][1] == "Compare in the bright region"


def test_a_follow_up_question_continues_on_the_same_frame(analyze, tmp_path: Path) -> None:  # noqa: F811
    window, page, context = analyze
    llm = ScriptedLlm([
        turn(call("find_peaks", curve="radial")),
        turn(report(("peaks", "done"), summary="Two lamellar orders out of plane.")),
        turn(call("propose_operations", operations=[
            {"tool": "set_sector_widths", "arguments": {"in_plane_half_width_deg": 5.0, "out_of_plane_half_width_deg": 5.0},
             "title": "Narrower sectors", "why": "Separate the orientations better."},
        ])),
        turn(report(("peaks", "done"), summary="Narrower sectors would separate them.")),
    ])
    controller = _controller(window, page, context, _services(tmp_path, llm), backend=BACKEND_API)
    assert not controller.follow_up("too early")  # nothing to follow up yet
    assert controller.run(AnalysisGoals(goals=tuple(GOALS), permission=PERMISSION_PREVIEW))
    _wait(lambda: controller.outcome is not None, 60)
    assert controller.panel.follow_button.isEnabled()
    controller.panel.follow_edit.setText("How can I separate the orientations better?")
    controller.panel.follow_button.click()
    _wait(lambda: controller.outcome is not None and controller.outcome.results.operations, 60)
    question = llm.requests[2]["messages"][0]["content"]
    assert "Follow-up question: How can I separate the orientations better?" in question
    assert "Two lamellar orders out of plane." in question  # the previous report as context
    (card,) = controller.panel.operations.cards
    assert card.title_label.text() == "Narrower sectors"
