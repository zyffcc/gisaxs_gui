"""Undo / Redo of the Analyze set-up: the history, what a step is called, and the page."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from src.gimap.features.analyze.application import Corrections, CutRegion, GisaxsCutSettings, GiwaxsSettings, MaskShape
from src.gimap.features.analyze.presentation.setup_history import MAX_STEPS, Setup, SetupHistory, describe

BASE = Setup("giwaxs", "p", 0.2, None, Corrections(), GiwaxsSettings(), GisaxsCutSettings(), 1)
RING = CutRegion("Ring q 1.100", (1.07, 1.13))


def test_every_change_is_a_step_and_a_drag_is_one() -> None:
    history = SetupHistory()
    assert not history.observe(BASE, 0.0)  # the first set-up is where the history starts
    one = replace(BASE, giwaxs=replace(BASE.giwaxs, regions=(RING,)))
    assert history.observe(one, 10.0) and history.next_undo() == "cut regions"
    moved = [replace(one, gisaxs=replace(BASE.gisaxs, horizontal_row=float(row))) for row in (100, 101, 102)]
    for step, setup in enumerate(moved):  # a band dragged: one step
        history.observe(setup, 20.0 + 0.1 * step)
    assert history.next_undo() == "GISAXS cuts"
    assert history.undo() == (one, "GISAXS cuts")
    assert history.undo() == (BASE, "cut regions")
    assert history.undo() is None and history.next_redo() == "cut regions"
    assert history.redo() == (one, "cut regions")
    history.observe(replace(one, incidence_deg=0.3), 40.0)  # a new change: nothing left to redo
    assert history.next_redo() is None and history.next_undo() == "αi"


def test_an_edit_that_cancels_the_last_one_is_its_own_step() -> None:
    history = SetupHistory()
    history.observe(BASE, 0.0)
    added = replace(BASE, giwaxs=replace(BASE.giwaxs, regions=(RING,)))
    history.observe(added, 1.0)
    history.observe(BASE, 1.1)  # removed at once: Undo must bring the region back
    assert history.undo() == (added, "cut regions")


def test_the_same_kind_after_a_pause_is_a_new_step() -> None:
    history = SetupHistory()
    history.observe(BASE, 0.0)
    history.observe(replace(BASE, incidence_deg=0.3), 1.0)
    history.observe(replace(BASE, incidence_deg=0.4), 5.0)
    assert history.undo()[0].incidence_deg == 0.3


def test_the_history_keeps_the_last_steps_only() -> None:
    history = SetupHistory()
    history.observe(BASE, 0.0)
    for step in range(MAX_STEPS + 20):
        history.observe(replace(BASE, sum_count=step + 2), 10.0 * (step + 1))
    undone = 0
    while history.undo() is not None:
        undone += 1
    assert undone == MAX_STEPS


def test_a_step_says_what_changed() -> None:
    mask = replace(BASE, corrections=replace(BASE.corrections, mask_shapes=(MaskShape("rectangle", ((0, 0), (5, 5))),)))
    assert describe(BASE, mask) == "mask"
    both = replace(mask, giwaxs=replace(BASE.giwaxs, regions=(RING,)))
    assert describe(BASE, both) == "mask and cut regions"
    assert describe(BASE, replace(both, mode="gisaxs")) == "set-up"
    assert describe(BASE, replace(BASE, corrections=replace(BASE.corrections, mirror_fill=True))) == "mirror filling"
    assert describe(BASE, replace(BASE, beam_center=(1.0, 2.0))) == "beam centre"


@pytest.fixture
def page(tmp_path: Path):
    from src.gimap.features.analyze.bootstrap import create_analyze_view_model
    from src.gimap.features.analyze.presentation.page import AnalyzePage
    from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
    from src.gimap.shared.geometry import InstrumentProfile
    from tests.test_analyze_workspace import _app, _context
    from tests.test_assistant_calibration import SHAPE, giwaxs_frame, save_tiff
    from tests.test_giwaxs_workspace import GEOMETRY, _settle

    _app()
    frame = save_tiff(tmp_path / "sample_001.tif", giwaxs_frame(seed=3))
    profile = InstrumentProfile("synthetic", GEOMETRY, None, SHAPE)
    page = AnalyzePage(create_analyze_view_model(_context(InMemoryInstrumentProfileRepository([profile]))))
    page.resize(1400, 900)
    page.show()
    page.set_mode_choice("giwaxs")
    page.add_paths([str(frame)])
    _settle(page, lambda: page.view_model.state.analysis is not None and page.view_model.state.analysis.reduction is not None)
    page.settle = lambda condition=lambda: True: _settle(page, condition)
    yield page
    page.tasks.wait(30)
    page.dispose()
    page.close()


def test_undo_and_redo_on_the_page(page) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    state = page.view_model.state
    assert not page.undo_button.isEnabled()
    page.view_model.add_region(RING)
    page.run_analysis()
    page.settle(lambda: state.analysis.reduction.curve("region1") is not None)
    assert page.undo_button.isEnabled() and page.undo_button.toolTip() == "Undo: cut regions (Ctrl+Z)"

    assert page.undo_setup()
    assert "Undone: cut regions" in page.status_text()
    page.settle(lambda: state.analysis.reduction.curve("region1") is None)
    assert state.giwaxs.regions == ()
    assert page.redo_button.isEnabled() and not page.undo_button.isEnabled()

    page.setFocus()
    QTest.keyClick(page, Qt.Key_Z, Qt.ControlModifier | Qt.ShiftModifier)  # redo
    page.settle(lambda: state.analysis.reduction.curve("region1") is not None)
    assert state.giwaxs.regions == (RING,)

    page.view_model.set_incidence(0.35)
    page.run_analysis()
    page.settle()
    assert page.undo_button.toolTip().startswith("Undo: αi")
    QTest.keyClick(page, Qt.Key_Z, Qt.ControlModifier)
    page.settle()
    assert state.incidence_deg != 0.35 and state.giwaxs.regions == (RING,)
    assert page.incidence_spin.value() != pytest.approx(0.35)  # the controls follow
