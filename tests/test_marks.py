"""Marks on images and plots: every kind can be hidden and shown again, and the ones you made removed."""

from __future__ import annotations

from pathlib import Path

import pytest
from PyQt5.QtWidgets import QMenu

from src.gimap.app.presentation.components.marks import MarkLayers
from tests.test_analyze_workspace import _app


class Item:
    def __init__(self):
        self.visible = False

    def setVisible(self, visible):  # noqa: N802 - Qt API
        self.visible = bool(visible)


def _menu_texts(marks: MarkLayers) -> list[str]:
    menu = QMenu()
    marks._fill(menu)
    return [action.text() for action in menu.actions() if action.text()]


def test_a_hidden_kind_keeps_what_the_page_wants() -> None:
    _app()
    marks = MarkLayers()
    marks.add_layer("beam", "Beam centre")
    marks.add_layer("box", "q box")
    beam, box = Item(), Item()
    assert marks.present() == []  # nothing drawn yet: nothing to offer
    marks.track("beam", beam)
    assert beam.visible and marks.present() == ["beam"]
    marks.set_visible("beam", False)
    marks.track("beam", beam, True)  # redrawn while hidden: stays hidden
    assert not beam.visible and marks.wanted(beam)
    marks.set_visible("beam", True)
    assert beam.visible
    marks.track("beam", beam, False)  # the page no longer wants it: switching on does not bring it back
    marks.set_visible("beam", False)
    marks.set_visible("beam", True)
    assert not beam.visible
    marks.track("box", box)
    marks.set_hidden(["box"])
    assert not box.visible and marks.hidden() == ["box"]


def test_the_menu_offers_the_kinds_present_all_at_once_and_the_removals() -> None:
    _app()
    marks = MarkLayers()
    removed = []
    marks.add_layer("beam", "Beam centre")
    marks.add_layer("bands", "Cut bands")
    marks.add_removal("Remove the q Box", lambda: removed.append(True), lambda: True)
    assert _menu_texts(marks) == ["No marks on this view yet", "Remove the q Box"]
    beam, band = Item(), Item()
    marks.track("beam", beam)
    marks.track("bands", band)
    assert _menu_texts(marks) == ["Beam centre", "Cut bands", "Show All Marks", "Hide All Marks", "Remove the q Box"]
    seen = []
    marks.changed.connect(lambda key, visible: seen.append((key, visible)))
    marks._all(False)
    assert not beam.visible and not band.visible and seen == [("beam", False), ("bands", False)]
    marks._all(True)
    assert beam.visible and band.visible


def test_the_detector_view_and_the_plots_follow_their_marks() -> None:
    from src.gimap.app.presentation.components import CurvePlot, DetectorView

    _app()
    view = DetectorView()
    view.show_beam_center(10.0, 20.0)
    view.show_horizontal_band(5.0, 8.0)
    assert view.beam_target.isVisibleTo(view.graphics.ci) or view.marks.wanted(view.beam_target)
    view.marks.set_visible("beam", False)
    view.show_beam_center(11.0, 21.0)
    assert not view.beam_target.isVisible() and view.marks.wanted(view.beam_target)
    assert "Beam centre" in _menu_texts(view.marks) and "Colour bar" in _menu_texts(view.marks)
    view.marks.set_visible("colorbar", False)
    assert not view.color_bar.isVisible()
    plot = CurvePlot("")
    plot.show_window(0.1, 0.2)
    plot.marks.set_visible("window", False)
    plot.show_window(0.1, 0.3)
    assert not plot.x_window.isVisible() and plot.marks.wanted(plot.x_window)
    view.close()
    plot.close()


@pytest.fixture
def page(tmp_path: Path):
    from src.gimap.features.analyze.bootstrap import create_analyze_view_model
    from src.gimap.features.analyze.presentation.page import AnalyzePage
    from src.gimap.integrations.state import InMemoryInstrumentProfileRepository, InMemorySettingsRepository
    from src.gimap.shared.geometry import InstrumentProfile
    from tests.test_assistant_calibration import SHAPE, giwaxs_frame, save_tiff
    from tests.test_giwaxs_workspace import GEOMETRY, _settle
    from tests.test_session_robustness import _page

    _app()
    frame = save_tiff(tmp_path / "sample_001.tif", giwaxs_frame(seed=4))
    settings = InMemorySettingsRepository({})
    profiles = InMemoryInstrumentProfileRepository([InstrumentProfile("synthetic", GEOMETRY, None, SHAPE)])

    def make():
        made = _page(settings, profiles)
        made.set_mode_choice("giwaxs")
        made.add_paths([str(frame)])
        _settle(made, lambda: made.view_model.state.analysis is not None and made.view_model.state.analysis.reduction is not None)
        made.settle = lambda condition=lambda: True: _settle(made, condition)
        return made

    first = make()
    first.make = make
    yield first
    first.tasks.wait(30)
    first.dispose()
    first.close()
    assert create_analyze_view_model and AnalyzePage


def test_analyze_remembers_hidden_marks_and_removes_what_you_made(page) -> None:
    from src.gimap.features.analyze.application import CutRegion

    view = page.detector_view
    assert view.marks.is_visible("beam") and view.marks.wanted(view.beam_target)
    view.marks.set_visible("beam", False, notify=True)
    page.top_plot.marks.set_visible("window", False, notify=True)
    again = page.make()  # the next session: the same choices
    assert not again.detector_view.marks.is_visible("beam") and not again.top_plot.marks.is_visible("window")
    assert not again.detector_view.beam_target.isVisible()
    again.dispose()
    again.close()

    page.view_model.add_region(CutRegion("Ring q 1.100", (1.07, 1.13)))
    page.run_analysis()
    page.settle(lambda: page.view_model.state.analysis.reduction.curve("region1") is not None)
    menu = QMenu()
    view.marks._fill(menu)
    actions = {action.text(): action for action in menu.actions()}
    assert actions["Remove All Cut Regions"].isEnabled() and not actions["Remove the q Box"].isEnabled()
    actions["Remove All Cut Regions"].trigger()
    page.settle(lambda: page.view_model.state.analysis.reduction.curve("region1") is None)
    assert page.view_model.state.giwaxs.regions == ()
    assert page.undo_setup()  # Undo brings them back
    page.settle(lambda: page.view_model.state.analysis.reduction.curve("region1") is not None)

    view.marks.set_visible("pixels", False, notify=True)
    page.sources_button.setChecked(True)  # asking for the sources shows the pixel overlay again
    assert view.marks.is_visible("pixels")


def test_the_q_map_shows_the_beam_centre_and_the_horizon(page) -> None:
    import math

    geometry = page.view_model.state.analysis.geometry
    page.set_view(1)  # the q map
    page.settle()
    view = page.detector_view
    assert view.marks.wanted(view.beam_target) and not view.beam_target.movable
    position = view.beam_target.pos()
    assert abs(position.x()) < 1e-9 and abs(position.y()) < 1e-9  # the direct beam: q∥ = qz = 0
    k = 2 * math.pi / geometry.wavelength_angstrom
    assert view.horizon_line.value() == pytest.approx(k * math.sin(geometry.incidence_rad), rel=1e-6)
    (x0, x1), (y0, y1) = view.view_box.viewRange()
    assert x0 <= 0.0 <= x1 and y0 <= 0.0 <= y1  # Fit view keeps it in sight, below the map
    page.set_view(0)  # back on the detector: the centre can be dragged again
    page.settle()
    assert view.beam_target.movable
