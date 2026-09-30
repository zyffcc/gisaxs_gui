"""Colour limits: dragged or typed limits stay, auto for every frame or once, per view, and in batch pictures."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.gimap.app.presentation.components.levels import (
    LevelControl,
    auto_levels,
    to_display,
    to_intensity,
)
from tests.test_analyze_workspace import _app

RNG = np.random.default_rng(3)
FRAME = RNG.gamma(2.0, 50.0, size=(200, 300)).astype(np.float32)


def test_the_rules() -> None:
    values = np.arange(1000, dtype=float).reshape(20, 50)
    assert auto_levels(values, "minmax") == (0.0, 999.0)
    low, high = auto_levels(values, "p5")
    assert low == pytest.approx(49.95) and high == pytest.approx(949.05)
    low, high = auto_levels(values, "std3")
    assert low == 0.0 and high == 999.0  # mean ± 3σ, clamped to the data
    assert auto_levels(np.full((4, 4), np.nan)) is None
    assert to_intensity(to_display((5.0, 5000.0), True), True) == pytest.approx((5.0, 5000.0))
    assert to_display((0.0, 100.0), True) == (-4.0, 2.0)  # a zero low limit in log: six decades below


def test_fixed_limits_stay_for_other_frames_and_contexts_keep_their_own() -> None:
    _app()
    control = LevelControl()
    control.set_context("detector")
    first = control.levels_for(FRAME, log=False)
    control.edited(10.0, 200.0, log=False)  # dragged
    assert not control.state.auto
    brighter = FRAME * 10
    assert control.levels_for(brighter, log=False) == (10.0, 200.0)  # the next frame: the same limits
    assert control.levels_for(brighter, log=True) == pytest.approx((1.0, math.log10(200.0)))  # the same in log
    control.set_context("qmap")
    assert control.state.auto and control.levels_for(brighter, log=False) != (10.0, 200.0)
    control.set_context("detector")
    control.set_auto(True)
    assert control.levels_for(brighter, log=False) == pytest.approx(auto_levels(brighter))
    control.set_auto(False)  # unticking keeps what is shown now
    assert control.state.fixed == pytest.approx(auto_levels(brighter))
    control.fix(300.0, 20.0)  # typed, in any order
    assert control.levels_for(FRAME, log=False) == (20.0, 300.0)
    control.set_auto(True)
    control.auto_once(FRAME, log=False)
    assert not control.state.auto and control.state.fixed == pytest.approx(first)


def test_the_view_keeps_dragged_limits_and_updates_itself_quietly() -> None:
    from src.gimap.app.presentation.components import DetectorView

    _app()
    view = DetectorView()
    edits = []
    view.color_bar.edited.levels.connect(lambda low, high: edits.append((low, high)))
    view.log_check.setChecked(False)
    view.set_image(FRAME, context="detector")
    view.set_image(FRAME * 2, context="detector", keep_view=True)
    assert edits == []  # the view's own updates are not edits
    view.color_bar._programmatic = False  # as when the person drags the handles
    view.color_bar.region.setRegion((15.0, 250.0))
    assert edits and view.levels.state.fixed == pytest.approx((15.0, 250.0))
    view.set_image(FRAME * 5, context="detector", keep_view=True)  # a new frame: the limits stay
    assert view.color_bar.levels() == pytest.approx((15.0, 250.0))
    assert tuple(view.image_item.levels) == pytest.approx((15.0, 250.0))
    assert view.auto_levels_button.text() == "Levels (fixed)"
    view.levels.set_auto(True)
    assert view.color_bar.levels() == pytest.approx(auto_levels(FRAME * 5))
    view.close()


def test_batch_pictures_use_the_limits_on_screen_or_their_own(tmp_path, monkeypatch) -> None:
    from src.gimap.features.analyze.application import BatchChoices
    from src.gimap.features.analyze.bootstrap import create_analyze_view_model
    from src.gimap.features.analyze.presentation.page import AnalyzePage
    from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
    from src.gimap.shared.geometry import InstrumentProfile
    from tests.test_analyze_workspace import _context
    from tests.test_assistant_calibration import SHAPE, giwaxs_frame, save_tiff
    from tests.test_giwaxs_workspace import GEOMETRY, _settle

    folder = tmp_path / "run"
    for index in range(3):
        save_tiff(folder / f"run_{index:03d}.tif", giwaxs_frame(seed=index) * (1 + index))
    profile = InstrumentProfile("synthetic", GEOMETRY, None, SHAPE)
    page = AnalyzePage(create_analyze_view_model(_context(InMemoryInstrumentProfileRepository([profile]))))
    page.show()
    page.set_mode_choice("giwaxs")
    page.add_paths([str(folder)])
    _settle(page, lambda: page.view_model.state.analysis is not None and page.view_model.state.analysis.reduction is not None)
    page.detector_view.levels.set_context("detector")
    page.detector_view.levels.fix(5.0, 500.0)
    written = []
    figures = page.view_model.figures
    original = figures.write_frame
    monkeypatch.setattr(figures, "write_frame", lambda *args, **kw: (written.append(kw), original(*args, **kw))[1])
    for scale in ("screen", "auto"):
        written.clear()
        out = tmp_path / scale
        page.run_batch(out, BatchChoices(tables=False, detector_image=True, image_scale=scale), stem="run")
        _settle(page, lambda: not page.batch_running(), timeout_s=120)
        record = json.loads((out / "run_batch.json").read_text(encoding="utf-8"))
        levels = [kw.get("levels") for kw in written]
        if scale == "screen":
            assert levels == [[5.0, 500.0]] * 3 and record["pictures"]["detector"] == [5.0, 500.0]
        else:
            assert levels == [None] * 3 and record["pictures"]["detector"] is None
        assert len(list((out / "images").glob("*_detector.png"))) == 3
    assert "detector 5 … 500" in page.batch_display_text()
    page.dispose()
    page.close()


def test_the_series_map_keeps_its_limits_per_curve(tmp_path) -> None:
    import warnings

    from src.gimap.features.analyze.bootstrap import create_analyze_view_model
    from src.gimap.features.analyze.presentation.page import AnalyzePage
    from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
    from src.gimap.shared.geometry import InstrumentProfile
    from tests.test_analyze_workspace import _context
    from tests.test_assistant_calibration import SHAPE, giwaxs_frame, save_tiff
    from tests.test_giwaxs_workspace import GEOMETRY, _settle

    folder = tmp_path / "series"
    for index in range(4):
        save_tiff(folder / f"s_{index:03d}.tif", giwaxs_frame(seed=index) * (1 + 0.5 * index))
    page = AnalyzePage(create_analyze_view_model(_context(InMemoryInstrumentProfileRepository(
        [InstrumentProfile("synthetic", GEOMETRY, None, (400, 400))]))))
    page.show()
    page.set_mode_choice("giwaxs")
    page.add_paths([str(folder)])
    _settle(page, lambda: page.view_model.state.analysis is not None and page.view_model.state.analysis.reduction is not None)
    view = page.series_map_view

    def build(key):
        page.series_curve_combo.setCurrentIndex(page.series_curve_combo.findData(key))
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)  # no “Mean of empty slice” from empty columns
            page.build_series_map()
            _settle(page, lambda: not page.batch_running() and page._series_map is not None, timeout_s=120)

    build("radial")
    view.color_bar._programmatic = False
    view.color_bar.region.setRegion((0.2, 0.9))  # dragged
    build("radial")  # rebuilt: the limits stay
    assert view.color_bar.levels() == pytest.approx((0.2, 0.9))
    build("in_plane")  # another curve: its own, automatic
    assert view.levels.state.auto and view.color_bar.levels() != pytest.approx((0.2, 0.9))
    build("radial")  # back: the fixed limits of I(q)
    assert view.color_bar.levels() == pytest.approx((0.2, 0.9))
    page.dispose()
    page.close()
