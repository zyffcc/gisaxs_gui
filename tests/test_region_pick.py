"""Custom cuts placed by hand: a click snapped to the peak, a rectangle drawn on the cake, snapping,
and cut sets saved for the next data set."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.gimap.features.analyze.domain.region_pick import RING, SECTOR, SPOT, peak_near, pick_region
from tests.test_giwaxs_workspace import _settle, page  # noqa: F401 - the page fixture


def test_a_peak_is_found_from_a_click_beside_it_on_a_sloping_background() -> None:
    rng = np.random.default_rng(1)
    x = np.linspace(0.5, 1.5, 500)
    y = 200 * np.exp(-x) + 80 * np.exp(-4 * np.log(2) * (x - 1.0) ** 2 / 0.02**2)  # FWHM 0.02
    y += 60 * np.exp(-4 * np.log(2) * (x - 1.06) ** 2 / 0.02**2)  # a neighbour 0.06 away
    y = y + rng.normal(0, 1.0, x.size)
    peak = peak_near(x, y, 0.992, search=0.1)  # clicked on the flank
    assert peak.center == pytest.approx(1.0, abs=0.002) and peak.fwhm == pytest.approx(0.02, rel=0.25)
    assert peak.high < 1.045  # the window stops before the neighbour
    assert peak_near(x, 200 * np.exp(-x) + rng.normal(0, 1.0, x.size), 1.0, search=0.1) is None  # no peak there
    assert peak_near(x, y, 1.03, search=0.1, reach=0.01) is None  # the nearest peak is too far from the click
    edge = x < 0.95  # this band of the detector ends at 0.95: a click at 1.0 has no data under it
    assert peak_near(x[edge], y[edge], 1.0, search=0.1) is None
    spike = 200 * np.exp(-x) + rng.normal(0, 1.0, x.size)
    spike[300] += 40.0  # one bright bin (edge or hot pixels) is not a peak
    assert peak_near(x, spike, x[300], search=0.1) is None


def test_ring_sector_and_spot_from_clicks_on_the_synthetic_film() -> None:
    from src.gimap.features.analyze.domain import giwaxs_maps
    from tests.test_giwaxs_workspace import GEOMETRY
    from tests.test_assistant_calibration import SHAPE, giwaxs_frame

    maps = giwaxs_maps(SHAPE, GEOMETRY)
    image = giwaxs_frame()
    usable = maps.above_horizon
    ring = pick_region(RING, 1.09, 50.0, maps, image, usable, name="r").region
    assert ring.q_range[0] < 1.10 < ring.q_range[1] and ring.q_range[1] - ring.q_range[0] < 0.2
    assert ring.chi_range == (0.0, 90.0)
    lamella = pick_region(SPOT, 0.41, -4.0, maps, image, usable, name="s")
    assert lamella.q_peak.center == pytest.approx(0.40, abs=0.01)
    assert lamella.region.chi_range[0] == 0.0 and 10.0 < lamella.region.chi_range[1] < 30.0  # the 15° spread
    sector = pick_region(SECTOR, 0.6, 60.0, maps, image, usable, name="c").region
    assert sector.q_range is None and sector.chi_range == pytest.approx((55.0, 65.0))


def test_click_on_the_q_map_adds_a_snapped_ring_with_undo_in_its_name(page) -> None:
    page.set_view(1)  # q map: (q∥, qz)
    page.pick_region("ring")
    assert page.region_pick_buttons["ring"].isChecked() and page.shape_layer.kind == "point"
    q, chi = 1.08, math.radians(40.0)
    page.shape_layer._finish([(q * math.sin(chi), q * math.cos(chi))])
    _settle(page, lambda: len(page.view_model.state.giwaxs.regions) == 1 and len(page._region_rows) == 5)
    region = page.view_model.state.giwaxs.regions[0]
    assert region.name.startswith("Ring q 1.1") and region.q_range[0] < 1.10 < region.q_range[1]
    assert not page.region_pick_buttons["ring"].isChecked()
    assert page.region_list.currentRow() == 4 and "d " in page.region_range_label.text()


def test_click_on_the_plot_and_draw_on_the_cake(page) -> None:
    page.pick_region("ring")
    page.top_plot.positionClicked.emit(0.41, 100.0)
    _settle(page, lambda: len(page.view_model.state.giwaxs.regions) == 1)
    assert page.view_model.state.giwaxs.regions[0].q_range[0] < 0.40 < page.view_model.state.giwaxs.regions[0].q_range[1]
    page.draw_region()
    _settle(page, lambda: getattr(page, "_cake", None) is not None and page.detector_view.has_image())
    page.shape_layer._finish([(0.30, -12.0), (0.50, -30.0)])  # below χ = 0: folded onto |χ|
    _settle(page, lambda: len(page.view_model.state.giwaxs.regions) == 2)
    drawn = page.view_model.state.giwaxs.regions[1]
    assert drawn.q_range == pytest.approx((0.30, 0.50)) and drawn.chi_range == pytest.approx((12.0, 30.0)) and drawn.both_sides
    assert page.view_model.state.corrections.mask_shapes == ()  # a region, not a mask


def test_snap_to_peak_and_cut_sets(page, tmp_path: Path) -> None:
    from src.gimap.features.analyze.application import CutRegion

    page.view_model.add_region(CutRegion("off", (1.13, 1.17), (0.0, 90.0), True))  # beside the 1.10 ring
    page.run_analysis()
    _settle(page, lambda: len(page._region_rows) == 5)
    page.region_list.setCurrentRow(4)
    page.region_snap_button.click()
    _settle(page, lambda: page.view_model.state.giwaxs.regions[0].q_range[0] < 1.10)
    snapped = page.view_model.state.giwaxs.regions[0]
    assert snapped.name == "off" and snapped.q_range[0] < 1.10 < snapped.q_range[1]
    saved = page.save_cuts(tmp_path / "cuts.json")
    record = json.loads(saved.read_text())
    assert record["format"] == "gimap-cut-regions" and record["regions"][0]["name"] == "off"
    page.view_model.set_regions([])
    assert page.load_cuts(saved)
    _settle(page, lambda: len(page._region_rows) == 5)
    assert page.view_model.state.giwaxs.regions[0] == snapped
