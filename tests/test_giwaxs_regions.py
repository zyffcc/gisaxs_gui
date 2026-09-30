"""GIWAXS cut regions, the unwrapped (cake) view, drawn masks, mirror filling and peak tracking."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from src.gimap.features.analyze.domain import (
    POLYGON,
    RECTANGLE,
    CutRegion,
    GiwaxsSettings,
    MaskShape,
    cake_map,
    curve_source_mask,
    giwaxs_maps,
    mirror_fill,
    rasterize,
    reduce_giwaxs,
    region_outline,
    series_map,
    track_peak,
)
from src.gimap.shared.geometry import DetectorGeometry
from tests.test_assistant_calibration import CENTER, DISTANCE_M, PIXEL, SHAPE, WAVELENGTH, giwaxs_frame

GEOMETRY = DetectorGeometry(PIXEL, PIXEL, DISTANCE_M, CENTER[0], CENTER[1], WAVELENGTH, 0.2)


@pytest.fixture(scope="module")
def frame():
    image = giwaxs_frame()
    return image, np.ones(image.shape, dtype=bool), giwaxs_maps(image.shape, GEOMETRY)


def test_a_region_gives_i_of_q_and_i_of_chi_from_its_own_pixels(frame) -> None:
    image, valid, maps = frame
    ring = CutRegion("Ring 1.10", (1.05, 1.15), (0.0, 90.0), True)
    lamella = CutRegion("Lamella", (0.35, 0.45), (0.0, 20.0), True)
    reduction = reduce_giwaxs(image, valid, GEOMETRY, GiwaxsSettings(regions=(ring, lamella)), maps=maps)
    along_q, along_chi = reduction.curve("region1"), reduction.curve("region1_chi")
    assert along_q.x.min() >= 1.05 - 1e-6 and along_q.x.max() <= 1.15 + 1e-6
    assert abs(along_q.x[np.nanargmax(along_q.intensity)] - 1.10) < 0.01
    assert along_chi.x_label == "|χ| (°)" and along_chi.x.min() >= 0 and along_chi.x.max() <= 90
    measured = along_chi.intensity[np.isfinite(along_chi.intensity)]
    assert measured.std() / measured.mean() < 0.35  # an isotropic ring: I(χ) roughly flat where measured
    lamella_chi = reduction.curve("region2_chi")
    assert lamella_chi.x.max() <= 20 and lamella_chi.title.startswith("Lamella: I(χ)")
    source = curve_source_mask(along_q, valid, maps)
    assert int(source.sum()) == int(along_q.pixels.sum())  # the image shows exactly the averaged pixels
    signed = CutRegion("Left", (1.05, 1.15), (-90.0, -10.0), False)
    left = reduce_giwaxs(image, valid, GEOMETRY, GiwaxsSettings(regions=(signed,)), maps=maps).curve("region1_chi")
    assert left.x_label == "χ (°)" and left.x.max() <= -10


def test_regions_reject_empty_ranges_and_outline_in_q() -> None:
    with pytest.raises(ValueError):
        CutRegion("empty", (0.5, 0.5))
    assert CutRegion("wide", None, (-30.0, 120.0), True).chi_range == (0.0, 90.0)
    x, z = region_outline((1.0, 1.2), (0.0, 90.0))
    assert np.allclose(np.hypot(x, z).max(), 1.2) and np.allclose(np.hypot(x, z).min(), 1.0)


def test_the_cake_shows_a_ring_as_a_line_of_constant_q(frame) -> None:
    image, valid, maps = frame
    cake = cake_map(image, valid & maps.above_horizon, maps)
    q_axis, chi_axis = cake.axes()
    assert cake.image.shape == (360, 800) and chi_axis[0] < -89 and chi_axis[-1] > 89
    profile = np.nanmean(cake.image[:, q_axis > 0.9], axis=0)
    assert abs(q_axis[q_axis > 0.9][np.nanargmax(profile)] - 1.10) < 0.01
    missing_wedge = cake.image[np.abs(chi_axis) < 1, :][:, q_axis > 1.5]
    assert np.isnan(missing_wedge).mean() > 0.5  # next to the surface normal at large q nothing is measured


def test_drawn_masks_cover_pixel_centres() -> None:
    rectangle = MaskShape(RECTANGLE, ((10.0, 20.0), (15.0, 30.0)))
    triangle = MaskShape(POLYGON, ((0.0, 0.0), (10.0, 0.0), (0.0, 10.0)))
    masked = rasterize((rectangle, triangle), (50, 50))
    assert masked[20:30, 10:15].all() and masked[20:30, 10:15].sum() == 50
    assert masked[0, 0] and masked[1, 7] and not masked[9, 9]
    assert "Rectangle: x 10–15" in rectangle.describe()
    with pytest.raises(ValueError):
        MaskShape(POLYGON, ((0, 0), (1, 1)))


def test_mirror_filling_takes_gaps_from_the_other_side() -> None:
    columns = np.arange(40, dtype=np.float32)
    center = 20.0  # the direct beam between columns 19 and 20
    image = np.tile(np.abs(columns + 0.5 - center), (5, 1)).astype(np.float32)  # symmetric in x
    valid = np.ones(image.shape, dtype=bool)
    valid[:, 5:8] = False  # a gap on the left
    valid[:, 32:35] = False  # its mirror (columns 32–34 ↔ 7–5) is also a gap for rows 0–1 only
    valid[2:, 32:35] = True
    filled_image, filled_valid, filled = mirror_fill(np.where(valid, image, np.nan), valid, center)
    assert filled[2:, 5:8].all() and not filled[:2, 5:8].any()  # rows 0–1: both sides empty, nothing invented
    assert np.allclose(filled_image[2:, 5:8], image[2:, 5:8])
    assert filled_valid.sum() == valid.sum() + filled.sum()
    _image, _valid, shifted = mirror_fill(image, valid, 20.25)  # fractional centre: interpolated
    assert np.allclose(_image[2:, 6], 0.5 * (image[2:, 33] + image[2:, 34]))


def test_a_peak_is_tracked_through_a_series() -> None:
    q = np.linspace(0.5, 1.5, 201)
    centres = np.linspace(0.95, 1.05, 11)
    curves = [(q, 10 + 0.5 * q + 100 * np.exp(-0.5 * ((q - c) / 0.02) ** 2)) for c in centres]
    track = track_peak(series_map(curves, [f"f{i}" for i in range(11)], x_label="q (Å⁻¹)", curve="radial"), 0.85, 1.15)
    assert np.allclose(track.position, centres, atol=0.004)
    assert np.allclose(track.fwhm, 2.3548 * 0.02, rtol=0.15)
    assert np.allclose(track.area, 100 * 0.02 * math.sqrt(2 * math.pi), rtol=0.05)
    assert [name for name, _values in track.table()] == ["position", "fwhm", "area", "height"]


def test_the_frame_pipeline_applies_drawn_masks_and_mirror_filling(tmp_path) -> None:
    from src.gimap.features.analyze.application import AnalysisRequest, AnalyzeFrame
    from src.gimap.features.analyze.domain import Corrections
    from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
    from src.gimap.shared.geometry import InstrumentProfile
    from tests.test_analyze_giwaxs_options import FakeFrames

    image = giwaxs_frame()
    image[:, 60:66] = -1  # a module gap on the left
    path = tmp_path / "frame.tif"
    path.write_bytes(b"x")
    frames = FakeFrames({str(path): image})
    profile = InstrumentProfile("Test", GEOMETRY, "Test", SHAPE)
    analyze = AnalyzeFrame(frames, InMemoryInstrumentProfileRepository([profile]))
    plain = analyze(AnalysisRequest(path, mode="giwaxs"))
    assert not plain.valid[:, 60:66].any() and plain.filled_pixels is None
    box = MaskShape(RECTANGLE, ((300.0, 0.0), (320.0, 50.0)))
    request = AnalysisRequest(path, mode="giwaxs", corrections=Corrections(mask_shapes=(box,), mirror_fill=True))
    filled = analyze(request, loaded=plain)
    assert filled.drawn_mask[0:50, 300:320].all()
    assert filled.filled_pixels[0:50, 300:320].all()  # a drawn mask is refilled from the mirror side too
    assert filled.filled_pixels is not None and filled.filled_pixels[:, 60:66].mean() > 0.9
    assert filled.valid[:, 60:66].mean() > 0.9
    again = analyze(replace(request, corrections=Corrections()), loaded=filled)
    assert not again.valid[:, 60:66].any() and again.filled_pixels is None  # back to the detector's own mask
