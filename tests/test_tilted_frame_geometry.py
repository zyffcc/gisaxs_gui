"""Detector rays through the horizon, a measured mirror donor, and the XRR specular spot."""

from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
import pytest

from src.gimap.features.analyze.application.gisaxs_cuts import UNIT_PIXEL, axis_from_unit, axis_to_unit
from src.gimap.features.analyze.domain.curve_errors import PixelErrors
from src.gimap.features.analyze.domain.gisaxs import ROW_BAND_WARNING, gisaxs_maps, reduce_gisaxs, vertical_cut
from src.gimap.features.analyze.domain.models import GisaxsCutSettings
from src.gimap.features.analyze.domain.preprocess import mirror_fill
from src.gimap.features.analyze.domain.sources import curve_source_mask
from src.gimap.features.xrr.domain import SpecularGeometry, extract_circular_roi, specular_pixel
from src.gimap.shared.geometry import DetectorGeometry
from src.gimap.shared.geometry.conventions import canonical_from_index_center
from src.gimap.shared.geometry.q_mapping import geometry_rays, lab_rays


FLAT = DetectorGeometry(1.3e-4, 1.1e-4, 0.12, 31.7, 36.8, 1.1, 0.4)
TILTED = FLAT.with_rotations(0.12, -0.07, 0.22)
SHAPE = (65, 79)


@pytest.mark.parametrize("geometry", [TILTED, replace(TILTED, rot3_rad=-0.38), FLAT])
def test_horizon_display_and_pixel_units_use_the_zero_exit_angle_line(geometry):
    (x0, y0), angle = geometry.horizon_line()
    for col in (1.0, 21.5, 77.0):
        row = geometry.horizon_row(col)
        assert row == pytest.approx(y0 + (col - x0) * math.tan(math.radians(angle)), abs=1e-12)
        assert geometry.exit_angle_at(col, row) == pytest.approx(0.0, abs=1e-14)
        qy = axis_from_unit(geometry, UNIT_PIXEL, "qy", [col])
        assert axis_to_unit(geometry, UNIT_PIXEL, "qy", qy) == pytest.approx([col], abs=1e-10)
    if geometry is FLAT:
        assert geometry.horizon_row(1) == FLAT.beam_center_y_px - FLAT.distance_m * math.tan(FLAT.incidence_rad) / FLAT.pixel_size_y_m


def test_detector_overlay_draws_a_sloping_horizon_and_resets_on_the_q_map(qt_application):
    from src.gimap.app.presentation.components.detector_view import DetectorView
    from src.gimap.features.analyze.presentation.bindings.display import DisplayMixin

    view = DetectorView()
    panel = SimpleNamespace(detector_view=view)
    marks = {"beam_center": (TILTED.beam_center_x_px, TILTED.beam_center_y_px), "horizon_row": TILTED.horizon_row()}
    analysis = SimpleNamespace(geometry=TILTED, kind="giwaxs", reduction=SimpleNamespace(kind="giwaxs", markers=marks))
    try:
        DisplayMixin._show_overlays(panel, analysis)
        line = view.horizon_line
        assert line.angle == pytest.approx(TILTED.horizon_line()[1])
        assert TILTED.exit_angle_at(line.pos().x(), line.pos().y()) == pytest.approx(0, abs=1e-12)
        DisplayMixin._show_q_map_marks(panel, analysis, SimpleNamespace(x_label="q∥"), keep_view=True)
        assert line.angle == 0
        assert line.value() == pytest.approx(TILTED.wavevector_inv_angstrom * math.sin(TILTED.incidence_rad))
    finally:
        view.close()
        view.deleteLater()


@pytest.mark.parametrize("critical", [0.0, 0.2])
def test_tilted_vertical_band_and_source_mask_follow_the_exit_angle_at_each_pixel(critical):
    geometry = TILTED.with_critical_angle(critical)
    image = np.ones(SHAPE)
    valid = np.ones(SHAPE, dtype=bool)
    valid[24:29, 30] = False
    cut = vertical_cut(image, valid, geometry, center_column=34.0, half_width_px=6)
    rows, cols = np.indices(SHAPE)
    exact = np.array([[geometry.exit_angle_at(j + 0.5, i + 0.5) >= math.radians(critical)
                       for j in range(SHAPE[1])] for i in range(SHAPE[0])])
    mask = exact & valid & (cols >= 28) & (cols < 40)
    assert np.array_equal(curve_source_mask(cut, valid, gisaxs_maps(SHAPE, geometry)), mask)
    assert cut.pixels.sum() == mask.sum()
    assert 0 < mask.sum(axis=1).max() and ((mask.sum(axis=1) > 0) & (mask.sum(axis=1) < 12)).any()

    variance = (1 + rows + 2 * cols).astype(float)
    cut_with_errors = vertical_cut(image, valid, geometry, center_column=34.0, half_width_px=6,
                                   counts=PixelErrors(model="sigma", variance=variance))
    np.testing.assert_array_equal(cut_with_errors.x, cut.x)
    np.testing.assert_array_equal(cut_with_errors.pixels, cut.pixels)
    np.testing.assert_allclose(np.sum((cut_with_errors.pixels * cut_with_errors.sigma)**2), variance[mask].sum())


def test_horizontal_cut_warns_when_horizon_changes_by_half_a_pixel_across_frame():
    image = np.full(SHAPE, 5.0)
    valid = np.ones(SHAPE, bool)
    settings = GisaxsCutSettings(horizontal_row=20.5, vertical_column=53.5)
    for roll, warn in [(0, False), (0.001, False), (0.02, True), (math.pi/2, True)]:
        reduction = reduce_gisaxs(image, valid, FLAT.with_rotations(0, 0, roll), settings)
        assert (ROW_BAND_WARNING in reduction.warnings) == warn


def test_the_untilted_vertical_band_keeps_its_previous_whole_row_boundary():
    image = np.arange(np.prod(SHAPE)).reshape(SHAPE)
    valid = np.ones(SHAPE, bool)
    cut = vertical_cut(image, valid, FLAT, center_column=53, half_width_px=4)
    stop = math.floor(FLAT.beam_center_y_px - FLAT.distance_m * math.tan(FLAT.incidence_rad) / FLAT.pixel_size_y_m)
    assert cut.region["rows"] == (0, stop)
    np.testing.assert_array_equal(cut.intensity[::-1], image[:stop, 49:57].mean(axis=1))


def test_tilted_mirror_has_the_same_exit_angle_and_opposite_lateral_ray():
    # A linear detector signal returns its exact value under bilinear interpolation.
    rows, cols = np.indices(SHAPE)
    data = (rows * 10 + cols).astype(np.float32)
    i, j = 22, 19
    ray = np.array(geometry_rays(TILTED, (i, i+1), (j, j+1))).reshape(3)
    reflected = ray * [1, -1, 1]
    col, up = np.asarray(TILTED.in_plane_axes())
    origin = np.array([TILTED.distance_m, 0, 0])
    # Independently solve the linear system for the detector displacements and ray scale.
    dx, dy, _scale = np.linalg.solve(np.stack([col, up, -reflected], axis=1), -origin)
    x = TILTED.beam_center_x_px + dx/TILTED.pixel_size_x_m - 0.5
    y = TILTED.beam_center_y_px - dy/TILTED.pixel_size_y_m - 0.5
    valid = np.ones(SHAPE, bool)
    valid[i,j] = False
    res, _, filled = mirror_fill(data, valid, TILTED.beam_center_x_px, geometry=TILTED)
    assert filled[i,j] and res[i,j] == pytest.approx(10*y + x, abs=1e-4)
    valid[math.floor(y), math.ceil(x)] = False
    assert not mirror_fill(data, valid, TILTED.beam_center_x_px, geometry=TILTED)[2][i,j]
    old = mirror_fill(data, valid, FLAT.beam_center_x_px)
    for x, y in zip(old, mirror_fill(data, valid, FLAT.beam_center_x_px, geometry=FLAT)):
        np.testing.assert_array_equal(x, y)


def test_pipeline_mirrors_the_corrected_intensity_of_the_measured_donor(tmp_path):
    from src.gimap.features.analyze.application import AnalysisRequest, AnalyzeFrame, Corrections
    from src.gimap.features.analyze.domain import intensity_factor
    from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
    from src.gimap.shared.geometry import InstrumentProfile
    from tests.test_analyze_giwaxs_options import FakeFrames

    geometry = TILTED.with_rotations(0.48, -0.07, 0.22)
    profile = InstrumentProfile("Test", geometry, "Test", SHAPE)
    factor = intensity_factor(SHAPE, geometry, solid_angle=True)
    data = np.float32(1000.0) * factor  # a sample with constant corrected scattering
    data[22, 19] = -1
    path = tmp_path / "uniform.tif"
    path.write_bytes(b"x")
    analysis = AnalyzeFrame(FakeFrames({str(path): data}), InMemoryInstrumentProfileRepository([profile]))(
        AnalysisRequest(path, mode="giwaxs", corrections=Corrections(mirror_fill=True, solid_angle=True, bad_pixels=False)))
    assert analysis.filled_pixels[22, 19]
    assert analysis.data[22, 19] == pytest.approx(1000, abs=2e-4)


@pytest.mark.parametrize("direction, orientation", [(1, 3), (-1, 3), (-1, 1), (-1, 2), (-1, 4)])
def test_xrr_roi_satisfies_specular_lab_ray_for_all_three_rotations(direction, orientation):
    geom = SpecularGeometry(FLAT.distance_m, 12, FLAT.pixel_size_x_m, FLAT.pixel_size_y_m,
                            32.5, 30.5, direction, TILTED.rot1_rad, TILTED.rot2_rad, TILTED.rot3_rad, orientation)
    theta = 0.51
    cx, cy = canonical_from_index_center(geom.beam_center_x_px, geom.beam_center_y_px)
    x, y = canonical_from_index_center(*specular_pixel(theta, geom))
    detector = replace(TILTED.with_beam_center(cx, cy), orientation=orientation)
    ray = lab_rays(detector, (x-cx)*geom.pixel_size_x_m, (cy-y)*geom.pixel_size_y_m)
    assert x != cx and ray[1] == pytest.approx(0.0, abs=1e-14)
    assert math.degrees(math.atan2(ray[2], ray[0])) == pytest.approx(-direction * 2 * theta)
    flat = replace(geom, rot1_rad=0, rot2_rad=0, rot3_rad=0, orientation=3)
    assert specular_pixel(theta, flat) == (flat.beam_center_x_px, flat.beam_center_y_px +
                                            direction * flat.distance_m * math.tan(math.radians(2*theta))/flat.pixel_size_y_m)
    no_ray = specular_pixel(90, geom)
    assert all(math.isnan(v) for v in no_ray)
    assert extract_circular_roi(np.ones(SHAPE), None, *no_ray, 1, "sum").valid_pixels == 0
