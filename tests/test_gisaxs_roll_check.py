import math

import pytest

from src.gimap.features.analyze.application.readout import detector_point, rod_rotation_deg
from src.gimap.shared.geometry import DetectorGeometry
from tests.test_analyze_geometry_adjustment import page  # noqa: F401


@pytest.mark.parametrize("roll", [-22.0, 0.0, 18.0, 158.0])
@pytest.mark.parametrize("orientation", [1, 2, 3, 4])
def test_two_points_on_a_tilted_rod_recover_the_fixed_rotation(roll, orientation):
    truth = DetectorGeometry(150e-6, 80e-6, 0.2, 126.5, 200.5, 1.0, 0.5,
                             0.16, -0.12, math.radians(roll), orientation=orientation)
    (_, rx, _), (_, ry, _) = truth.in_plane_axes()
    def rod_point(rows_above):
        dx = -ry / rx * rows_above * truth.pixel_size_y_m / truth.pixel_size_x_m
        return truth.beam_center_x_px + dx, truth.beam_center_y_px - rows_above
    first, second = rod_point(10), rod_point(100)
    guess = truth.with_rotations(truth.rot1_rad, truth.rot2_rad, math.radians(roll + 3))
    assert rod_rotation_deg(guess, first, second) == pytest.approx(roll)
    assert rod_rotation_deg(guess, second, first) == pytest.approx(roll)
    assert detector_point(truth, *second, "gisaxs")["qy"] == pytest.approx(0, abs=1e-14)
    assert abs(detector_point(guess, *second, "gisaxs")["qy"]) > 1e-3


def test_check_actions_measure_the_displayed_frame_and_leave_the_geometry_intact(page):
    geometry = page.view_model.state.analysis.geometry
    x = geometry.beam_center_x_px
    page.check_rod_action.trigger()
    page.shape_layer._finish([(x, 10.5), (x, 90.5)])
    assert "rot3 = 0.000°" in page.roll_result.text()
    assert "0 and 0 Å⁻¹" in page.roll_result.text()
    page.check_band_action.trigger()
    y = geometry.horizon_row()
    page.shape_layer._finish([(100, y), (175, y)])
    assert "0.0000°" in page.roll_result.text()
    page.check_band_action.trigger()
    page.shape_layer._finish([(100, y), (150, y - 15)])
    assert "difference is 0.0000°" not in page.roll_result.text()
    assert page.view_model.state.analysis.geometry == geometry
    page.check_rod_action.trigger()
    page.shape_layer.cancel_drawing()
    assert page.roll_result.isHidden()
