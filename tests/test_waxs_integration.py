import numpy as np

from src.gimap.features.waxs.application import (
    IntegrateWaxsImage,
    IntegrateWaxsImageRequest,
)
from src.gimap.features.waxs.domain import (
    angle_between,
    circle_cut_profile,
    integrate_image,
    line_cut_profile,
    normalize_angle_deg,
    smooth_curve,
)


def _geometry():
    return {
        "incidence": 0.2,
        "center_x": 2.0,
        "center_y": 2.0,
        "distance": 1000.0,
        "pixel_x": 100.0,
        "pixel_y": 100.0,
        "wavelength": 1.0,
        "qr_min": -121.0,
        "qr_max": -121.0,
        "qz_min": -121.0,
        "qz_max": -121.0,
    }


def test_radial_integration_constant_image_preserves_mean():
    x, intensity = integrate_image(
        np.ones((5, 5)),
        _geometry(),
        {"mode": "radial", "bins": 6, "x_axis": "pixel"},
        -1e12,
        1e12,
    )

    assert len(x) == len(intensity) > 1
    np.testing.assert_allclose(intensity, 1.0)


def test_line_cut_orientation_and_mask_are_stable():
    image = np.arange(25, dtype=float).reshape(5, 5)

    x, intensity = line_cut_profile(image, 2.5, 2.5, 4, 2, 0, 20)

    np.testing.assert_array_equal(x, [0.0, 1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(intensity, np.mean(image[1:4, :], axis=0))


def test_circle_cut_supports_wraparound_angle_sector():
    angles = normalize_angle_deg(np.array([-10.0, 0.0, 10.0, 180.0]))
    selected = angle_between(angles, 350.0, 10.0)
    assert selected.tolist() == [True, True, True, False]

    x, intensity = circle_cut_profile(
        np.ones((7, 7)),
        3,
        3,
        1,
        3,
        350,
        10,
        4,
        mode="radial",
        mask_min=0,
        mask_max=2,
    )
    assert len(x) == len(intensity) > 0
    np.testing.assert_allclose(intensity, 1.0)


def test_integration_use_case_dispatches_line_and_smoothing():
    image = np.arange(49, dtype=float).reshape(7, 7)
    request = IntegrateWaxsImageRequest(
        image=image,
        geometry=_geometry(),
        integration={"mode": "radial", "bins": 8, "smooth": True, "smooth_window": 3},
        mask_min=0,
        mask_max=100,
        cut_kind="line",
        selection={"center_x": 3.5, "center_y": 3.5, "width": 5, "height": 2},
    )

    curve = IntegrateWaxsImage().execute(request)
    raw_x, raw_y = line_cut_profile(image, 3.5, 3.5, 5, 2, 0, 100)

    np.testing.assert_array_equal(curve.x, raw_x)
    np.testing.assert_allclose(curve.intensity, smooth_curve(raw_y, 3))
