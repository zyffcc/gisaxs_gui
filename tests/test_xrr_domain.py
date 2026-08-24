"""Scientific regression tests for XRR point extraction."""

import math

import numpy as np

from src.gimap.features.xrr.domain import (
    HC_KEV_ANGSTROM,
    SpecularGeometry,
    extract_circular_roi,
    qz_from_theta,
    specular_pixel,
)


def _geometry(direction=-1):
    return SpecularGeometry(
        distance_m=1.0,
        energy_kev=12.0,
        pixel_size_x_m=100e-6,
        pixel_size_y_m=100e-6,
        beam_center_x_px=50.0,
        beam_center_y_px=60.0,
        vertical_direction=direction,
    )


def test_specular_qz_uses_sample_theta_and_two_theta_detector_motion():
    theta = 0.5
    expected_qz = 4 * math.pi * math.sin(math.radians(theta)) / (
        HC_KEV_ANGSTROM / 12.0
    )
    assert qz_from_theta(theta, 12.0) == pytest.approx(expected_qz)

    x, y = specular_pixel(theta, _geometry())
    expected_offset = math.tan(math.radians(2 * theta)) / 100e-6
    assert x == 50.0
    assert y == pytest.approx(60.0 - expected_offset)
    assert specular_pixel(theta, _geometry(1))[1] == pytest.approx(60.0 + expected_offset)


def test_circular_roi_supports_single_pixel_sum_mean_and_invalid_mask():
    image = np.arange(49, dtype=float).reshape(7, 7)
    single = extract_circular_roi(image, None, 3.0, 2.0, 0, "sum")
    assert single.intensity == image[2, 3]
    assert single.valid_pixels == 1

    mask = np.zeros_like(image, dtype=bool)
    mask[2, 3] = True
    circle = extract_circular_roi(image, mask, 3.0, 3.0, 1, "mean")
    expected = np.mean([image[3, 3], image[3, 2], image[3, 4], image[4, 3]])
    assert circle.valid_pixels == 4
    assert circle.intensity == pytest.approx(expected)


def test_outside_detector_roi_is_an_explicit_missing_measurement():
    result = extract_circular_roi(np.ones((5, 5)), None, 40.0, -20.0, 2, "sum")
    assert result.valid_pixels == 0
    assert math.isnan(result.intensity)


import pytest
