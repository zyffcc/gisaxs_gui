"""Stored beam-centre conventions, expressed in the canonical frame."""

from __future__ import annotations

import numpy as np
import pytest

from src.gimap.shared.geometry import DetectorGeometry, grazing_q_map
from src.gimap.shared.geometry.conventions import (
    canonical_from_fitting_center,
    canonical_from_index_center,
    index_center_from_canonical,
)


def test_index_and_fitting_centres_describe_the_same_pixel() -> None:
    rows = 1679
    # Direct beam at the centre of pixel (row 1400, column 737), numpy indexing.
    assert canonical_from_index_center(737, 1400) == (737.5, 1400.5)
    assert index_center_from_canonical(737.5, 1400.5) == (737, 1400)
    # The former Cut & Fitting page counted that row from the bottom: 1679 - 1 - 1400.
    assert canonical_from_fitting_center(737, 278, rows) == (737.5, 1400.5)


@pytest.mark.parametrize("x,y", [(0.0, 0.0), (12.25, 3.75), (-4.0, 30.5)])
def test_index_conversions_round_trip(x: float, y: float) -> None:
    assert index_center_from_canonical(*canonical_from_index_center(x, y)) == (
        pytest.approx(x),
        pytest.approx(y),
    )


def test_stored_fitting_centre_puts_q_zero_where_the_fitting_page_did() -> None:
    """A beam row counted from the bottom lands on the same pixel in the canonical frame."""
    shape = (201, 151)
    center_x, center_y = canonical_from_fitting_center(75.0, 40.0, shape[0])
    geometry = DetectorGeometry(
        pixel_size_x_m=172e-6,
        pixel_size_y_m=172e-6,
        distance_m=4.2,
        beam_center_x_px=center_x,
        beam_center_y_px=center_y,
        wavelength_angstrom=1.033,
        incidence_deg=0.0,
    )
    canonical = grazing_q_map(shape, geometry)
    assert np.unravel_index(np.argmin(canonical.q), shape) == (shape[0] - 1 - 40, 75)
