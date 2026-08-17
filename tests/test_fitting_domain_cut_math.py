import numpy as np

from src.gimap.features.fitting.domain.cut_math import (
    extract_pixel_profile,
    extract_q_profile,
    pixel_region_bounds,
    sample_q_mesh_line,
)
from src.gimap.features.fitting.domain.models import CutSelection


def test_pixel_cut_preserves_origin_and_finite_mean_behavior():
    image = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, np.nan, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]
    )
    selection = CutSelection(1.5, 1.5, 2.0, 2.0, "horizontal")

    assert pixel_region_bounds(image.shape, selection) == (0, 2, 1, 3)
    intensity, pixels = extract_pixel_profile(image, selection)

    np.testing.assert_allclose(pixels, [0, 1, 2])
    np.testing.assert_allclose(intensity, [9.0, 12.0, 11.0])


def test_q_cut_matches_direct_legacy_column_means():
    image = np.arange(1.0, 17.0).reshape(4, 4)
    qy = np.tile(np.array([-1.5, -0.5, 0.5, 1.5]), (4, 1))
    qz = np.tile(np.array([0.0, 1.0, 2.0, 3.0])[:, None], (1, 4))
    selection = CutSelection(0.0, 1.5, 2.0, 2.0, "horizontal")

    intensity, q_line, indices = extract_q_profile(image, qy, qz, selection)

    np.testing.assert_allclose(indices, [1, 2])
    np.testing.assert_allclose(q_line, [-0.5, 0.5])
    np.testing.assert_allclose(intensity, [8.0, 9.0])


def test_fractional_pixel_sampling_preserves_distinct_q_values():
    qy = np.tile(np.linspace(-2.0, 2.0, 5), (3, 1))
    actual = sample_q_mesh_line(
        qy,
        [1.25, 1.75],
        orientation="horizontal",
        image_shape=qy.shape,
    )

    np.testing.assert_allclose(actual, [-0.75, -0.25])
    assert actual[0] != actual[1]
