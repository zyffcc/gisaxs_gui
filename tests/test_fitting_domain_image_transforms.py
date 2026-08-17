import numpy as np

from src.gimap.features.fitting.domain.image_transforms import (
    apply_input_image_options,
    apply_threshold_mask,
    finite_log_profiles,
    finite_mean_axis,
    mirror_fill_detector_gaps,
)


def test_image_transforms_match_legacy_reference_arrays():
    image = np.array([[1.0, np.nan, 8.0], [-1.0, 4.0, 6.0]], dtype=np.float32)
    expected = np.array([[np.nan, 4.0, 6.0], [np.nan, np.nan, 8.0]], dtype=np.float32)

    actual = apply_input_image_options(
        image,
        flip_ud=True,
        threshold_enabled=True,
        threshold_min=2.0,
        threshold_max=8.0,
    )

    np.testing.assert_allclose(actual, expected, equal_nan=True)
    np.testing.assert_allclose(
        apply_threshold_mask(image, enabled=False),
        image,
        equal_nan=True,
    )


def test_finite_mean_and_log_profiles_preserve_mask_weighting():
    image = np.array([[1.0, np.nan, 9.0], [3.0, 5.0, np.nan]])

    np.testing.assert_allclose(finite_mean_axis(image, axis=0), [2.0, 5.0, 9.0])
    vertical, horizontal = finite_log_profiles(image)
    expected_log = np.array(
        [[0.0, np.nan, np.log10(9.0)], [np.log10(3.0), np.log10(5.0), np.nan]]
    )
    np.testing.assert_allclose(vertical, np.nansum(expected_log, axis=1))
    np.testing.assert_allclose(horizontal, np.nansum(expected_log, axis=0))


def test_mirror_gap_fill_matches_legacy_center_and_margin_rules():
    image = np.array([[10.0, -1.0, 30.0, 40.0, 50.0]])

    filled = mirror_fill_detector_gaps(image, center_x=2.0)
    np.testing.assert_allclose(filled, [[10.0, 40.0, 30.0, 40.0, 50.0]])

    with_margin = mirror_fill_detector_gaps(image, center_x=2.0, gap_margin_px=1)
    np.testing.assert_allclose(with_margin, [[50.0, 40.0, 30.0, 40.0, 50.0]])
