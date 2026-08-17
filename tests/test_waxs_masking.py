import numpy as np

from src.gimap.features.waxs.domain import (
    estimate_display_limits,
    percentile_limits,
    prepare_display_array,
)


def test_linear_mask_and_vertical_flip_preserve_threshold_semantics():
    image = np.array([[0.0, 2.0], [4.0, np.inf]], dtype=float)

    result = prepare_display_array(
        image,
        log_scale=False,
        mask_min=1.0,
        mask_max=4.0,
        flip_vertical=True,
    )

    np.testing.assert_allclose(result[0, 0], 4.0)
    assert np.isnan(result[0, 1])
    assert np.isnan(result[1, 0])
    np.testing.assert_allclose(result[1, 1], 2.0)


def test_log_display_applies_linear_mask_before_log10():
    image = np.array([0.0, 1.0, 10.0, 100.0, -2.0])

    result = prepare_display_array(
        image,
        log_scale=True,
        mask_min=1.0,
        mask_max=10.0,
        flip_vertical=False,
    )

    assert np.isnan(result[0]) and np.isnan(result[3]) and np.isnan(result[4])
    np.testing.assert_allclose(result[1:3], [0.0, 1.0])


def test_percentile_and_estimated_limits_handle_constant_and_invalid_images():
    constant = percentile_limits(np.ones((3, 3)))
    invalid = percentile_limits(np.full((2, 2), np.nan))
    estimated = estimate_display_limits(
        np.array([1.0, 10.0, 100.0, np.nan]),
        log_scale=True,
        mask_min=1.0,
        mask_max=100.0,
        stride_hint=1,
    )

    assert constant[0] == 1.0
    assert constant[1] > constant[0]
    assert invalid is None
    assert estimated is not None
    assert 0.0 <= estimated[0] < estimated[1] <= 2.0
