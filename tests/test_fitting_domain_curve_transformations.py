import numpy as np
import pytest

from src.gimap.features.fitting.domain.curve_transformations import (
    filter_axis,
    filter_for_display,
    interpolate_series,
    normalize_intensity,
    q_values_for_display,
    q_values_for_model,
    sort_filter_pairs,
)


def _legacy_sort_filter_pairs(x_values, intensity_values):
    x_array = np.asarray(x_values, dtype=float).reshape(-1)
    y_array = np.asarray(intensity_values, dtype=float).reshape(-1)
    count = min(x_array.size, y_array.size)
    x_array = x_array[:count]
    y_array = y_array[:count]
    finite = np.isfinite(x_array) & np.isfinite(y_array)
    x_array = x_array[finite]
    y_array = y_array[finite]
    order = np.argsort(x_array, kind="mergesort")
    x_array = x_array[order]
    y_array = y_array[order]
    unique, inverse, counts = np.unique(x_array, return_inverse=True, return_counts=True)
    summed = np.zeros(unique.size, dtype=float)
    np.add.at(summed, inverse, y_array)
    return unique, summed / counts


def test_sort_filter_pairs_matches_legacy_fixture():
    q = [2.0, np.nan, -1.0, 2.0, 0.5]
    intensity = [20.0, 99.0, 4.0, 24.0, 8.0]
    expected_q, expected_intensity = _legacy_sort_filter_pairs(q, intensity)

    actual_q, actual_intensity, rows = sort_filter_pairs(q, intensity)

    np.testing.assert_allclose(actual_q, expected_q)
    np.testing.assert_allclose(actual_intensity, expected_intensity)
    assert rows is None


@pytest.mark.parametrize("method", ["Linear", "Quadratic", "Spline", "unknown"])
def test_interpolation_preserves_endpoints_and_count(method):
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = x**2
    target = np.linspace(0.0, 3.0, 9)

    result = interpolate_series(x, y, target, method)

    assert result.shape == target.shape
    np.testing.assert_allclose(result[[0, -1]], y[[0, -1]])


def test_axis_filter_runs_before_negative_display_sort():
    q = np.array([-3.0, 2.0, -1.0, 0.0, 4.0])
    intensity = np.array([30.0, 20.0, 10.0, 0.0, 40.0])

    negative_q, negative_i = filter_axis(q, intensity, "negative")
    raw, plotted, plotted_i, mode = filter_for_display(q, intensity, "negative")

    np.testing.assert_allclose(negative_q, [-3.0, -1.0])
    np.testing.assert_allclose(negative_i, [30.0, 10.0])
    np.testing.assert_allclose(raw, [-1.0, -3.0])
    np.testing.assert_allclose(plotted, [1.0, 3.0])
    np.testing.assert_allclose(plotted_i, [10.0, 30.0])
    assert mode == "negative"


def test_q_units_and_normalization_preserve_existing_definitions():
    q_angstrom = np.array([0.1, 0.2])
    np.testing.assert_allclose(q_values_for_model(q_angstrom, "angstrom"), [1.0, 2.0])
    np.testing.assert_allclose(
        q_values_for_display(q_angstrom, "angstrom", "angstrom"),
        q_angstrom,
    )
    np.testing.assert_allclose(normalize_intensity(np.array([1.0, 4.0])), [0.25, 1.0])
