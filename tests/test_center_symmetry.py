import numpy as np
import pytest

from src.gimap.features.fitting.domain.center_symmetry import optimize_horizontal_center
from src.gimap.features.fitting.presentation.bindings.center_symmetry import CenterSymmetryMixin


def image_at(center=173.35):
    x = np.arange(361) - center
    y = 2 + 100 * np.exp(-((x / 18) ** 2)) + 20 * np.exp(-(((abs(x) - 67) / 11) ** 2))
    return np.tile(y, (7, 1))


def test_subpixel_center_with_gap_and_hot_pixel_does_not_change_data():
    data = image_at()
    data[:, 85:95] = -1
    data[2, 220] = 1e10
    before = data.copy()
    result = optimize_horizontal_center(data, (0, 6, 0, 360), 180)
    assert abs(result.center_x - 173.35) < 0.1
    assert result.score_after < result.score_before / 20
    np.testing.assert_array_equal(data, before)


def test_frozen_insitu_components_are_accepted_as_json_options():
    from src.gimap.features.fitting.application.workflow_v5 import validate_options

    assert validate_options({"components": ()})["components"] == []
    assert validate_options({"components": (2, 2)})["components"] == [2, 2]


def test_masked_nan_gap_and_y_band_selection():
    data = np.vstack((image_at(150), image_at(173.35)))
    data[7:, 100:108] = np.nan
    result = optimize_horizontal_center(data, (7, 13, 0, 360), 180)
    assert abs(result.center_x - 173.35) < 0.1


@pytest.mark.parametrize(
    "data,region,center",
    [
        (np.ones((7, 361)), (0, 6, 0, 360), 180),
        (image_at(), (0, 6, 0, 360), 2),
        (image_at(), (0, 6, 0, 4), 2),
        (image_at(), (0, 99, 0, 360), 180),
    ],
)
def test_unsupported_or_unidentifiable_center_is_not_applied(data, region, center):
    with pytest.raises(ValueError):
        optimize_horizontal_center(data, region, center)


@pytest.mark.parametrize(
    "file,enabled,busy,count",
    [
        ("example.cbf", True, False, 1),
        ("example.CBF", True, False, 1),
        ("example.cbf", False, False, 0),
        ("example.cbf", True, True, 0),
        ("example.nxs", True, False, 0),
    ],
)
def test_automatic_yoneda_only_on_enabled_single_cbf_load(file, enabled, busy, count):
    from types import SimpleNamespace

    binding = CenterSymmetryMixin()
    calls = []
    binding.ui = SimpleNamespace(
        gisaxsAutoYonedaOnLoadCheckBox=SimpleNamespace(isChecked=lambda: enabled)
    )
    binding._insitu_workflow_busy = busy
    binding._auto_find_center = lambda: calls.append(True)
    binding._auto_yoneda_after_load(file)
    assert len(calls) == count
