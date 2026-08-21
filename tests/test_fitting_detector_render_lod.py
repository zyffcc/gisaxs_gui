from __future__ import annotations

import numpy as np
import pytest

from src.gimap.features.fitting.presentation.detector_render_lod import (
    DetectorRenderLevel,
    choose_detector_render_level,
    detector_render_cell_budget,
)


def test_large_detector_render_level_stays_within_screen_budget() -> None:
    level = choose_detector_render_level((2048, 2048), max_cells=100_000)

    assert level.stride > 1
    assert level.rendered_cells <= 100_000
    assert level.rendered_cells < 2048 * 2048


def test_render_level_samples_intensity_and_q_grids_with_one_stride() -> None:
    intensity = np.arange(48).reshape(6, 8)
    qy = intensity + 100
    qz = intensity + 200
    level = choose_detector_render_level(intensity.shape, max_cells=12)

    assert level == DetectorRenderLevel((6, 8), stride=2)
    np.testing.assert_array_equal(level.sample(intensity), intensity[::2, ::2])
    np.testing.assert_array_equal(level.sample(qy), qy[::2, ::2])
    np.testing.assert_array_equal(level.sample(qz), qz[::2, ::2])


def test_small_detector_keeps_full_display_resolution() -> None:
    level = choose_detector_render_level((64, 80), max_cells=20_000)

    assert level.stride == 1
    assert level.rendered_shape == (64, 80)


def test_render_level_rejects_misaligned_coordinate_grid() -> None:
    level = DetectorRenderLevel((10, 12), stride=2)

    with pytest.raises(ValueError, match="shape mismatch"):
        level.sample(np.zeros((9, 12)))


def test_viewport_budget_is_bounded() -> None:
    assert detector_render_cell_budget(10, 10) == 20_000
    assert detector_render_cell_budget(10_000, 10_000) == 180_000
