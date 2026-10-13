"""Stages and odd frames of a series of curves (``shared/series_stages``), on series with a known truth."""

from __future__ import annotations

import time

import numpy as np
import pytest

from src.gimap.shared.series_stages import comparable, find_stages, odd_frames, progress

Q = np.linspace(0.1, 3.0, 400)


def _peak(centre: float, width: float = 0.04) -> np.ndarray:
    return np.exp(-0.5 * ((Q - centre) / width) ** 2)


def _series(rows: int = 120, seed: int = 3, kinks=(40, 80)) -> np.ndarray:
    """In log I (what is compared): a peak at 1.0 grows fast until the first kink, then slowly; a peak at
    2.0 grows from the second kink on."""
    t = np.arange(rows, dtype=float)
    first, second = kinks
    growth_a = np.where(t < first, t / first, 1.0 + (t - first) / (10.0 * first))
    growth_b = np.where(t < second, 0.0, (t - second) / first)
    log = np.log10(50.0 * Q ** -2)[None, :] + growth_a[:, None] * _peak(1.0) + growth_b[:, None] * _peak(2.0)
    return 10 ** log * (1.0 + 0.01 * np.random.default_rng(seed).normal(size=log.shape))


def test_the_stages_are_where_the_change_turns_and_say_what_grows() -> None:
    stages = find_stages(Q, _series())
    assert stages.suggested == 3 and stages.count == 3
    first, second = stages.boundaries[3]
    assert abs(first - 40) <= 3 and abs(second - 80) <= 3
    assert stages.unexplained[3] < 0.05 < stages.unexplained[2]
    change = stages.stage_changes()[1]  # stage 2 → 3: the new peak
    assert change.stage == 3 and abs(change.rises[0][0] - 2.0) < 0.05 and change.rises[0][1] > 50
    assert "Stage 2 → 3" in change.text()
    assert [stages.stage_of(row) for row in (0, 39, 45, 79, 85, 119)] == [1, 1, 2, 2, 3, 3]
    assert stages.ranges() == [(0, first - 1), (first, second - 1), (second, 119)]
    representatives = stages.stage_representatives()
    assert len(representatives) == 3 and all(a <= r <= b for r, (a, b) in zip(representatives, stages.ranges()))
    two = stages.with_count(2)
    assert two.count == 2 and len(two.edges) == 3 and stages.count == 3  # a copy, the original unchanged
    with pytest.raises(ValueError):
        stages.with_count(42)


def test_odd_frames_are_found_first_and_a_narrow_difference_is_the_detector() -> None:
    image = _series()
    image[25, 300:303] *= 30.0  # a hot spot for one frame: three bins
    image[100] *= np.exp(-Q)  # a frame whose whole curve is different
    stages = find_stages(Q, image)
    odd = {frame.row: frame for frame in stages.odd}
    assert set(odd) == {25, 100}
    assert odd[25].narrow and "detector" in odd[25].reason and abs(odd[25].q - Q[301]) < 0.02
    assert not odd[100].narrow and "before and after" in odd[100].reason
    assert stages.suggested == 3  # the odd frames make no stage of their own
    assert stages.stage_of(100) == 3 and 100 not in stages.stage_representatives()


def test_an_overall_intensity_change_is_not_a_new_structure() -> None:
    t = np.arange(60, dtype=float)
    shape = 50.0 * Q ** -2 + 300.0 * _peak(1.2)
    image = (1.0 + t / 6.0)[:, None] * shape[None, :]
    image *= 1.0 + 0.01 * np.random.default_rng(1).normal(size=image.shape)
    stages = find_stages(Q, image)
    assert stages.suggested == 1 and not stages.odd
    tenfold = np.log10(1.0 + 59 / 6.0)  # ×10.8 over the series: kept as the level, not as stages
    assert stages.level[-1] - stages.level[0] == pytest.approx(tenfold, abs=0.02)
    assert stages.half_row is None  # no change of shape: nothing to be half done
    with_level = comparable(Q, image, shape_only=False)
    assert with_level.x.mean(axis=1)[-1] - with_level.x.mean(axis=1)[0] == pytest.approx(tenfold, abs=0.02)


def test_the_q_range_and_missing_data_decide_the_bins_compared() -> None:
    image = _series()
    image[:, :20] = np.nan  # no frame has data at the lowest q
    image[::2, 390:] = np.nan  # half of the frames lack the highest q
    data = comparable(Q, image)
    assert data.q.min() >= Q[20] and data.q.max() < Q[390]
    narrow = comparable(Q, image, q_range=(1.5, 0.5))
    assert narrow.q.min() >= 0.5 and narrow.q.max() <= 1.5
    with pytest.raises(ValueError, match="q range"):
        comparable(Q, image, q_range=(3.5, 4.0))
    with pytest.raises(ValueError, match="three frames"):
        find_stages(Q, image[:2])
    assert odd_frames(comparable(Q, image[:3])) == ()


def test_progress_finds_half_and_ninety_percent_of_a_saturating_change() -> None:
    t = np.arange(400, dtype=float)
    values = 1.0 - np.exp(-t / 60.0)
    half, ninety = progress(values, t.astype(int))
    assert abs(half - 60.0 * np.log(2)) <= 3 and abs(ninety - 60.0 * np.log(10)) <= 5  # start: first 5 frames
    assert progress(np.ones(20), np.arange(20)) == (None, None)


def test_a_long_series_is_cut_quickly() -> None:
    image = _series(3200, seed=5, kinks=(1000, 2400))
    tick = time.perf_counter()
    stages = find_stages(Q, image)
    assert time.perf_counter() - tick < 20
    assert stages.suggested == 3
    assert abs(stages.boundaries[3][0] - 1000) <= 20 and abs(stages.boundaries[3][1] - 2400) <= 20
