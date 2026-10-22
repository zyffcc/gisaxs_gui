"""Odd frames (``shared/series_stages``): a detector glitch in a few points is odd even where it is too small to
show over the whole curve; a real change of the sample, however sudden, is not."""

from __future__ import annotations

import numpy as np

from src.gimap.features.compare.domain import SeriesData, compare
from src.gimap.shared.series_stages import comparable, find_stages, odd_frames
from src.gimap.shared.series_stages.comparable import ODD_RATIO, SPIKE_RATIO

Q = np.linspace(0.1, 3.0, 600)
HOT = slice(450, 456)  # 6 of 600 points (1 %): a few hot detector columns in a horizontal cut


def _peak(centre: float, width: float = 0.05) -> np.ndarray:
    return np.exp(-0.5 * ((Q - centre) / width) ** 2)


def _film(rows: int = 40, *, seed: int = 1, glitch=(), noise: float = 0.05) -> np.ndarray:
    """A film growing in situ (nucleation → growth → coarsening: three stages), 5 % noise per point;
    ``glitch``: frames whose hot points are 25× too bright."""
    t = np.arange(rows) / (rows - 1)
    grown = 1.0 / (1.0 + np.exp(-(t - 0.45) * 12.0))
    log = (np.log10(50.0 * Q ** -2)[None, :] + (0.2 + 1.5 * grown)[:, None] * _peak(1.0)
           + (grown ** 2)[:, None] * _peak(1.6, 0.1))
    image = 10 ** log * (1.0 + noise * np.random.default_rng(seed).normal(size=log.shape))
    for row in glitch:
        image[row, HOT] *= 25.0
    return image


def _odd(image: np.ndarray) -> list[int]:
    return [frame.row for frame in odd_frames(comparable(Q, image))]


def test_a_glitch_near_the_end_is_odd_for_the_detector_makes_no_stage_and_stays_out_of_the_end_state() -> None:
    image = _film(glitch=(33,))  # frame 34 of 40: six frames after it
    data = comparable(Q, image)
    # Over the whole curve the glitch is diluted: 1.4 in log10 at 1 % of the points is under 5× a step.
    steps = np.sqrt(np.mean(np.diff(data.x, axis=0) ** 2, axis=1))
    sides = [data.x[33] - np.median(data.x[30:33], axis=0), data.x[33] - np.median(data.x[34:37], axis=0)]
    assert min(np.sqrt(np.mean(side ** 2)) for side in sides) < ODD_RATIO * np.median(steps)

    stages, clean = find_stages(Q, image), find_stages(Q, _film())
    assert [frame.row for frame in stages.odd] == [33] and not clean.odd
    frame = stages.odd[0]
    assert frame.narrow and "probably the detector" in frame.reason and frame.z > SPIKE_RATIO
    assert Q[HOT][0] <= frame.q <= Q[HOT][-1]
    # Not a stage of its own: the stages are those of the same film without the glitch.
    assert stages.suggested == clean.suggested == 3
    assert stages.boundaries[3] == clean.boundaries[3] and stages.stage_of(33) == 3
    assert 33 not in stages.stage_representatives()

    comparison = compare([SeriesData("glitch", Q, image, tuple(f"f{i}" for i in range(40))),
                          SeriesData("clean", Q, _film(seed=2), tuple(f"g{i}" for i in range(40)))])
    result = comparison.results[0]
    assert [frame.row for frame in result.odd] == [33] and result.odd[0].narrow
    kept = [row for row in range(29, 40) if row != 33]  # the last 10 frames that are not odd
    assert np.allclose(result.end_curve[HOT], image[kept][:, HOT].mean(axis=0))
    with_glitch = image[30:40][:, HOT].mean(axis=0)  # what the end state was with the glitch in it: ~3.4×
    assert np.all(with_glitch > 3.0 * result.end_curve[HOT])
    assert np.all(np.abs(result.end_curve[HOT] / _film()[kept][:, HOT].mean(axis=0) - 1.0) < 0.1)


def test_a_real_change_is_not_odd_however_sudden_or_narrow() -> None:
    t = np.arange(40, dtype=float)[:, None]
    noise = 1.0 + 0.05 * np.random.default_rng(5).normal(size=(40, Q.size))
    base = 50.0 * Q ** -2
    narrow = np.zeros(Q.size)
    narrow[HOT] = 1.0
    changes = {
        # the same 25× in the same points as the glitch, from frame 34 on: it stays, so it is the sample
        "a step near the end": base * 25.0 ** (narrow * (t >= 33)),
        "a step in the middle": base * 25.0 ** (narrow * (t >= 20)),
        "a narrow peak that grows fast": base * 10 ** (3.0 * t / 40 * _peak(1.5, 0.01)),
        "the whole curve tilts at once": base * np.exp(-0.5 * Q * (t >= 20)),
        "a peak that switches on in three frames": base * 10 ** (_peak(1.5, 0.01) / (1.0 + np.exp(-(t - 20.0) / 0.8))),
        "nothing happens": np.repeat(base[None, :], 40, axis=0),
    }
    for name, image in changes.items():
        assert _odd(image * noise) == [], name
    # A still series of few counts (Poisson, 2–10 per point): its noise is no spike either.
    counts = np.random.default_rng(7).poisson(np.repeat((2.0 + 8.0 * _peak(1.0, 0.3))[None, :], 200, axis=0))
    assert _odd(counts.astype(float)) == []


def test_glitches_in_a_row_and_at_either_end() -> None:
    assert _odd(_film(glitch=(20, 21))) == [20, 21]  # each outvoted by the clean frames on its sides
    assert _odd(_film(glitch=(20, 21, 22))) == [20, 21, 22]  # the middle one first, then the two without it
    assert _odd(_film(glitch=(38, 39))) == [38, 39]  # the last one first, then the one before it without it
    assert _odd(_film(glitch=(0, 1))) == [0, 1]
    assert _odd(_film(glitch=(0,))) == [0] and _odd(_film(glitch=(39,))) == [39]
    assert _odd(_film(glitch=(5, 33))) == [5, 33]
    assert _odd(_film(glitch=(20, 21, 22, 23))) == []  # four alike are a state of their own, not odd frames
