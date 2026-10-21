"""One preparation of a signed cut for every fit (``shared/signed_cut.py``): which halves a fit takes, the
mean that never bridges a hole of the other half, a missing half, merging, and what the note says."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pytest

from src.gimap.shared import signed_cut
from src.gimap.shared.signed_cut import (
    FIT_HALVES,
    MAX_FIT_POINTS,
    MISSING_HALF,
    MissingHalf,
    choose_halves,
    fit_curve,
    fit_halves,
    local_steps,
    mean_of_halves,
    merge_points,
    prepare_cut,
)

STEP = 0.01


def _curve(q):
    """A smooth curve and its Poisson-like σ on ``q`` (|q| ascending)."""
    intensity = 1000.0 * np.exp(-q / 0.5) + 5.0
    return q, intensity, np.sqrt(intensity)


def _half(q, factor=1.0):
    q, intensity, sigma = _curve(np.asarray(q, dtype=float))
    return q, factor * intensity, sigma


# -- which halves a fit takes ---------------------------------------------------------------------

def test_a_fit_takes_the_mean_or_one_half_and_every_other_choice_becomes_the_default() -> None:
    assert FIT_HALVES == ("mean", "positive", "negative")
    assert [fit_halves(choice) for choice in FIT_HALVES] == list(FIT_HALVES)
    assert fit_halves("both") == fit_halves("both_abs") == fit_halves(None) == fit_halves("sideways") == "mean"
    assert fit_halves("both_abs", "negative") == "negative" and fit_halves("positive", "negative") == "positive"


def test_a_missing_half_is_a_value_error_with_one_sentence_per_half() -> None:
    assert issubclass(MissingHalf, ValueError)
    assert MISSING_HALF == {"positive": "No points on the q > 0 half", "negative": "No points on the q < 0 half"}


# -- the mean of the two halves -------------------------------------------------------------------

def test_mirrored_halves_give_the_plain_mean_with_the_error_of_a_mean() -> None:
    grid = np.arange(1, 201) * STEP
    positive, negative = _half(grid), _half(grid, 1.1)
    mean = mean_of_halves(positive, negative)
    assert np.array_equal(mean.q, grid) and mean.averaged == 200 and mean.single == 0
    assert np.allclose(mean.intensity, 0.5 * (positive[1] + negative[1]))
    assert np.allclose(mean.sigma, 0.5 * np.hypot(positive[2], negative[2]))
    assert set(mean.halves.tolist()) == {2}


def test_the_mean_does_not_interpolate_across_a_hole_of_the_other_half() -> None:
    grid = np.arange(1, 201) * STEP
    positive = _half(grid)
    with_hole = (grid < 0.4) | (grid > 0.9)  # the q < 0 half has no points from 0.4 to 0.9
    negative = _half(grid[with_hole], 1.5)
    mean = mean_of_halves(positive, negative)
    assert np.array_equal(mean.q, grid) and np.all(np.diff(mean.q) > 0)
    hole = ~with_hole
    # In the hole the measured half stands alone, with its own σ — not a mean with a value drawn across it.
    assert np.array_equal(mean.intensity[hole], positive[1][hole]) and np.array_equal(mean.sigma[hole], positive[2][hole])
    assert np.all(mean.halves[hole] == 1) and np.all(mean.halves[with_hole] == 2)
    assert np.allclose(mean.intensity[with_hole], 1.25 * positive[1][with_hole])
    assert mean.averaged == int(with_hole.sum()) and mean.single == int(hole.sum()) and mean.averaged + mean.single == 200


def test_points_of_the_other_half_in_a_hole_of_the_longer_half_are_kept() -> None:
    grid = np.arange(1, 201) * STEP
    long = (grid < 0.5) | (grid > 0.6)  # the q > 0 half: more points, but a hole from 0.5 to 0.6
    positive = _half(grid[long])
    short = np.arange(1, 121) * STEP + 0.003  # the q < 0 half: fewer points, on its own grid, to 1.2
    negative = _half(short, 0.8)
    mean = mean_of_halves(positive, negative)
    assert np.all(np.diff(mean.q) > 0)
    in_hole = (mean.q > 0.5) & (mean.q < 0.6)
    assert in_hole.sum() >= 9 and np.all(mean.halves[in_hole] == -1)  # measured there by the q < 0 half only
    kept = np.isin(short, mean.q[in_hole])
    assert np.array_equal(mean.intensity[in_hole], negative[1][kept]) and np.array_equal(mean.sigma[in_hole], negative[2][kept])
    beyond = mean.q > 1.2 + STEP  # past the end of the shorter half: the longer one alone
    assert np.all(mean.halves[beyond] == 1) and np.array_equal(mean.intensity[beyond], positive[1][positive[0] > 1.2 + STEP])


def test_a_half_that_starts_later_or_reaches_further_is_not_extended() -> None:
    positive = _half(np.arange(30, 201) * STEP)  # starts at 0.3 (a beam stop on this side)
    negative = _half(np.arange(1, 151) * STEP + 0.004, 2.0)  # from 0.014 to 1.504, fewer points
    mean = mean_of_halves(positive, negative)
    below = mean.q < 0.3
    assert below.sum() == 29 and np.all(mean.halves[below] == -1)  # the q < 0 half is not dropped below 0.3 …
    assert np.array_equal(mean.intensity[below], negative[1][:29])  # … and not averaged with an extended value
    assert np.all(mean.halves[mean.q > 1.51] == 1)
    assert mean.q.size == 171 + 29 and mean.averaged == int(((positive[0] >= 0.3) & (positive[0] <= 1.504)).sum())


def test_one_missing_point_is_not_a_hole_but_two_are() -> None:
    grid = np.arange(1, 101) * STEP
    positive = _half(grid)
    one, two = np.delete(grid, [40]), np.delete(grid, [40, 41])
    assert mean_of_halves(positive, _half(one)).averaged == 100
    mean = mean_of_halves(positive, _half(two))
    assert mean.averaged == 98 and np.flatnonzero(mean.halves == 1).tolist() == [40, 41]


def test_holes_are_judged_by_the_step_around_them() -> None:
    grid = np.geomspace(0.01, 2.0, 200)  # steps grow 200-fold along the curve: no interval is a hole
    assert mean_of_halves(_half(grid), _half(grid * 1.001)).averaged == 199  # all but the first (before the other's start)
    steps = local_steps(np.delete(grid, np.arange(100, 110)))
    widths = np.diff(np.delete(grid, np.arange(100, 110)))
    assert widths[99] > 2.5 * steps[99] and np.all(np.delete(widths, 99) < 2.5 * np.delete(steps, 99))


def test_a_few_scattered_points_of_a_half_are_not_a_curve_to_interpolate_on() -> None:
    grid = np.arange(1, 201) * STEP
    dense, sparse = _half(grid), _half(grid[9::20], 2.0)  # every 20th point of the other half is left
    mean = mean_of_halves(dense, sparse)
    assert np.array_equal(mean.q, grid) and mean.averaged == 10  # a mean only where both have a point
    both = mean.halves == 2
    assert np.array_equal(np.flatnonzero(both), np.arange(9, 200, 20)) and np.allclose(mean.intensity[both], 1.5 * dense[1][both])
    assert np.array_equal(mean.intensity[~both], dense[1][~both]) and np.array_equal(mean.sigma[~both], dense[2][~both])
    # With the scattered half giving the |q|, the dense one fills what lies between its points.
    turned = mean_of_halves(sparse, dense, grid="positive")
    assert np.array_equal(turned.q, grid) and turned.averaged == 10 and np.all(turned.halves[~both] == -1)
    assert np.allclose(turned.intensity, mean.intensity)


def test_without_sigma_on_either_half_the_mean_has_none() -> None:
    grid = np.arange(1, 51) * STEP
    positive, negative = _half(grid), _half(grid)
    assert mean_of_halves(positive[:2] + (None,), negative).sigma is None
    assert mean_of_halves(positive, negative[:2] + (None,)).sigma is None


def test_a_half_of_one_point_is_no_grid_and_no_point_comes_twice() -> None:
    """The beam next to the detector edge: one measured column on one side of it. Nothing can be interpolated
    to or from one point: it is averaged with the other half's point at the same |q| (a mirrored grid), and
    whichever half is asked to give the |q| of the result, no |q| comes twice."""
    grid = np.arange(1, 301) * 0.001
    many = _half(grid, 2.0)
    for at, averaged in ((grid[0], 1), (grid[4], 1), (grid[4] + 0.00035, 0)):  # on a point of the other half; between two
        one = _half(np.array([at]))
        for positive, negative, sign in ((one, many, -1), (many, one, 1)):
            for chosen in (None, "positive", "negative"):
                mean = mean_of_halves(positive, negative, grid=chosen)
                assert np.array_equal(mean.q, grid) and np.all(np.diff(mean.q) > 0), (at, sign, chosen)
                both = mean.halves == 2
                assert mean.averaged == averaged == int(both.sum()) and np.all(mean.halves[~both] == sign)
                assert np.array_equal(mean.intensity[~both], many[1][~both]) and np.array_equal(mean.sigma[~both], many[2][~both])
                assert np.allclose(mean.intensity[both], 0.5 * (many[1][both] + one[1][0]))  # two measurements, once
                assert np.allclose(mean.sigma[both], 0.5 * np.hypot(many[2][both], one[2][0]))
    # Where the other half has no points — a hole, beyond its last point — the one point is kept as it is.
    hole = (grid < 0.1) | (grid > 0.12)
    mean = mean_of_halves(_half(np.array([0.11])), _half(grid[hole]), grid="positive")
    assert mean.q.size == int(hole.sum()) + 1 and mean.averaged == 0 and np.all(np.diff(mean.q) > 0)
    assert mean.halves[mean.q == 0.11].tolist() == [1]
    beyond = mean_of_halves(_half(grid), _half(np.array([0.4])))
    assert beyond.q.size == 301 and beyond.q[-1] == 0.4 and beyond.halves[-1] == -1 and beyond.averaged == 0
    # One point on each side: their mean when they are at the same |q|, else both of them.
    assert mean_of_halves(_half(grid[:1]), _half(grid[:1], 3.0)).intensity == pytest.approx(2.0 * _half(grid[:1])[1])
    apart = mean_of_halves(_half(grid[:1]), _half(grid[5:6]))
    assert apart.q.tolist() == [grid[0], grid[5]] and apart.halves.tolist() == [1, -1]


def test_a_half_without_points_leaves_the_other_half_as_it_is() -> None:
    grid = np.arange(1, 51) * STEP
    measured, nothing = _half(grid), (np.zeros(0), np.zeros(0), np.zeros(0))
    for positive, negative, sign in ((measured, nothing, 1), (nothing, measured, -1)):
        for chosen in (None, "positive", "negative"):
            mean = mean_of_halves(positive, negative, grid=chosen)
            assert np.array_equal(mean.q, grid) and np.array_equal(mean.intensity, measured[1])
            assert np.array_equal(mean.sigma, measured[2]) and np.all(mean.halves == sign) and mean.averaged == 0
    assert mean_of_halves(measured, nothing[:2] + (None,)).sigma is not None  # a half that is not there has no σ to miss
    assert mean_of_halves(nothing, nothing).q.size == 0


# -- merging --------------------------------------------------------------------------------------

def test_more_points_than_a_fit_takes_are_merged_in_groups() -> None:
    q, intensity, sigma = _curve(np.arange(1, 2501) * 0.001)
    x, y, s, group = merge_points(q, intensity, sigma, 1000)
    assert group == 3 and x.size == 833 and np.all(np.diff(x) > 0)
    weights = 1.0 / sigma[:3] ** 2
    assert x[0] == pytest.approx(q[:3].mean()) and y[0] == pytest.approx((weights * intensity[:3]).sum() / weights.sum())
    assert s[0] == pytest.approx(1.0 / np.sqrt(weights.sum()))
    x, y, s, group = merge_points(q, intensity, None, 1000)
    assert s is None and y[0] == pytest.approx(intensity[:3].mean())
    assert merge_points(q[:1000], intensity[:1000], sigma[:1000], 1000)[3] == 1


def test_points_are_not_merged_with_a_sigma_that_cannot_weight_them() -> None:
    """1/σ² of a σ that is 0 or missing made one merged point NaN, and the fit then spoke of q and intensity:
    the curve is refused before, with the sentence the fit has for such a σ."""
    from src.gimap.shared.signed_cut import BAD_SIGMA

    assert BAD_SIGMA == "Every supplied sigma must be finite and positive"
    q, intensity, sigma = _curve(np.arange(1, 2501) * 0.001)
    for value in (0.0, np.nan, -1.0, np.inf):
        bad = sigma.copy()
        bad[1700] = value
        with pytest.raises(ValueError, match=f"^{BAD_SIGMA}$"):
            merge_points(q, intensity, bad, 1000)
        assert merge_points(q[1000:2000], intensity[1000:2000], bad[1000:2000], 1000)[3] == 1  # nothing to merge: as it comes
    # The cut of a fit that takes every point (the quick fit): refused for the points it would use, short or long.
    for points in (2523, 401):
        q, intensity, sigma, _pixels = _cut(points)
        at = int(np.flatnonzero(q > 0)[points // 4])  # a point of the q > 0 half
        for value in (0.0, np.nan):
            bad = sigma.copy()
            bad[at] = value
            for side in ("mean", "positive"):
                with pytest.raises(ValueError, match=f"^{BAD_SIGMA}$"):
                    prepare_cut(q, intensity, bad, None, side, every_point=True)
            other = prepare_cut(q, intensity, bad, None, "negative", every_point=True)  # that point is not among its points
            assert np.all(np.isfinite(other.intensity)) and np.all(other.sigma > 0)
            made_up = prepare_cut(q, intensity, bad, None, "positive")  # the automatic analysis: a relative σ stands in
            assert np.all(np.isfinite(made_up.intensity)) and np.all(made_up.sigma > 0)


# -- the curve of a fit (Å⁻¹) ---------------------------------------------------------------------

def _cut(points=401, low=-0.2, high=0.2):
    q = np.linspace(low, high, points)
    intensity = 100 * np.exp(-np.abs(q) / 0.05) + 5
    return q, intensity, np.sqrt(intensity), np.full(q.shape, 10.0)


def test_the_note_of_a_mean_without_holes_is_written_as_before() -> None:
    q, intensity, sigma, pixels = _cut(451, -0.2, 0.25)  # the q > 0 half reaches further
    x, y, s, note = fit_curve(q, intensity, sigma, pixels, "mean")
    assert note == "mean of both halves up to |qy| = 0.2 Å⁻¹; then the qy > 0 half alone to 0.25 Å⁻¹"
    assert x.size == 250 and np.all(np.diff(x) > 0) and x[0] == pytest.approx(0.001) and x[-1] == pytest.approx(0.25)
    both = x <= 0.2 + 1e-9
    assert np.allclose(y[both], intensity[q > 0][:200]) and np.allclose(s[both], sigma[q > 0][:200] / np.sqrt(2))
    assert np.allclose(s[~both], sigma[q > 0][200:])  # alone: its own σ
    cut = prepare_cut(q, intensity, sigma, pixels, "mean")
    assert (cut.halves_used, cut.points, cut.averaged, cut.single, cut.merged) == ("mean", 250, 200, 50, 1)
    assert cut.note == note


def test_one_half_is_only_that_half_and_never_stands_in_for_the_other() -> None:
    q, intensity, sigma, pixels = _cut()
    x, y, _s, note = fit_curve(q, intensity * np.where(q < 0, 3.0, 1.0), sigma, pixels, "negative")
    assert note == "the qy < 0 half" and x.size == 200 and np.all(np.diff(x) > 0)
    assert np.allclose(y, 3.0 * intensity[q < 0][::-1])
    assert fit_curve(q, intensity, sigma, pixels, "positive")[3] == "the qy > 0 half"
    only_negative = q < 0
    with pytest.raises(MissingHalf, match="^No points on the q > 0 half$"):
        fit_curve(q[only_negative], intensity[only_negative], sigma[only_negative], None, "positive")
    with pytest.raises(MissingHalf, match="^No points on the q < 0 half$"):
        fit_curve(q[~only_negative], intensity[~only_negative], sigma[~only_negative], None, "negative")
    gone = np.where(q > 0, 0.0, pixels)  # the q > 0 half behind a module gap: no pixels
    with pytest.raises(MissingHalf, match="q > 0"):
        fit_curve(q, intensity, sigma, gone, "positive")
    with pytest.raises(ValueError, match="Unknown side"):
        fit_curve(q, intensity, sigma, pixels, "both_abs")


def test_the_mean_of_a_cut_with_one_half_is_that_half_and_says_so() -> None:
    q, intensity, sigma, _pixels = _cut()
    keep = q < 0
    cut = prepare_cut(q[keep], intensity[keep], sigma[keep], None, "mean")
    assert cut.halves_used == "negative" and cut.points == 200 and cut.averaged == 0
    assert cut.note == "the qy < 0 half; the qy > 0 half has no points"
    with pytest.raises(ValueError, match="no measured points"):
        prepare_cut(q[:0], intensity[:0], sigma[:0], None, "mean")


def test_a_hole_in_one_half_is_counted_in_the_note_and_not_filled() -> None:
    q, intensity, sigma, pixels = _cut()
    intensity = intensity * np.where(q < 0, 2.0, 1.0)
    pixels[101:150] = 0.0  # a gap in the q < 0 half: no pixels from |qy| = 0.051 to 0.099
    cut = prepare_cut(q, intensity, sigma, pixels, "mean")
    hole = np.zeros(200, dtype=bool)
    hole[50:99] = True  # the same |qy| on the q > 0 half
    assert cut.points == 200 and np.allclose(cut.q, q[q > 0])
    assert np.array_equal(cut.intensity[hole], intensity[q > 0][hole]) and np.array_equal(cut.sigma[hole], sigma[q > 0][hole])
    assert np.allclose(cut.intensity[~hole], 1.5 * intensity[q > 0][~hole])
    assert cut.averaged == 151 and cut.single == 49
    assert cut.note == ("mean of both halves up to |qy| = 0.2 Å⁻¹; 151 points averaged, 49 from one half alone "
                        "(49 of them inside that range, where the other half has no points)")


def _detector_cut(points: int = 400, step: float = 0.0005, positive: float = 0.35, negative: float = 0.65):
    """A cut as a detector gives it: the beam centre is not on a pixel edge, so the |qy| of the two halves lie
    between each other (here a third of a column apart)."""
    q = np.concatenate([-(np.arange(points) + negative)[::-1] * step, (np.arange(points) + positive) * step])
    intensity = 1000 * np.exp(-np.abs(q) / 0.05) + 5
    return q, intensity, np.sqrt(intensity)


def test_halves_that_start_a_fraction_of_a_step_apart_have_no_hole_to_count_in_the_note() -> None:
    q, intensity, sigma = _detector_cut()
    cut = prepare_cut(q, intensity, sigma, None, "mean")
    # The first point of the q > 0 half lies before the first of the q < 0 half: one measurement, kept as it is —
    # and nothing to say about holes for it (the numbers are there all the same).
    assert cut.note == "mean of both halves up to |qy| = 0.2 Å⁻¹; then the qy < 0 half alone to 0.2 Å⁻¹"
    assert (cut.halves_used, cut.points, cut.averaged, cut.single) == ("mean", 401, 399, 2)
    assert cut.intensity[0] == intensity[q > 0][0] and cut.sigma[0] == sigma[q > 0][0]
    # A half that begins later (a beam stop on its side) is counted: the note's ranges do not say it.
    behind = (q < 0) | (q > 0.015)
    cut = prepare_cut(q[behind], intensity[behind], sigma[behind], None, "mean")
    assert cut.note == ("mean of both halves up to |qy| = 0.2 Å⁻¹; then the qy < 0 half alone to 0.2 Å⁻¹; 370 points averaged, "
                        "31 from one half alone (30 of them inside that range, where the other half has no points)")
    # Halves that share no |qy| at all — each where the other has a gap, or one stray point before the other half
    # begins — are both kept, and "mean" is not left standing alone for points of which none is one.
    apart = (q < -0.1) | ((q > 0) & (q < 0.05))
    cut = prepare_cut(q[apart], intensity[apart], sigma[apart], None, "mean")
    assert (cut.halves_used, cut.points, cut.averaged, cut.single) == ("mean", 300, 0, 300) and np.all(np.diff(cut.q) > 0)
    assert cut.note.endswith("0 points averaged, 300 from one half alone (100 of them inside that range, where the other half has no points)")
    stray = (q > 0.001) | (q == q[q < 0].max())
    cut = prepare_cut(q[stray], intensity[stray], sigma[stray], None, "mean")
    assert (cut.points, cut.averaged) == (int((q > 0.001).sum()) + 1, 0) and "0 points averaged" in cut.note


def test_a_cut_with_one_column_on_one_side_of_the_beam_has_no_point_twice() -> None:
    """One measured column next to the beam on one side, the whole other half: the fit was refused for a
    duplicate |q| (the column counted twice), or the column was lost."""
    grid = np.arange(1, 301) * 0.001
    names = {1: "qy > 0", -1: "qy < 0"}
    for offset in (0.0, 0.00035):  # the column at a |qy| of the other half (the beam on a pixel edge), and between two
        for sign in (1, -1):  # the one column at qy > 0, then at qy < 0
            q = np.concatenate([-sign * grid[::-1], [sign * (grid[0] + offset)]])
            intensity = 1000 * np.exp(-np.abs(q) / 0.05) + 5
            sigma = np.sqrt(intensity)
            for every_point in (True, False):
                cut = prepare_cut(q, intensity, sigma, None, "mean", every_point=every_point)
                assert cut.points == 300 and np.all(np.diff(cut.q) > 0) and np.allclose(cut.q, grid)
                if offset:  # nothing of that column can be averaged: the other half is what there is to fit
                    assert (cut.halves_used, cut.averaged, cut.single) == ("negative" if sign > 0 else "positive", 0, 300)
                    assert cut.note == f"the {names[-sign]} half; the {names[sign]} half is too sparse to average with"
                else:
                    assert (cut.halves_used, cut.averaged, cut.single) == ("mean", 1, 299)
                    assert cut.sigma[0] == pytest.approx(sigma[-1] / np.sqrt(2)) and np.allclose(cut.sigma[1:], sigma[:299][::-1])
                    assert cut.note == f"mean of both halves up to |qy| = 0.001 Å⁻¹; then the {names[-sign]} half alone to 0.3 Å⁻¹"


def test_a_long_cut_is_merged_for_every_choice_of_halves() -> None:
    q, intensity, sigma, _pixels = _cut(2523)  # 1261 points on each half, as on a wide detector
    for side in FIT_HALVES:
        cut = prepare_cut(q, intensity, sigma, None, side)
        assert 8 <= cut.points <= MAX_FIT_POINTS and cut.merged == 2 and np.all(np.diff(cut.q) > 0)
        assert cut.note.endswith(f"2 neighbouring points merged ({cut.points} points for the fit)")
    x, _y, _s, note = fit_curve(q, intensity, sigma, None, "mean", max_points=200)
    assert x.size == 180 and "7 neighbouring points merged (180 points for the fit)" in note


def test_without_sigma_the_prepared_cut_has_none_and_the_curve_of_the_assistant_a_relative_one() -> None:
    q, intensity, _sigma, _pixels = _cut(2523)
    cut = prepare_cut(q, intensity, None, None, "mean")
    assert cut.sigma is None and cut.points <= MAX_FIT_POINTS
    x, y, s, _note = fit_curve(q, intensity, None, None, "positive")
    assert s.shape == y.shape and np.allclose(s, 0.1 * y, rtol=1e-6)
    bad = np.sqrt(intensity)
    bad[::7] = 0.0  # σ missing for some points: the usual relative σ of the others stands in, as before
    _x, y, s, _note = fit_curve(q[:401], intensity[:401], bad[:401], None, "negative")
    assert np.all(np.isfinite(s)) and np.all(s > 0)


# -- the halves the automatic analysis chooses ----------------------------------------------------

def test_the_halves_are_chosen_as_before() -> None:
    q, intensity, _sigma, pixels = _cut()
    choice = choose_halves(q, intensity, pixels)
    assert (choice.side, choice.fit_side) == ("mean", "mean") and "they are averaged where both exist" in choice.reason
    gappy = pixels.copy()
    gappy[q < -0.01] = 0  # the qy < 0 half behind a module gap
    shadowed = intensity.copy()
    shadowed[q < -0.01] = np.nan
    choice = choose_halves(q, shadowed, gappy)
    assert (choice.side, choice.fit_side) == ("positive", "positive") and choice.reason.startswith("The qy < 0 half reaches only")
    differ = choose_halves(q, intensity * np.where(q < 0, 1.5, 1.0), pixels)
    assert differ.side == "both_abs" and differ.fit_side in ("negative", "positive") and "differ by 50 %" in differ.reason
    only = choose_halves(q[q > 0], intensity[q > 0], pixels[q > 0])
    assert only.side == only.fit_side == "positive" and only.reason == "Only the qy > 0 half is on the detector."
    q, intensity, _sigma, pixels = _cut(451, -0.2, 0.25)
    longer = choose_halves(q, intensity, pixels)
    assert longer.side == "mean" and "; beyond 0.2 Å⁻¹ the longer qy > 0 half continues alone to 0.25 Å⁻¹" in longer.reason


def test_the_automatic_analysis_uses_these_functions() -> None:
    from src.gimap.features.assistant import domain
    from src.gimap.features.assistant.domain import gisaxs_cut

    for name in ("HalvesChoice", "MAX_FIT_POINTS", "MIN_COVERAGE", "MIN_REACH", "MISMATCH_LIMIT", "choose_halves", "fit_curve"):
        assert getattr(gisaxs_cut, name) is getattr(signed_cut, name), name
    assert domain.choose_halves is choose_halves and domain.fit_curve is fit_curve
    assert gisaxs_cut.Spacing is domain.Spacing and gisaxs_cut.spacing_from_peaks is domain.spacing_from_peaks


def test_every_sentence_about_the_halves_is_still_written_in_its_translated_form() -> None:
    """The Results tab shows these sentences in the interface language by their form (``guided_words``): a
    form the shared module no longer writes so would silently stay English."""
    from src.gimap.features.assistant.domain.gisaxs_cut import HALVES_SENTENCES
    from src.gimap.features.assistant.presentation.guided_words import SENTENCES

    written = []
    for node in ast.walk(ast.parse(Path(signed_cut.__file__).read_text(encoding="utf-8"))):
        if isinstance(node, ast.JoinedStr):
            written.append("\0".join(part.value if isinstance(part, ast.Constant) else "" for part in node.values))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            written.append(node.value)
    text = "\n".join(written)

    def found(piece: str) -> bool:  # as written, or in two strings the code joins (a sentence inside a sentence)
        return piece in text or any(piece[:index].strip() in text and piece[index:].strip() in text
                                    for index in range(4, len(piece) - 3))

    from src.gimap.features.assistant.domain.gisaxs_cut import HOLES_SENTENCE

    assert len(HALVES_SENTENCES) >= 10 and all(template in SENTENCES for template in HALVES_SENTENCES)
    for template in (*HALVES_SENTENCES, HOLES_SENTENCE):
        for piece in re.split(r"\{\w+\}", template):
            piece = piece.strip(" .;:()")
            if len(piece) >= 12:
                assert found(piece), (template, piece)


def _shown_in_chinese(note: str) -> str:
    """The note as the Results tab of the automatic analysis shows it in the Chinese interface."""
    from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language
    from src.gimap.features.assistant.presentation.guided_words import parts

    apply_language("zh")
    try:
        return parts(note)
    finally:
        apply_language(DEFAULT_LANGUAGE)


def test_the_note_of_a_cut_as_a_detector_gives_it_is_shown_completely_in_chinese() -> None:
    """The halves of a real cut never start at the same |qy|: its note must not carry a part the Results tab
    has no form for — and the fixed sentences about a half are entries of the table."""
    q, intensity, sigma = _detector_cut()
    one_column = np.concatenate([-np.arange(1, 301)[::-1] * 0.001, [0.00135]])
    notes = [
        fit_curve(q, intensity, sigma, None, "mean")[3],
        fit_curve(q[q > 0], intensity[q > 0], sigma[q > 0], None, "mean")[3],
        fit_curve(q[q < 0], intensity[q < 0], sigma[q < 0], None, "mean")[3],
        fit_curve(one_column, np.full(301, 50.0), np.full(301, 7.0), None, "mean")[3],
        fit_curve(-one_column, np.full(301, 50.0), np.full(301, 7.0), None, "mean")[3],
        prepare_cut(*_cut(2523)[:3], None, "negative").note,
    ]
    assert notes[1] == "the qy > 0 half; the qy < 0 half has no points" and "too sparse" in notes[3] and "merged" in notes[5]
    for note in notes:
        shown = _shown_in_chinese(note)
        assert not re.search(r"[A-Za-z]{4,}", shown.replace("qy", "")), (note, shown)
    assert _shown_in_chinese(notes[1]) != _shown_in_chinese(notes[2])


def test_the_count_of_a_hole_has_its_chinese_once_the_results_tab_knows_the_form(monkeypatch) -> None:
    """The part of the note that counts the points next to a hole is a sentence with numbers: the Results tab
    shows it by its form. The form and its Chinese are this package's (``gisaxs_cut.HOLES_SENTENCE``, the table);
    the list of forms the tab knows is ``guided_words.SENTENCES``."""
    from src.gimap.app.presentation.i18n import ZH
    from src.gimap.features.assistant.domain.gisaxs_cut import HOLES_SENTENCE
    from src.gimap.features.assistant.presentation import guided_words

    q, intensity, sigma, pixels = _cut()
    pixels[101:150] = 0.0  # a gap in the q < 0 half
    note = fit_curve(q, intensity, sigma, pixels, "mean")[3]
    assert "151 points averaged, 49 from one half alone (49 of them inside that range" in note
    chinese = ZH[HOLES_SENTENCE]
    assert set(re.findall(r"\{\w+\}", chinese)) == set(re.findall(r"\{\w+\}", HOLES_SENTENCE)) == {"{averaged}", "{single}", "{inside}"}
    if HOLES_SENTENCE not in guided_words.SENTENCES:  # as the tab will know it
        monkeypatch.setattr(guided_words, "SENTENCES", (*guided_words.SENTENCES, HOLES_SENTENCE))
    guided_words._ordered.cache_clear()
    try:
        shown = _shown_in_chinese(note)
    finally:
        monkeypatch.undo()
        guided_words._ordered.cache_clear()
    assert "151" in shown and "49" in shown and not re.search(r"[A-Za-z]{4,}", shown.replace("qy", "")), shown


def test_the_shared_module_needs_numpy_only() -> None:
    imported = set()
    for node in ast.walk(ast.parse(Path(signed_cut.__file__).read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= {"__future__", "dataclasses", "math", "typing", "numpy"}
