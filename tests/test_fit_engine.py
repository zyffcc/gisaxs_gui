"""The single-curve model and fitting engine: the manual model's curve, relative spreads, fits and errors."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.gimap.features.fitting.domain.fit_engine import FitData, FitStopped, fit, residuals
from src.gimap.features.fitting.domain.fit_model import (
    FitModel,
    evaluate,
    from_manual,
    model_from_dict,
    model_to_dict,
    new_component,
    to_manual,
)
from src.gimap.features.fitting.domain.scattering_model import make_mixed_model

Q = np.linspace(0.05, 2.0, 300)


def _truth(family="sphere") -> FitModel:
    model = FitModel((new_component(family, radius=6.0, distance=40.0),))
    values = {(0, "Int"): 1000.0, (0, "sigma_R"): 0.15, (0, "sigma_D"): 0.3,
              ("globals", "background"): 2.0, ("globals", "res_amplitude"): 300.0,
              ("globals", "res_width"): 0.03, ("globals", "res_exponent"): 4.0}
    if family == "cylinder":
        values.update({(0, "h"): 15.0, (0, "sigma_h"): 0.2})
    return model.with_values(values)


def _manual_curve(model: FitModel, q) -> np.ndarray:
    mapping = to_manual(model)
    order = {"Sphere": ("intensity", "radius", "sigma_radius", "diameter", "sigma_diameter"),
             "Cylinder": ("intensity", "radius", "sigma_radius", "height", "sigma_height", "diameter", "sigma_diameter"),
             "Vertical Cylinder": ("intensity", "radius", "sigma_radius", "diameter", "sigma_diameter")}
    params = []
    for component in mapping.components:
        params += [component.parameters[key] for key in order[component.shape]]
    g = mapping.global_parameters
    params += [g["background"], g["sigma_res"], g["nu_res"], g["int_res"], g["k_value"]]
    spec = [component.shape.lower().replace(" ", "_") for component in mapping.components]
    return make_mixed_model(spec)(q, *params)


@pytest.mark.parametrize("family", ["sphere", "cylinder", "vertical_cylinder"])
def test_the_model_is_the_manual_model_with_relative_spreads(family) -> None:
    model = _truth(family)
    np.testing.assert_allclose(evaluate(model, Q), _manual_curve(model, Q), rtol=1e-12)
    back = from_manual(to_manual(model))  # spreads: relative → nm → relative
    np.testing.assert_allclose(evaluate(back, Q), evaluate(model, Q), rtol=1e-12)
    assert back.components[0].value("sigma_R") == pytest.approx(0.15)


def test_a_model_is_saved_and_read_back() -> None:
    model = _truth("cylinder").with_parameter((0, "D"), free=False, lower=10.0, upper=90.0)
    again = model_from_dict(json.loads(json.dumps(model_to_dict(model))))
    assert again.get((0, "D")) == model.get((0, "D"))
    np.testing.assert_allclose(evaluate(again, Q), evaluate(model, Q))


def _noisy(model: FitModel, seed: int = 4) -> FitData:
    exact = evaluate(model, Q)
    sigma = np.sqrt(exact) + 0.5
    return FitData.prepare(Q, exact + np.random.default_rng(seed).normal(0.0, sigma), sigma)


def test_a_local_fit_finds_the_parameters_with_errors_that_cover_them() -> None:
    truth = _truth()
    data = _noisy(truth)
    start = truth.with_values({(0, "R"): 5.0, (0, "D"): 46.0, (0, "sigma_R"): 0.25, (0, "Int"): 1.0})
    result = fit(start, data, method="local")
    assert result.converged and result.weighting == "sigma"
    assert result.chi2_reduced == pytest.approx(1.0, abs=0.25)  # noise at the level of σ
    for key in ("R", "D", "sigma_R"):
        value, error = result.model.get((0, key)).value, result.errors[(0, key)]
        assert 0 < error < 0.2 * truth.get((0, key)).value
        assert abs(value - truth.get((0, key)).value) < 4 * error, (key, value, error)
    assert result.model.get((0, "Int")).value == pytest.approx(1000.0, rel=0.05)  # solved, not searched


def test_a_huge_parameter_does_not_wipe_out_the_errors_of_the_others() -> None:
    # In a series the resolution peak can run off to a vanishing width with an amplitude of 1e28 or more:
    # it must not set the finite-difference step (or the cut-off of the inverse) of R.
    from src.gimap.features.fitting.domain.fit_engine import _errors, _Problem

    truth = _truth().with_values({("globals", "res_amplitude"): 0.0})
    data = _noisy(truth)
    runaway = truth.with_values({("globals", "res_amplitude"): 1e28, ("globals", "res_width"): 0.0021,
                                 ("globals", "res_exponent"): 20.0})
    reference, _ = _errors(truth, data, _Problem(truth, data, None, None, 1))
    errors, _ = _errors(runaway, data, _Problem(runaway, data, None, None, 1))
    for key in ("R", "sigma_R", "Int"):
        assert reference[(0, key)] / 3 < errors[(0, key)] < 3 * reference[(0, key)], key


def test_a_wide_search_escapes_a_bad_start() -> None:
    truth = _truth()
    data = _noisy(truth, seed=9)
    start = truth.with_values({(0, "R"): 1.5, (0, "D"): 300.0})
    local = fit(start, data, method="local")
    wide = fit(start, data, method="global", max_evaluations=1500)
    assert wide.chi2_reduced < 1.5 and wide.chi2_reduced <= local.chi2_reduced
    assert wide.model.get((0, "R")).value == pytest.approx(6.0, rel=0.05)


def test_without_sigma_the_residuals_are_relative() -> None:
    truth = _truth()
    data = FitData.prepare(Q, evaluate(truth, Q))
    assert data.weighting == "relative"
    result = fit(truth.with_values({(0, "R"): 5.5}), data)
    assert result.log_rmse < 1e-4 and np.max(np.abs(residuals(result.model, data))) < 1e-3


def test_fixed_parameters_stay_and_a_stop_keeps_the_best_so_far() -> None:
    truth = _truth()
    data = _noisy(truth)
    start = truth.with_values({(0, "R"): 5.0}).with_parameter((0, "D"), free=False, value=41.0)
    result = fit(start, data)
    assert result.model.get((0, "D")).value == 41.0 and (0, "D") not in result.errors
    calls = []

    def stop():
        calls.append(1)
        return len(calls) > 5

    stopped = fit(start, data, method="global", stop=stop)
    assert stopped.stopped and not stopped.converged and "Stopped" in stopped.message
    with pytest.raises(ValueError, match="finite range"):
        fit(start.with_parameter((0, "R"), upper=math.inf), data)
    assert FitStopped.__name__ == "FitStopped"


# -- the points of a signed cut: one preparation for every fit (shared/signed_cut.py) ---------------

def _cut_curve(points: int = 240, *, hole=None, only=None, negative=1.0, shift=1.0):
    """A cut through the beam as Fitting loads it (q in nm⁻¹): the q < 0 half ``negative`` times as intense;
    ``hole``: a |q| range the q > 0 half has no points in; ``only``: the one half the cut holds."""
    from src.gimap.features.fitting.application.single_fit import Curve

    grid = np.linspace(0.08, 2.0, points)
    exact = evaluate(_truth(), grid)
    there = np.ones(points, dtype=bool) if hole is None else (grid < hole[0]) | (grid > hole[1])
    halves = {"negative": (-grid[::-1], negative * exact[::-1], np.sqrt(exact[::-1])),
              "positive": (grid[there], exact[there], np.sqrt(exact[there]))}
    q, intensity, sigma = (np.concatenate(parts) for parts in zip(*(halves[name] for name in halves if only in (None, name))))
    return Curve.from_arrays(q * shift, intensity, sigma, name="cut.dat", unit="nm"), grid, exact, there


def test_the_mean_of_a_cut_leaves_a_hole_of_one_half_to_the_other_half() -> None:
    from src.gimap.features.fitting.application.single_fit import prepare_curve, prepare_points

    curve, grid, exact, there = _cut_curve(hole=(0.4, 0.9), negative=1.4)
    prepared = prepare_points(curve, "mean")
    data = prepared.data
    assert np.allclose(data.q, grid) and prepared.halves_used == "mean"
    # In the hole the half that was measured stands alone, with its own σ: nothing is interpolated across it.
    assert np.allclose(data.intensity[~there], 1.4 * exact[~there]) and np.allclose(data.sigma[~there], np.sqrt(exact[~there]))
    assert np.allclose(data.intensity[there], 1.2 * exact[there])
    assert np.allclose(data.sigma[there], 0.5 * np.hypot(np.sqrt(exact[there]), np.sqrt(exact[there])))
    assert prepared.averaged == int(there.sum()) and prepared.points == 240 and prepared.left_out == 0
    assert np.array_equal(prepare_curve(curve, "mean").intensity, data.intensity)
    # Inside the hole only one half is fitted, and the account of the points says so.
    inside = prepare_points(curve, "mean", (0.5, 0.8))
    assert inside.halves_used == "negative" and inside.averaged == 0 and inside.points == int(((grid >= 0.5) & (grid <= 0.8)).sum())


def test_both_halves_on_q_and_one_half_are_the_points_of_the_file() -> None:
    from src.gimap.features.fitting.application.single_fit import SERIES_SIDES, SIDES, prepare_points

    curve, grid, exact, _there = _cut_curve(negative=1.4)
    both = prepare_points(curve, "both")
    assert both.points == 480 and both.halves_used == "both" and both.averaged == 0  # as Refine and Search fit them
    assert np.allclose(np.unique(both.data.q), grid)
    positive, negative = prepare_points(curve, "positive"), prepare_points(curve, "negative", strict=True)
    assert np.allclose(positive.data.intensity, exact) and np.allclose(negative.data.intensity, 1.4 * exact)
    assert (positive.halves_used, negative.halves_used) == ("positive", "negative")
    assert [key for key, _text in SIDES] == ["mean", "both", "positive", "negative"]
    assert SERIES_SIDES == (("mean", "Mean of both halves"), ("positive", "q > 0 half"), ("negative", "q < 0 half"))


def test_a_chosen_half_without_points_is_missing_and_never_the_other_half() -> None:
    from src.gimap.features.fitting.application.single_fit import MissingHalf, prepare_curve, prepare_points

    curve, grid, exact, _there = _cut_curve(only="negative")
    with pytest.raises(MissingHalf, match="^No points on the q > 0 half$"):
        prepare_points(curve, "positive", strict=True)
    with pytest.raises(ValueError, match="^No points on the q > 0 half$"):
        prepare_curve(curve, "positive", (0.2, 1.5), strict=True)
    with pytest.raises(MissingHalf, match="^No points on the q < 0 half$"):
        prepare_points(_cut_curve(only="positive")[0], "negative", strict=True)
    assert prepare_points(curve, "negative", strict=True).points == 240
    mean = prepare_points(curve, "mean", strict=True)  # the mean of a cut with one half: that half, named
    assert mean.points == 240 and mean.halves_used == "negative" and mean.averaged == 0
    # Single analysis shows no choice of halves for a curve with one half: whatever is left of a choice
    # made for another curve, the half there is is fitted — and named.
    single = prepare_points(curve, "positive")
    assert single.points == 240 and single.halves_used == "negative"
    assert prepare_points(curve, "both").halves_used == "negative"


def test_left_out_points_are_left_out_in_a_frame_whose_q_differs_in_the_last_digits() -> None:
    from src.gimap.features.fitting.application.single_fit import left_out_points, point_key, prepare_points

    first, grid, _exact, _there = _cut_curve()
    names = frozenset(point_key(value) for value in grid[100:110])
    assert prepare_points(first, "positive", None, names).points == 230
    other = _cut_curve(shift=1.0 + 2e-7)[0]  # the next frame of the series: the same columns, q off in the 7th digit
    assert not np.isin(np.array([point_key(value) for value in other.q[other.q > 0]]), np.array(sorted(names))).any()
    for side in ("positive", "negative", "mean"):
        prepared = prepare_points(other, side, None, names)
        assert prepared.points == 230 and prepared.left_out == 10, side
        assert not ((prepared.data.q > grid[99] * 1.000001) & (prepared.data.q < grid[110] * 0.999999)).any()
    both = prepare_points(other, "both", None, names)
    assert both.points == 460 and both.left_out == 20  # both measurements at each of those |q|
    in_range = prepare_points(other, "positive", (grid[105] * 0.9999, 3.0), names)
    assert in_range.left_out == 5 and in_range.points == 240 - 105 - 5
    # A name where the frame has no point (a hole) means no point; nor does one more than half a step away.
    holed = _cut_curve(hole=(grid[95], grid[115]))[0]
    assert prepare_points(holed, "positive", None, names).left_out == 0
    step = grid[1] - grid[0]
    assert left_out_points(grid, {point_key(grid[50] + 0.4 * step)}).nonzero()[0].tolist() == [50]
    assert not left_out_points(grid, {point_key(grid[0] - 0.6 * step), point_key(grid[-1] + 0.6 * step)}).any()
    assert not left_out_points(grid, ()).any() and not left_out_points(grid[:0], names).any()


def _counted_cut(points: int, seed: int = 4):
    """A cut through the beam as Analyze hands it over (q in Å⁻¹, counts with Poisson noise, every column an
    observation — zero counts too), made with the app's own model; ``points`` on each half."""
    q_nm = np.linspace(0.06, 2.4, points)
    exact = evaluate(_truth(), q_nm) * 5.0
    rng = np.random.default_rng(seed)
    left, right = rng.poisson(exact[::-1]).astype(float), rng.poisson(exact).astype(float)
    counts = np.concatenate([left, right])
    return np.concatenate([-q_nm[::-1], q_nm]) / 10.0, counts, np.sqrt(np.maximum(counts, 1.0))


def test_the_quick_fit_takes_a_cut_of_any_length_on_the_mean_or_one_half() -> None:
    """The real fit: a cut of 2500 points (1250 on each half, as on a wide detector) and a one-sided curve of
    1400 — both used to be refused for their length, the cut also for holding both halves."""
    from src.gimap.features.fitting.bootstrap import create_quick_fit

    quick_fit = create_quick_fit()
    q, counts, sigma = _counted_cut(1250)
    made_of = {}
    for halves in ("mean", "positive", "negative"):
        rows = quick_fit(q, counts, sigma, components=(1,), halves=halves)
        best = rows[0]
        made_of[halves] = best["preparation"]
        assert best["components"][0]["params"]["R"] == pytest.approx(6.0, rel=0.05), halves
        assert best["components"][0]["params"]["D"] == pytest.approx(40.0, rel=0.1), halves
        assert best["best_chi2_weighted"] < 2.0 and len(best["native_q"]) == 625 and min(best["native_q"]) > 0
        assert best["prepared"].endswith("2 neighbouring points merged (625 points for the fit)")
        assert all(row["prepared"] == best["prepared"] for row in rows)
    assert made_of["mean"] == {"halves": "mean", "points": 625, "averaged": 1250, "one_half": 0, "merged": 2}
    assert made_of["positive"] == {"halves": "positive", "points": 625, "averaged": 0, "one_half": 1250, "merged": 2}
    assert made_of["negative"]["halves"] == "negative"
    q, counts, sigma = _counted_cut(1400)
    right = q > 0
    best = quick_fit(q[right], counts[right], sigma[right], components=(1,))[0]  # 1400 points on |q|
    assert best["components"][0]["params"]["R"] == pytest.approx(6.0, rel=0.05) and best["best_chi2_weighted"] < 2.0
    assert best["prepared"] == "2 neighbouring points merged (700 points for the fit)" and len(best["native_q"]) == 700
    assert best["preparation"] == {"halves": "", "points": 700, "averaged": 0, "one_half": 1400, "merged": 2}


def test_the_quick_fit_prepares_the_curve_it_is_given(monkeypatch) -> None:
    from src.gimap.features.fitting.bootstrap import create_quick_fit
    from src.gimap.features.fitting.infrastructure.adapters import experimental_fit
    from src.gimap.shared.signed_cut import MissingHalf

    def fitted(item, _options, _report, _cancelled, _evaluated=None):  # the points the fit would be made on
        return [{"native_q": item["q"], "observed": item["observed"], "sigma": item["sigma"],
                 "sigma_estimated": item["sigma_estimated"]}]

    monkeypatch.setattr(experimental_fit, "fit_candidates", fitted)
    quick_fit = create_quick_fit()
    q, counts, sigma = _counted_cut(250)
    right = q > 0
    # Without a word on the halves, and with the choices that are for looking, a signed cut is fitted on its mean.
    for halves in (None, "both", "both_abs"):
        row = quick_fit(q, counts, sigma, halves=halves)[0]
        assert row["preparation"] == {"halves": "mean", "points": 250, "averaged": 250, "one_half": 0, "merged": 1}
        assert row["prepared"] == "mean of both halves up to |qy| = 0.24 Å⁻¹"
        assert np.allclose(row["native_q"], 10.0 * q[right]) and np.allclose(row["observed"], 0.5 * (counts[right] + counts[~right][::-1]))
    empty = counts.copy()
    empty[[100, 400]] = 0.0  # a column without a count is an observation: it is neither dropped nor a hole
    assert quick_fit(q, empty, sigma, halves="positive")[0]["observed"][150] == 0.0
    row = quick_fit(q, empty, sigma, halves="mean")[0]
    assert row["preparation"]["averaged"] == 250 and row["observed"][150] == 0.5 * counts[99]
    # A half that is not there is missing: the other one is not fitted in its place.
    with pytest.raises(MissingHalf, match="^No points on the q < 0 half$"):
        quick_fit(q[right], counts[right], sigma[right], halves="negative")
    with pytest.raises(MissingHalf, match="^No points on the q > 0 half$"):
        quick_fit(q[~right], counts[~right], sigma[~right], halves="positive")
    alone = quick_fit(q[right], counts[right], sigma[right], halves="mean")[0]
    assert alone["preparation"]["halves"] == "positive" and alone["prepared"] == "the qy > 0 half; the qy < 0 half has no points"
    # A one-sided curve given without a word on the halves is fitted as it comes (the q < 0 half on |q|).
    for half in (right, ~right):
        row = quick_fit(q[half], counts[half], sigma[half])[0]
        assert row["prepared"] == "" and row["preparation"] == {"halves": "", "points": 250, "averaged": 0, "one_half": 250, "merged": 1}
        assert np.allclose(row["native_q"], 10.0 * q[right]) and np.allclose(np.sort(row["observed"]), np.sort(counts[half]))
    # No σ: the fit estimates it as before; a σ that is missing for a point is not made up.
    assert quick_fit(q, counts, None)[0]["sigma_estimated"] and not quick_fit(q, counts, sigma)[0]["sigma_estimated"]
    broken = sigma.copy()
    broken[300] = 0.0
    with pytest.raises(ValueError, match="sigma must be finite and positive"):
        quick_fit(q, counts, broken, halves="positive")


def _prepared_only(monkeypatch):
    """The quick fit with the points it would be fitted on in place of the fit."""
    from src.gimap.features.fitting.bootstrap import create_quick_fit
    from src.gimap.features.fitting.infrastructure.adapters import experimental_fit

    def fitted(item, _options, _report, _cancelled, _evaluated=None):
        return [{"native_q": item["q"], "observed": item["observed"], "sigma": item["sigma"]}]

    monkeypatch.setattr(experimental_fit, "fit_candidates", fitted)
    return create_quick_fit()


def test_the_quick_fit_gives_the_same_reason_for_a_bad_sigma_whatever_the_length_of_the_cut(monkeypatch) -> None:
    """More than 1000 points on a half are merged first: a σ of 0 (or none) for one of them made a merged point
    NaN and the fit said “q and intensity must … contain finite values” — the wrong cause."""
    quick_fit = _prepared_only(monkeypatch)
    for points in (400, 1250):
        q, counts, sigma = _counted_cut(points)
        at = int(np.flatnonzero(q > 0)[points // 3])
        for value in (0.0, np.nan):
            broken = sigma.copy()
            broken[at] = value
            for halves in ("mean", "positive", None):
                with pytest.raises(ValueError, match="^Every supplied sigma must be finite and positive$"):
                    quick_fit(q, counts, broken, halves=halves)
            row = quick_fit(q, counts, broken, halves="negative")[0]  # the q < 0 half does not hold that point
            assert np.all(np.isfinite(row["observed"])) and np.all(np.asarray(row["sigma"]) > 0)
            right = q > 0  # … and a curve with one side, taken as it comes
            with pytest.raises(ValueError, match="^Every supplied sigma must be finite and positive$"):
                quick_fit(q[right], counts[right], broken[right])


def test_the_quick_fit_takes_a_cut_with_one_column_on_one_side_of_the_beam(monkeypatch) -> None:
    """The beam centre on a pixel edge (mirrored |q|) and one measured column on one side: that column was
    counted twice and the fit refused for “duplicate q values”."""
    quick_fit = _prepared_only(monkeypatch)
    q, counts, sigma = _counted_cut(400)
    for sign in (1, -1):
        column = q[q * sign > 0][np.argmin(np.abs(q[q * sign > 0]))]
        keep = (q * sign < 0) | (q == column)
        for halves in ("mean", None):  # None: what a caller gives that has no word on the halves
            row = quick_fit(q[keep], counts[keep], sigma[keep], halves=halves)[0]
            assert len(row["native_q"]) == 400 and np.all(np.diff(row["native_q"]) > 0)
            assert row["preparation"] == {"halves": "mean", "points": 400, "averaged": 1, "one_half": 399, "merged": 1}
            both = counts[q == column][0], counts[q == -column][0]
            assert row["observed"][0] == pytest.approx(0.5 * sum(both))
        # … and off the columns of the other side (the beam elsewhere in the pixel): the other half is fitted.
        moved = np.where(q == column, q * 1.35, q)
        row = quick_fit(moved[keep], counts[keep], sigma[keep], halves="mean")[0]
        assert row["preparation"]["halves"] == ("negative" if sign > 0 else "positive") and len(row["native_q"]) == 400
        assert row["prepared"].endswith("half is too sparse to average with") and np.all(np.diff(row["native_q"]) > 0)


def test_the_quick_fit_names_the_half_that_has_too_few_points(monkeypatch) -> None:
    """Three points on the chosen q < 0 half used to be refused as “positive: need 8–1000 measured points”."""
    from src.gimap.app.presentation.i18n import ZH
    from src.gimap.features.fitting.application.single_fit import QUICK_FIT_MIN_POINTS, TOO_FEW_FOR_QUICK_FIT
    from src.gimap.features.fitting.application.workflow_v5 import default_options
    from src.gimap.features.fitting.infrastructure.adapters.workflow_v5 import prepare_sides

    quick_fit = _prepared_only(monkeypatch)
    q, counts, sigma = _counted_cut(300)
    for sign, name in ((-1, "q < 0"), (1, "q > 0")):
        few = np.sort(np.abs(q[q * sign > 0]))[2]
        keep = (q * sign < 0) | (np.abs(q) <= few)  # three points on that half, the whole other half
        halves = "negative" if sign < 0 else "positive"
        with pytest.raises(ValueError, match=f"^Only 3 points on the {name} half: this fit needs at least 8$"):
            quick_fit(q[keep], counts[keep], sigma[keep], halves=halves)
        assert len(quick_fit(q[keep], counts[keep], sigma[keep], halves="mean")[0]["native_q"]) == 300
    right = (q > 0) & (q < np.sort(q[q > 0])[5])
    with pytest.raises(ValueError, match="^Only 5 points in the curve: this fit needs at least 8$"):
        quick_fit(q[right], counts[right], sigma[right])
    small = np.abs(q) < np.sort(q[q > 0])[4]
    with pytest.raises(ValueError, match="^Only 4 points in the mean of both halves: this fit needs at least 8$"):
        quick_fit(q[small], counts[small], sigma[small])
    assert set(TOO_FEW_FOR_QUICK_FIT) == {"positive", "negative", "mean", ""}
    for template in TOO_FEW_FOR_QUICK_FIT.values():  # English here; a page shows it with trf
        assert "{points}" in ZH[template] and "{minimum}" in ZH[template] and ZH[template] != template
    # The least the fit itself takes is this number; what else it refuses is said without the name it has for
    # the curve it was handed ("positive: …" — the curve is on |q|, whichever half it is).
    options = {**default_options(), "q_unit": "A^-1", "side": "positive"}
    x = np.linspace(0.01, 0.1, QUICK_FIT_MIN_POINTS)
    assert len(prepare_sides(x, np.ones(x.size), np.ones(x.size), options)[0]["q"]) == QUICK_FIT_MIN_POINTS
    with pytest.raises(ValueError, match="need 8"):
        prepare_sides(x[1:], np.ones(x.size - 1), np.ones(x.size - 1), options)
    twice = np.concatenate([q[q > 0], q[q > 0][:3]])
    with pytest.raises(ValueError, match="^duplicate q values"):
        quick_fit(twice, np.ones(twice.size), np.ones(twice.size))


def test_the_quick_fit_counts_the_points_it_fits(monkeypatch) -> None:
    """A point at q = 0 is on neither side of the beam: the fit does not take it, and it is not counted."""
    quick_fit = _prepared_only(monkeypatch)
    q, counts, sigma = _counted_cut(300)
    right = q > 0
    row = quick_fit(np.concatenate([[0.0], q[right]]), np.concatenate([[9.0], counts[right]]),
                    np.concatenate([[3.0], sigma[right]]))[0]
    assert len(row["native_q"]) == 300 and row["preparation"]["points"] == 300 and row["preparation"]["one_half"] == 300
