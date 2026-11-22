"""Exit angles belong to the substrate selection; shared fitted values cost degrees of freedom only once."""

from dataclasses import replace

import numpy as np
import pytest

from src.gimap.features.fitting.application import joint_fit as joint
from src.gimap.features.fitting.application.single_fit import FitModel, evaluate, new_component


@pytest.mark.parametrize("film_ac", [0.0, 0.09])
def test_cutoff_uses_substrate_and_keeps_aligned_points_at_the_limit(film_ac):
    substrate = joint.substrate_of("custom", np.radians(0.2)**2 / 2, 1e-8)
    model = FitModel((new_component("sphere"),))
    angles = np.array([0.7, 0.3, 0.39999, 0.4, 0.8, 1.1])
    sin_c2 = np.sin(np.radians(film_ac))**2
    q = 2 * np.pi / 0.1 * (np.sqrt(np.sin(np.radians(0.5))**2 - sin_c2)
                             + np.sqrt(np.sin(np.radians(angles))**2 - sin_c2))
    q[-1] *= -1  # a signed q_z below the horizon must not fold onto the vertical cut
    y = np.arange(1, 7, dtype=float)
    spec = joint.CurveSpec("vertical", q, y, y + 10, "vertical", alpha_i=0.5, wavelength=0.1,
                           q_parallel=0.3, critical_angle=film_ac)
    made = joint.curve_model(spec, model, substrate, False)
    data = joint.curve_data(spec, made)
    assert made.cut.min_exit_angle_deg == pytest.approx(0.4)
    assert np.array_equal(data.q, q[[3, 0, 4]])
    assert np.array_equal(data.intensity, y[[3, 0, 4]]) and np.array_equal(data.sigma, data.intensity + 10)
    assert np.array_equal(joint.plot_points(spec, made)[0], data.q)
    full = joint.curve_model(spec, model, substrate, False, min_exit_angle_ac=0.0)
    assert joint.curve_data(spec, full).q.size == 5
    masked = replace(spec, sigma=np.where(angles < 0.4, 1.0, np.nan))
    with pytest.raises(ValueError, match="Lower the limit"):
        joint.curve_data(masked, made)
    horizontal = replace(spec, kind="horizontal")
    assert joint.joint_curves([horizontal], model, substrate, min_exit_angle_ac=100)[0].data.q.size == len(q)


def test_vertical_chi2_warning_uses_vertical_degrees_of_freedom():
    base = FitModel((new_component("sphere"),))
    for path, _ in base.parameters():
        base = base.with_parameter(path, free=path == ("globals", "background"))
    substrate = joint.substrate_of("Si")
    qh, qv = np.linspace(0.05, 2.0, 200), np.linspace(1.0, 1.8, 20)
    specs = [joint.CurveSpec("horizontal", qh, qh, np.ones_like(qh)),
             joint.CurveSpec("vertical", qv, qv, np.ones_like(qv), "vertical", alpha_i=0.4, q_parallel=0.3)]
    specs = [replace(spec, intensity=10 + evaluate(joint.curve_model(spec, base, substrate, True), spec.q))
             for spec in specs]
    specs[1] = replace(specs[1], intensity=specs[1].intensity + np.tile([1.7, -1.7], 10))
    result = joint.run_joint_fit(specs, base, substrate, set())
    assert result.curve_dof == pytest.approx((199, 19))
    assert result.chi2_reduced < 3 and result.curve_chi2[1] < 3 < result.reduced_chi2_for(1)
    record = joint.joint_record(result, specs, substrate)
    assert record["curves"][1]["chi2_reduced"] == pytest.approx(20 * 1.7**2 / 19)
    assert "13–60%" in record["needs_attention"][0] and "6–20%" in record["needs_attention"][0]
    on_limit = replace(result, curve_dof=(199, 20), curve_chi2=(100, 3))
    assert joint.needs_attention(on_limit, specs) == []
    log_fit = replace(result, weightings=("sigma", "relative"))
    assert joint.needs_attention(log_fit, specs) == []
    assert joint.HEIGHT_ASSUMPTION in record["model_assumptions"]


def test_shared_radius_costs_one_degree_of_freedom_across_both_curves():
    base = FitModel((new_component("sphere", structure=False),))
    for path, _ in base.parameters():
        base = base.with_parameter(path, free=path == (0, "R"))
    q = np.linspace(0.05, 2.0, 40)
    y = evaluate(base, q)
    # Identical q grids; the first curve has twice the precision, hence 4/5 of the leverage.
    result = joint.run_joint_fit([joint.CurveSpec("a", q, y, np.ones_like(q)),
                                  joint.CurveSpec("b", q, y, 2 * np.ones_like(q))], base, None, {(0, "R")})
    assert result.curve_dof == pytest.approx((40 - 0.8, 40 - 0.2))


def test_changed_height_below_cutoff_warns_even_with_small_vertical_chi2():
    # Deliberately give the low-angle part a different height, with the same high-angle curve in both fits.
    base = FitModel((new_component("vertical_cylinder", heights=True, structure=False),))
    base = base.with_values({(0, "h"): 6., (0, "Int"): 1e3})
    for path, _ in base.parameters():
        base = base.with_parameter(path, free=path == (0, "h"))
    substrate = joint.substrate_of("custom", np.radians(0.2)**2 / 2, 1e-8)
    angles = np.linspace(.05, 1.2, 96)
    q = (2 * np.pi / .1) * (np.sin(np.radians(.4)) + np.sin(np.radians(angles)))
    spec = joint.CurveSpec("vertical", q, np.ones(q.size), np.ones(q.size), "vertical", alpha_i=.4, q_parallel=.3)
    made = joint.curve_model(spec, base, substrate, True)
    y = np.where(angles < .4, evaluate(made.with_values({(0, "h"): 5.}), q), evaluate(made, q))
    spec = replace(spec, intensity=y)
    primary = joint.run_joint_fit([spec], base, substrate, set())
    full = joint.run_joint_fit([spec], base, substrate, set(), min_exit_angle_ac=0.)
    row, = joint.height_range_rows(primary)
    assert primary.reduced_chi2_for(0) < 3 and row["warn"]
    assert row["selected_nm"] == pytest.approx(6.)
    assert row["alternate_nm"] == pytest.approx(full.models[0].get((0, "h")).value)
    assert joint.height_range_rows(full)[0]["alternate_nm"] == pytest.approx(row["selected_nm"])
    record = joint.joint_record(primary, [spec], substrate)
    assert record["height_range_check"]["alternate_points_per_curve"] == [q.size]
    assert "h_1 depends on the vertical fit range" in record["needs_attention"][0]

    # The 2% test is still available with a missing covariance; a shift below 2% can exceed 2σ.
    for change, sigma, warn in ((0.11, np.nan, False), (0.13, np.nan, True), (0.06, .01, True), (0.06, .1, False)):
        check = replace(primary.height_range_check, estimates={(0, "h"): (6. - change, sigma)})
        trial = replace(primary, errors={(joint.SHARED, (0, "h")): sigma}, height_range_check=check)
        assert joint.height_range_rows(trial)[0]["warn"] is warn

    above = angles >= .4
    cropped = replace(spec, q=q[above], intensity=y[above], sigma=np.ones(above.sum()))
    same = joint.run_joint_fit([cropped], base, substrate, set())
    assert same.height_range_check.unavailable == joint.SAME_POINTS
