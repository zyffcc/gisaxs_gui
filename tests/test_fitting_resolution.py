"""Fitting's Gaussian q resolution σ_q: the particle terms convolved with a Gaussian, the cut taken as symmetric."""

from __future__ import annotations

import numpy as np
import pytest
from dataclasses import replace
from scipy.integrate import quad

from src.gimap.features.fitting.domain import fit_model as fm

Q = np.linspace(0.02, 2.0, 400)


def _model(family: str = "sphere") -> fm.FitModel:
    return fm.FitModel(components=(fm.new_component(family, radius=4.0, distance=13.0),))


def _brute_force(model: fm.FitModel, sigma: float) -> np.ndarray:
    fine = np.linspace(-2.2, 2.2, 44000)
    values = fm.evaluate(model, np.abs(fine))
    weights = np.exp(-0.5 * ((fine[None, :] - Q[:, None]) / sigma) ** 2)
    return (weights @ values) / weights.sum(axis=1)


@pytest.mark.parametrize("family", ["sphere", "vertical_cylinder"])
def test_the_smeared_curve_is_the_convolution_of_the_symmetric_cut(family) -> None:
    model = _model(family)
    smeared = fm.evaluate(model.with_parameter(("globals", "q_resolution"), value=0.02), Q)
    np.testing.assert_allclose(smeared, _brute_force(model, 0.02), rtol=1e-3)


def test_no_resolution_by_default_and_the_parts_add_up() -> None:
    model = _model()
    assert model.globals["q_resolution"].value == 0 and not model.globals["q_resolution"].free
    smeared = model.with_parameter(("globals", "q_resolution"), value=0.01).with_parameter(
        ("globals", "res_amplitude"), value=5.0)
    total, parts = fm.evaluate(smeared, Q, parts=True)
    np.testing.assert_allclose(total, np.sum(list(parts.values()), axis=0), rtol=1e-12)
    assert fm.model_from_dict(fm.model_to_dict(smeared)).globals["q_resolution"].value == 0.01


def test_file_widths_match_independent_convolution_including_a_sharp_structure_peak():
    """One |q| has two measured widths. The reference evaluates the monodisperse sphere analytically."""
    component = fm.new_component("sphere", radius=4., distance=15.)
    model = fm.FitModel((component,)).with_values({(0, "sigma_R"): 0, (0, "sigma_nn"): 0.035,
                                                   ("globals", "res_amplitude"): 3, ("globals", "background"): 2})
    peak = 2 * np.pi / 15.
    q = np.array([0.011, peak, peak, 0.77, 1.1])
    widths = np.array([0.035, 0.012, 0, 0.06, 0.03])

    def particle(x):
        u = 4 * x
        f = 3 * (np.sin(u) - u * np.cos(u)) / u**3 if abs(u) > 1e-3 else 1 - u**2/10
        log_p = -0.5 * (x * 15 * 0.035)**2
        structure = (-np.expm1(2*log_p) / (np.expm1(log_p)**2 + 4*np.exp(log_p)*np.sin(x*15/2)**2)
                     if x else 0.035**2)
        return f**2 * structure

    reference = np.array([
        quad(lambda z: np.exp(-z*z/2) * particle(x + width*z) / np.sqrt(2*np.pi),
             -9., 9., epsabs=1e-12, limit=400)[0] if width else particle(x)
        for x, width in zip(q, widths)])
    model = replace(model.with_parameter(("globals", "q_resolution"), value=0.08, free=True), resolution_mode="file")
    total, terms = fm.evaluate(model, q, dq=widths, parts=True)
    np.testing.assert_allclose(terms["1: Sphere"], reference, rtol=2e-6)
    np.testing.assert_allclose(total, reference + 2 + 3 / (1 + (q / 0.02)**3), rtol=2e-6)
    assert ("globals", "q_resolution") not in dict(model.parameters())
    with pytest.raises(ValueError, match="dQ"):
        fm.evaluate(model, q)


@pytest.mark.parametrize("method", ["local", "global"])
def test_manual_scales_residuals_and_exports_share_file_resolution(method):
    from src.gimap.features.fitting.application import fit_report, single_fit as sf

    curve_q = np.linspace(0.02, 0.9, 45)
    widths = np.linspace(0, 0.075, curve_q.size)
    model = replace(_model(), resolution_mode="file", weighting="poisson")
    for path, parameter in model.parameters():
        if parameter.free and path != (0, "Int"):
            model = model.with_parameter(path, free=False)
    expected = fm.evaluate(model.with_values({(0, "Int"): 15.}), curve_q, dq=widths)
    curve = sf.Curve.from_arrays(curve_q, expected, dq=widths, name="Synthetic", unit="nm")
    data = sf.prepare_curve(curve, weighting=model.weighting, resolution=model.resolution_mode)
    result, _, _ = sf.fit_resolving(model, data, method=method)
    assert result.model.components[0].value("Int") == pytest.approx(15., rel=1e-10)
    options = dict(side="mean", q_range=None, result=result)
    record = sf.fit_record(result.model, data, curve, **options)
    assert record["resolution"]["dq_nm^-1"] == pytest.approx(widths)
    assert record["points"]["weighting_choice"] == "poisson"
    assert fm.model_from_dict(record["model"]) == result.model
    report = fit_report.report(result.model, data, curve, **options)
    assert "Poisson" in report and "dQ from the file" in report and "Gaussian, by number" in report
    columns, table = sf.export_table(result.model, data)
    np.testing.assert_allclose(table[:, columns.index("model")], expected)
    np.testing.assert_allclose(table[:, columns.index("residual")], 0., atol=1e-12)

