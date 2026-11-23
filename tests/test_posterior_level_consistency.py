"""The forward curve, fitted scales and physical support used by Predict at each substrate level."""

from dataclasses import replace

import numpy as np
import pytest

from src.gimap.features.fitting.application import single_predict, substrate_level
from src.gimap.features.fitting.domain import fit_engine, fit_model as fm, posterior_check
from src.gimap.features.fitting.domain.cut_model.geometry import Geometry
from src.gimap.features.fitting.domain.posterior import encoding as E, infer, refine, reflection_disk
from src.gimap.features.fitting.domain.substrates import Substrate


def _cut(level):
    q = np.linspace(0.021, 1.4, 87)
    angles = dict(alpha_i=0.37, alpha_f=0.19, wavelength=0.11)
    born = Geometry("born", angles["wavelength"], angles["alpha_i"], angles["alpha_f"])
    options = [{}, {"qz": born.qz_born}, {"qz": born.qz_born, "qz_minus": born.k_i - born.k_f},
               {"grazing": angles, "substrate": Substrate("Si")}]
    return infer.Cut.prepare(q, np.ones_like(q), np.ones_like(q), **options[level - 1]), angles


def _values():
    v = np.full(E.DIMS, np.nan)
    v[:6] = 4.7, 0.16, 14.0, 0.14, 16.1, 0.10
    v[E.GLOBAL] = 0.014, 1.97, 0.012, 0.031, -0.67, 0.34
    return v


@pytest.mark.parametrize("level", [1, 2, 3, 4])
@pytest.mark.parametrize("family", E.FAMILIES)
def test_likelihood_and_refinement_use_same_resolved_curve(level, family):
    data, angles = _cut(level)
    values = _values()
    index = E.composition_index([(family, True)])
    like = infer.Likelihood(data, index)
    # The width is near the upper end of the prior; convolution materially changes the correlation peak.
    x = like.columns(values)[0]
    sharp = values.copy()
    sharp[E.GLOBAL.start + 2] = 0.0002
    assert np.max(np.abs(like.columns(sharp)[0, 0] - x[0])) > 0.001
    second = values.copy()
    second[E.GLOBAL.start + 2] *= 0.7
    np.testing.assert_allclose(like.columns([second, values])[1], x, rtol=1e-12)
    data = replace(data, intensity=1120 * x[0] + 47 * x[1] + 3)
    model = refine.fit_model(values, index, data, grazing=angles)
    for candidate in (model, fm.model_from_dict(fm.model_to_dict(model))):
        geometry = fm.grazing(candidate) or Geometry()
        assert geometry.mode == ("eia", "born", "two_path", "dwba")[level - 1]
        if family == "cylinder":
            assert geometry.random_cylinder_qz() == pytest.approx(data.qz if level in (2, 4) else 0)
        forms = fm.particle_forms(candidate, data.q, None)
        np.testing.assert_allclose(forms[0] / forms[0].max(), x[0], rtol=2e-11)
        np.testing.assert_allclose(fm.evaluate(candidate, data.q), data.intensity, rtol=2e-11)
    assert model.globals["refl_amp"].free == (level == 3 and family == "sphere")


def test_uniform_disk_area_inverse_and_level_three_mcmc_target():
    # Every square of half-width a maps to the disk of radius a. Uniform area gives Pr(|r| < a) = a².
    axis = (np.arange(100) + 0.5) / 50 - 1
    square = np.stack(np.meshgrid(axis, axis), axis=-1)
    physical = reflection_disk.from_square(square)
    radius = np.linalg.norm(physical, axis=-1)
    for a in (0.2, 0.5, 0.8):
        assert np.mean(radius < a) == pytest.approx(a**2)
    np.testing.assert_allclose(reflection_disk.to_square(physical), square, atol=2e-15)
    data, _angles = _cut(3)
    c = E.composition_index([("sphere", True)])
    v = _values()
    target = posterior_check.PredictTarget(data.q, data.intensity, data.sigma, composition=c,
                                          centre=v, qz=data.qz, qz_minus=data.qz_minus)
    full = target._full(target.z_star)
    for z in ([0.2, -0.4], [4.0, 3.0]):
        full[:, -2:] = z
        sample = infer.physical(E.from_z(full, [data.ratio], c), data.q_max, disk=True)
        np.testing.assert_allclose(target.values(full[:, target.dims]), sample[:, target.dims])
        assert np.hypot(*sample[0, -2:]) <= 1
        log_l = target.likelihood(sample)[0][0]
        assert target(full[:, target.dims])[0] == pytest.approx(log_l + E.log_prior(full, c)[0])
    v[-2:] = 0.8, 0.8
    assert target.likelihood(v)[0][0] == -np.inf


def test_saved_born_model_retains_transfer_until_measured_angles_are_supplied():
    data, measured = _cut(2)
    c = E.composition_index([("cylinder", False)])
    model = refine.fit_model(_values(), c, data, grazing={"wavelength": None})
    model = fm.model_from_dict(fm.model_to_dict(model))
    g = {key: model.globals[key].value for key in measured}
    same = substrate_level.prediction_model(model, {"grazing": g})
    assert single_predict.cut_qz_minus(same) is None
    assert fm.grazing(same).random_cylinder_qz() == pytest.approx(data.qz)
    measured_model = substrate_level.prediction_model(same, {"grazing": measured})
    assert single_predict.cut_qz_minus(measured_model) == pytest.approx(_cut(3)[0].qz_minus)
    assert fm.grazing(measured_model).random_cylinder_qz() == 0


def test_refinement_keeps_the_encoded_order_radius_ratios_and_window():
    data, _ = _cut(2)
    v = _values()
    v[6:12] = 5.2, 0.18, 13.0, 0.21, 27.0, 0.19
    c = E.composition_index([("sphere", True), ("cylinder", True)])
    model = refine.fit_model(v, c, data)
    problem = fit_engine._Problem(model, fit_engine.FitData.prepare(data.q, data.intensity, data.sigma),
                                  None, None, 1)
    ends = np.arange(len(problem.nonlinear)) % 2
    for t in (ends, 1 - ends, np.zeros_like(ends), np.ones_like(ends)):
        fields = {path: x for (path, _), x in zip(problem.nonlinear, problem.decode(t))}
        params = refine.model_values(model.with_values(fields), c)
        assert params is not None
        unit = E.to_unit(infer.dimensionless(params[None], data.q_max), [data.ratio], c)
        valid = unit[np.isfinite(unit)]
        assert np.min(valid) >= -1e-6 and np.max(valid) <= 1 + 1e-6
