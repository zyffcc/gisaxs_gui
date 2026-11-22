"""Independent fibre integrals, low-angle extrapolation and systematic uncertainty."""

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad

from src.gimap.features.analyze.domain.crystal import PoleFit, fit_pole_figure, fold_profile
from src.gimap.features.analyze.domain.crystal import pole_models as models
from src.gimap.features.analyze.domain.crystal.pole_systematics import orientation_systematics
from src.gimap.features.assistant.domain import Profile, ring_orientation


@pytest.mark.parametrize("m", [-60., 0., 35., 2000., 60000.])
@pytest.mark.parametrize("director", [0., 32., 90.])
def test_tilted_maier_saupe_matches_independent_azimuth_quadrature(m, director):
    angles = np.radians([0., 24., director, 59., 89.5, 90.])
    tilt = math.radians(director)

    def reference(chi):
        def density(psi):
            u = math.cos(chi) * math.cos(tilt) + math.sin(chi) * math.sin(tilt) * math.cos(psi)
            return math.exp(m * u**2 - max(m, 0.))

        return quad(density, 0., math.pi, epsabs=1e-13, points=[0.01, math.pi / 2, math.pi - 0.01])[0] / math.pi

    actual = models.oriented_shape(angles, models.MAIER_SAUPE, m, director)
    np.testing.assert_allclose(actual, [reference(a) for a in angles], rtol=1e-7, atol=1e-10)


@pytest.mark.parametrize("width", [0.3, 18., 150.])
def test_gaussian_mosaic_averages_angles_to_the_director_and_normalises(width):
    chi = np.radians([18., 54., 70.])
    director = math.radians(54.)
    sigma = math.radians(width) / math.sqrt(8. * math.log(2.))

    def reference(theta, psi):
        dot = math.cos(director) * math.cos(theta) + math.sin(director) * math.sin(theta) * math.cos(psi)
        gamma = math.acos(min(1., max(-1., dot)))
        return math.exp(-0.5 * (gamma / sigma)**2) + math.exp(-0.5 * ((math.pi - gamma) / sigma)**2)

    expected = [quad(lambda psi: reference(a, psi), 0., math.pi, epsabs=1e-13)[0] / math.pi for a in chi]
    np.testing.assert_allclose(models.oriented_shape(chi, models.GAUSSIAN_MOSAIC, width, 54.), expected,
                               rtol=1e-7, atol=1e-10)
    integral = quad(lambda a: models.crystallites_per_radian(a, models.GAUSSIAN_MOSAIC, .25, width, 54.),
                    0., math.pi / 2., points=[director])[0]
    assert integral == pytest.approx(1., abs=1e-6)


@pytest.mark.parametrize("model,width", [(models.MAIER_SAUPE, 2000.), (models.GAUSSIAN_MOSAIC, 2.5)])
def test_pole_moments_and_axial_symmetry_at_an_oblique_tilt(model, width):
    tilt = 62.7
    chi = np.radians(np.linspace(0., 90., 901))
    rho = models.pole_density(chi, model, .24, width, tilt)
    np.testing.assert_allclose(models.pole_density(-chi, model, .24, width, tilt), rho, rtol=1e-10)
    np.testing.assert_allclose(models.pole_density(math.pi - chi, model, .24, width, tilt), rho, rtol=1e-10)
    integrate = getattr(np, "trapezoid", None) or np.trapz
    assert 2. * math.pi * integrate(rho * np.sin(chi), chi) == pytest.approx(1., abs=1e-6)
    solid = integrate((3 * np.cos(chi)**2 - 1) / 2 * rho * np.sin(chi), chi) / integrate(rho * np.sin(chi), chi)
    assert models.hermans(model, .24, width, tilt) == pytest.approx(solid, abs=1e-6)
    along_ring = integrate(rho * np.cos(chi)**2, chi) / integrate(rho, chi)
    assert models.order_2d(model, .24, width, tilt) == pytest.approx(2 * along_ring - 1, abs=1e-6)


def test_tilted_maier_saupe_girdle_fit_recovers_negative_m():
    chi = np.arange(.5, 90., 1.)
    counts = 1200. * models.binned(chi - .5, chi + .5, models.MAIER_SAUPE, .3, -18., 33.)
    profile = fold_profile(chi, counts / np.sin(np.radians(chi)), np.ones(chi.size))
    fit = fit_pole_figure(profile, models.MAIER_SAUPE, chi0_deg=33., width=40.)
    assert fit.ok and fit.converged
    assert fit.values["fraction_iso"] == pytest.approx(.3, abs=1e-5)
    assert fit.values["width"] == pytest.approx(-18., abs=1e-5)


def test_measured_f_preserves_signed_net_intensities_and_requires_positive_total():
    chi = np.arange(.5, 90., 1.)
    sigma = np.full(90, .2)
    net = np.where(chi < 45., -1., 3.)
    weights = net * np.sin(np.radians(chi))
    expected = np.average((3 * np.cos(np.radians(chi))**2 - 1.) / 2., weights=weights)
    ring = ring_orientation(Profile.of(chi, 10. + net, sigma), q_window=(.9, 1.1), background=10.)
    assert ring.herman == pytest.approx(expected)
    assert fold_profile(chi, 10. + net, sigma, 10.).herman_measured() == pytest.approx(expected)
    net *= -1
    ring = ring_orientation(Profile.of(chi, 10. + net, sigma), q_window=(.9, 1.1), background=10.)
    assert ring.herman is None and "nonpositive" in ring.reason
    assert fold_profile(chi, 10. + net, sigma, 10.).herman_measured() is None


def test_filling_extrapolates_only_the_wedge_and_propagates_the_fit_covariance():
    chi = np.arange(-89.5, 90., 1.)
    background = 70.
    signal = 2000. * models.pole_density(np.radians(abs(chi)), models.GAUSSIAN, .4, 18., 0.)
    profiles = {}
    for threshold, maximum in [(0.5, 89.), (1., 87.), (2., 84.)]:
        keep = (abs(chi) > 5.) & (abs(chi) < maximum) & (abs(abs(chi) - 45.) > 2.)
        profiles[threshold] = chi[keep], background + signal[keep], np.full(keep.sum(), 4.)
    profile = fold_profile(*profiles[1.], background)
    fit = fit_pole_figure(profile, models.GAUSSIAN)
    assert fit.converged and not fit.quality_reasons
    scenarios = orientation_systematics(profiles, background, background_sigma=3., nominal_fit=fit)
    assert scenarios["background_step"] == 3. and len(scenarios["scenarios"]) == 9 and not scenarios["incomplete"]
    center = next(s for s in scenarios["scenarios"] if s["threshold_multiplier"] == 1. and s["background"] == background)
    raw = ring_orientation(Profile.of(*profiles[1.]), q_window=(.99, 1.01), background=background)
    assert center["measured"] == pytest.approx(raw.herman)
    p2 = 0.5 * (3. * np.cos(np.radians(profile.chi))**2 - 1.)
    counts = np.where(profile.measured, profile.value, 0.)
    wedge = profile.chi < 5.
    params = fit.values
    counts[wedge] = math.radians(1.) * params["scale"] * models.binned(
        profile.chi[wedge] - .5, profile.chi[wedge] + .5, models.GAUSSIAN, params["fraction_iso"], params["width"], 0.)
    assert center["model_fill"] == pytest.approx(np.average(p2, weights=counts))
    assert center["wedge_sigma"] > 0. and center["model_fill"] > center["measured"]
    assert scenarios["range"][0] <= center["measured"] < center["model_fill"] + center["wedge_sigma"] <= scenarios["range"][1]
    nominal_only = orientation_systematics({1.: profiles[1.]}, background, background_sigma=0., nominal_fit=fit)
    assert nominal_only["incomplete"] and nominal_only["scenarios"][0]["model_fill"] is not None
    no_fit = orientation_systematics({1.: profiles[1.]}, background, nominal_fit=PoleFit(False, "gaussian"))
    assert no_fit["background_step"] == pytest.approx(.2 * background) and no_fit["incomplete"]
    assert all(s["measured"] is not None for s in no_fit["scenarios"])
    assert all(s["model_fill"] is None for s in no_fit["scenarios"])


def test_fit_quality_records_each_reason_and_respects_a_fixed_fraction():
    good = PoleFit(True, models.GAUSSIAN, values={"fwhm_deg": 100.}, chi2_red=5., fixed=("fraction_iso",), converged=True)
    assert not good.quality_reasons
    bad = replace(good, converged=False, chi2_red=5.01, values={"fwhm_deg": 100.1}, fixed=(), at_bound=("fraction_iso",))
    assert set(bad.record()["quality_reasons"]) == {
        "fit did not converge", "reduced χ² exceeds 5", "FWHM exceeds 100°", "isotropic fraction at a bound"}
    assert replace(good, errors={"width": math.nan}).quality_reasons == ("fit parameters are not identifiable",)


def test_measured_error_and_range_use_same_cap_even_when_pole_fit_fails():
    from src.gimap.features.analyze.application.pole_figure import PoleFigure
    from src.gimap.features.analyze.presentation.indexing_texts import pole_result_text

    chi = np.arange(.5, 90., 1.)
    value = 3. + np.where(chi < 80., 2., 11.)
    error = np.linspace(.4, 1., chi.size)
    cap = 80.
    mask = chi < cap
    direct = ring_orientation(Profile.of(chi[mask], value[mask], error[mask]), q_window=(.9, 1.1), background=3.)
    profile = fold_profile(chi, value, error, 3.)
    fit = fit_pole_figure(profile, chi_max=cap)
    assert fit.record()["herman_measured"] == pytest.approx(direct.herman)
    assert fit.record()["herman_measured_err"] == pytest.approx(direct.herman_err)
    assert fit.weighted_coverage == pytest.approx(direct.weighted_coverage)
    # Only three bins, enough for a measured ratio and its error but insufficient for a pole fit.
    failed = fit_pole_figure(profile, chi_max=3.)
    assert not failed.ok and failed.measured_moment[1] > 0.
    result = PoleFigure(failed, (.9, 1.1), 3., "given", False, 0., 3.)
    text = pole_result_text(result)
    assert "not determined" in text and "Measured f" in text and "(stat)" in text


@pytest.mark.parametrize("nominal", [
    PoleFit(False, models.GAUSSIAN),
    PoleFit(True, models.GAUSSIAN, converged=True, chi2_red=5.1),
    PoleFit(True, models.GAUSSIAN, converged=True, at_bound=("fraction_iso",)),
    PoleFit(True, models.GAUSSIAN, converged=True, values={"fwhm_deg": 101.}),
])
def test_no_scenario_uses_a_model_fill_when_the_nominal_pole_is_not_determined(nominal, monkeypatch):
    from src.gimap.features.analyze.domain.crystal import pole_systematics

    chi = np.arange(4.5, 87., 1.)
    curve = chi, np.full(chi.size, 3.), np.ones(chi.size)
    monkeypatch.setattr(pole_systematics, "fit_pole_figure", lambda *a, **kw: pytest.fail("unexpected model fill"))
    result = orientation_systematics({k: curve for k in (.5, 1., 2.)}, 1., nominal_fit=nominal)
    assert not result["model_fill_allowed"]
    assert all(s["model_fill"] is None and s["wedge_sigma"] is None for s in result["scenarios"])
    measured = [s["measured"] for s in result["scenarios"]]
    assert result["range"] == [min(measured), max(measured)]


def test_a_poor_scenario_fit_keeps_only_its_measured_moment(monkeypatch):
    from src.gimap.features.analyze.domain.crystal import pole_systematics

    chi = np.arange(4.5, 87., 1.)
    curve = chi, np.full(chi.size, 10.), np.ones(chi.size)
    profile = fold_profile(*curve, 2.)
    nominal = PoleFit(True, models.GAUSSIAN, profile=profile, converged=True,
                      values={"herman_wedge_fill": .2}, errors={"herman_wedge_fill": .01})
    poor = replace(nominal, chi2_red=18., values={"herman_wedge_fill": .8})
    monkeypatch.setattr(pole_systematics, "fit_pole_figure", lambda *args, **kwargs: poor)
    result = orientation_systematics({k: curve for k in (.5, 1., 2.)}, 2., background_sigma=0., nominal_fit=nominal)
    assert result["model_fill_allowed"] and result["range"][1] == pytest.approx(.21)
    assert [s["model_fill"] for s in result["scenarios"]] == [None, .2, None]
    assert result["scenarios"][0]["quality_reasons"] == ["reduced χ² exceeds 5"]


def test_an_attenuated_horizon_is_excluded_from_both_moments_and_pole_fits():
    # The exit-angle cut already removed |χ| ≥ 88°. Two thresholds disagree on the 83–84° bin.
    chi = np.arange(-87.5, 88., 1.)
    angle = abs(chi)
    net = np.where(angle >= 84., 4., np.where(angle >= 83., 8., 100.))
    net[(angle == 45.5) | (angle == 81.5)] = 2.  # isolated weak bins still count
    background = 20.
    curve = chi, background + net, np.ones(chi.size)
    horizon = 87.9
    profile = fold_profile(*curve, background, horizon_chi_deg=horizon)
    fit = fit_pole_figure(profile)
    assert profile.horizon_excluded_deg == (84, 88) and fit.points == 84
    np.testing.assert_array_equal(profile.measured, profile.chi < 84.)
    ring = ring_orientation(Profile.of(*curve), q_window=(.9, 1.1), background=background, horizon_chi_deg=horizon)
    weights = net[angle < 84.] * np.sin(np.radians(angle[angle < 84.]))
    expected = np.average((3. * np.cos(np.radians(angle[angle < 84.]))**2 - 1.) / 2., weights=weights)
    assert ring.herman == pytest.approx(expected)
    assert fit.measured_moment[0] == pytest.approx(expected)

    result = orientation_systematics({1.: curve}, background, background_sigma=0., nominal_fit=fit,
                                     horizon_angles={1.: horizon})
    assert result["horizon_excluded_deg"] == (84, 88)
    scenarios = {s["horizon_fraction"]: s for s in result["scenarios"]}
    assert scenarios[0.]["horizon_excluded_deg"] is None
    assert scenarios[.05]["measured"] == pytest.approx(expected)
    assert scenarios[.10]["horizon_excluded_deg"] == (83, 88)
    assert result["range"][0] == pytest.approx(scenarios[0.]["measured"])
    assert result["range"][1] == pytest.approx(scenarios[.10]["measured"])

    # The measured sector ends well short of the exit cut. Weak bins at a detector edge stay measured.
    keep = angle < 80.
    y = np.where(angle[keep] > 77., background + 1., background + 100.)
    cropped = fold_profile(chi[keep], y, np.ones_like(y), background, horizon_chi_deg=horizon)
    assert cropped.horizon_excluded_deg is None and cropped.measured.sum() == 80
