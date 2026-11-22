"""The four-term DWBA level of manual fits, the vertical cut and the fit of several curves together:

* ``domain.substrates``: δ and β of the table against CXRO's values at Cu Kα, its range, typed values, the record;
* the substrate level of ``fit_model``: the DWBA geometry of ``cut_model`` exactly; a model without a substrate
  unchanged and written as before;
* ``domain.vertical_cut``: the exit angles of q_z (vacuum and refracted), the four paths against an independent
  sum, the size average converged, upright cylinders against a direct double sum over radii and heights, the
  structure factor at the cut's q_∥, the families it refuses;
* ``domain.joint_fit``: a horizontal and a vertical cut of upright cylinders recover the height the horizontal cut
  does not see, each curve with its own scale and background; shared values; refusals.

BornAgain itself: ``tools/benchmarks/joint_dwba_bornagain.py`` (Physics and Methods 9.17, 9.18).
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from src.gimap.features.fitting.domain import fit_engine as fe
from src.gimap.features.fitting.domain import fit_model as fm
from src.gimap.features.fitting.domain import joint_fit as jf
from src.gimap.features.fitting.domain.cut_model import Geometry, population_terms
from src.gimap.features.fitting.domain.substrates import CUSTOM, Substrate, optical_constants
from src.gimap.features.fitting.domain.vertical_cut import VerticalCut, cylinder_averages, sphere_averages

CU_KALPHA = 1.239841984 / 8.04879
LAMBDA, ALPHA_I = 0.1045, 0.4


# -- the table -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("material, delta, beta", [
    ("Si", 7.58e-6, 1.73e-7),       # CXRO, 2.33 g/cm³
    ("SiO2", 7.13e-6, 9.2e-8),      # CXRO, 2.2 g/cm³
    ("Au", 4.71e-5, 4.85e-6),       # CXRO, 19.32 g/cm³
])
def test_table_gives_cxros_index_at_cu_kalpha(material, delta, beta):
    d, b = optical_constants(material, CU_KALPHA)
    assert d == pytest.approx(delta, rel=0.01)
    assert b == pytest.approx(beta, rel=0.02)


def test_table_scales_with_wavelength_away_from_edges_and_refuses_outside_its_range():
    d8, _b = optical_constants("Si", CU_KALPHA)
    d16, _b = optical_constants("Si", CU_KALPHA / 2)
    assert d16 / d8 == pytest.approx(0.25 * 14.0949 / 14.2558, rel=2e-3)  # δ ∝ λ² f₁
    glass = optical_constants("glass", CU_KALPHA)[0]
    assert 6e-6 < glass < 9e-6
    with pytest.raises(ValueError, match="type δ and β"):
        optical_constants("Si", 0.5)  # 2.5 keV
    with pytest.raises(ValueError):
        Substrate("diamond")
    with pytest.raises(ValueError):
        Substrate(CUSTOM, 0.0, 0.0)


def test_substrate_record_round_trips():
    for substrate in (Substrate("Au"), Substrate(CUSTOM, 3.4e-6, 4e-8)):
        assert Substrate.from_dict(substrate.to_dict()) == substrate
    assert Substrate.from_dict(None) is None
    assert Substrate(CUSTOM, 3.4e-6, 4e-8).optical(0.2) == (3.4e-6, 4e-8)  # typed: at the cut's λ, as given
    assert Substrate("Si").critical_angle_deg(LAMBDA) == pytest.approx(math.degrees(math.sqrt(2 * optical_constants(
        "Si", LAMBDA)[0])))


# -- the substrate level of Fitting ------------------------------------------------------------------------------


def _horizontal(model: fm.FitModel, alpha_f: float = 0.15) -> fm.FitModel:
    return model.with_values({("globals", "alpha_i"): ALPHA_I, ("globals", "alpha_f"): alpha_f,
                              ("globals", "wavelength"): LAMBDA})


def test_substrate_level_is_cut_models_four_term_dwba():
    q = np.linspace(0.05, 2.5, 200)
    model = _horizontal(fm.FitModel((fm.new_component("sphere", radius=4.0, structure=False),),
                                    substrate=Substrate("Si")))
    geometry = fm.grazing(model)
    delta, beta = optical_constants("Si", LAMBDA)
    assert geometry == Geometry("dwba", LAMBDA, ALPHA_I, 0.15, delta=delta, beta=beta)
    reference = population_terms("sphere", q, radius=4.0, radius_spread=0.2, geometry=geometry)[0]
    assert np.allclose(fm.evaluate(model, q), reference, rtol=1e-12, atol=0)
    assert len(geometry.sphere_terms()) == 4
    assert fm.qz_plus_r(model)[0][1] == pytest.approx(geometry.qz_born * 4.0)


def test_substrate_needs_both_angles_and_leaves_other_models_as_they_were():
    q = np.linspace(0.05, 2.0, 50)
    base = fm.FitModel((fm.new_component("sphere", radius=3.0, structure=False),))
    plain = _horizontal(base).with_values({("globals", "refl_amp"): 0.5})
    known = replace(plain, substrate=Substrate("Si"))
    assert fm.grazing(plain).mode == "two_path"
    assert fm.grazing(known).mode == "dwba"
    assert not np.allclose(fm.evaluate(plain, q), fm.evaluate(known, q))
    no_angles = replace(base, substrate=Substrate("Si"))
    assert fm.grazing(no_angles) is None  # without the angles: the effective interface, as before
    assert np.array_equal(fm.evaluate(no_angles, q), fm.evaluate(base, q))
    assert "substrate" not in fm.model_to_dict(plain)  # a model without one is written as in earlier versions
    again = fm.model_from_dict(fm.model_to_dict(known))
    assert again.substrate == Substrate("Si") and np.array_equal(fm.evaluate(again, q), fm.evaluate(known, q))


def test_upright_cylinder_heights_are_parameters_only_when_asked():
    plain = fm.new_component("vertical_cylinder", radius=4.0)
    tall = fm.new_component("vertical_cylinder", radius=4.0, heights=True)
    assert "h" not in plain.keys() and {"h", "sigma_h"} <= set(tall.keys())
    record = fm.model_to_dict(fm.FitModel((tall,)))
    assert record["components"][0]["heights"] is True
    assert "heights" not in fm.model_to_dict(fm.FitModel((plain,)))["components"][0]
    assert fm.model_from_dict(record).components[0].keys() == tall.keys()
    q = np.linspace(0.05, 2.0, 40)  # the horizontal cut does not see the height
    assert np.array_equal(fm.evaluate(fm.FitModel((plain,)), q), fm.evaluate(fm.FitModel((tall,)), q))


@pytest.mark.parametrize("grazing", [False, True])
@pytest.mark.parametrize("interference", ["paracrystal", "hard_sphere"])
def test_horizontal_da_against_independent_radius_height_quadrature(grazing, interference):
    from scipy.special import j1
    from src.gimap.features.fitting.domain import model_library
    from src.gimap.features.fitting.domain.cut_model.structure import paracrystal

    q = np.array([0.08, 0.27, 0.5, 0.8])
    cylinder = fm.new_component("vertical_cylinder", radius=4, distance=14, heights=True, interference=interference)
    model = fm.FitModel((replace(cylinder, coupling="da"),), substrate=Substrate("Si")).with_values({
        (0, "sigma_R"): 0.15, (0, "h"): 6.0, (0, "sigma_h"): 0.2})
    if grazing:
        model = _horizontal(model)
    terms = (fm.grazing(model) or Geometry()).sphere_terms()
    if interference == "paracrystal":
        structure = paracrystal(q, 14.0, cylinder.value("sigma_nn"))[0]
    else:
        structure = model_library.hard_sphere(q, cylinder.value("R_hs"), cylinder.value("phi"))[0]

    # Independent Gaussian counts over ±4σ, integrating the complex amplitude of each (R,H) pair.
    x, weights = np.polynomial.legendre.leggauss(64)
    weights *= np.exp(-0.5 * (4*x)**2)
    weights /= weights.sum()
    number = np.outer(weights, weights)
    radii, heights = 4*(1 + 4*0.15*x), 6*(1 + 4*0.2*x)
    axial = sum(c * heights * np.sinc(w * heights / (2*np.pi)) * np.exp(-0.5j*w*heights) for c, w in terms)
    norm = np.sum(number * np.abs(np.outer(radii**2, axial))**2)
    direct = []
    for qi, si in zip(q, structure):
        amp = np.outer(radii**2 * 2*j1(qi*radii) / (qi*radii), axial)
        direct.append((np.sum(number * np.abs(amp)**2) + abs(np.sum(number * amp))**2 * (si - 1)) / norm)
    assert np.allclose(fm.evaluate(model, q), direct, rtol=2e-5)

    single_height = model.with_values({(0, "sigma_h"): 0.0})
    radial_only = replace(model, components=(replace(model.components[0], heights=False),))
    assert np.array_equal(fm.evaluate(single_height, q), fm.evaluate(radial_only, q))


# -- the vertical cut ----------------------------------------------------------------------------------------------


def _cut(q_parallel: float = 0.3, critical: float = 0.0) -> VerticalCut:
    delta, beta = optical_constants("Si", LAMBDA)
    return VerticalCut(q_parallel, LAMBDA, ALPHA_I, delta, beta, critical)


def test_exit_angles_invert_qz_in_vacuum_and_refracted():
    cut = _cut()
    k = 2 * math.pi / LAMBDA
    alpha_f = np.array([0.05, 0.2, 0.8, 1.5])
    qz = k * (math.sin(math.radians(ALPHA_I)) + np.sin(np.radians(alpha_f)))
    assert np.allclose(cut.exit_angles_deg(qz), alpha_f, rtol=1e-12)
    assert np.isnan(cut.exit_angles_deg(np.array([0.5 * k * math.sin(math.radians(ALPHA_I))]))).all()
    critical = 0.12
    refracted = _cut(critical=critical)
    sc2 = math.sin(math.radians(critical)) ** 2
    s_i, s_f = math.sin(math.radians(ALPHA_I)) ** 2, np.sin(np.radians(alpha_f)) ** 2
    qz_film = k * (np.sqrt(np.maximum(s_f - sc2, 0)) + math.sqrt(s_i - sc2))
    above = alpha_f > critical
    assert np.allclose(refracted.exit_angles_deg(qz_film[above]), alpha_f[above], rtol=1e-10)


def test_four_paths_against_an_independent_sum():
    cut = _cut()
    k = 2 * math.pi / LAMBDA
    delta, beta = optical_constants("Si", LAMBDA)
    alpha_f = 0.31
    qz = np.array([k * (math.sin(math.radians(ALPHA_I)) + math.sin(math.radians(alpha_f)))])
    form, _c = sphere_averages(0.3, *cut.paths(qz), "number_gauss", 4.0, 0.0, False)
    geometry = Geometry("dwba", LAMBDA, ALPHA_I, alpha_f, delta=delta, beta=beta)  # the same four terms, one q
    amplitude = 0j
    for c, wave in geometry.sphere_terms():
        x = math.hypot(0.3, wave) * 4.0
        amplitude += c * 3 * (math.sin(x) - x * math.cos(x)) / x**3 * np.exp(-1j * wave * 4.0)
    assert form[0] == pytest.approx(abs(amplitude) ** 2, rel=1e-12)
    # and the horizontal model of the same point (its sizes on their own rule)
    reference = population_terms("sphere", [0.3], radius=4.0, radius_spread=0.2, geometry=geometry)[0][0]
    assert sphere_averages(0.3, *cut.paths(qz), "number_gauss", 4.0, 0.2, False)[0][0] == pytest.approx(reference,
                                                                                                         rel=1e-5)


def test_upright_cylinders_against_a_direct_double_sum():
    cut = _cut(0.4)
    k = 2 * math.pi / LAMBDA
    alpha_f = np.array([0.1, 0.3, 0.7, 1.1])
    qz = k * (math.sin(math.radians(ALPHA_I)) + np.sin(np.radians(alpha_f)))
    coefficients, wave = cut.paths(qz)
    form, coherent = cylinder_averages(0.4, coefficients, wave, "number_gauss", 4.0, 0.15, 6.0, 0.15, True)
    from scipy.special import j1

    radii = np.linspace(4.0 * 0.4, 4.0 * 1.6, 1601)
    heights = np.linspace(6.0 * 0.4, 6.0 * 1.6, 1601)
    n_r = np.exp(-0.5 * ((radii - 4.0) / 0.6) ** 2)
    n_h = np.exp(-0.5 * ((heights - 6.0) / 0.9) ** 2)
    n_r, n_h = n_r / n_r.sum(), n_h / n_h.sum()  # number weights summing to 1, as the coherent term needs
    disc = 2 * j1(0.4 * radii) / (0.4 * radii)
    total_sq = np.zeros(alpha_f.size)
    total = np.zeros(alpha_f.size, complex)
    for point in range(alpha_f.size):
        amplitude = sum(c * np.sinc(w * heights / (2 * np.pi)) * np.exp(-0.5j * w * heights)
                        for c, w in zip(coefficients[:, point], wave[:, point]))
        height_sq = np.sum(n_h * heights**2 * np.abs(amplitude) ** 2) / np.sum(n_h * heights**2)
        total_sq[point] = np.sum(n_r * radii**4 * disc**2) / np.sum(n_r * radii**4) * height_sq
        total[point] = (np.sum(n_r * radii**2 * disc) * np.sum(n_h * heights * amplitude)
                        / math.sqrt(np.sum(n_r * radii**4) * np.sum(n_h * heights**2)))
    assert np.allclose(form, total_sq, rtol=2e-5)
    assert np.allclose(coherent, np.abs(total) ** 2, rtol=2e-5)


def test_vertical_model_terms_structure_and_refusals():
    cut = _cut()
    k = 2 * math.pi / LAMBDA
    qz = k * (math.sin(math.radians(ALPHA_I)) + np.sin(np.radians(np.linspace(0.05, 1.2, 60))))
    sphere = fm.new_component("sphere", radius=4.0, distance=14.0)
    model = fm.FitModel((sphere,), cut=cut).with_values({(0, "sigma_R"): 0.0, (0, "sigma_nn"): 0.2})
    bare = replace(model, components=(replace(sphere, structure=False),)).with_values({(0, "sigma_R"): 0.0})
    from src.gimap.features.fitting.domain.cut_model.structure import paracrystal

    factor = paracrystal([0.3], 14.0, 0.2)[0, 0]
    assert np.allclose(fm.evaluate(model, qz), factor * fm.evaluate(bare, qz), rtol=1e-12)
    da = replace(model, components=(replace(model.components[0], coupling="da"),))
    assert np.allclose(fm.evaluate(da, qz), fm.evaluate(model, qz), rtol=1e-10)  # one size: DA = LMA
    q_sigma = model.with_values({("globals", "q_resolution"): 0.01})  # no resolution in a vertical cut
    assert np.array_equal(fm.evaluate(q_sigma, qz), fm.evaluate(model, qz))
    with pytest.raises(ValueError, match="spheres and upright cylinders"):
        fm.evaluate(fm.FitModel((fm.new_component("cylinder", radius=4.0),), cut=cut), qz)
    with pytest.raises(ValueError, match="height"):
        fm.evaluate(fm.FitModel((fm.new_component("vertical_cylinder", radius=4.0),), cut=cut), qz)
    with pytest.raises(ValueError):
        VerticalCut(0.3, LAMBDA, ALPHA_I, 0.0, 0.0)


# -- the fit of several curves ---------------------------------------------------------------------------------


def _cylinder_curves(seed: int = 3):
    rng = np.random.default_rng(seed)
    cut = _cut()
    k = 2 * math.pi / LAMBDA
    q = np.linspace(0.05, 2.5, 250)
    qz = k * (math.sin(math.radians(ALPHA_I)) + np.sin(np.radians(np.linspace(0.05, 1.3, 180))))
    component = fm.new_component("vertical_cylinder", radius=4.0, distance=14.0, heights=True)
    truth = fm.FitModel((component,)).with_values({
        (0, "sigma_R"): 0.15, (0, "h"): 6.0, (0, "sigma_h"): 0.1, (0, "sigma_nn"): 0.2, (0, "Int"): 2e4,
        ("globals", "background"): 3.0})
    truth = truth.with_parameter(("globals", "res_amplitude"), value=0.0, free=False)
    for key in ("res_width", "res_exponent"):
        truth = truth.with_parameter(("globals", key), free=False)
    vertical = replace(truth, cut=cut).with_values({(0, "Int"): 5e3, ("globals", "background"): 10.0})
    curves = []
    for name, model, x in (("horizontal", truth, q), ("vertical", vertical, qz)):
        y = rng.poisson(fm.evaluate(model, x)).astype(float)
        curves.append(jf.JointCurve(name, fe.FitData.prepare(x, y, np.sqrt(np.maximum(y, 1.0))), model))
    return truth, curves


def test_joint_fit_recovers_the_height_only_the_vertical_cut_sees():
    truth, curves = _cylinder_curves()
    start = {(0, "R"): 5.0, (0, "h"): 9.0, (0, "sigma_R"): 0.25, (0, "D"): 16.0}
    started = [replace(c, model=c.model.with_values(start)) for c in curves]
    result = jf.fit_curves(started, method="local")
    assert result.converged and not result.stopped
    shared = jf.SHARED
    for path, value in (((0, "h"), 6.0), ((0, "R"), 4.0), ((0, "D"), 14.0)):
        fitted = result.models[0].get(path).value
        assert result.models[1].get(path).value == fitted
        assert abs(fitted - value) < 4 * result.errors[(shared, path)] + 0.02 * value
    assert 0.7 < result.chi2_reduced < 1.4
    assert len(result.curve_chi2) == 2 and result.points == (250, 180)
    # each curve has its own scale and background
    assert result.models[0].components[0].value("Int") != pytest.approx(result.models[1].components[0].value("Int"))
    assert (1, ("globals", "background")) in result.errors and (0, (0, "Int")) in result.errors
    # the horizontal cut alone does not see the height
    alone = fe.fit(started[0].model, curves[0].data, method="local")
    assert alone.model.get((0, "h")).value == pytest.approx(9.0) or not math.isfinite(alone.errors.get((0, "h"),
                                                                                                         math.nan))


def test_shared_values_and_refusals():
    truth, curves = _cylinder_curves()
    assert jf.default_shared(truth) == frozenset({(0, k) for k in ("R", "sigma_R", "h", "sigma_h", "D", "sigma_nn")})
    moved = [curves[0], replace(curves[1], model=curves[1].model.with_values({(0, "R"): 7.0}))]
    assert jf.shared_curves(moved, {(0, "R")})[1].model.get((0, "R")).value == 4.0
    assert jf.shared_curves(moved, set())[1].model.get((0, "R")).value == 7.0
    with pytest.raises(ValueError, match="cannot be shared"):
        jf.shared_curves(curves, {(0, "Int")})
    other = replace(curves[1], model=fm.FitModel((fm.new_component("sphere", radius=4.0),), cut=_cut()))
    with pytest.raises(ValueError, match="particles differ"):
        jf.fit_curves([curves[0], other])
    # a per-curve value: each curve its own D
    result = jf.fit_curves(curves, shared=jf.default_shared(truth) - {(0, "D")}, max_evaluations=60)
    assert (0, (0, "D")) in result.errors and (1, (0, "D")) in result.errors
    assert (jf.SHARED, (0, "D")) not in result.errors
