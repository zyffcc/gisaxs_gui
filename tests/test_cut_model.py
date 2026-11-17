"""cut_model, the one forward model of a 1D cut: normalisation, convergence, the independent truth, conventions."""

from __future__ import annotations

import functools
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import j1

from src.gimap.features.fitting.domain import scattering_model as old
from src.gimap.features.fitting.domain.cut_model import (
    Geometry,
    Population,
    Settings,
    conventions,
    evaluate,
    population_terms,
    sizes,
)
from src.gimap.features.fitting.domain.cut_model.structure import paracrystal

Q = np.linspace(0.02, 3.0, 300)
BENCHMARKS = Path(__file__).resolve().parents[1] / "tools" / "benchmarks"


@pytest.mark.parametrize("family", ["sphere", "vertical_cylinder", "cylinder"])
@pytest.mark.parametrize("law", ["number_gauss", "lognormal", "intensity_gauss"])
def test_every_family_scatters_one_at_q_zero(family, law) -> None:
    value = population_terms(family, np.array([1e-7]), radius=5.0, radius_spread=0.3, height=12.0,
                             height_spread=0.2, law=law)
    assert value[0, 0] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("family", ["sphere", "vertical_cylinder"])
@pytest.mark.parametrize("law", ["number_gauss", "lognormal", "intensity_gauss"])
def test_the_size_average_agrees_with_the_fully_resolved_one(family, law) -> None:
    """Below q_cut the average has converged (tables: their interpolation, ≤ 1e-4); above it the envelope leaves out
    only the ringing of the hard cut of a distribution by number, which grows with σR/R (``envelope``). Sizes by
    intensity are always resolved directly."""
    for radius, spread, tolerance in ((2.0, 0.05, 1e-4), (7.0, 0.2, 1e-4), (12.0, 0.35, 1e-3), (40.0, 0.5, 1e-2)):
        default = population_terms(family, Q, radius=radius, radius_spread=spread, law=law)
        fine = population_terms(family, Q, radius=radius, radius_spread=spread, law=law, panels=1024)
        np.testing.assert_allclose(default, fine, rtol=3e-5 if law == "intensity_gauss" else tolerance)


def test_the_random_cylinder_average_agrees_with_the_fully_resolved_one() -> None:
    q = Q[::3]
    for radius, spread, height, tolerance in ((5.0, 0.2, 40.0, 1e-4), (12.0, 0.35, 30.0, 1e-3)):
        common = dict(radius=radius, radius_spread=spread, height=height, height_spread=0.2)
        np.testing.assert_allclose(population_terms("cylinder", q, **common),
                                   population_terms("cylinder", q, panels=256, **common), rtol=tolerance)


@pytest.mark.parametrize("name", ["sphere", "disc", "sinc"])
def test_an_envelope_is_the_average_over_one_oscillation(name) -> None:
    from src.gimap.features.fitting.domain.cut_model import envelope, shapes

    exact, smooth, period = {
        "sphere": (lambda x: shapes.sphere_amplitude(x) ** 2, envelope.sphere_square, np.pi),
        "disc": (lambda x: shapes.disc_amplitude(x) ** 2, envelope.disc_square, np.pi),
        "sinc": (lambda x: shapes.sinc_half(x) ** 2, envelope.sinc_square, 2 * np.pi),
    }[name]
    for centre in (200.0, 600.0):
        x = np.linspace(centre - 12 * period, centre + 12 * period, 200001)
        window = np.exp(-(((x - centre) / (3 * period)) ** 2))  # a Gaussian over several periods: no leakage
        assert np.sum(window * exact(x)) / np.sum(window * smooth(x)) == pytest.approx(1.0, rel=1e-4)
    pair = envelope.sphere_pair(np.array([50.0]), np.array([50.0]))
    assert pair[0] == pytest.approx(envelope.sphere_square(np.array([50.0]))[0], rel=1e-12)


def test_a_random_cylinder_of_one_size_is_the_orientation_integral() -> None:
    def disc(x):
        return 1.0 if abs(x) < 1e-9 else 2 * j1(x) / x

    def half(x):
        return 1.0 if abs(x) < 1e-9 else math.sin(x / 2) / (x / 2)

    q = np.array([0.05, 0.6, 1.4, 2.5])
    for radius, height in ((3.0, 10.0), (2.0, 60.0)):
        reference = [quad(lambda u, x=x: disc(x * radius * math.sqrt(1 - u * u)) ** 2 * half(x * height * u) ** 2,
                          0, 1, limit=2000, epsrel=1e-12)[0] for x in q]
        got = population_terms("cylinder", q, radius=radius, radius_spread=0.0, height=height, height_spread=0.0)[0]
        np.testing.assert_allclose(got, reference, rtol=3e-5)


def test_a_batch_gives_the_rows_one_by_one() -> None:
    rng = np.random.default_rng(3)
    radius, spread = np.exp(rng.uniform(0, 3, 12)), rng.uniform(0.02, 0.5, 12)
    batch = population_terms("sphere", Q, radius=radius, radius_spread=spread, distance=3 * radius,
                             distance_spread=0.2, coupling="da")
    for row in range(12):
        one = population_terms("sphere", Q, radius=radius[row], radius_spread=spread[row], distance=3 * radius[row],
                               distance_spread=0.2, coupling="da")[0]
        np.testing.assert_allclose(batch[row], one, rtol=1e-12)


def test_the_paracrystal_is_hosemanns_and_fittings_with_sqrt_2pi() -> None:
    np.testing.assert_allclose(paracrystal(Q, 20.0, 0.25)[0], old.structure_factor_1d(Q, 20.0, 20.0 * 0.25 / math.sqrt(2 * math.pi)),
                               rtol=1e-12)
    p = np.exp(-0.5 * (Q * 0.25 * 20.0) ** 2)
    np.testing.assert_allclose(paracrystal(Q, 20.0, 0.25)[0], (1 - p * p) / (1 - 2 * p * np.cos(Q * 20.0) + p * p), rtol=1e-10)
    assert paracrystal(np.array([0.0]), 20.0, 0.25)[0, 0] == pytest.approx(0.0625)
    assert np.all(paracrystal(Q, 0.0, 0.25) == 1.0)


@pytest.mark.parametrize("family", ["sphere", "vertical_cylinder"])
def test_da_and_lma_agree_for_one_size(family) -> None:
    """(Randomly oriented cylinders of one size differ: their orientation is a disorder of its own.)"""
    common = dict(radius=4.0, radius_spread=0.0, distance=13.0, distance_spread=0.2)
    np.testing.assert_allclose(population_terms(family, Q, coupling="da", **common),
                               population_terms(family, Q, coupling="lma", **common), rtol=1e-9)


@pytest.fixture(scope="module")
def truth():
    sys.path.insert(0, str(BENCHMARKS))
    import yoneda_truth

    yoneda_truth.size_rule = functools.partial(yoneda_truth.size_rule, n=600)
    return yoneda_truth


@pytest.mark.parametrize("scattering, mode", [("born_qz", "born"), ("dwba", "dwba")])
@pytest.mark.parametrize("coupling", ["lma", "da"])
def test_agrees_with_the_independent_truth_model(truth, scattering, mode, coupling) -> None:
    """tools/benchmarks/yoneda_truth.py shares no code; it has an arbitrary absolute scale."""
    geometry = truth.Geometry(wavelength_nm=0.1045, alpha_i_deg=0.4)
    delta, beta = geometry.index()
    ours = Geometry(mode, 0.1045, 0.4, geometry.alpha_f, delta=delta, beta=beta)
    q = Q[::3]
    for family, film in (("sphere", dict(R=4.0, sR=0.2, D=14.0, s_nn=0.2)),
                         ("vertical_cylinder", dict(R=4.0, sR=0.25, D=15.0, s_nn=0.2, cyl_h=7.0))):
        reference = truth.particle_intensity(q, truth.Case(geometry, truth.Film(coupling=coupling, scattering=scattering,
                                                                                **film)))
        got = population_terms(family, q, radius=film["R"], radius_spread=film["sR"], distance=film["D"],
                               distance_spread=film["s_nn"], coupling=coupling, geometry=ours)[0]
        ratio = got / reference
        np.testing.assert_allclose(ratio / np.median(ratio), 1.0, rtol=1e-4)  # the tables' interpolation


def test_two_paths_without_reflection_are_the_born_term() -> None:
    two = Geometry("two_path", 0.1, 0.4, 0.15, 0.0, 0.0)
    born = Geometry("born", 0.1, 0.4, 0.15)
    np.testing.assert_allclose(population_terms("sphere", Q, radius=5.0, radius_spread=0.2, geometry=two),
                               population_terms("sphere", Q, radius=5.0, radius_spread=0.2, geometry=born),
                               rtol=1e-4)  # two paths: direct quadrature; one path: the tables
    np.testing.assert_allclose(population_terms("sphere", Q, radius=5.0, radius_spread=0.0, geometry=two),
                               population_terms("sphere", Q, radius=5.0, radius_spread=0.0, geometry=born), rtol=1e-12)


def test_the_dwba_exit_angle_defaults_to_the_critical_angle() -> None:
    geometry = Geometry("dwba", 0.1, 0.4, None, delta=3e-6, beta=5e-8)
    assert geometry.exit_angle_deg == pytest.approx(math.degrees(math.sqrt(6e-6)))
    assert abs(geometry.fresnel(geometry.exit_angle_deg)) > 0.5


def test_fitting_formulas_on_a_fine_rule_are_reproduced() -> None:
    """earlier Fitting on 20001 nodes (its own default rule aliases: Physics and Methods 9.3)."""
    for radius, spread in ((3.0, 0.1), (6.0, 0.25)):
        np.testing.assert_allclose(
            population_terms("sphere", Q, radius=radius, radius_spread=spread, law="intensity_gauss")[0],
            old.sphere_form_factor_pd(Q, radius, radius * spread, n_samples=20001), rtol=1e-4)
    legacy = old.vertical_cylinder_form_factor_pd(Q, 4.0, 0.2, n_samples=20000, nsig=4.0)
    at_zero = old.vertical_cylinder_form_factor_pd(np.array([1e-9]), 4.0, 0.2, n_samples=20000, nsig=4.0)
    np.testing.assert_allclose(population_terms("vertical_cylinder", Q, radius=4.0, radius_spread=0.2)[0],
                               legacy / at_zero, rtol=1e-4)


def test_the_resolution_is_the_convolution_of_the_symmetric_cut() -> None:
    population = Population("sphere", 4.0, 0.15, distance=14.0, distance_spread=0.3)
    sigma = 0.02
    smeared = evaluate([population], Settings(q_resolution=sigma), Q)
    grid = np.linspace(-4.0, 4.0, 40001)
    raw = population_terms("sphere", np.abs(grid) + 1e-12, radius=4.0, radius_spread=0.15, distance=14.0,
                           distance_spread=0.3)[0]
    step = grid[1] - grid[0]
    brute = [np.sum(raw * np.exp(-0.5 * ((grid - x) / sigma) ** 2)) * step / (sigma * math.sqrt(2 * math.pi)) for x in Q]
    np.testing.assert_allclose(smeared, brute, rtol=1e-3)


def test_the_total_is_the_sum_of_its_parts() -> None:
    populations = [Population("sphere", 3.0, 0.2, scale=2.0, distance=10.0, distance_spread=0.2),
                   Population("vertical_cylinder", 9.0, 0.3, scale=0.5)]
    settings = Settings(background=0.01, tail_amplitude=5.0, tail_width=0.03, tail_exponent=2.5, k=1.5)
    total, parts = evaluate(populations, settings, Q, parts=True)
    np.testing.assert_allclose(total, sum(parts["populations"]) + parts["tail"] + parts["background"], rtol=1e-12)


def test_size_moments_by_number_and_by_intensity() -> None:
    report = sizes.size_report("lognormal", 5.0, 0.3, 6)
    exact = sizes.lognormal_moments(5.0, 0.3, 6)
    assert report["number"][0] == pytest.approx(exact["number"][0], rel=1e-4)
    assert report["intensity"][0] == pytest.approx(exact["intensity"][0], rel=1e-3)
    assert report["intensity"][1] == pytest.approx(0.3, rel=1e-2)
    narrow = sizes.size_report("number_gauss", 5.0, 0.1, 6)
    assert narrow["number"][0] == pytest.approx(5.0, rel=1e-6) and narrow["number"][1] == pytest.approx(0.1, rel=1e-3)
    assert sizes.size_report("intensity_gauss", 5.0, 0.4, 6)["number"] is None  # rises towards R = 0 by number


def test_conventions() -> None:
    assert conventions.sigma_d_from_sigma_nn(conventions.sigma_nn_from_sigma_d(0.13)) == pytest.approx(0.13)
    assert conventions.fit1d_w(20.0, 0.25) == pytest.approx(5.0)
    legacy = old.vertical_cylinder_form_factor_pd(np.array([1e-9]), 5.0, 0.2)[0]
    assert conventions.legacy_vertical_cylinder_scale(5.0, 0.2) == pytest.approx(legacy, rel=1e-9)


def test_invalid_input_is_refused() -> None:
    with pytest.raises(ValueError):
        population_terms("cube", Q, radius=1.0, radius_spread=0.1)
    with pytest.raises(ValueError):
        population_terms("sphere", Q, radius=-1.0, radius_spread=0.1)
    with pytest.raises(ValueError):
        Geometry("dwba", 0.1, 0.4, 0.2)  # no substrate
    with pytest.raises(ValueError):
        Population("sphere", 4.0, coupling="ssca")


def test_the_tables_are_the_direct_average_and_a_stale_file_is_not_used(tmp_path, monkeypatch) -> None:
    """The tabulated path and the direct quadrature are the same definition (resolved below q_cut, the envelope
    above); a prebuilt table is used only when it was built with the current ``PARAMETERS``."""
    from src.gimap.features.fitting.domain.cut_model import model, tables

    q = np.geomspace(0.02, 4.0, 300)
    for law in ("number_gauss", "lognormal"):
        for radius, spread in ((3.0, 0.05), (20.0, 0.3)):
            fast, fast_c = model.size_averages("sphere", q, radius=radius, radius_spread=spread, law=law, coherent=True)
            monkeypatch.setattr(tables, "usable", lambda law, rel: np.zeros(np.atleast_1d(rel).size, bool))
            slow, slow_c = model.size_averages("sphere", q, radius=radius, radius_spread=spread, law=law, coherent=True)
            monkeypatch.undo()
            np.testing.assert_allclose(fast, slow, rtol=1e-3 if spread > 0.25 else 2e-4)
            np.testing.assert_allclose(fast_c, slow_c, atol=1e-4 * float(slow.max()))
    monkeypatch.setattr(tables, "DATA", tmp_path)
    stale = tables.path("number_gauss", "sinc", tmp_path)
    np.savez(stale, parameters=np.array("{}"))
    assert tables._load("number_gauss", "sinc") is None



def test_born_at_a_given_qz():
    """The training simulator's Born term at a dimensionless q_z: the same paths as a beam at those angles."""
    geometry = Geometry.born_at(0.37)
    assert geometry.mode == "born" and geometry.qz_born == pytest.approx(0.37, rel=1e-12)
    assert geometry.sphere_terms()[0][1] == pytest.approx(0.37, rel=1e-12)
    assert Geometry.born_at(0.0) == Geometry()
    with pytest.raises(ValueError):
        Geometry.born_at(2.5)
