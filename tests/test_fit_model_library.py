"""The models for manual fitting (``domain.model_library``) against independent implementations:

* Percus–Yevick hard spheres: the Fourier transform of Wertheim's direct correlation function, integrated
  numerically, and the closed form of Kinning & Thomas as SasView's ``hardsphere`` writes it;
* core–shell spheres: SasView's ``core_shell_sphere`` amplitude written with scattering-length densities, averaged
  over a Gaussian of the core radius by a dense trapezoid rule;
* the 2D hexagonal paracrystal: BornAgain's formula (complex, integrated by adaptive quadrature), and BornAgain itself
  where it is installed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad

from src.gimap.features.fitting.domain import fit_model as fm
from src.gimap.features.fitting.domain import model_library as L
from src.gimap.features.fitting.domain.cut_model import population_terms


# -- Percus–Yevick ---------------------------------------------------------------------------------------------


def _wertheim(q: float, radius: float, fraction: float) -> float:
    """S = 1/(1 − n ĉ(q)), ĉ the 3D Fourier transform of c(r) = −(λ₁ − 6φλ₂ s + ½φλ₁ s³), s = r/σ < 1 (Wertheim 1963)."""
    sigma = 2.0 * radius
    lam1 = (1 + 2 * fraction) ** 2 / (1 - fraction) ** 4
    lam2 = -((1 + 0.5 * fraction) ** 2) / (1 - fraction) ** 4
    density = 6.0 * fraction / (math.pi * sigma**3)

    def integrand(r):
        s = r / sigma
        c = -(lam1 + 6 * fraction * lam2 * s + 0.5 * fraction * lam1 * s**3)
        return 4 * math.pi * r * r * c * (math.sin(q * r) / (q * r) if q * r > 1e-12 else 1.0)

    transform = quad(integrand, 0.0, sigma, epsabs=1e-13, epsrel=1e-12, limit=200)[0]
    return 1.0 / (1.0 - density * transform)


def _kinning_thomas(q: float, radius: float, fraction: float) -> float:
    """SasView's hardsphere.c (Kinning & Thomas 1984), its non-series branch."""
    d = (1.0 / (1.0 - fraction)) ** 2
    a = ((1.0 + 2.0 * fraction) * d) ** 2
    x = abs(q * radius * 2.0)
    x2, x4 = x * x, x**4
    b = -6.0 * fraction * ((1.0 + 0.5 * fraction) * d) ** 2
    g = 0.5 * fraction * a
    s, c = math.sin(x), math.cos(x)
    ff = ((g * ((4.0 * x2 - 24.0) * x * s - (x4 - 12.0 * x2 + 24.0) * c + 24.0) / x2
           + b * (2.0 * x * s - (x2 - 2.0) * c - 2.0)) / x + a * (s - x * c)) / x
    return 1.0 / (1.0 + fraction * 24.0 * ff / x2)


@pytest.mark.parametrize("fraction", [0.05, 0.2, 0.35, 0.5])
def test_percus_yevick_equals_the_transform_of_wertheims_direct_correlation(fraction):
    radius = 5.0
    q = np.array([1e-4, 0.02, 0.09, 0.1, 0.2, 0.45, 0.7, 1.3, 2.5])
    ours = L.hard_sphere(q, radius, fraction)[0]
    reference = np.array([_wertheim(value, radius, fraction) for value in q])
    assert ours == pytest.approx(reference, rel=1e-8, abs=1e-10)
    assert ours[0] == pytest.approx((1 - fraction) ** 4 / (1 + 2 * fraction) ** 2, rel=1e-6)  # compressibility


def test_percus_yevick_equals_sasviews_closed_form_and_is_smooth_where_the_series_takes_over():
    radius, fraction = 7.0, 0.3
    q = np.linspace(0.08, 3.0, 200)  # x = 2qR from 1.1: the closed form is accurate there
    ours = L.hard_sphere(q, radius, fraction)[0]
    assert ours == pytest.approx([_kinning_thomas(value, radius, fraction) for value in q], rel=1e-9)
    edge = L.SERIES_BELOW / (2 * radius)  # the two branches meet without a step
    near = L.hard_sphere(np.array([edge * (1 - 1e-9), edge * (1 + 1e-9)]), radius, fraction)[0]
    assert near[0] == pytest.approx(near[1], rel=1e-9)
    assert np.all(L.hard_sphere(q, radius, 0.0) == 1.0)


# -- core–shell spheres ----------------------------------------------------------------------------------------


def _sasview_core_shell(q, core, shell, sld_core, sld_shell, sld_solvent):
    """SasView core_shell_sphere: F = 3 V_c (ρc − ρs) j₁(qr_c)/(qr_c) + 3 V_s (ρs − ρ_solv) j₁(qr_s)/(qr_s)."""
    def term(radius, contrast):
        x = q * radius
        volume = 4.0 / 3.0 * math.pi * radius**3
        return 3.0 * volume * contrast * (np.sin(x) - x * np.cos(x)) / x**3

    return term(core, sld_core - sld_shell) + term(core + shell, sld_shell - sld_solvent)


def _sasview_forward(core, shell, sld_core, sld_shell, sld_solvent):
    """F(0) = V_c (ρc − ρs) + V_s (ρs − ρ_solv) (the limit of 3 j₁(x)/x is 1)."""
    volume = lambda radius: 4.0 / 3.0 * math.pi * radius**3  # noqa: E731
    return volume(core) * (sld_core - sld_shell) + volume(core + shell) * (sld_shell - sld_solvent)


def test_core_shell_equals_sasviews_amplitude_for_one_size_and_for_a_size_distribution():
    q = np.geomspace(0.02, 3.0, 300)
    sld_core, sld_shell, sld_solvent = 4.0, 1.5, 0.5
    ratio = (sld_core - sld_solvent) / (sld_shell - sld_solvent)
    amplitude = _sasview_core_shell(q, 6.0, 2.5, sld_core, sld_shell, sld_solvent)
    forward = _sasview_forward(6.0, 2.5, sld_core, sld_shell, sld_solvent)
    ours, _c = L.core_shell(q, 6.0, 0.0, 2.5, ratio)
    assert ours[0] == pytest.approx(amplitude**2 / forward**2, rel=1e-9)

    radii = np.linspace(6.0 * (1 - 4 * 0.1), 6.0 * (1 + 4 * 0.1), 4001)  # a Gaussian of the core radius by number
    weights = np.exp(-0.5 * ((radii - 6.0) / 0.6) ** 2)
    weights /= weights.sum()
    amplitudes = np.array([_sasview_core_shell(q, r, 2.5, sld_core, sld_shell, sld_solvent) for r in radii])
    forwards = np.array([_sasview_forward(r, 2.5, sld_core, sld_shell, sld_solvent) for r in radii])
    expected = weights @ amplitudes**2 / (weights @ forwards**2)
    mean = (weights @ amplitudes) ** 2 / (weights @ forwards**2)
    ours, coherent = L.core_shell(q, 6.0, 0.1, 2.5, ratio, coherent=True)
    assert ours[0] == pytest.approx(expected, rel=2e-5)
    assert coherent[0] == pytest.approx(mean, rel=2e-5, abs=1e-9)  # (C(0) = 1)


def test_core_shell_limits_are_spheres():
    q = np.geomspace(0.05, 2.0, 120)
    homogeneous = L.core_shell(q, 4.0, 0.0, 2.0, 1.0)[0][0]  # ratio 1: one sphere of the outer radius
    no_shell = L.core_shell(q, 4.0, 0.0, 0.0, 3.0)[0][0]
    for radius, curve in ((6.0, homogeneous), (4.0, no_shell)):
        sphere = population_terms("sphere", q, radius=radius, radius_spread=0.0)[0]
        assert curve == pytest.approx(sphere, rel=1e-9, abs=1e-14)


# -- the 2D hexagonal paracrystal ----------------------------------------------------------------------------


def _bornagain_formula(q: float, distance: float, spread: float) -> float:
    """BornAgain's Interference2DParacrystal for infinite domains, integrated over ξ: Re[(1 + φ)/(1 − φ)] per lattice
    vector (a₂ at 120°), φ = exp(i q·a) · exp(−q²ω²/2)."""
    omega = spread * distance

    def product(xi):
        p = math.exp(-0.5 * (q * omega) ** 2)
        out = 1.0
        for angle in (xi, xi + 2.0 * math.pi / 3.0):
            phi = p * complex(math.cos(q * distance * math.cos(angle)), math.sin(q * distance * math.cos(angle)))
            out *= ((1.0 + phi) / (1.0 - phi)).real
        return out

    return quad(product, 0.0, math.pi, limit=4000, epsabs=1e-13, epsrel=1e-11)[0] / math.pi


@pytest.mark.parametrize("distance,spread", [(20.0, 0.1), (12.0, 0.05), (35.0, 0.25)])
def test_hexagonal_paracrystal_equals_bornagains_formula(distance, spread):
    q = np.array([0.03, 0.15, 0.36, 0.6, 1.1, 2.0]) * 20.0 / distance
    ours = L.hexagonal_paracrystal(q, distance, spread)[0]
    reference = np.array([_bornagain_formula(value, distance, spread) for value in q])
    assert ours == pytest.approx(reference, rel=1e-6)
    assert L.hexagonal_paracrystal(np.array([50.0]), distance, spread)[0][0] == pytest.approx(1.0, abs=1e-6)


def test_hexagonal_paracrystal_equals_a_bornagain_simulation():
    ba = pytest.importorskip("bornagain")
    distance, spread, wavelength, alpha = 20.0, 0.1, 0.1, 0.2
    count = 60

    def simulate(paracrystal: bool):
        particle = ba.Particle(ba.RefractiveMaterial("p", (0.5, 0.5, 0.5), 1e-5, 1e-7), ba.Sphere(0.5 * ba.nm))
        if paracrystal:
            layout = ba.Paracrystal2D(particle, ba.HexagonalLattice2D(distance * ba.nm, 0.0), 0.0, 0.0, 0.0)
            layout.setIntegrationOverXi(True)
            pdf = ba.Profile2DGauss(spread * distance * ba.nm, spread * distance * ba.nm, 0.0)
            layout.setProbabilityDistributions(pdf, pdf)
        else:
            layout = ba.Dilute2D(0.001, particle)
        top = ba.Layer(ba.Vacuum())
        top.deposit2D(layout)
        sample = ba.Sample()
        sample.addLayer(top)
        sample.addLayer(ba.Layer(ba.Vacuum()))
        k = 2 * math.pi / wavelength
        exit_angle = math.radians(alpha)
        phi_max = math.asin(1.5 / (k * math.cos(exit_angle)))
        detector = ba.SphericalDetector(count, 1e-4, phi_max, 1, exit_angle - 1e-7, exit_angle + 1e-7)
        result = ba.ScatteringSimulation(ba.Beam(1.0, wavelength * ba.nm, alpha * ba.deg), sample, detector).simulate()
        values = np.asarray(result.npArray() if hasattr(result, "npArray") else result.intensities(), float).reshape(-1)
        phi = 1e-4 + (np.arange(count) + 0.5) * (phi_max - 1e-4) / count
        q = np.hypot(k * math.cos(exit_angle) * np.sin(phi), k * math.cos(exit_angle) * (np.cos(phi) - 1.0))
        return q, values

    q, structured = simulate(True)
    _q, dilute = simulate(False)
    density = 2.0 / (math.sqrt(3.0) * distance**2)  # BornAgain's lattice density over the dilute one
    ratio = structured / dilute * 0.001 / density
    assert L.hexagonal_paracrystal(q, distance, spread)[0] == pytest.approx(ratio, rel=2e-3)


# -- in Fitting's model --------------------------------------------------------------------------------------


def test_the_new_models_are_in_the_model_and_its_file_and_the_old_ones_are_unchanged():
    q = np.geomspace(0.05, 2.0, 80)
    plain = fm.new_component("sphere", radius=4.0, distance=15.0)
    assert plain.interference == "paracrystal" and set(plain.params) == {"Int", "R", "sigma_R", "D", "sigma_nn"}
    assert fm.component_form(plain, q) == pytest.approx(population_terms(
        "sphere", q, radius=4.0, radius_spread=0.2, distance=15.0, distance_spread=0.25)[0], rel=1e-12)
    shell = fm.new_component("core_shell_sphere", radius=4.0, interference="hard_sphere")
    assert shell.keys() == ("Int", "R", "sigma_R", "t_shell", "core_contrast", "R_hs", "phi")
    assert shell.value("R_hs") == pytest.approx(1.1 * (4.0 + 2.0))
    hexagonal = fm.with_interference(fm.new_component("vertical_cylinder", radius=3.0), "hex2d")
    model = fm.FitModel((shell, hexagonal))
    total = fm.evaluate(model, q)
    assert np.all(np.isfinite(total)) and np.all(total > 0)
    again = fm.model_from_dict(fm.model_to_dict(model))
    assert [c.interference for c in again.components] == ["hard_sphere", "hex2d"]
    assert fm.evaluate(again, q) == pytest.approx(total, rel=1e-12)
    assert "interference" not in fm.model_to_dict(fm.FitModel((plain,)))["components"][0]  # earlier files as they were
    with pytest.raises(ValueError, match="no equivalent"):
        fm.to_manual(model)
