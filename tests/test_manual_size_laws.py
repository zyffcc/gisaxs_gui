"""Direct manual size averages against adaptive integrations of the sphere intensity by number."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import j1, roots_legendre, spherical_jn
from scipy.stats import gamma, lognorm, norm

from src.gimap.features.fitting.application.single_predict import predict_request
from src.gimap.features.fitting.domain import fit_model as fm
from src.gimap.features.fitting.domain.cut_model import manual_cylinder, shapes, sizes
from src.gimap.features.fitting.domain.fit_engine import FitData


def _expectation(law, width, f):
    # x is radius divided by the input size. Integrate in standard deviations for the normal laws and in x for Gamma.
    if law == "gaussian":
        return quad(lambda z: f(1 + width*z) * norm.pdf(z), max(-4, -1/width), 4, epsabs=1e-13)[0]
    if law == "lognormal_median":
        return quad(lambda z: f(math.exp(width*z)) * norm.pdf(z), -12, 12, epsabs=1e-13, limit=500)[0]
    dist = gamma(1 / width**2, scale=width**2)
    return quad(lambda x: f(x) * dist.pdf(x), 0, np.inf, epsabs=1e-13, limit=500)[0]


@pytest.mark.parametrize("law", ["gaussian", "lognormal_median", "schulz"])
@pytest.mark.parametrize("width", [0.08, 0.3, 0.5])
def test_manual_sphere_matches_independent_quadrature(law, width):
    radius = 3.0
    component = fm.new_component("sphere", radius=radius, structure=False, law=law)
    model = fm.FitModel((component,)).with_values({(0, "sigma_R"): width})
    q = np.array([0.0, 0.04, 0.45, 1.498, 3.5, 9.0])
    norm6 = _expectation(law, width, lambda x: x**6)
    expected = [1.0]
    for qi in q[1:]:
        def scattered(x):
            arg = qi * radius * x
            return x**6 * (3 * spherical_jn(1, arg) / arg)**2 if arg else 0.0
        expected.append(_expectation(law, width, scattered) / norm6)
    np.testing.assert_allclose(fm.evaluate(model, q), expected, rtol=1e-4, atol=0)
    # Compare the actual normalized node weights and the radius/spread in the record with number moments.
    r, weights = sizes.nodes(law, radius, width, power=6, panels=32)
    assert weights.min() >= 0 and weights.sum() == pytest.approx(1)
    n = _expectation(law, width, lambda x: 1.0)
    m = _expectation(law, width, lambda x: x) / n
    cv = math.sqrt(_expectation(law, width, lambda x: (x - m)**2) / n) / m
    np.testing.assert_allclose(fm.number_size(model.components[0]), [radius*m, cv], rtol=1e-4)
    np.testing.assert_allclose(sizes.moments(r, weights, 6)["number"], [[radius*m], [cv]], rtol=1e-4)


@pytest.mark.parametrize("law", ["gaussian", "lognormal_median", "schulz"])
def test_zero_width_is_one_size_and_predict_receives_its_original_gaussian(law):
    c = fm.new_component("sphere", structure=False, law=law)
    model = fm.FitModel((c,)).with_values({(0, "sigma_R"): 0.0})
    q = np.array([0.2, 0.5, 1.0])
    x = q * c.value("R")
    expected = (3 * spherical_jn(1, x) / x)**2
    saved = fm.model_to_dict(model)
    assert saved["schema"] == "gimap_fit_model_v3"
    np.testing.assert_allclose(fm.evaluate(fm.model_from_dict(saved), q), expected, rtol=1e-12)
    data = FitData.prepare(q, expected)
    assert predict_request(data, model, "model").payload["law"] == "number_gauss"


def test_legacy_lognormal_still_reads_mean_and_relative_width():
    c = fm.new_component("sphere", law="lognormal")
    saved = fm.model_to_dict(fm.FitModel((c,)))
    assert saved["schema"] == "gimap_fit_model_v2"
    old = fm.model_from_dict(saved).components[0]
    assert fm.number_size(old) == pytest.approx((5.0, 0.2), rel=1e-3)
    assert fm.parameter_info(old, "sigma_R").label == "σR/R"
    median = fm.new_component("sphere", law="lognormal_median")
    assert fm.parameter_info(median, "sigma_R").label == "σ(ln R)"


@pytest.mark.parametrize("law", ["gaussian", "lognormal_median", "schulz"])
def test_random_cylinder_matches_direct_radius_height_and_orientation_averages(law):
    radius, height = 2.7, 7.0
    q = np.array([0.001, 0.4, 1.5, 3.1])
    model = fm.FitModel((fm.new_component("cylinder", radius=radius, structure=False, law=law),))
    model = model.with_values({(0, "h"): height, (0, "sigma_R"): 0.5, (0, "sigma_h"): 0.3})
    nodes, weights = roots_legendre(600)

    def intensity_nodes(centre, width, power):
        if law == "schulz":
            x = (1 + nodes) * 12
            w = weights * gamma.pdf(x, 1 / width**2, scale=width**2)
        else:
            lo, hi = (max(-4, -1 / width), 4) if law == "gaussian" else (-10, 10)
            z = lo + (hi-lo)*(1+nodes)/2
            x = 1 + z*width if law == "gaussian" else np.exp(z*width)
            w = weights * norm.pdf(z)
        w = w * x**power
        return centre * x, w / w.sum()

    r, wr = intensity_nodes(radius, 0.5, 4)
    h, wh = intensity_nodes(height, 0.3, 2)
    u = (1 + nodes)/2
    expected = []
    for qi in q:
        x = qi*np.sqrt(1-u**2)[:, None]*r
        radial = ((2*j1(x)/x)**2) @ wr
        axial = (np.sinc(qi*u[:, None]*h / (2*np.pi))**2) @ wh
        expected.append(np.sum(weights * radial * axial) / 2)
    np.testing.assert_allclose(fm.evaluate(model, q), expected, rtol=1e-4)


@pytest.mark.parametrize("law", ["lognormal_median", "schulz"])
@pytest.mark.parametrize("power, amplitude", [(2, shapes.sinc_half), (4, shapes.disc_amplitude)])
def test_broad_cylinder_factor_at_high_q_matches_a_resolved_real_axis_integral(law, power, amplitude):
    width = 0.5
    distribution = lognorm(width) if law == "lognormal_median" else gamma(1/width**2, scale=width**2)
    # In units of the median / mean, integrate to 80 on a fine uniform composite Gauss rule.
    x, weight = roots_legendre(10)
    edges = np.linspace(0, 80, 16385)
    half = edges[1] / 2
    r = (edges[:-1, None] + half * (1+x)).ravel()
    w = np.broadcast_to(weight * half, (edges.size-1, x.size)).ravel() * distribution.pdf(r)
    grid = np.array([2.0, 26.0, 160.0, 400.0])
    f = amplitude(r[:, None] * grid)
    normp = np.sum(w * r**power)
    square = (w * r**power) @ (f*f) / normp
    mean = (w * r**(power/2)) @ f / np.sqrt(normp)
    actual, actual_mean = manual_cylinder._rotated_factor(grid, law, 1.0, width, power, coherent=True)
    np.testing.assert_allclose(actual, square, rtol=1e-4)
    np.testing.assert_allclose(actual_mean, mean, rtol=1e-4, atol=1e-8)
