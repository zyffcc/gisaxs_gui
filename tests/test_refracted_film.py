import cmath
import math

import numpy as np
import pytest
from scipy.integrate import quad

from src.gimap.features.analyze.domain.intensity import film_absorption_factor
from src.gimap.features.analyze.domain.refracted_film import RefractedFilm


@pytest.mark.parametrize("d", [0., 180., np.inf])
def test_both_waves_far_above_critical_angle_recover_table_2(d):
    delta, beta, wavelength = 2e-6, 1.5e-7, 1.2
    si = math.sin(math.radians(18.))
    sf = np.sin(np.radians([12., 18., 25., 72.]))
    mu_per_nm = 40 * np.pi * beta / wavelength
    factor = RefractedFilm(sf, si, delta, beta, wavelength).factor(d)
    np.testing.assert_allclose(factor, film_absorption_factor(sf, si, mu_per_nm*d), rtol=6e-5)
    assert factor[1] == pytest.approx(1.)


def test_below_critical_intensity_matches_depth_quadrature():
    delta, beta, wavelength, depth = 3.e-6, 1.7e-7, .9, 110.
    si, sf = np.sin(np.radians([.44, .08]))
    k = 2 * np.pi / (.1 * wavelength)

    def scattered(s):
        normal = cmath.sqrt(s*s - 2*delta + 2j*beta)
        return lambda z: 2*s / (s + normal) * cmath.exp(1j*k*normal*z)

    incident, outgoing = scattered(si), scattered(sf)
    specular = quad(lambda z: abs(incident(z))**4, 0., depth)[0]
    expected = quad(lambda z: abs(incident(z)*outgoing(z))**2, 0., depth)[0] / specular
    waves = RefractedFilm(sf, si, delta, beta, wavelength)
    assert waves.factor(depth) == pytest.approx(expected, rel=1e-10)
    assert waves.factor(depth) < waves.factor(0.) / 8


def test_lossless_limit_and_horizon():
    sf = [np.nan, -.1, 0., .001, .02, 1.1]
    waves = RefractedFilm(sf, .02, 2e-6, 0., 1.)
    finite, thick = waves.factor(50.), waves.factor()
    assert np.isnan(finite[[0, 1, 2, 5]]).all()
    assert 0 < finite[3] < waves.factor(0.)[3]
    assert thick[3] == 0.  # finite evanescent depth divided by divergent specular depth
    assert finite[4] == thick[4] == pytest.approx(1.)
    below = RefractedFilm(.001, .001, 2e-6, 0., 1.)
    assert below.factor() == pytest.approx(1.)
