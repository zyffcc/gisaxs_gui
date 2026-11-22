"""Incident and reciprocal exit waves at the vacuum surface of a uniform film."""
from __future__ import annotations

import math

import numpy as np


REFRACTED_FILM_FORMULA = (
    "|ti tf|^2 (1-exp(-a*d))/a, a=2 Im(kzi+kzf), kz=k sqrt(sin^2(alpha)-2 delta+2i beta), "
    "t=2 k sin(alpha)/(k sin(alpha)+kz), relative to af=ai, no substrate reflection"
)


def _surface_wave(sine, delta, beta):
    """Return kz/k and the intensity transmission from vacuum into the film."""
    if not (math.isfinite(delta) and math.isfinite(beta) and beta >= 0.):
        raise ValueError("Film optics requires finite δ and nonnegative finite β.")
    sine = np.asarray(sine, dtype=float)
    valid = (sine > 0.) & (sine <= 1.)
    s = np.where(valid, sine, 0.)
    # With β >= 0, the principal root has Im >= 0 (the wave decays into the film).
    normal = np.sqrt(s*s - 2*delta + 2j*beta)
    amplitude = np.divide(2*s, s + normal, out=np.zeros_like(normal), where=valid)
    return normal, np.abs(amplitude)**2


def exit_transmission(sin_exit, delta, beta):
    """Reciprocal Fresnel intensity |t_f|², zero for rays outside the exit hemisphere."""
    return _surface_wave(sin_exit, delta, beta)[1]


def _depth_integral(rate, thickness):
    """Integral of exp(−rate*z) from 0 to thickness, including transparent rays."""
    return np.divide(-np.expm1(-rate * thickness), rate,
                     out=np.full_like(rate, thickness), where=rate > 0.)


class RefractedFilm:
    """Precompute the fields and depth decay rates for repeated thickness evaluations.

    The film is homogeneous. Only the transmitted wave at its upper surface
    enters each path. Wavelength is in Å and the intensity decay rates are in nm⁻¹.
    """

    def __init__(self, sin_exit, sin_in, delta, beta, wavelength_angstrom):
        if not 0. < sin_in <= 1.:
            raise ValueError("Film optics requires a positive incidence angle.")
        if not (math.isfinite(wavelength_angstrom) and wavelength_angstrom > 0.):
            raise ValueError("Film optics requires a positive finite wavelength.")
        sin_exit = np.asarray(sin_exit, dtype=float)
        self._valid = (sin_exit > 0.) & (sin_exit <= 1.)
        incident, self.incident_transmission = _surface_wave(sin_in, delta, beta)
        outgoing, self.exit_transmission = _surface_wave(sin_exit, delta, beta)
        k = 20 * np.pi / wavelength_angstrom
        self.mu_nm = 2*k * beta
        self.incident_decay_nm = 2*k * incident.imag
        self._rate = self.incident_decay_nm + 2*k * outgoing.imag
        self._specular_rate = 2 * self.incident_decay_nm

    def factor(self, thickness_nm=np.inf):
        """Depth-integrated intensity divided by its value at αf = αi.

        |ti|² cancels in this ratio. Zero thickness gives the thin-film limit,
        infinity the semi-infinite limit. Rays at or below the horizon give NaN.
        """
        if np.isnan(thickness_nm) or thickness_nm < 0.:
            raise ValueError("Film thickness must be nonnegative.")
        if np.isinf(thickness_nm):
            # For transparent propagating rays both integrals grow as d.
            depth_ratio = np.divide(self._specular_rate, self._rate,
                                    out=np.ones_like(self._rate), where=self._rate > 0.)
        elif thickness_nm == 0.:
            depth_ratio = 1.
        else:
            depth_ratio = (_depth_integral(self._rate, thickness_nm)
                           / _depth_integral(self._specular_rate, thickness_nm))
        return np.where(self._valid, self.exit_transmission / self.incident_transmission * depth_ratio, np.nan)
