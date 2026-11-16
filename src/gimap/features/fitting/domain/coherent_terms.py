"""The coherent term |<F>|² of the decoupling approximation (DA), for each family of Fitting's model.

I = <|F|²> + |<F>|² (S − 1), the averages over the particles by number. The sphere and the random cylinder of
``scattering_model`` are normalised per size (F(0) = 1) on intensity weights w_j: by number p_j ∝ w_j / V_j², so,
divided by <V²> as P is, |<F>|² / <V²> = (Σ w_j F_j / V_j)² / Σ (w_j / V_j²). A single size gives F², so DA and LMA
agree there. The vertical cylinder keeps R⁴ in P: its weights already count particles.
"""

from __future__ import annotations

import numpy as np

from .scattering_model import _cylinder_radial_amplitude, _F_sphere_amp, _gaussian_grid, _orientations


def _coherent(weights, amplitudes, volumes):
    """(Σ w F / V)² / Σ (w / V²) along the size axis 1 of ``amplitudes`` (complex allowed)."""
    w = np.asarray(weights, float)
    v = np.asarray(volumes, float)
    mean = np.sum(amplitudes * (w / v)[np.newaxis, :], axis=1)
    return np.abs(mean) ** 2 / np.sum(w / v**2)


def sphere_coherent_pd(q, R, sigma_R, n_samples=25, nsig=4.0, paths=None):
    """|<F>|² of :func:`sphere_form_factor_pd` (or of :func:`sphere_two_path_pd` with ``paths`` =
    (q₊, q₋, k_f, |r|, arg r)), normalised like it."""
    q = np.asarray(q, dtype=float)
    Rs, W = _gaussian_grid(R, sigma_R, nsig=nsig, n=n_samples, clip_min=0.0)
    Rs = np.maximum(Rs, 1e-9)
    if paths is None:
        amplitudes = np.column_stack([_F_sphere_amp(q, Ri) for Ri in Rs])
    else:
        q_plus, q_minus, _k_f, refl_amp, refl_phase = paths
        q_p, q_m = np.hypot(q, q_plus), np.hypot(q, q_minus)
        r = refl_amp * np.exp(1j * refl_phase)
        amplitudes = np.column_stack([_F_sphere_amp(q_p, Ri) * np.exp(-1j * q_plus * Ri)
                                      + r * _F_sphere_amp(q_m, Ri) * np.exp(-1j * q_minus * Ri) for Ri in Rs])
    return _coherent(W, amplitudes, Rs**3)


def vertical_cylinder_coherent_pd(q, R, sigma_R, n_samples=25, nsig=3.0):
    """|<A>|² of :func:`vertical_cylinder_form_factor_pd`, whose weights count particles (R⁴ is in its P)."""
    from scipy.special import j1

    q = np.asarray(q, dtype=float)
    Rs, W = _gaussian_grid(R, float(R) * float(sigma_R), nsig=nsig, n=int(n_samples) + 1, clip_min=0.0)
    q_col = q[:, np.newaxis]
    denom = np.where(np.abs(q_col) < 1e-30, 1e-30, q_col)
    mean = np.sum(W * (Rs * j1(q_col * Rs) / denom), axis=1)
    return mean**2 * 1e-6


def cylinder_coherent_pd(q, R, sigma_R, h, sigma_h, n_R=13, n_h=13, nsig=4.0, n_orient=24):
    """|<F>|² of :func:`cylinder_form_factor_pd`: the amplitude averaged over radius, height and orientation."""
    q = np.asarray(q, dtype=float)
    Rs, WR = _gaussian_grid(R, sigma_R, nsig=nsig, n=n_R, clip_min=0.0)
    Hs, WH = _gaussian_grid(h, sigma_h, nsig=nsig, n=n_h, clip_min=0.0)
    Rs, Hs = np.maximum(Rs, 1e-9), np.maximum(Hs, 1e-9)
    alphas, weights = _orientations(n_orient)
    radial = _cylinder_radial_amplitude(q[:, None, None] * Rs[None, :, None] * np.sin(alphas)[None, None, :])
    axial = np.sinc(q[:, None, None] * Hs[None, :, None] * np.cos(alphas)[None, None, :] / 2.0 / np.pi)
    # by number: radius weights ∝ WR / R⁴, height weights ∝ WH / h² (V ∝ R² h), amplitudes ∝ V F
    radial_mean = np.sum(radial * (WR / Rs**2)[None, :, None], axis=1) / np.sqrt(np.sum(WR / Rs**4))
    axial_mean = np.sum(axial * (WH / Hs)[None, :, None], axis=1) / np.sqrt(np.sum(WH / Hs**2))
    return np.sum(radial_mean * axial_mean * weights[None, :], axis=1) ** 2


__all__ = ["cylinder_coherent_pd", "sphere_coherent_pd", "vertical_cylinder_coherent_pd"]
