"""Noise models for synthetic 1D curves."""

from __future__ import annotations

import numpy as np


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def add_noise(
    I_clean,
    rng: np.random.Generator,
    poisson_scale_min: float = 10.0,
    poisson_scale_max: float = 200.0,
    rel_noise_min: float = 0.001,
    rel_noise_max: float = 0.02,
):
    I_clean_safe = np.maximum(np.asarray(I_clean, dtype=np.float64), 1e-30)
    ref = max(float(np.median(I_clean_safe)), 1e-30)
    poisson_scale = _log_uniform(rng, poisson_scale_min, poisson_scale_max)
    expected_counts = np.clip(I_clean_safe / ref * poisson_scale, 0.0, 1e9)
    noisy_counts = rng.poisson(expected_counts)
    I_poisson = noisy_counts / poisson_scale * ref
    rel_sigma = float(rng.uniform(rel_noise_min, rel_noise_max))
    I_noisy = I_poisson * np.exp(rng.normal(0.0, rel_sigma, size=I_poisson.shape))
    sigma_poisson = np.sqrt(np.maximum(noisy_counts, 1.0)) / poisson_scale * ref
    sigma_total = np.sqrt(sigma_poisson**2 + (rel_sigma * I_noisy) ** 2)
    return np.clip(I_noisy, 1e-30, np.inf), np.clip(sigma_total, 1e-30, np.inf)
