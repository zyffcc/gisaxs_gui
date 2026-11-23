"""Equal-area coordinates for the level-3 reflection prior (concentric square-to-disk map)."""

from __future__ import annotations

import numpy as np


def _sector(x, y):
    """Dominant axis and the signed ratio of minor to major coordinate in [-1, 1]."""
    horizontal = np.abs(x) > np.abs(y)
    major, minor = np.where(horizontal, x, y), np.where(horizontal, y, x)
    ratio = np.divide(minor, major, out=np.zeros_like(major), where=major != 0)
    return horizontal, major, ratio


def from_square(values):
    """Map the uniform encoding square [-1,1]² to uniform area in the physical unit disk."""
    horizontal, radius, ratio = _sector(values[..., 0], values[..., 1])
    angle = (np.pi / 4) * ratio
    major, minor = radius * np.cos(angle), radius * np.sin(angle)
    return np.stack((np.where(horizontal, major, minor), np.where(horizontal, minor, major)), axis=-1)


def to_square(values):
    """Inverse map for a physical (Re r, Im r) used to centre a proposal or MCMC walker."""
    x, y = values[..., 0], values[..., 1]
    horizontal, major, ratio = _sector(x, y)
    radius = np.copysign(np.hypot(x, y), major)
    minor = radius * (4 / np.pi) * np.arctan(ratio)
    return np.stack((np.where(horizontal, radius, minor), np.where(horizontal, minor, radius)), axis=-1)
