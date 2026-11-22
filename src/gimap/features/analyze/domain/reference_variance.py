"""Independent reference measurements propagated through a dark/background subtraction and flat division."""

import numpy as np

from .flat_field import FlatField


def reference_variance(signal: np.ndarray, *, background: np.ndarray | None, background_scale: float,
                       dark: np.ndarray | None, dark_coefficient: float,
                       flat: FlatField | None, flat_dark_scale: float) -> np.ndarray | None:
    """Variance beyond sample noise for Y = J/F, J = I − s_B B + c_K K.

    The mean m used to normalise the flat is treated as fixed. The same measurement K appears in
    J and F = (D_ff − rK)/m, so its derivative must be combined before squaring. The independent
    Poisson terms for B, K and D_ff use their measured counts before subtracting references.
    """
    if background is None and dark is None and flat is None:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse = 1.0 if flat is None else 1.0 / flat.factor
        result = np.zeros_like(signal, dtype=np.float32)
        if background is not None:
            result += (background_scale * inverse)**2 * np.maximum(background, 0)
        dark_derivative = dark_coefficient * inverse
        if flat is not None:
            d_signal_d_factor = signal * inverse**2
            result += d_signal_d_factor**2 * flat.independent_variance
            dark_derivative = dark_derivative + d_signal_d_factor * (flat_dark_scale / flat.mean)
        if dark is not None:
            result += dark_derivative**2 * np.maximum(dark, 0)
    return result
