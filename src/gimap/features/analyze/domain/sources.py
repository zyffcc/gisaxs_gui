"""Which pixels a reduced curve comes from, so the image can show it.

The masks repeat the selections of ``gisaxs.py`` and ``giwaxs.py`` from the region each curve
records (bands of rows and columns; χ sectors, q windows and q∥–qz boxes above the horizon),
so ``mask.sum()`` equals the pixels counted in the curve's bins.
"""

from __future__ import annotations

import numpy as np

from .giwaxs import GiwaxsMaps
from .models import Curve
from .regions import region_mask

GIWAXS_KEYS = ("radial", "in_plane", "out_of_plane", "azimuthal", "sector", "sector_chi", "box_q", "box_qz", "box_qpar")


def _band(shape: tuple[int, int], rows: tuple[int, int], columns: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[int(rows[0]):int(rows[1]), int(columns[0]):int(columns[1])] = True
    return mask


def curve_source_mask(curve: Curve, valid: np.ndarray, maps: GiwaxsMaps | None = None) -> np.ndarray:
    """Boolean mask (frame shape) of the pixels averaged into ``curve``."""
    valid = np.asarray(valid, dtype=bool)
    region = dict(curve.region)
    if curve.key in ("horizontal", "vertical"):
        return valid & _band(valid.shape, region["rows"], region["columns"])
    if "cut_region" in region:
        if maps is None:
            raise ValueError("GIWAXS curves need the q maps of the frame.")
        return region_mask(region["q"], region["chi_deg"], region["both_sides"], maps, valid & maps.above_horizon)
    if curve.key not in GIWAXS_KEYS:
        raise ValueError(f"No source region is known for the curve {curve.key!r}.")
    if maps is None:
        raise ValueError("GIWAXS curves need the q maps of the frame.")
    usable = valid & maps.above_horizon
    chi, q = maps.chi_deg, maps.q
    if curve.key == "radial":
        return usable
    if curve.key == "in_plane":
        return usable & (np.abs(chi) >= region["chi_deg"][0])
    if curve.key == "out_of_plane":
        return usable & (np.abs(chi) <= region["chi_deg"][1])
    if curve.key == "azimuthal":
        low, high = region["q_window"]
        return usable & (q >= low) & (q <= high) & (chi >= -90.0) & (chi <= 90.0)
    if curve.key in ("sector", "sector_chi"):
        chi_low, chi_high = region["chi_deg"]
        q_low, q_high = region["q"]
        return usable & (chi >= chi_low) & (chi <= chi_high) & (q >= q_low) & (q <= q_high)
    par_low, par_high = region["q_parallel"]
    z_low, z_high = region["qz"]
    return (
        usable
        & (maps.q_parallel >= par_low) & (maps.q_parallel <= par_high)
        & (maps.qz >= z_low) & (maps.qz <= z_high)
    )


def source_labels(masks: list[np.ndarray]) -> np.ndarray:
    """One uint8 image: 0 outside every mask, ``i + 1`` where mask ``i`` is the last to cover the pixel.

    Later masks are drawn on top, so pass the widest first (the full ring before its sectors).
    """
    if not masks:
        raise ValueError("No masks.")
    labels = np.zeros(masks[0].shape, dtype=np.uint8)
    for index, mask in enumerate(masks[:254]):
        labels[mask] = index + 1
    return labels


__all__ = ["GIWAXS_KEYS", "curve_source_mask", "source_labels"]
