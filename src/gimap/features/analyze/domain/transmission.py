"""SAXS/WAXS in transmission: full azimuthal I(q), I(χ) round the whole ring, a qy–qz map; no horizon.

A frame measured in transmission (a calibrant such as silver behenate or LaB6, or a sample in transmission) shows
rings that run round the direct beam, also below it: there is no sample surface, hence no horizon, no Yoneda band
and no missing wedge. Every valid pixel takes part.

Coordinates (the laboratory frame; the incidence angle plays no part):

* q = 4π sin θ / λ of every pixel (GIMaP's pixel→q mapping, tilted detectors included,
  ``shared/geometry/q_mapping``; the same |q| as ``transmission_map``);
* qy (horizontal, positive to the right of the image) and qz (vertical, positive up) are the components of
  q = k_f − k_i across the beam, from the same mapping at αi = 0;
* χ = atan2(qz, qy) in degrees: 0° to the right of the beam, 90° up, ±180° to the left, −90° down — the azimuth
  on the detector (``shared/geometry/q_mapping.TransmissionMap``). It is not the GIWAXS χ (0° along the surface
  normal).

The curves: ``radial`` I(q) over the whole ring; ``azimuthal`` I(χ) from −180° to 180° (1° bins) in a q window
(the strongest ring, or the one set); the cut regions drawn by the person (χ in the convention above); the q map
in qy–qz. Bins, errors and sparse-bin merging are those of the GIWAXS reduction (``giwaxs._profile``).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.gimap.shared.geometry import DetectorGeometry
from src.gimap.shared.geometry.q_mapping import grazing_q_region

from .binning import binned_mean_2d, map_grid_shape
from .curve_errors import PixelErrors
from .giwaxs import BLOCK_ROWS, CHI_BIN_DEG, MAP_SHAPE, GiwaxsMaps, _profile, _radial_bins, _pixel_q_step, strongest_ring
from .giwaxs import _in_two_theta
from .models import TRANSMISSION, X_AXIS_TWO_THETA, GiwaxsSettings, ReciprocalSpaceMap, Reduction
from .regions import region_curves, region_key

CHI_RANGE = (-180.0, 180.0)
CAKE_SHAPE = (720, 800)
"""(χ bins, q bins) of the transmission cake: 0.5° in χ, as the GIWAXS cake."""
QY_LABEL = "qy (Å⁻¹)"


def transmission_maps(shape: tuple[int, int], geometry: DetectorGeometry) -> GiwaxsMaps:
    """q, qy (in ``q_parallel``), qz and χ of every pixel in float32 (row blocks); every pixel is usable."""
    flat = replace(geometry, incidence_deg=0.0, critical_angle_deg=0.0, sample_roll_deg=0.0)
    rows, columns = int(shape[0]), int(shape[1])
    q = np.empty((rows, columns), dtype=np.float32)
    qy = np.empty_like(q)
    qz = np.empty_like(q)
    chi = np.empty_like(q)
    for start in range(0, rows, BLOCK_ROWS):
        stop = min(rows, start + BLOCK_ROWS)
        block = grazing_q_region(flat, (start, stop), (0, columns))
        q[start:stop] = block.q
        qy[start:stop] = block.qy
        qz[start:stop] = block.qz
        chi[start:stop] = np.degrees(np.arctan2(block.qz, block.qy))
    return GiwaxsMaps(q, qy, qz, chi, np.ones((rows, columns), dtype=bool), _pixel_q_step(q))


def reduce_transmission(
    image: np.ndarray,
    valid: np.ndarray,
    geometry: DetectorGeometry,
    settings: GiwaxsSettings | None = None,
    *,
    maps: GiwaxsMaps | None = None,
    with_map: bool = True,
    counts: bool | np.ndarray | PixelErrors = True,
) -> Reduction:
    """The transmission curves (see the module docstring); ``counts`` as in ``reduce_giwaxs``."""
    settings = settings or GiwaxsSettings()
    maps = maps or transmission_maps(image.shape, geometry)
    usable = np.asarray(valid, dtype=bool)
    warnings_out: list[str] = []
    if not usable.any():
        return Reduction(TRANSMISSION, (), {}, ("No valid pixels.",))
    q_low = float(maps.q[usable].min())
    q_high = float(maps.q[usable].max())
    bins = int(settings.bins) if settings.bins else _radial_bins(maps, usable, q_low, q_high)
    radial = _profile(
        "radial", "I(q)", "q (Å⁻¹)", maps.q, image, usable,
        low=q_low, high=q_high, bins=bins, region={"sector": "full", "chi_deg": CHI_RANGE}, counts=counts,
    )
    curves = [radial]
    window = settings.chi_q_window or strongest_ring(radial)
    if window is None:
        warnings_out.append("No diffraction ring found for I(χ); drag a q window to choose one.")
    else:
        low, high = sorted(float(value) for value in window)
        ring = usable & (maps.q >= low) & (maps.q <= high)
        curves.append(_profile(
            "azimuthal", f"I(χ), q = {low:.3f}–{high:.3f} Å⁻¹", "χ (°)", maps.chi_deg, image, ring,
            low=CHI_RANGE[0], high=CHI_RANGE[1], bins=int(round((CHI_RANGE[1] - CHI_RANGE[0]) / CHI_BIN_DEG)),
            region={"q_window": (low, high)}, merge=False, counts=counts,
        ))
    for index, region in enumerate(settings.regions):
        curves.extend(region_curves(
            region, region_key(index), maps, image, usable, q_low=q_low, q_high=q_high, bins=bins, counts=counts,
        ))
    for curve in curves:
        if curve.is_empty:
            warnings_out.append(f"{curve.title}: no valid pixels in this sector.")
    if settings.x_axis == X_AXIS_TWO_THETA:
        radial_keys = {"radial"} | {region_key(index) for index in range(len(settings.regions))}
        curves = [_in_two_theta(curve, geometry.wavelength_angstrom) if curve.key in radial_keys else curve
                  for curve in curves]
    rsm = None
    if with_map:
        qy, qz = maps.q_parallel[usable], maps.qz[usable]
        grid = binned_mean_2d(
            qy, qz, image[usable],
            x_range=(float(qy.min()), float(qy.max()) + 1e-12), y_range=(float(qz.min()), float(qz.max()) + 1e-12),
            shape=map_grid_shape(maps.q_parallel, maps.qz, usable, float(qy.max() - qy.min()),
                                 float(qz.max() - qz.min()), MAP_SHAPE),
        )
        rsm = ReciprocalSpaceMap(np.flipud(grid), (float(qy.min()), float(qy.max())), (float(qz.min()), float(qz.max())),
                                 x_label=QY_LABEL)
    markers = {
        "beam_center": (geometry.beam_center_x_px, geometry.beam_center_y_px),
        "chi_q_window": window,
        "radial_bins": {"bins": bins, "q_range": (q_low, q_high), "chosen": "set" if settings.bins else "auto"},
        "chi_convention": "detector azimuth: 0° right of the beam, 90° up (transmission)",
    }
    return Reduction(TRANSMISSION, tuple(curves), markers, tuple(warnings_out), rsm)


__all__ = ["CAKE_SHAPE", "CHI_RANGE", "QY_LABEL", "reduce_transmission", "transmission_maps"]
