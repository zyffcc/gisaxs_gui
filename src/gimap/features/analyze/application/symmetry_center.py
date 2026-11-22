"""Locate the direct-beam column from the symmetry of the selected horizontal GISAXS cut."""

from dataclasses import replace

import numpy as np

from src.gimap.shared.geometry.q_inverse import positions_of_q

from ..domain import GISAXS, SymmetryCenter, curve_source_mask, gisaxs_maps, symmetric_center_x
from .models import FrameAnalysis


def refine_center_by_symmetry(analysis: FrameAnalysis, *, cancelled=None) -> SymmetryCenter:
    geometry, reduction = analysis.geometry, analysis.reduction
    if analysis.kind != GISAXS or reduction is None or geometry is None:
        raise ValueError("Symmetry refinement needs a GISAXS frame with a geometry.")
    curve = reduction.curve("horizontal")
    if curve is None:
        raise ValueError("This frame has no horizontal cut.")
    if "gisaxs_cut" in curve.region:
        # Subtract the geometric offset of the incidence plane at the band height:
        # the measured mirror axis there is displaced from the direct beam.
        maps = gisaxs_maps(analysis.shape, geometry)
        band = curve_source_mask(curve, analysis.valid, maps)
        qz = np.mean(curve.region["qz_edges"])
        rod, _ = positions_of_q(geometry, 0.0, qz)
        result = symmetric_center_x(analysis.data, band, (0, band.shape[0]), float(rod), cancelled=cancelled)
        offset = geometry.beam_center_x_px - float(rod)
        return replace(result, x_px=result.x_px + offset, initial_x_px=geometry.beam_center_x_px,
                       coarse_x_px=result.coarse_x_px + offset)
    rows = tuple(int(value) for value in curve.region.get("rows", (0, 0)))
    return symmetric_center_x(analysis.data, analysis.valid, rows, geometry.beam_center_x_px, cancelled=cancelled)
