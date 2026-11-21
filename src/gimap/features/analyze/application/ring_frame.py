"""Use case: does the open frame show complete rings about the direct beam (a calibrant in transmission)?"""

from __future__ import annotations

from ..domain.ring_frame import rings_about_beam
from .models import FrameAnalysis


def ring_frame(analysis: FrameAnalysis) -> dict:
    """``{"complete_rings", "is_ring_frame", "rings"}`` of the frame about its beam centre; nothing changes."""
    if analysis is None or analysis.geometry is None:
        raise ValueError("Open a frame with a geometry first.")
    geometry = analysis.geometry
    return rings_about_beam(analysis.data, analysis.valid, geometry.beam_center_x_px, geometry.beam_center_y_px).to_dict()


__all__ = ["ring_frame"]
