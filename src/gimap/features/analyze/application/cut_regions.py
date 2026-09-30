"""Cut regions as a person places them: a click snapped to the peak there, a region snapped again,
and region sets as plain dicts (the form the AI tool ``set_cut_regions`` and cut-set files use).

A cut-set file is JSON: ``{"format": "gimap-cut-regions", "version": 1, "regions": [{name, q_min,
q_max, chi_min_deg, chi_max_deg, both_sides}, …]}``; ``q_min``/``q_max`` are ``null`` for every q.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..domain import GISAXS, CutRegion, Pick, pick_region, snap_region
from .models import FrameAnalysis
from .use_cases import AnalyzeFrame

CUT_SET_FORMAT = "gimap-cut-regions"


class PlaceRegions:
    """Thread-safe helpers over the cached q maps of ``AnalyzeFrame``."""

    def __init__(self, analyze_frame: AnalyzeFrame):
        self._analyze = analyze_frame

    def _inputs(self, analysis: FrameAnalysis):
        if analysis.reduction is None or analysis.reduction.kind == GISAXS or analysis.geometry is None:
            raise ValueError("Cut regions need a GIWAXS frame with a geometry.")
        maps = self._analyze.maps_of(analysis)
        usable = np.asarray(analysis.valid, dtype=bool) & maps.above_horizon
        return maps, analysis.data, usable

    def at_pixel(self, analysis: FrameAnalysis, x: float, y: float) -> tuple[float, float]:
        """``(q, χ)`` of a detector position (view coordinates: pixel ``i`` spans ``[i, i + 1)``)."""
        maps = self._analyze.maps_of(analysis)
        rows, columns = maps.q.shape
        row = int(np.clip(np.floor(float(y)), 0, rows - 1))
        column = int(np.clip(np.floor(float(x)), 0, columns - 1))
        return float(maps.q[row, column]), float(maps.chi_deg[row, column])

    def pick(self, analysis: FrameAnalysis, kind: str, q: float, chi: float, *, name: str, chi_band=None) -> Pick:
        maps, values, usable = self._inputs(analysis)
        return pick_region(kind, q, chi, maps, values, usable, name=name, chi_band=chi_band)

    def snap(self, analysis: FrameAnalysis, region: CutRegion) -> Pick:
        maps, values, usable = self._inputs(analysis)
        return snap_region(region, maps, values, usable)


def region_to_dict(region: CutRegion) -> dict[str, Any]:
    q_min, q_max = region.q_range if region.q_range is not None else (None, None)
    return {
        "name": region.name, "q_min": q_min, "q_max": q_max,
        "chi_min_deg": region.chi_range[0], "chi_max_deg": region.chi_range[1], "both_sides": region.both_sides,
    }


def regions_from_dicts(items: Iterable[Mapping[str, Any]]) -> list[CutRegion]:
    """``CutRegion`` s from dicts (``ValueError`` names the first bad one)."""
    regions = []
    for index, item in enumerate(items):
        try:
            q_min, q_max = item.get("q_min"), item.get("q_max")
            regions.append(CutRegion(
                str(item.get("name") or f"Region {index + 1}"),
                None if q_min is None or q_max is None else (float(q_min), float(q_max)),
                (float(item["chi_min_deg"]), float(item["chi_max_deg"])),
                bool(item.get("both_sides", True)),
            ))
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"Region {index + 1} is not valid: {exc}") from exc
    return regions


def cut_set(regions: Sequence[CutRegion]) -> dict[str, Any]:
    return {"format": CUT_SET_FORMAT, "version": 1, "regions": [region_to_dict(region) for region in regions]}


def regions_from_cut_set(data: Any) -> list[CutRegion]:
    """The regions of a cut-set file (or of a bare list of region dicts)."""
    if isinstance(data, Mapping):
        if data.get("format") not in (None, CUT_SET_FORMAT):
            raise ValueError("This is not a GIMaP cut-set file.")
        data = data.get("regions")
    if not isinstance(data, list):
        raise ValueError("A cut-set file holds a list of regions.")
    return regions_from_dicts(data)


__all__ = ["CUT_SET_FORMAT", "PlaceRegions", "cut_set", "region_to_dict", "regions_from_cut_set", "regions_from_dicts"]
