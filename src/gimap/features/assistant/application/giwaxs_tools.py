"""GIWAXS tools of the catalog beyond the standard cuts: the person's cut regions.

A region is a q range × a χ range (optionally both sides ±χ, folded onto |χ|); each gives the
curves ``regionN`` (I(q)) and ``regionN_chi`` (I(χ)). The list goes through ``_tracked`` like every
other setting, so it can be previewed, applied and undone.
"""

from __future__ import annotations

from .models import ToolInputError, ToolOutcome

MAX_REGIONS = 12


def region_arguments(region: dict) -> dict:
    """A region as the tool and the status describe it (checked and completed)."""
    try:
        chi_min, chi_max = float(region["chi_min_deg"]), float(region["chi_max_deg"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ToolInputError("Each region needs chi_min_deg and chi_max_deg (degrees).") from exc
    both = bool(region.get("both_sides", True))
    if both and min(chi_min, chi_max) < 0:
        raise ToolInputError("With both_sides the χ range is on |χ| (0–90°); use both_sides=false for a signed range.")
    q_min, q_max = region.get("q_min"), region.get("q_max")
    if (q_min is None) != (q_max is None):
        raise ToolInputError("Give both q_min and q_max, or neither (all q).")
    return {
        "name": str(region.get("name") or "Region"), "q_min": None if q_min is None else float(q_min),
        "q_max": None if q_max is None else float(q_max), "chi_min_deg": chi_min, "chi_max_deg": chi_max,
        "both_sides": both,
    }


class GiwaxsToolsMixin:
    """Needs ``workbench``, ``results`` and ``_ok``."""

    def _tool_set_cut_regions(self, regions) -> ToolOutcome:
        if not isinstance(regions, list) or len(regions) > MAX_REGIONS:
            raise ToolInputError(f"regions must be a list of at most {MAX_REGIONS} regions.")
        cleaned = [region_arguments(region) for region in regions]
        status = self.workbench.set_cut_regions(cleaned)
        self.results.status = status
        names = ", ".join(region["name"] for region in cleaned) or "none"
        return self._ok(status, f"cut regions: {names}")


__all__ = ["GiwaxsToolsMixin", "MAX_REGIONS", "region_arguments"]
