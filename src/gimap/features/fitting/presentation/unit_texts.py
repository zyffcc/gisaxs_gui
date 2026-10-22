"""How Fitting's pages say the unit of q a curve file was read in (``application.curve_units``: the unit its header
states, else the setting “q unit of files without one in their header”): Single analysis' Curve step, In-situ
series' Curves step and selected frame, the 1D Predict window. The same words everywhere, translated."""

from __future__ import annotations

from src.gimap.app.presentation.i18n import tr, trf

UNIT_SHOWN = {"angstrom": "Å⁻¹", "nm": "nm⁻¹", "A^-1": "Å⁻¹", "nm^-1": "nm⁻¹"}
"""A unit as Fitting names it (``angstrom`` / ``nm``) or as 1D Predict's option does (``A^-1`` / ``nm^-1``)."""
UNIT_LABEL = "q unit of files without one in their header"
READ_IN = "q read in {unit}: {origin}"


def read_in(unit: str, origin: str) -> str:
    """“q read in Å⁻¹: stated in the file”, in the interface language; ``""`` when nobody said where it came from."""
    if not origin or unit not in UNIT_SHOWN:
        return ""
    return trf(READ_IN, unit=UNIT_SHOWN[unit], origin=tr(origin))


__all__ = ["READ_IN", "UNIT_LABEL", "UNIT_SHOWN", "read_in"]
