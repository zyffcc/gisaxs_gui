"""Format the frame and its resolved geometry for the Analyze Geometry step."""

from typing import Callable

from ..application import TECHNIQUE_NAMES
from .texts import flat_detector_summary, tilt_summary


def geometry_summary(analysis, translate: Callable[[str], str] = str) -> str:
    """The frame and its geometry in one line. ``translate`` turns each English template into the interface
    language before it is filled (the page passes ``tr``); names, numbers and units are kept. A frame whose
    file names no detector is described by its size alone."""
    if analysis is None:
        return translate("No frame loaded")
    rows, columns = analysis.shape
    head = " ".join(part for part in (analysis.detector_name, f"{rows}×{columns}") if part)
    geometry = analysis.geometry
    if geometry is None:
        return translate("{frame} · no geometry").format(frame=head)
    profile = analysis.resolution.profile
    origin = {
        "matched": translate("profile “{name}” (matched)").format(name=profile.name) if profile else "",
        "chosen": translate("profile “{name}”").format(name=profile.name) if profile else "",
        "file": translate("{name} beside the frame").format(name=profile.name) if profile else "",
    }.get(analysis.resolution.how, "")
    parts = [
        head, f"D = {geometry.distance_m * 1e3:.1f} mm", f"λ = {geometry.wavelength_angstrom:.4f} Å",
        f"αi = {geometry.incidence_deg:.3f}°", f"ψ = {geometry.sample_roll_deg:g}°",
        translate(TECHNIQUE_NAMES.get(analysis.kind, (analysis.kind or "").upper())), origin,
    ]
    flat = getattr(analysis, "flat_detector", None)  # a .poni tilt the flat geometry leaves out
    parts.extend((*tilt_summary(geometry, translate), *flat_detector_summary(flat, translate)))
    return " · ".join(part for part in parts if part)
