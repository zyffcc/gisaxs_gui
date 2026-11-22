"""The curves of a frame as NXcanSAS (``shared.nxcansas``), for SasView, SASfit and other NeXus programs.

One file per frame, ``<stem>_nxcansas.h5``, with one SASentry per curve. Only curves against a q go in: Q of
NXcanSAS is a momentum transfer (1/angstrom, 1/nm or 1/m), so I(χ) and a curve shown against 2θ are left out —
the CSV files keep them. Each entry holds:

* Q in 1/angstrom as Analyze computed it, I in 1/cm for absolute intensities, in arbitrary units otherwise.
  Idev is the statistical standard error of the point (``error_model`` of the record).
* the title says the frame, the curve and which q its Q is: a horizontal cut keeps its signed q_y (both halves
  of the cut), a vertical cut has q_z, a box I(q∥) q_par, the others |q|;
* the wavelength, the sample–detector distance, the pixel size and, for grazing incidence, αi (as the sample's
  pitch) when the frame has a geometry;
* the frame's record (``analysis_metadata`` plus the curve's key, title, axis and region) as the JSON ``data`` of
  an NXnote in SASprocess: the same record the ``_analysis.json`` beside a CSV export holds.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Sequence

from src.gimap.shared.export_text import export_text
from src.gimap.shared.nxcansas import ARBITRARY, PER_ANGSTROM, Instrument, SasCurve
from src.gimap.shared.version import SOFTWARE as RELEASE

from ..domain import Curve

Q_AXES = {
    "qy (Å⁻¹)": "q_y (signed: both halves of the cut, not |q|)",
    "qz (Å⁻¹)": "q_z",
    "q∥ (Å⁻¹)": "q_par (in the sample plane)",
    "q (Å⁻¹)": "|q|",
}
"""What the Q of a curve is, by the axis Analyze gives it (the curves against an angle are not among them)."""
SUFFIX = "_nxcansas.h5"
ERRORS = {
    "poisson": "Poisson: the square root of the counts summed in the bin over the number of pixels",
    "pixel_scatter": "the scatter of the bin's pixels about their mean over the square root of their number",
    "scaled_poisson": "Poisson carried through the intensity corrections",
}


def nxcansas_name(stem: str) -> str:
    """The file of a frame's curves: ``<stem>_nxcansas.h5``."""
    return f"{stem}{SUFFIX}"


def against_q(curve: Curve) -> bool:
    """The curve's x is a q in Å⁻¹ (an NXcanSAS Q), not an angle."""
    return curve.x_label in Q_AXES and not curve.is_empty


def instrument(metadata: Mapping[str, Any]) -> Instrument:
    """Wavelength, distance, pixel size and αi of the frame's geometry (nothing without one)."""
    geometry = metadata.get("geometry") or {}
    if not geometry:
        return Instrument(detector_name=str(metadata.get("detector_name") or ""))

    def number(key: str) -> Optional[float]:
        value = geometry.get(key)
        return None if value is None else float(value)

    pixels = (number("pixel_size_x_m"), number("pixel_size_y_m"))
    return Instrument(
        wavelength_angstrom=number("wavelength_angstrom"), distance_m=number("distance_m"),
        detector_name=str(metadata.get("detector_name") or ""),
        pixel_size_m=pixels if None not in pixels else None, incidence_deg=number("incidence_deg"),
    )


def sas_curves(curves: Sequence[Curve], metadata: Mapping[str, Any], *, run: str,
               written_by: str = "Analyze") -> list[SasCurve]:
    """The curves against q of ``curves`` as NXcanSAS curves of the frame ``run`` (its export stem), each with the
    frame's record ``metadata`` (``analysis_metadata``); ``written_by``: Analyze or the command line."""
    known = instrument(metadata)
    calibrated = bool((metadata.get("normalisation") or {}).get("standard_calibration"))
    relative = metadata.get("output_intensity_unit") == "relative"
    unit = "1/cm" if calibrated and not relative else ARBITRARY
    intensity_description = ("absolute differential scattering cross section (1/cm)" if calibrated
                             else "the mean detector counts per pixel of the point (arbitrary units)")
    if relative:
        intensity_description = "relative intensity after reference-peak scaling"
    error = ERRORS.get(str(metadata.get("error_model") or ""), "as the record's error_model says")
    out = []
    for curve in curves:
        if not against_q(curve):
            continue
        axis = Q_AXES[curve.x_label]
        record = {**dict(metadata), "curve": {"key": curve.key, "title": curve.title, "x_label": curve.x_label,
                                             "region": curve.region}}
        out.append(SasCurve(
            title=export_text(f"{run}: {curve.title}; Q is {axis}"),
            q=curve.x, intensity=curve.intensity, sigma=curve.sigma, q_unit=PER_ANGSTROM, intensity_unit=unit,
            run=run, sample=run, instrument=known,
            process="GIMaP Analyze" if written_by == "Analyze" else f"GIMaP Analyze ({written_by})",
            description=export_text(
                f"Reduced by {RELEASE}: {curve.title}. Q: {axis}, in 1/angstrom. I: {intensity_description}. "
                f"Idev: the standard error of that mean ({error}). The record "
                "of the frame (settings, geometry, corrections, inputs with SHA-256) is the NXnote gimap_record."),
            record=json.dumps(record, ensure_ascii=True, sort_keys=True, default=str),
            date=str(metadata.get("created") or ""),
        ))
    return out


__all__ = ["Q_AXES", "SUFFIX", "against_q", "instrument", "nxcansas_name", "sas_curves"]
