"""NXcanSAS curve files (HDF5) in Fitting: the curve a source names, the unit of its q, the preview of Import Curves.

A file may hold several curves (one per SASdata group; GIMaP writes one SASentry per curve). The source's
``InSituSourceFrame.frame_index`` says which, counted from 0 in the order ``shared.nxcansas.read_nxcansas`` gives;
a plain path is the first. The unit of q is the one the file states on Q (``@units``): 1/angstrom and 1/nm are
read as they are, 1/m is converted to 1/nm (× 1e-9). A Q without a unit Fitting knows states none: the setting
for files without one applies, as for a text file. Idev (the dataset I's ``uncertainties`` names) is the σ.
Points the file's mask leaves out are not in the curve (the source record counts them); points without a finite
Q or I are left out as unreadable rows of a text file are (``rows_left_out``).

What each curve is (``curve_roles``): the role the file states (``SasCurve.role``); a file of fits GIMaP wrote before
it stated roles is known by its titles (``FIT_TITLES``: “…: the points fitted”, “…: the model”).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from src.gimap.shared.nxcansas import READ_SUFFIXES, q_unit, read_curve, read_nxcansas

from ...application.curve_sources import CurveTablePreview
from ...application.models import InSituSourceFrame
from ...domain import CurveData

UNIT_READ = {"angstrom": "angstrom", "nm": "nm", "m": "nm"}
"""Fitting's name of the unit a curve is read in, by the unit of Q in the file."""
SCALE = {"m": 1e-9}
"""Q in 1/m times this is Q in 1/nm."""
PREVIEW_ROWS = 10
FIT_TITLES = {": the points fitted": "data", ": the model": "model"}
"""The ends of the titles of Fitting's Save Data and Fit (and of a series' fits) before the files said the role."""


def curve_role(curve) -> str:
    """``data`` / ``model`` as the file says (``SasCurve.role``), else by the end of a title GIMaP wrote; empty."""
    if curve.role:
        return curve.role
    return next((role for end, role in FIT_TITLES.items() if str(curve.title).endswith(end)), "")


def is_nxcansas_file(path) -> bool:
    """A file Fitting reads as NXcanSAS (by its suffix; its contents are checked when it is read)."""
    return Path(path).suffix.lower() in READ_SUFFIXES


def stated_unit(path, index: int = 0) -> Optional[str]:
    """``"angstrom"`` / ``"nm"`` for the curve ``index`` of the file, ``None`` when its Q states no unit Fitting
    knows. ``OSError`` for a file that is not HDF5, ``ValueError`` for one without such a curve."""
    return UNIT_READ.get(q_unit(read_curve(path, index).q_unit) or "")


def load_curve(source: InSituSourceFrame, setting: str) -> CurveData:
    """The curve the source names; ``setting``: the unit of q for a Q that states none."""
    curves = read_nxcansas(source.path)
    if not 0 <= source.frame_index < len(curves):
        read_curve(source.path, source.frame_index)  # says why: no NXcanSAS curve, or not that many
    curve = curves[source.frame_index]
    unit = q_unit(curve.q_unit)
    q = np.asarray(curve.q, dtype=float) * SCALE.get(unit or "", 1.0)
    intensity = np.asarray(curve.intensity, dtype=float)
    sigma = None if curve.sigma is None else np.asarray(curve.sigma, dtype=float)
    masked = np.zeros(q.shape, dtype=bool) if curve.mask is None else np.asarray(curve.mask, dtype=bool)
    finite = np.isfinite(q) & np.isfinite(intensity)
    keep = finite & ~masked
    left_out = int(np.count_nonzero(~finite))
    record = {
        "path": str(source.path), "format": "NXcanSAS", "group": curve.group, "title": curve.title,
        "curve": int(source.frame_index) + 1, "curves_in_file": len(curves), "q_units": curve.q_unit,
        "intensity_units": curve.intensity_unit, "sigma_source": "none" if sigma is None else "Idev",
        "masked_points_left_out": int(np.count_nonzero(finite & masked)),
    }
    if unit == "m":
        record["q_converted"] = "Q in 1/m read as 1/nm (x 1e-9)"
    return CurveData(
        q=q[keep], intensity=intensity[keep], error=None if sigma is None else sigma[keep],
        q_source_unit=UNIT_READ.get(unit or "", setting), source_path=source.token,
        observation={"curve_source": record}, rows_left_out=left_out,
    )


def preview(path) -> CurveTablePreview:
    """What Import Curves shows of the file: its curves (one source each) and the first rows of the first."""
    path = Path(path)
    curves = read_nxcansas(path)
    if not curves:
        read_curve(path, 0)  # says that the file holds no NXcanSAS curve
    first = curves[0]
    names = (f"Q ({first.q_unit or 'no unit'})", f"I ({first.intensity_unit or 'no unit'})")
    columns = [first.q, first.intensity]
    if first.sigma is not None:
        names += ("Idev",)
        columns.append(first.sigma)
    rows = tuple(tuple(f"{value:.6g}" for value in row) for row in zip(*(column[:PREVIEW_ROWS] for column in columns)))
    return CurveTablePreview(
        path=path, column_names=names, sample_rows=rows, column_count=len(names), suggested_layout="single",
        unit=UNIT_READ.get(q_unit(first.q_unit) or ""), has_header=True,
        uncertainty_columns=(2,) if first.sigma is not None else (), intensity_columns=(1,),
        curve_titles=tuple(curve.title or f"curve {number}" for number, curve in enumerate(curves, 1)),
        curve_roles=tuple(curve_role(curve) for curve in curves),
    )


__all__ = ["FIT_TITLES", "SCALE", "UNIT_READ", "curve_role", "is_nxcansas_file", "load_curve", "preview",
           "stated_unit"]
