"""Analyze ▸ Send Both Cuts to Fitting: the horizontal and the vertical cut of a GISAXS frame, for a fit of the two
together in Fitting (Physics and Methods 9.18).

The horizontal cut is written as Send to Fitting writes it (``<stem>_fit_input.dat``). The vertical cut goes to
``<stem>_fit_input_qz.dat`` — a name a folder's ``*_fit_input.dat`` does not match, so a series never takes it for a
frame — with q_z in Å⁻¹ and, in its ``# observation:`` line, where it was taken (``cut_geometry``): ``cut`` =
``vertical``, the incidence angle, the wavelength (nm), the in-plane q of its column band on the sample horizon
(nm⁻¹), the film's critical angle when q_z is refracted, and the exit angle of the horizontal cut's band centre.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

from src.gimap.shared.geometry.q_inverse import qy_on_horizon

from ..domain.gisaxs import exit_angle_deg_at_row
from ..domain.vertical_band import rod_columns
from .export_record import analysis_metadata, export_stem, fit_input_observation

GISAXS = "gisaxs"
NOT_GISAXS = "Both cuts are those of a GISAXS frame: this frame has no horizontal and vertical cut."
NO_VERTICAL = "The vertical cut of this frame is empty: set it in the Cuts step first."
NO_HORIZON = "The vertical band has no unique intersection with the sample horizon. Its in-plane q is undefined."
VERTICAL_SUFFIX = "_fit_input_qz.dat"


def cut_geometry(analysis) -> dict:
    """Where the frame's two cuts lie (module docstring); raises ``ValueError`` when the frame has no two cuts."""
    reduction, geometry = analysis.reduction, getattr(analysis, "geometry", None)
    if reduction is None or geometry is None or str(analysis.kind).lower() != GISAXS:
        raise ValueError(NOT_GISAXS)
    horizontal, vertical = reduction.curve("horizontal"), reduction.curve("vertical")
    if horizontal is None or horizontal.is_empty:
        raise ValueError(NOT_GISAXS)
    if vertical is None or vertical.is_empty:
        raise ValueError(NO_VERTICAL)
    row = horizontal.region.get("center_row")
    if row is None:
        rows = horizontal.region.get("rows") or (0, 0)
        row = 0.5 * (float(rows[0]) + float(rows[1]))
    columns = vertical.region["columns"]
    column = vertical.region.get("center_column", 0.5 * sum(columns))
    qy = qy_on_horizon(geometry, float(column))
    if not math.isfinite(qy):
        raise ValueError(NO_HORIZON)
    values = {
        "cut": "vertical",
        "alpha_i_deg": float(geometry.incidence_deg),
        "wavelength_nm": float(geometry.wavelength_angstrom) / 10.0,
        "q_parallel_nm": abs(float(qy)) * 10.0,
        "critical_angle_deg": float(getattr(geometry, "critical_angle_deg", 0.0) or 0.0),
        "horizontal_alpha_f_deg": float(exit_angle_deg_at_row(geometry, float(row))),
    }
    if "rows" in vertical.region:
        values["band_crosses_specular"] = rod_columns(geometry, vertical.region["rows"], columns) is not None
    return values


def write_both_cuts(export, analysis, destination, halves: Optional[str] = None) -> tuple[Path, Path, dict]:
    """(horizontal file, vertical file, ``cut_geometry``): the horizontal cut by ``export.fit_input`` (Analyze's
    ``ExportFrame``), the vertical one beside it by the same writer."""
    geometry = cut_geometry(analysis)
    if not all(math.isfinite(value) for value in geometry.values() if isinstance(value, float)):
        raise ValueError(NOT_GISAXS)
    horizontal = export.fit_input(analysis, destination, halves=halves)
    curve = analysis.reduction.curve("vertical")
    metadata = analysis_metadata(analysis, curve_keys=["fit_input_qz"])
    observation = {**fit_input_observation(analysis), "source": "native_detector_rows", **geometry}
    observation = {key: value for key, value in observation.items() if key != "halves"}
    path = Path(destination) / f"{export_stem(analysis)}{VERTICAL_SUFFIX}"
    from .export_record import _for_fitting  # the curve as Fitting takes it (q in Å⁻¹)

    metadata.update(observation=observation, files=[path.name])
    export.writer.write_record(path.with_suffix(".json"), metadata)
    vertical = export.writer.write_xy(_for_fitting(curve), path, metadata)
    return Path(horizontal), Path(vertical), geometry


__all__ = ["NOT_GISAXS", "NO_VERTICAL", "VERTICAL_SUFFIX", "cut_geometry", "write_both_cuts"]
