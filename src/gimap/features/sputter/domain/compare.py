"""Several Analyze results side by side (pure Python): Compare Scans of the Sputter page.

A *compared scan* is one result of the analysis package (``client.ClientResult``) with the name the user gives it
(by default its scan and substrate), the user's choice of solution per window, and the model-free readings of the
package's report (``report.csv``: the film thickness from the fringes along the exit angle, the grain height from the
grain fringe, the optical density of the metal from its Yoneda peak, the lateral scale from the knee of the
horizontal cut; only the values the package marks as read from the data, not its estimates).

Every curve is a quantity against the nominal deposit of each window (nm: rate x time, the package's value) or the
time since the onset of the deposition: two scans sputtered at different rates are compared at the same amount of
metal. The growth table lists the package's growth descriptors of each scan (``process.DESCRIPTORS``) and the metal
amount at which each stage starts. GIMaP computes none of these values: it reads them and places them side by side.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, field
from typing import Optional

from .client import ClientResult, number
from .process import DESCRIPTORS, STAGE_NAMES

READ = "reading"
"""The source the report gives a value read from the data (not ``estimate``)."""
CURVES = (("R_nm", "Radius R", "nm", "model"), ("H_nm", "Height H", "nm", "model"),
          ("D_nm", "Distance D", "nm", "model"), ("gap_nm", "Gap", "nm", "model"),
          ("coverage", "Coverage", "", "model"), ("base_nm", "Continuous layer t_b", "nm", "model"),
          ("d_film_nm", "Film thickness (fringes)", "nm", READ),
          ("h_grain_nm", "Grain height (its fringe)", "nm", READ),
          ("yoneda_rel", "Optical density (α/α_c)²", "", READ),
          ("L_lateral_nm", "Lateral scale 2π/q", "nm", READ))
"""(key, name, unit, source) of the curves: ``model`` the solution shown of each window, ``reading`` the report."""
CURVE_NAMES = {key: name for key, name, _unit, _source in CURVES}
AXES = (("nominal", "Nominal deposit (nm)"), ("time", "Time since the onset (s)"))
"""(key, label) of the horizontal axis."""
STAGE_ROW = "{stage} starts at (metal amount, nm)"
"""A row of the growth table: the metal amount of the first window of a stage."""
NO_SCAN = "Tick at least one result to compare."
COMPARE_TEMPLATES = (STAGE_ROW, NO_SCAN)


@dataclass(frozen=True)
class ComparedScan:
    name: str
    result: ClientResult
    readings: dict = field(default_factory=dict)
    """Key -> ((time s, nominal nm, value), …) read from the report."""
    choices: dict = field(default_factory=dict)
    """The user's choice of solution per window (``ClientResult`` keys), as on the main screen."""

    @property
    def folder(self):
        return self.result.folder


def default_name(result: ClientResult) -> str:
    """The scan and the substrate (``scan · PEO``), or the folder's name."""
    scan = result.scan.removeprefix("gui_") or result.folder.parent.name or result.folder.name
    return f"{scan} · {result.substrate}" if result.substrate else scan


def report_readings(text: str) -> dict:
    """Key -> ((time s, nominal nm, value), …) of the model-free readings in the text of ``report.csv`` (the
    package's ``report.write_csv``): only values whose source is ``reading``. {} for an empty or foreign text."""
    try:
        rows = list(csv.DictReader(io.StringIO(text or "")))
    except csv.Error:
        return {}
    out: dict = {}
    for row in rows:
        time_s, nominal = number(row.get("t")), number(row.get("h_nom"))
        if time_s is None:
            continue
        for key, _name, _unit, source in CURVES:
            if source != READ or (row.get(f"{key}_source") or "").strip() != READ:
                continue
            value = number(row.get(key))
            if value is not None:
                out.setdefault(key, []).append((time_s, nominal, value))
    return {key: tuple(sorted(points)) for key, points in out.items()}


def _x(window, axis: str) -> Optional[float]:
    if axis == "time":
        return window.since_onset_s if window.since_onset_s is not None else window.time_s
    return window.nominal_nm


def series(scan: ComparedScan, key: str, axis: str = "nominal") -> tuple:
    """(x, values) of one quantity of one scan, in time order; a window without the quantity is left out."""
    xs, ys = [], []
    if dict((k, s) for k, _n, _u, s in CURVES).get(key) == READ:
        onset = _onset(scan.result)
        for time_s, nominal, value in scan.readings.get(key, ()):
            x = nominal if axis == "nominal" else (time_s - onset if onset is not None else time_s)
            if x is not None:
                xs.append(x)
                ys.append(value)
        return xs, ys
    if not scan.result.shows(key):
        return xs, ys
    for window in scan.result.windows:
        entry = window.chosen(scan.choices)
        value = entry.value(key) if entry is not None else None
        x = _x(window, axis)
        if value is not None and x is not None and not window.on_demand:
            xs.append(x)
            ys.append(value)
    return xs, ys


def _onset(result: ClientResult) -> Optional[float]:
    """The time of the onset (time minus time since the onset of a window that has both)."""
    for window in result.windows:
        if window.since_onset_s is not None:
            return window.time_s - window.since_onset_s
    return None


def curves(scans) -> list:
    """(key, name, unit, source) of the curves at least one scan has, in the order of ``CURVES``."""
    return [item for item in CURVES if any(series(scan, item[0])[0] or series(scan, item[0], "time")[0]
                                           for scan in scans)]


def growth_rows(scans) -> list:
    """(name, fields, unit, (value or None per scan)) of the growth table: the descriptors at least one scan has, then
    the metal amount at which each stage starts (name ``STAGE_ROW`` with ``fields`` {stage: the stage's name}; the
    name of a descriptor has no fields)."""
    rows = []
    for key, name, unit in DESCRIPTORS:
        values = tuple((scan.result.process.descriptors.get(key) if scan.result.process is not None else None)
                       for scan in scans)
        if any(value is not None for value in values):
            rows.append((name, {}, unit, values))
    stages = []
    for scan in scans:
        for stage in (scan.result.process.stages if scan.result.process is not None else ()):
            if stage.stage not in stages and stage.stage != "after":
                stages.append(stage.stage)
    for stage in stages:
        values = []
        for scan in scans:
            found = next((item for item in (scan.result.process.stages if scan.result.process is not None else ())
                          if item.stage == stage), None)
            values.append(found.h_start if found is not None else None)
        rows.append((STAGE_ROW, {"stage": STAGE_NAMES.get(stage, stage)}, "nm", tuple(values)))
    return rows


def compare_csv(scans, axis: str = "nominal") -> str:
    """Every curve of every scan as rows (scan, quantity, unit, source, x, value), after ``#`` lines that say where
    each scan's values come from."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    out.write("# GIMaP Sputter ▸ Compare Scans: values of the analysis package side by side; GIMaP computes none.\n")
    for scan in scans:
        result = scan.result
        out.write(f"# {scan.name}: folder {result.folder}; scan {result.scan}; metal {result.metal}; "
                  f"substrate {result.substrate}; model family {result.family}; mode {result.mode}\n")
    out.write("# model: the solution shown of each window (the user's choice, else the default); reading: the "
              "package's report.csv, values read from the data\n")
    label = dict(AXES)[axis]
    writer.writerow(["scan", "quantity", "unit", "source", label, "value"])
    for scan in scans:
        for key, name, unit, source in CURVES:
            for x, value in zip(*series(scan, key, axis)):
                writer.writerow([scan.name, name, unit, source, _plain(x), _plain(value)])
    return out.getvalue()


def growth_csv(scans) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    out.write("# GIMaP Sputter ▸ Compare Scans: growth descriptors of the analysis package (process analysis of each "
              "result); metal amounts in nm as the package gives them\n")
    writer.writerow(["quantity", "unit", *(scan.name for scan in scans)])
    for name, fields, unit, values in growth_rows(scans):
        writer.writerow([name.format(**fields), unit, *("" if value is None else _plain(value) for value in values)])
    return out.getvalue()


def _plain(value) -> str:
    return "" if value is None or not math.isfinite(float(value)) else f"{float(value):.6g}"


__all__ = ["AXES", "COMPARE_TEMPLATES", "CURVES", "CURVE_NAMES", "ComparedScan", "NO_SCAN", "READ", "STAGE_ROW",
           "compare_csv", "curves", "default_name", "growth_csv", "growth_rows", "report_readings", "series"]
