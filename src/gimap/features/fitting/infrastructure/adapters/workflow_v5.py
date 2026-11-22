"""The 1D Predict window's job: every half of every curve with Predict parameters (the proposal network and its
posterior, ``posterior_job``), and the files of the run (request, per curve, summary). Also ``prepare_sides`` (the
halves of a curve as the fits take them) and ``read_curve``."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src.gimap.shared.reproduction import file_reference, replay_header
from src.gimap.shared.version import SOFTWARE
from ...application.workflow_v5 import FROM_REQUEST, q_unit_of_file, validate_options
from ...application import curve_intake
from .curve_units import file_q_unit
from .scattering_curve_files import load_xy_any
from .run_provenance import model_provenance


def read_curve(path: Path, *, details: bool = False):
    """``(q, intensity, sigma or None)`` of a text curve, read by the reader of the Fit page
    (``scattering_curve_files.load_xy_any``: any header lines, commas / tabs / semicolons / spaces, a fourth column
    of pixel counts), so that a file that opens there is fitted here. ``details``: also the number of rows inside
    the data that could not be read as numbers (left out, and said in the curve's notes)."""
    loaded = load_xy_any(str(path))
    values = loaded.q, loaded.I, loaded.err
    left_out = loaded.meta.get("rows_left_out", loaded.meta.get("bad_lines", 0))  # not numbers, or NaN / inf
    return (*values, int(left_out)) if details else values


def prepare_sides(q, intensity, sigma, options):
    q, y = np.asarray(q, float).reshape(-1), np.asarray(intensity, float).reshape(-1)
    if q.shape != y.shape or not np.isfinite(q).all() or not np.isfinite(y).all():
        raise ValueError("q and intensity must have equal lengths and contain finite values")
    q = q * (10.0 if options["q_unit"] == "A^-1" else 1.0)
    estimated = sigma is None
    if not estimated:
        sigma = np.asarray(sigma, float).reshape(-1)
        if sigma.shape != y.shape or not np.isfinite(sigma).all() or np.any(sigma <= 0):
            raise ValueError("Every supplied sigma must be finite and positive")
    result = []
    for side, sign in (("positive", 1), ("negative", -1)):
        if options["side"] not in ("both", side):
            continue
        indices = np.flatnonzero(q * sign > 0)
        if len(indices) == 0:
            continue
        indices = indices[np.argsort(np.abs(q[indices]), kind="stable")]
        negative, zero = curve_intake.nonpositive(y[indices])
        if estimated:  # no σ: the relative σ of Predict parameters, on the points ln I has (as the Fit page fits)
            indices = indices[y[indices] > 0]
        qq = np.abs(q[indices])
        side_sigma = curve_intake.relative_sigma(qq, y[indices]) if estimated and len(indices) else (
            np.empty(0) if estimated else sigma[indices])
        if not 8 <= len(indices) <= 1000:
            raise ValueError(
                f"{side}: need 8–1000 measured points (received {len(indices)}); no automatic interpolation"
            )
        if np.any(np.diff(qq) <= 0):
            raise ValueError(
                f"{side}: duplicate q values; combine replicates with their uncertainties first"
            )
        norm = float(np.max(y[indices]))
        if norm <= 0:
            raise ValueError(f"{side}: at least one positive intensity is required")
        result.append(
            dict(
                side=side,
                sign=sign,
                q=qq,
                observed=y[indices],
                sigma=side_sigma,
                normalizer=norm,
                indices=indices,
                sigma_estimated=estimated,
                sigma_level=curve_intake.sigma_level(qq, y[indices]) if estimated else None,
                negative=negative,
                zero=zero,
            )
        )
    if not result:
        raise ValueError("No nonzero q points in the selected side")
    return result


SIGMA_SOURCES = {False: "given with the curve",
                 True: "estimated: a relative sigma from the scatter of ln I about its trend"}
"""``sigma_source`` of a candidate row (``sigma_relative``: the estimate's level, None for a given σ)."""


def side_notes(item, unit, unreadable: int = 0) -> list[dict]:
    """What is to be said of one half as it was taken in (``application.curve_intake``; each note a dict with
    ``kind``, ``text`` and its numbers): an implausible q range in a unit the file did not state, the σ estimated
    for a file without one, the negative and zero intensities (left out without σ, kept with it), unreadable rows."""
    level = item.get("sigma_level")  # None: σ came with the curve
    return [note.as_dict() for note in curve_intake.intake_notes(
        item["q"], unit=unit.get("q_unit_read") or "", unit_from=unit.get("q_unit_from") or "", level=level,
        negative=item["negative"], zero=item["zero"], points=len(item["q"]), left_out=level is not None,
        unreadable=unreadable)]


def curve_units(sources, options) -> list[dict]:
    """The unit every curve's q is read in, before any is read (``application.workflow_v5.q_unit_of_file``: a file
    in the unit its header states, else the option ``q_unit``; the current cut in nm⁻¹, as Fitting sends it)."""
    units = []
    for source in sources:
        if source is None:
            units.append(dict(file="Current curve", q_unit_read=options["q_unit"], q_unit_from=FROM_REQUEST))
        else:
            units.append(dict(file=Path(source).name, source=source, **q_unit_of_file(file_q_unit, source, options)))
    return units


def run_workflow_job(payload, report, is_cancelled):
    """One isolated process for a full text batch, with per-file durable results. ``request.json`` and
    ``fitting_run_summary.json`` name the unit every curve's q was read in (``q_units``), as do the file's record
    in ``batch_summary.json`` and its candidates (``q_unit_read``, ``q_unit_from``)."""
    started = time.perf_counter()
    options = validate_options(payload.get("options", {}))
    output = Path(payload["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    sources = payload.get("files") or [None]
    units = curve_units(sources, options)
    model_record = model_provenance(options["method"])
    (output / "request.json").write_text(
        json.dumps({**payload, "options": options, "software_version": SOFTWARE,
                    "randomness": {"policy": "Predict parameters: numpy generator, fixed seed", "seed": 0},
                    "model": model_record, "q_units": units}, indent=2, allow_nan=False), encoding="utf-8"
    )
    from .posterior_job import fit_and_write

    records, all_rows, fit_records = [], [], []
    for index, source in enumerate(sources):
        if is_cancelled():
            raise RuntimeError("Prediction cancelled")
        tick = time.perf_counter()
        name = Path(source).name if source else "Current curve"
        directory = output / f"{index + 1:04d}_{Path(name).stem}"
        unit = {key: units[index][key] for key in ("q_unit_read", "q_unit_from")}
        try:
            if units[index].get("error"):
                raise ValueError(units[index]["error"])
            file_options = {**options, "q_unit": unit["q_unit_read"]}  # what every record of this curve says
            q, y, sigma, unreadable = (
                read_curve(Path(source), details=True)
                if source
                else (payload["q"], payload["intensity"], payload.get("sigma"), 0)
            )
            source_ref = file_reference(source) if source else None
            sides = prepare_sides(q, y, sigma, file_options)
            for item in sides:
                item["curve_notes"] = side_notes(item, unit, unreadable)
                item["source_reference"] = source_ref
            metadata = payload.get("observation_metadata", {}) if source is None else {}
            counts = None
            if metadata.get("source") == "native_cbf_columns" and "valid_pixel_counts" in metadata:
                counts = np.asarray(metadata["valid_pixel_counts"], float)
                if counts.shape != np.asarray(q).shape or not np.isfinite(counts).all() or np.any(counts <= 0):
                    raise ValueError("Native CBF valid_pixel_counts must match q and contain finite positive counts")
            for item in sides:
                item["observation_metadata"] = {k:v for k,v in metadata.items() if k != "valid_pixel_counts"}
                if counts is not None:
                    item["count"] = counts[item["indices"]]
            if payload.get("sigma_estimated"):
                for item in sides:
                    item["sigma_estimated"] = True
            rows = []
            for item in sides:
                side_folder = directory / item["side"]
                side_rows = fit_and_write(side_folder, item, file_options, report, is_cancelled)
                side_record = side_folder / "solutions.json"
                if side_record.is_file():
                    fit_records.append({"record_file": {**file_reference(side_record),
                                                        "path": side_record.relative_to(output).as_posix()}})
                for row in side_rows:
                    row.update(curve_notes=item["curve_notes"], sigma_source=SIGMA_SOURCES[item.get("sigma_level") is not None],
                               sigma_relative=item.get("sigma_level"))
                rows.extend(side_rows)
            for row in rows:
                row["file"] = name
                row.update(unit)
                row["file_seconds"] = time.perf_counter() - tick
                row["curve_logrmse_target"] = None
                row["curve_quality_passed"] = None
                row["curve_quality_note"] = (
                    "Observed-data residual includes measurement noise. Review peak positions "
                    "and overall shape; no mandatory logRMSE cutoff is applied."
                )
                row["quality_scope"] = "Full measured positive-intensity observations; excludes masked pixels and q=0. Not a composition/parameter accuracy guarantee."
            all_rows.extend(rows)
            records.append(
                dict(
                    file=name,
                    source=source,
                    status="complete",
                    seconds=time.perf_counter() - tick,
                    output_dir=str(directory),
                    candidates=len(rows),
                    curve_notes=[note for item in sides for note in item["curve_notes"]],
                    **unit,
                )
            )
        except Exception as exc:
            if is_cancelled() or not payload.get("files"):
                raise
            records.append(
                dict(
                    file=name,
                    source=source,
                    status="failed",
                    error=str(exc),
                    seconds=time.perf_counter() - tick,
                    **unit,
                )
            )
        (output / "batch_summary.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        (output / "top20_candidates.json").write_text(
            json.dumps(all_rows, indent=2, allow_nan=False), encoding="utf-8"
        )
        report(index + 1, len(sources), f"{index + 1}/{len(sources)} files finished — {name}", {})
    summary = dict(
        software_version=SOFTWARE, model=model_record,
        profile="Predict parameters (network + posterior)",
        runtime_seconds=time.perf_counter() - started,
        configured_candidates=len(all_rows),
        best_log_rmse=all_rows[0]["best_log_rmse"] if all_rows else None,
        exit_code=0 if all_rows else 1,
        q_units=units,
    )
    if fit_records:
        summary["reproduction"] = {**replay_header("batch", {}), "records": fit_records}
    (output / "fitting_run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return dict(output_dir=str(output), summary=summary, candidates=all_rows, records=records)
