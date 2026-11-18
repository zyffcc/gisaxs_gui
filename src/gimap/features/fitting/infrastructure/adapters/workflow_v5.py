"""The 1D Predict window's job: every half of every curve with Predict parameters (the proposal network and its
posterior, ``posterior_job``), and the files of the run (request, per curve, summary). Also ``prepare_sides`` (the
halves of a curve as the fits take them) and ``read_curve``."""

from __future__ import annotations

from src.gimap.shared.version import SOFTWARE

import json
import time
from pathlib import Path

import numpy as np


from ...application.workflow_v5 import FROM_REQUEST, q_unit_of_file, validate_options
from .curve_units import file_q_unit
from .run_provenance import model_provenance


def read_curve(path: Path):
    """Read numeric text with optional header, commas/tabs/spaces, no silent row loss."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            values = [float(x) for x in line.replace(",", " ").split()]
        except ValueError:
            if not rows and (
                any(word in line.lower() for word in ("q", "intensity", "sigma"))
                or line.lower().split() in (["x", "y"], ["x", "y", "err"])
            ):
                continue
            raise ValueError(f"Non-numeric row in {path.name}: {line[:80]}") from None
        if len(values) not in (2, 3, 4):
            raise ValueError("Curve files need q, intensity[, sigma[, pixels]] columns")
        rows.append(values)
    if not rows or len({len(row) for row in rows}) != 1:
        raise ValueError("Empty curve or inconsistent columns")
    a = np.asarray(rows, dtype=float)
    return a[:, 0], a[:, 1], a[:, 2] if a.shape[1] >= 3 else None


def prepare_sides(q, intensity, sigma, options):
    q, y = np.asarray(q, float).reshape(-1), np.asarray(intensity, float).reshape(-1)
    if q.shape != y.shape or not np.isfinite(q).all() or not np.isfinite(y).all():
        raise ValueError("q and intensity must have equal lengths and contain finite values")
    q = q * (10.0 if options["q_unit"] == "A^-1" else 1.0)
    estimated = sigma is None
    if estimated:
        floor = options["absolute_noise"] or max(float(np.max(np.abs(y))) * 0.001, 1e-12)
        sigma = np.hypot(options["relative_noise"] * np.abs(y), floor)
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
        qq = np.abs(q[indices])
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
                sigma=sigma[indices],
                normalizer=norm,
                indices=indices,
                sigma_estimated=estimated,
            )
        )
    if not result:
        raise ValueError("No nonzero q points in the selected side")
    return result


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

    records, all_rows = [], []
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
            q, y, sigma = (
                read_curve(Path(source))
                if source
                else (payload["q"], payload["intensity"], payload.get("sigma"))
            )
            sides = prepare_sides(q, y, sigma, file_options)
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
                rows.extend(fit_and_write(directory / item["side"], item, file_options, report, is_cancelled))
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
    (output / "fitting_run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return dict(output_dir=str(output), summary=summary, candidates=all_rows, records=records)
