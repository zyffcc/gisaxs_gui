"""Refine indexed cells on the native radial curves of individual detector frames."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import replace
from pathlib import Path

from src.gimap.shared.version import SOFTWARE

from ..domain import GIWAXS, TRANSMISSION
from ..domain.cell_tracks import ring_tracks, track_frame
from ..domain.crystal.refinement import refine_cell
from ..domain.series_map import RowFacts
from .export_record import analysis_metadata
from .record_values import plain
from .series_map import frame_key, row_label, series_row, series_times


def refine_series(frames, numbers, template, analyze, phases, results, source, *, check_cancel, progress):
    """Anchor tracking to the nearest selected frame to the indexing frame and follow each direction.

    The reductions use the selected settings, one physical frame at a time. Failed reads and fits
    keep their row. A frame for which the cell is undetermined never inherits the previous cell.
    """
    if len(frames) < 2:
        raise ValueError("Select at least two frames for cell refinement.")
    tracks = ring_tracks(results)
    if not tracks:
        raise ValueError("No indexed peaks are available for tracking.")
    systems = {result["phase"]: result["system"] for result in results if result.get("seeds")}
    cells = {phase.name: phase.reference_cell for phase in phases if phase.name in systems}
    reference = source["analysis"]
    first = numbers.get(frame_key(reference["source_file"], reference["frame_index"]), 1)

    def frame_number(index):
        return numbers[frame_key(*frames[index])]

    indices = range(len(frames))
    after = min(bisect_left(indices, first, key=frame_number), len(frames) - 1)
    anchor = min({after, max(0, after - 1)}, key=lambda index: (abs(frame_number(index) - first), index))
    angle = float(source["q_axis"].get("critical_angle_deg") or 0)
    template = replace(template, with_map=False, summed_frames=(),
                       corrections=replace(template.corrections, critical_angle_deg=angle or None))
    rows = [None] * len(frames)
    facts = [None] * len(frames)
    count = 0
    anchor_centres = {}
    for order in (indices[anchor:], reversed(indices[:anchor])):
        previous = dict(anchor_centres)
        for index in order:
            check_cancel()
            path, frame_index = frames[index]
            number = frame_number(index)
            request = replace(template, path=Path(path), frame_index=frame_index).with_scoped_roll()
            row = {"frame": number, "file": str(path), "frame_index": int(frame_index), "fits": {}, "tracks": []}
            facts[index] = RowFacts(number, str(path), frame_index)
            rows[index] = row
            try:
                analysis = analyze(request)
                label = row_label(analysis)
                facts[index] = RowFacts(number, str(path), frame_index, frame_count=analysis.frame_count,
                                        timestamp=label.timestamp, exposure_s=label.exposure_s,
                                        frame_time_s=label.frame_time_s)
                row.update(frame_count=analysis.frame_count, timestamp=label.timestamp or None,
                           frame_time_s=label.frame_time_s, exposure_s=label.exposure_s)
                record = analysis_metadata(analysis, with_reproduction=False)
                row["analysis"] = {k: record[k] for k in ("geometry", "source_files", "source_layout", "input_sha256",
                                   "corrections", "error_model", "binning", "measurement") if k in record}
                if analysis.kind not in (GIWAXS, TRANSMISSION) or analysis.geometry is None:
                    raise ValueError("This frame has no GIWAXS or transmission geometry.")
                if analysis.kind != reference["measurement"]:
                    raise ValueError("The frame geometry and the indexed frame use different measurement modes.")
                if abs(float(analysis.geometry.critical_angle_deg or 0) - angle) > 1e-9:
                    raise ValueError("The frame and the indexed peaks use different refraction corrections.")
                x, y, _ = series_row(analysis, "radial")
                sigma = analysis.reduction.curve("radial").sigma
                if sigma is None:
                    raise ValueError("This radial curve has no intensity uncertainties.")
                peaks, row["tracks"] = track_frame(x, y, sigma, tracks, previous, check_cancel)
                if index == anchor:
                    anchor_centres = dict(previous)
                for name, cell in cells.items():
                    row["fits"][name] = refine_cell(cell, systems[name], peaks.get(name, []))
            except InterruptedError:
                raise
            except (OSError, ValueError, RuntimeError) as error:
                row["reason"] = str(error)
            count += 1
            progress(count, len(frames))
    times, time_from = series_times(facts)
    for i, row in enumerate(rows):
        row["time_s"] = float(times[i]) if times is not None else None
    return {"kind": "GIMaP indexed cell series", "software_version": SOFTWARE, "indexing": source,
            "phases": [p.record() for p in phases if p.name in cells], "indexed_cells": results,
            "tracking": "Gaussian + linear background, intensity sigma, series-map centroid start; "
                        "half-window max(2.5 indexed FWHM, 5 native bin steps), clipped at midpoints between neighbouring peaks in the indexed frame",
            "anchor_frame": frame_number(anchor), "time_from": time_from, "frames": rows}


_COLUMNS = {"a": "a_A", "b": "b_A", "c": "c_A", "alpha": "alpha_deg", "beta": "beta_deg",
            "gamma": "gamma_deg", "V": "V_A3"}


def write_cell_series(writer, path, record):
    """One row per physical frame and phase; the full record retains the native peak-fit uncertainties."""
    path = Path(path)
    if path.suffix.lower() != ".csv":
        path = path.with_suffix(".csv")
    companion = path.with_suffix(".json")
    fitted = {key for row in record["frames"] for fit in row["fits"].values() for key in fit.get("values", {})}
    fitted.update(key for fit in record["indexed_cells"] for key in fit.get("values", {}))
    header = ["frame", "file", "frame_index", "time_s", "phase", "reason", "peaks_used", "chi2_red",
              "zero_shift_A^-1", "sigma_zero_shift_A^-1"]
    columns = []
    for key, unit_name in _COLUMNS.items():
        if key in fitted:
            columns.extend([("values", key, unit_name), ("errors", key, "sigma_" + unit_name)])
            if key in ("a", "b", "c", "V"):
                columns.extend([("strain", key, "strain_" + key), ("strain_errors", key, "sigma_strain_" + key)])
    header.extend(name for _, _, name in columns)
    rows = []
    for row in record["frames"]:
        for phase in record["phases"]:
            name = phase["name"]
            fit = row["fits"].get(name, {})
            data = [row["frame"], row["file"], row["frame_index"], row["time_s"], name,
                    row.get("reason", fit.get("reason", "")), len(fit.get("residuals", ())),
                    fit.get("chi2_red"), fit.get("zero_shift"), fit.get("zero_shift_error")]
            data.extend(fit.get(field, {}).get(key) for field, key, _ in columns)
            rows.append(["" if value is None else value for value in data])
    writer.write_record(companion, plain(record))
    writer.write_table(path, ["GIMaP indexed cell series; one-sigma uncertainties; blank = undetermined",
                              f"Indexed frame, CIF, reduction settings, peak tracks and residuals: {companion.name}"],
                       header, rows)
    return path
