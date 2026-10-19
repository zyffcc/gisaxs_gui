"""Read SEM images and explicitly mapped particle-size CSV files."""

from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..application import SemCsvSettings, SemImage, SemMeasurements, SemParticle


def source_record(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


class LocalSemImageReader:
    def read(self, path: Path) -> SemImage:
        path = Path(path)
        with Image.open(path) as image:
            mode = image.mode
            source = np.asarray(image).copy()
            # Palette indices are not intensities; explicitly convert their rendered colors.
            if mode == "P":
                source = np.asarray(image.convert("RGB")).copy()
        if source.ndim == 3 and source.shape[-1] in {3, 4}:
            grayscale = np.asarray(source[..., :3], float) @ np.array([0.2126, 0.7152, 0.0722])
            transform = "RGB luminance=0.2126R+0.7152G+0.0722B; alpha ignored"
        elif source.ndim == 2:
            grayscale = np.asarray(source, float)
            transform = "stored grayscale intensity preserved"
        else:
            raise ValueError("SEM image must contain one grayscale or RGB image")
        return SemImage(grayscale, str(path.resolve()), {
            **source_record(path), "stored_shape": list(source.shape),
            "stored_dtype": str(source.dtype), "image_mode": mode,
            "grayscale_transform": transform, "row_zero": "top",
        })


def _is_numeric_row(row):
    try:
        for value in row:
            if value.strip():
                float(value)
        return True
    except ValueError:
        return False


def _read_table(path, skip_rows, delimiter):
    if not isinstance(skip_rows, int) or skip_rows < 0:
        raise ValueError("CSV skipped rows must be a non-negative integer")
    if delimiter == "\\t":
        delimiter = "\t"
    if len(delimiter) != 1:
        raise ValueError("CSV delimiter must be one character")
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        numbered = [(number, line) for number, line in enumerate(stream, 1)
                    if number > skip_rows and not line.lstrip().startswith("#")]
    reader = csv.reader((line for _, line in numbered), delimiter=delimiter)
    rows = []
    for row in reader:
        if any(v.strip() for v in row):
            rows.append((numbered[reader.line_num - 1][0], row))
    if not rows:
        raise ValueError("CSV contains no data rows")
    first = rows[0][1]
    if _is_numeric_row(first):
        columns = tuple(f"Column {index + 1}" for index in range(len(first)))
        header = False
    else:
        columns = tuple(value.strip() or f"Column {index + 1}" for index, value in enumerate(first))
        rows = rows[1:]
        header = True
    if len(set(columns)) != len(columns):
        raise ValueError("CSV header contains duplicate column names")
    return columns, rows, header


class LocalSemCsvReader:
    def inspect(self, path: Path, skip_rows=0, delimiter=",") -> tuple[str, ...]:
        return _read_table(path, skip_rows, delimiter)[0]

    def read(self, path: Path, settings: SemCsvSettings) -> SemMeasurements:
        if settings.quantity not in {"radius", "diameter"}:
            raise ValueError("Choose whether the CSV column contains radius or diameter")
        if settings.unit not in {"nm", "um", "px"}:
            raise ValueError("CSV units must be explicitly selected as nm, um or px")
        if settings.unit == "px":
            nmpp = settings.nm_per_pixel
            if nmpp is None or not np.isfinite(nmpp) or nmpp <= 0:
                raise ValueError("Pixel CSV values require a positive explicit calibration in nm/pixel")
            unit_factor = float(nmpp)
        else:
            unit_factor = 1000.0 if settings.unit == "um" else 1.0
        radius_factor = unit_factor / (2 if settings.quantity == "diameter" else 1)
        columns, rows, header = _read_table(path, settings.skip_rows, settings.delimiter)
        if settings.value_column not in columns:
            raise ValueError("Selected particle-size column was not found in the CSV")
        column = columns.index(settings.value_column)
        own_table = {"label", "radius_nm", "diameter_nm", "accepted", "rejection_reason"}.issubset(columns)
        accepted_column = columns.index("accepted") if own_table else None
        reason_column = columns.index("rejection_reason") if own_table else None
        particles = []
        for row_number, row in rows:
            try:
                value = float(row[column])
                radius = value * radius_factor
                valid = np.isfinite(radius) and radius > 0
            except (IndexError, ValueError, OverflowError):
                radius, valid = None, False
            reason = "" if valid else "missing_non_numeric_non_finite_or_non_positive_value"
            accepted = valid
            if own_table:
                flag = row[accepted_column].strip().casefold() if len(row) > accepted_column else ""
                if flag not in {"true", "1"}:
                    accepted = False
                    original_reason = row[reason_column] if len(row) > reason_column else ""
                    reason = ";".join(item for item in (reason, original_reason or "previously_rejected") if item)
            particles.append(SemParticle(
                label=row_number, radius_nm=float(radius) if valid else None,
                diameter_nm=float(2 * radius) if valid else None, accepted=bool(accepted),
                rejection_reason=reason,
            ))
        values = np.array([particle.radius_nm for particle in particles if particle.accepted])
        recorded = asdict(settings)
        recorded.update({"has_header": header, "unit_factor_to_nm": unit_factor,
                         "selected_value_to_radius_nm_factor": radius_factor,
                         "comment_lines": "lines starting with # ignored",
                         "respected_exported_accepted_flags": own_table,
                         "radius_definition": "explicit_csv_column_mapping"})
        provenance = {}
        if own_table:
            with Path(path).open("r", encoding="utf-8-sig") as stream:
                for line in stream:
                    if line.lstrip().startswith("#"):
                        try:
                            note = json.loads(line.lstrip()[1:].strip())
                        except (json.JSONDecodeError, ValueError):
                            continue
                        if isinstance(note, dict) and "source" in note and "settings" in note:
                            provenance["upstream_provenance"] = note
                            break
        return SemMeasurements(values, tuple(particles),
                               {**source_record(Path(path)), "kind": "sem_particle_csv", **provenance}, recorded,
                               {"detected_count": len(particles), "accepted_count": len(values),
                                "rejected_count": len(particles) - len(values)})
