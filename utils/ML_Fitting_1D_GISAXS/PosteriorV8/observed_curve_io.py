"""Explicit real-curve ingestion for Posterior V8 inference and audits.

The neural encoder requires a positive uncertainty proxy as its third point
feature, while exact acceptance must distinguish measured uncertainties from
an imputed preprocessing value.  This module keeps those two meanings
separate: an unknown uncertainty can be imputed for encoder preprocessing but
never appears as ``ObservedCurve.sigma_log``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from .evaluation import ObservedCurve
from .preprocessing import DEFAULT_CONTRACT, PreprocessedCurve, preprocess_curve


OBSERVED_CURVE_IO_VERSION = "posterior_v8_explicit_real_curve_input_v1"
SUPPORTED_Q_UNITS = ("nm^-1", "angstrom^-1")
SUPPORTED_SIGMA_KINDS = ("missing", "absolute", "log")
DEFAULT_MISSING_RELATIVE_SIGMA = 0.015


@dataclass(frozen=True, eq=False)
class PreparedObservedCurve:
    """Paired exact-physics input, encoder tensors, and immutable provenance."""

    observed: ObservedCurve
    preprocessed: PreprocessedCurve
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.observed, ObservedCurve):
            raise TypeError("observed must be an ObservedCurve")
        if not isinstance(self.preprocessed, PreprocessedCurve):
            raise TypeError("preprocessed must be a PreprocessedCurve")
        if not isinstance(self.provenance, Mapping):
            raise TypeError("provenance must be a mapping")
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


def _vector(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    return result


def _q_scale(unit: str) -> tuple[str, float]:
    normalized = str(unit).strip().lower().replace("å", "angstrom")
    aliases = {
        "nm^-1": "nm^-1",
        "nm-1": "nm^-1",
        "1/nm": "nm^-1",
        "angstrom^-1": "angstrom^-1",
        "angstrom-1": "angstrom^-1",
        "a^-1": "angstrom^-1",
        "1/angstrom": "angstrom^-1",
    }
    try:
        canonical = aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            "q_unit must explicitly identify nm^-1 or angstrom^-1"
        ) from exc
    return canonical, 1.0 if canonical == "nm^-1" else 10.0


def _array_sha256(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(values, dtype="<f8")
    return sha256(canonical.tobytes()).hexdigest()


def prepare_observed_curve(
    q: Sequence[float] | np.ndarray,
    intensity: Sequence[float] | np.ndarray,
    sigma: Sequence[float] | np.ndarray | None = None,
    *,
    curve_id: str,
    source_kind: str = "real_cut_data",
    q_unit: str,
    sigma_kind: str | None = None,
    missing_relative_sigma: float = DEFAULT_MISSING_RELATIVE_SIGMA,
    q_range: Sequence[float] | None = None,
) -> PreparedObservedCurve:
    """Prepare one curve without promoting imputed uncertainty to evidence.

    ``sigma_kind='absolute'`` means intensity units. ``sigma_kind='log'``
    means a positive natural-log standard deviation, which is converted to an
    absolute proxy for encoder preprocessing. When sigma is absent, the
    relative proxy is used only by the encoder and exact acceptance receives
    ``sigma_log=None``.
    """

    q_array = _vector(q, "q")
    intensity_array = _vector(intensity, "intensity")
    if intensity_array.shape != q_array.shape:
        raise ValueError("q and intensity must have identical shapes")
    canonical_q_unit, scale = _q_scale(q_unit)
    q_nm = q_array * scale

    inferred_kind = "missing" if sigma is None else "absolute"
    kind = inferred_kind if sigma_kind is None else str(sigma_kind).strip().lower()
    if kind not in SUPPORTED_SIGMA_KINDS:
        raise ValueError(f"sigma_kind must be one of {SUPPORTED_SIGMA_KINDS}")
    if (sigma is None) != (kind == "missing"):
        raise ValueError("sigma must be absent exactly when sigma_kind is 'missing'")

    try:
        relative = float(missing_relative_sigma)
    except (TypeError, ValueError) as exc:
        raise ValueError("missing_relative_sigma must be finite and positive") from exc
    if not np.isfinite(relative) or relative <= 0.0:
        raise ValueError("missing_relative_sigma must be finite and positive")

    measured_sigma_log = None
    if kind == "missing":
        sigma_for_encoder = np.abs(intensity_array) * relative
    else:
        sigma_array = _vector(sigma, "sigma")
        if sigma_array.shape != q_array.shape:
            raise ValueError("sigma must have the same shape as q and intensity")
        if kind == "absolute":
            sigma_for_encoder = sigma_array
        else:
            sigma_for_encoder = np.abs(intensity_array) * sigma_array

    tiny = np.finfo(np.float64).tiny
    sigma_for_encoder = np.maximum(sigma_for_encoder, tiny)
    preprocessed = preprocess_curve(
        q_nm,
        intensity_array,
        sigma_for_encoder,
        q_range=q_range,
    )
    prepared_q, prepared_intensity, prepared_sigma = preprocessed.valid_arrays()
    if np.any(np.diff(prepared_q) <= 0.0):
        raise ValueError(
            "prepared q contains duplicates; upstream cut preparation must provide "
            "one strictly increasing value per paired observation"
        )
    if kind != "missing":
        measured_sigma_log = prepared_sigma / prepared_intensity

    observed = ObservedCurve(
        curve_id=curve_id,
        source_kind=source_kind,
        q=prepared_q,
        intensity=prepared_intensity,
        sigma_log=measured_sigma_log,
    )
    provenance = {
        "version": OBSERVED_CURVE_IO_VERSION,
        "curve_id": observed.curve_id,
        "source_kind": observed.source_kind,
        "source_q_unit": canonical_q_unit,
        "q_to_nm_inverse_scale": scale,
        "sigma_kind": kind,
        "exact_acceptance_has_measured_sigma_log": measured_sigma_log is not None,
        "encoder_uncertainty": {
            "source": "imputed_relative" if kind == "missing" else f"measured_{kind}",
            "missing_relative_sigma": relative if kind == "missing" else None,
        },
        "input": {
            "point_count": int(q_array.size),
            "q_sha256": _array_sha256(q_array),
            "intensity_sha256": _array_sha256(intensity_array),
            "sigma_sha256": None if sigma is None else _array_sha256(_vector(sigma, "sigma")),
        },
        "prepared": {
            "point_count": int(observed.q.size),
            "q_nm_inverse_sha256": _array_sha256(observed.q),
            "intensity_sha256": _array_sha256(observed.intensity),
            "sigma_log_sha256": (
                None if observed.sigma_log is None else _array_sha256(observed.sigma_log)
            ),
        },
        "preprocessing": dict(preprocessed.stats),
    }
    return PreparedObservedCurve(observed, preprocessed, provenance)


_TOKEN_SPLIT = re.compile(r"[\s,;]+")


def read_numeric_text(path: str | os.PathLike[str]) -> np.ndarray:
    """Read a numeric text table, allowing one leading non-numeric header."""

    source = Path(path)
    rows: list[list[float]] = []
    header_skipped = False
    width = None
    for line_number, original in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = original.strip()
        if not line or line.startswith("#"):
            continue
        tokens = [value for value in _TOKEN_SPLIT.split(line) if value]
        try:
            values = [float(value) for value in tokens]
        except ValueError as exc:
            if not rows and not header_skipped:
                header_skipped = True
                continue
            raise ValueError(f"non-numeric data at line {line_number}") from exc
        if len(values) < 2:
            raise ValueError(f"line {line_number} has fewer than two numeric columns")
        if width is None:
            width = len(values)
        elif len(values) != width:
            raise ValueError(f"inconsistent column count at line {line_number}")
        rows.append(values)
    if not rows:
        raise ValueError("text curve contains no numeric rows")
    return np.asarray(rows, dtype=np.float64)


def prepare_observed_text(
    path: str | os.PathLike[str],
    *,
    curve_id: str,
    q_unit: str,
    q_column: int = 0,
    intensity_column: int = 1,
    sigma_column: int | None = None,
    sigma_kind: str | None = None,
    missing_relative_sigma: float = DEFAULT_MISSING_RELATIVE_SIGMA,
) -> PreparedObservedCurve:
    """Load a text cut with explicit columns and attach file provenance."""

    source = Path(path).resolve()
    table = read_numeric_text(source)
    columns = (q_column, intensity_column) + (() if sigma_column is None else (sigma_column,))
    if any(isinstance(value, bool) or not isinstance(value, int) for value in columns):
        raise TypeError("column indices must be integers")
    if any(value < 0 or value >= table.shape[1] for value in columns):
        raise ValueError("requested column index is outside the text table")
    sigma = None if sigma_column is None else table[:, sigma_column]
    prepared = prepare_observed_curve(
        table[:, q_column],
        table[:, intensity_column],
        sigma,
        curve_id=curve_id,
        q_unit=q_unit,
        sigma_kind=sigma_kind,
        missing_relative_sigma=missing_relative_sigma,
    )
    provenance = dict(prepared.provenance)
    provenance["text_source"] = {
        "path": str(source),
        "sha256": sha256(source.read_bytes()).hexdigest(),
        "columns": {
            "q": q_column,
            "intensity": intensity_column,
            "sigma": sigma_column,
        },
    }
    return PreparedObservedCurve(
        prepared.observed,
        prepared.preprocessed,
        provenance,
    )


def write_reference_input(
    prepared: PreparedObservedCurve,
    output: str | os.PathLike[str],
) -> tuple[Path, Path]:
    """Atomically publish an exclusive NPZ plus auditable JSON sidecar."""

    if not isinstance(prepared, PreparedObservedCurve):
        raise TypeError("prepared must be a PreparedObservedCurve")
    target = Path(output).resolve()
    manifest = target.with_suffix(target.suffix + ".json")
    if target.exists() or manifest.exists():
        raise FileExistsError("reference input or its manifest already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "q": prepared.observed.q,
        "intensity": prepared.observed.intensity,
    }
    if prepared.observed.sigma_log is not None:
        arrays["sigma_log"] = prepared.observed.sigma_log
    descriptor = -1
    temporary = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp.npz", dir=target.parent
        )
        os.close(descriptor)
        descriptor = -1
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, target)
        temporary = None
        payload = {
            "schema": "gisaxs.posterior_v8.observed_curve_input/v1",
            "npz_file": target.name,
            "npz_sha256": sha256(target.read_bytes()).hexdigest(),
            "array_schema": {
                name: {"shape": list(value.shape), "dtype": value.dtype.str}
                for name, value in arrays.items()
            },
            "provenance": dict(prepared.provenance),
        }
        manifest.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        if target.exists() and not manifest.exists():
            target.unlink()
        raise
    return target, manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--curve-id", required=True)
    parser.add_argument("--q-unit", required=True)
    parser.add_argument("--q-column", type=int, default=0)
    parser.add_argument("--intensity-column", type=int, default=1)
    parser.add_argument("--sigma-column", type=int)
    parser.add_argument("--sigma-kind", choices=SUPPORTED_SIGMA_KINDS)
    parser.add_argument(
        "--missing-relative-sigma", type=float, default=DEFAULT_MISSING_RELATIVE_SIGMA
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    prepared = prepare_observed_text(
        args.input,
        curve_id=args.curve_id,
        q_unit=args.q_unit,
        q_column=args.q_column,
        intensity_column=args.intensity_column,
        sigma_column=args.sigma_column,
        sigma_kind=args.sigma_kind,
        missing_relative_sigma=args.missing_relative_sigma,
    )
    output, manifest = write_reference_input(prepared, args.output)
    print(json.dumps({"npz": str(output), "manifest": str(manifest)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MISSING_RELATIVE_SIGMA",
    "OBSERVED_CURVE_IO_VERSION",
    "PreparedObservedCurve",
    "main",
    "prepare_observed_curve",
    "prepare_observed_text",
    "read_numeric_text",
    "write_reference_input",
]
