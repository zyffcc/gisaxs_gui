"""Focused helpers for TOP-K prediction: prediction curve io."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from TrainSetBuild import schema
from TrainSetBuild.sampling import pad_2d, preprocess_curve
from Training.model import SlotQueryBase


def split_data_line(line: str):
    line = line.strip()
    if "\t" in line:
        return [part.strip() for part in line.split("\t") if part.strip()]
    if "," in line:
        return [part.strip() for part in line.split(",") if part.strip()]
    return line.split()


def load_curve(
    path: Path,
    drop_low_intensity_floor=False,
    low_intensity_floor_percentile=0.5,
    low_intensity_floor_factor=5.0,
):
    arr, names = load_numeric_table(path)
    original_n = int(arr.shape[0])
    if names:
        q_idx = find_column(names, ["q", "q_nm", "q_nm^-1", "q_1/nm", "x"], 0)
        i_idx = find_column(names, ["I", "intensity", "counts", "y"], 1)
        sigma_idx = find_column(names, ["sigma", "err", "error", "uncertainty", "dI"], -1)
    else:
        q_idx, i_idx, sigma_idx = 0, 1, 2 if arr.shape[1] >= 3 else -1
    if q_idx >= arr.shape[1] or i_idx >= arr.shape[1]:
        raise ValueError(f"Could not resolve q/I columns in {path}; names={names}")
    q = np.asarray(arr[:, q_idx], dtype=np.float64)
    I = np.asarray(arr[:, i_idx], dtype=np.float64)
    if sigma_idx >= 0 and sigma_idx < arr.shape[1] and sigma_idx not in (q_idx, i_idx):
        sigma_arr = np.asarray(arr[:, sigma_idx], dtype=np.float64)
    else:
        sigma_arr = np.maximum(0.05 * np.maximum(I, 1e-30), 1e-30)
    order = np.argsort(q)
    q, I, sigma_arr = q[order], I[order], sigma_arr[order]
    keep = (
        np.isfinite(q)
        & np.isfinite(I)
        & np.isfinite(sigma_arr)
        & (q > 0)
        & (I > 0)
        & (sigma_arr > 0)
    )
    finite_positive_n = int(np.sum(keep))
    floor_removed = 0
    floor = None
    if drop_low_intensity_floor:
        positive = I[keep]
        if positive.size > 0:
            floor = float(np.percentile(positive, float(low_intensity_floor_percentile)))
            if np.isfinite(floor) and floor > 0:
                before = int(np.sum(keep))
                keep = keep & (I > floor * float(low_intensity_floor_factor))
                floor_removed = before - int(np.sum(keep))
                if floor_removed:
                    print(
                        f"Low-intensity floor removed {floor_removed} points "
                        f"(percentile={low_intensity_floor_percentile}, factor={low_intensity_floor_factor}, floor={floor:.4g}).",
                        flush=True,
                    )
    if keep.sum() < 16:
        raise ValueError("Input curve has too few finite positive points.")
    debug = {
        "original_n_points": original_n,
        "after_finite_positive_n_points": finite_positive_n,
        "drop_low_intensity_floor": bool(drop_low_intensity_floor),
        "low_intensity_floor_percentile": float(low_intensity_floor_percentile),
        "low_intensity_floor_factor": float(low_intensity_floor_factor),
        "low_intensity_floor_value": None if floor is None else float(floor),
        "low_intensity_floor_removed_n_points": int(floor_removed),
        "after_low_intensity_floor_n_points": int(np.sum(keep)),
    }
    return q[keep], I[keep], sigma_arr[keep], debug


def token_is_float(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def load_numeric_table(path: Path):
    data_rows = []
    header = None
    comment_header = None
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                candidate = stripped.lstrip("#").strip()
                if candidate:
                    tokens = split_data_line(candidate)
                    if len(tokens) >= 2 and not all(token_is_float(t) for t in tokens[:2]):
                        comment_header = tokens
                continue
            tokens = split_data_line(stripped)
            if len(tokens) < 2:
                continue
            if all(token_is_float(t) for t in tokens[: min(3, len(tokens))]):
                data_rows.append([float(t) for t in tokens])
            elif header is None:
                header = tokens

    if not data_rows:
        raise ValueError(f"No numeric q/I rows found in {path}")
    width = min(len(row) for row in data_rows)
    if width < 2:
        raise ValueError(f"Need at least two numeric columns q and I in {path}")
    arr = np.asarray([row[:width] for row in data_rows], dtype=np.float64)
    names = header or comment_header
    # A comment such as ``# q (1/A) I err`` tokenizes to four labels for a
    # three-column table because the unit is separated from q.  Treat any
    # width mismatch as an ambiguous header instead of silently mapping I to
    # the uncertainty column.
    if names is not None and len(names) != width:
        names = None
    return arr, names


def _validate_model_contract(model_dir: Path, model, artifact: Path):
    manifest_path = model_dir / "manifest.json"
    config_path = model_dir / "model_config.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    )
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    expected_points = int(manifest.get("max_points", config.get("max_points", schema.MAX_POINTS)))
    expected_slots = int(manifest.get("max_slots", config.get("max_slots", schema.MAX_SLOTS)))
    expected_types = int(manifest.get("num_types", config.get("num_types", schema.NUM_TYPES)))
    actual = (schema.MAX_POINTS, schema.MAX_SLOTS, schema.NUM_TYPES)
    declared = (expected_points, expected_slots, expected_types)
    if declared != actual:
        raise RuntimeError(
            f"Model/schema mismatch: model declares max_points/max_slots/num_types={declared}, GUI inference expects {actual}."
        )
    expected_hash = str(manifest.get("sha256", "")).lower()
    if expected_hash and artifact.is_file():
        digest = hashlib.sha256()
        with artifact.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest().lower() != expected_hash:
            raise RuntimeError(f"Model checksum mismatch for {artifact}")
    input_names = {tensor.name.split(":", 1)[0].split("/")[-1] for tensor in model.inputs}
    missing_inputs = set(manifest.get("required_inputs", ())) - input_names
    output_names = set(getattr(model, "output_names", ()) or ())
    if isinstance(getattr(model, "output", None), dict):
        output_names.update(model.output)
    missing_outputs = set(manifest.get("required_outputs", ())) - output_names
    if missing_inputs or missing_outputs:
        raise RuntimeError(
            "Model architecture is incompatible: "
            f"missing inputs={sorted(missing_inputs)}, missing outputs={sorted(missing_outputs)}"
        )


def make_input(q, I, sigma_arr, cons):
    x, global_features = preprocess_curve(q, I, sigma_arr)
    n = min(len(q), schema.MAX_POINTS)
    mask = np.zeros(schema.MAX_POINTS, dtype=bool)
    mask[:n] = True
    batch = {
        "x": pad_2d(x[:n], schema.MAX_POINTS, 3)[None, ...],
        "point_mask": mask[None, ...],
        "global_features": global_features[None, ...],
    }
    for key, val in cons.items():
        batch[key] = val[None, ...]
    return batch


def find_column(names, aliases, default_idx):
    normalized = [normalize_col_name(n) for n in names]
    for alias in aliases:
        alias_norm = normalize_col_name(alias)
        for idx, name in enumerate(normalized):
            if name == alias_norm or alias_norm in name:
                return idx
    return default_idx


def normalize_col_name(name: str) -> str:
    return (
        name.strip()
        .lower()
        .lstrip("#")
        .replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
    )


def load_model(model_dir: Path, allow_unsafe_lambda: bool = False):
    model_dir = Path(model_dir)
    if model_dir.is_file():
        model_dir = model_dir.parent
    if model_dir.name == "saved_model" and (model_dir / "saved_model.pb").is_file():
        model_dir = model_dir.parent
    custom_objects = {"SlotQueryBase": SlotQueryBase}
    errors = []
    for candidate in [model_dir / "model.keras", model_dir / "saved_model"]:
        if candidate.exists():
            try:
                model = tf.keras.models.load_model(
                    candidate,
                    custom_objects=custom_objects,
                    compile=False,
                    safe_mode=not allow_unsafe_lambda,
                )
                _validate_model_contract(model_dir, model, candidate)
                print(f"Loaded and validated model artifact: {candidate}", flush=True)
                return model
            except ValueError as exc:
                message = str(exc)
                if "Lambda layer" in message and not allow_unsafe_lambda:
                    raise ValueError(
                        "Model contains Lambda layers and Keras safe deserialization blocked loading. "
                        "If you trust this model source, rerun with --allow_unsafe_lambda."
                    ) from exc
                errors.append((candidate, exc))
                print(
                    f"WARNING: failed to load {candidate}: {type(exc).__name__}: {exc}", flush=True
                )
            except Exception as exc:
                errors.append((candidate, exc))
                print(
                    f"WARNING: failed to load {candidate}: {type(exc).__name__}: {exc}", flush=True
                )
    if errors:
        detail = "\n".join([f"- {path}: {type(exc).__name__}: {exc}" for path, exc in errors])
        raise RuntimeError(f"No loadable model artifact found in {model_dir}. Tried:\n{detail}")
    raise FileNotFoundError(f"No saved_model or model.keras found in {model_dir}")
