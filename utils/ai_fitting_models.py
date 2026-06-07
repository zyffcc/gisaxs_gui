"""AI fitting model discovery and loading helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import json


@dataclass
class ModelInfo:
    name: str
    display_name: str
    model_dir: Path
    artifact_path: Path
    artifact_type: str
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    history_summary: Dict[str, Any] = field(default_factory=dict)


def default_ai_fitting_model_base_dirs(root: Path | None = None) -> List[Path]:
    root = Path.cwd() if root is None else Path(root)
    return [
        root / "modules" / "Fitting_1D_Model",
        root / "modules" / "Fitting_1D_model",
    ]


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        if path.is_file():
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    return {}


def _summarize_history(history: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for key in ("val_loss", "val_logrmse", "logRMSE", "loss"):
        values = history.get(key)
        if isinstance(values, list) and values:
            summary[key] = values[-1]
        elif values is not None:
            summary[key] = values
    return summary


def _first_known(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _display_name(name: str, artifact_type: str, config: Dict[str, Any], metadata: Dict[str, Any], history_summary: Dict[str, Any]) -> str:
    trained_k = _first_known(
        metadata.get("trained_K"),
        metadata.get("trained_k"),
        metadata.get("max_k"),
        metadata.get("K"),
        config.get("trained_K"),
        config.get("trained_k"),
        config.get("max_k"),
        config.get("K"),
    )
    max_points = _first_known(
        metadata.get("max_points"),
        metadata.get("n_points"),
        metadata.get("points"),
        config.get("max_points"),
        config.get("n_points"),
        config.get("points"),
    )
    parts = [name, artifact_type]
    if trained_k is not None:
        parts.append(f"K={trained_k}")
    if max_points is not None:
        parts.append(f"max_points={max_points}")
    loss = _first_known(
        metadata.get("val_loss"),
        metadata.get("val_logrmse"),
        metadata.get("logRMSE"),
        config.get("val_loss"),
        config.get("val_logrmse"),
        config.get("logRMSE"),
        history_summary.get("val_loss"),
        history_summary.get("val_logrmse"),
        history_summary.get("logRMSE"),
    )
    if loss is not None:
        try:
            parts.append(f"val={float(loss):.4g}")
        except Exception:
            parts.append(f"val={loss}")
    return " | ".join(parts)


def _model_info_for_artifact(model_dir: Path, artifact_path: Path, artifact_type: str) -> ModelInfo:
    config = _read_json(model_dir / "config.json")
    metadata = _read_json(model_dir / "metadata.json")
    history_summary = _summarize_history(_read_json(model_dir / "history.json"))
    name = model_dir.name if model_dir.is_dir() else artifact_path.stem
    return ModelInfo(
        name=name,
        display_name=_display_name(name, artifact_type, config, metadata, history_summary),
        model_dir=model_dir,
        artifact_path=artifact_path,
        artifact_type=artifact_type,
        config=config,
        metadata=metadata,
        history_summary=history_summary,
    )


def discover_model_in_path(path: Path) -> List[ModelInfo]:
    path = Path(path)
    infos: List[ModelInfo] = []
    seen: set[Path] = set()

    def add(model_dir: Path, artifact: Path, artifact_type: str) -> None:
        key = artifact.resolve() if artifact.exists() else artifact.absolute()
        if key in seen:
            return
        seen.add(key)
        infos.append(_model_info_for_artifact(model_dir, artifact, artifact_type))

    if path.is_file() and path.suffix.lower() == ".keras":
        add(path.parent, path, "keras")
        return infos

    if not path.is_dir():
        return infos

    if (path / "model.keras").is_file():
        add(path, path / "model.keras", "keras")

    for keras_file in sorted(path.glob("*.keras")):
        add(path, keras_file, "keras")

    saved_subdir = path / "saved_model"
    if (saved_subdir / "saved_model.pb").is_file() and (saved_subdir / "variables").is_dir():
        add(path, saved_subdir, "saved_model_subdir")

    if (path / "saved_model.pb").is_file() and (path / "variables").is_dir():
        add(path, path, "saved_model_root")

    return infos


def discover_ai_fitting_models(base_dirs: Iterable[Path]) -> List[ModelInfo]:
    models: List[ModelInfo] = []
    seen: set[Path] = set()
    for base_dir in base_dirs:
        base = Path(base_dir)
        if not base.exists():
            continue
        candidates = [base] if base.is_file() else [base, *[p for p in sorted(base.iterdir()) if p.is_dir() or p.suffix.lower() == ".keras"]]
        for candidate in candidates:
            for info in discover_model_in_path(candidate):
                key = info.artifact_path.resolve() if info.artifact_path.exists() else info.artifact_path.absolute()
                if key in seen:
                    continue
                seen.add(key)
                models.append(info)
    return models


def model_artifact_candidates(model_dir: Path) -> List[Path]:
    model_dir = Path(model_dir)
    candidates: List[Path] = []
    if model_dir.is_file() and model_dir.suffix.lower() == ".keras":
        candidates.append(model_dir)
        return candidates
    if not model_dir.is_dir():
        return candidates
    if (model_dir / "model.keras").is_file():
        candidates.append(model_dir / "model.keras")
    candidates.extend(p for p in sorted(model_dir.glob("*.keras")) if p not in candidates)
    if (model_dir / "saved_model" / "saved_model.pb").is_file():
        candidates.append(model_dir / "saved_model")
    if (model_dir / "saved_model.pb").is_file():
        candidates.append(model_dir)
    return candidates


def load_tensorflow_model_compatible(
    model_dir: Path,
    custom_objects: Dict[str, Any] | None = None,
    allow_unsafe_lambda: bool = True,
):
    """Load a Keras/SavedModel artifact and return ``(model, artifact_path)``.

    Raises RuntimeError with all attempted artifacts when loading fails.
    """
    import tensorflow as tf  # type: ignore

    attempts: List[Tuple[Path, str]] = []
    for candidate in model_artifact_candidates(Path(model_dir)):
        try:
            try:
                model = tf.keras.models.load_model(
                    str(candidate),
                    custom_objects=custom_objects,
                    compile=False,
                    safe_mode=not allow_unsafe_lambda,
                )
            except TypeError:
                model = tf.keras.models.load_model(
                    str(candidate),
                    custom_objects=custom_objects,
                    compile=False,
                )
            print(f"Loaded model artifact: {candidate}")
            return model, candidate
        except ValueError as exc:
            message = str(exc)
            if "Lambda layer" in message and not allow_unsafe_lambda:
                raise ValueError(
                    "Model contains Lambda layers and Keras safe deserialization blocked loading. "
                    "If you trust this model source, rerun with --allow_unsafe_lambda."
                ) from exc
            attempts.append((candidate, message))
        except Exception as exc:
            attempts.append((candidate, str(exc)))
            if (candidate / "saved_model.pb").is_file() if candidate.is_dir() else False:
                try:
                    loaded = tf.saved_model.load(str(candidate))
                    signature = loaded.signatures.get("serving_default")
                    if signature is None:
                        attempts.append((candidate, "SavedModel has no serving_default signature"))
                        continue

                    class SavedModelSignatureWrapper:
                        def __init__(self, fn):
                            self._fn = fn

                        def __call__(self, inputs, training=False):
                            del training
                            return self._fn(**inputs)

                    print(f"Loaded SavedModel serving signature: {candidate}")
                    return SavedModelSignatureWrapper(signature), candidate
                except Exception as sig_exc:
                    attempts.append((candidate, f"saved_model signature fallback failed: {sig_exc}"))

    detail = "\n".join(f"- {path}: {err}" for path, err in attempts) or "- no candidate artifacts found"
    raise RuntimeError(f"Failed to load AI fitting model from {model_dir}. Tried:\n{detail}")
