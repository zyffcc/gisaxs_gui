"""AI fitting model discovery and loading helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Mapping, Tuple
import json


class ModelRegistryError(RuntimeError):
    """Base error for friendly model discovery/loading failures."""


class ModelCompatibilityError(ModelRegistryError):
    """Raised when an artifact does not match its declared architecture."""


@dataclass(frozen=True)
class ModelContract:
    version: str = "legacy"
    architecture: str = "ML1DGISAXSSlotModel"
    max_points: int = 1000
    max_slots: int = 4
    num_types: int = 4
    supported_k: tuple[int, ...] = (1, 2, 3, 4)
    required_inputs: tuple[str, ...] = ()
    required_outputs: tuple[str, ...] = ()


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
    manifest: Dict[str, Any] = field(default_factory=dict)
    training_status: Dict[str, Any] = field(default_factory=dict)
    contract: ModelContract = field(default_factory=ModelContract)

    @property
    def model_id(self) -> str:
        return str(self.manifest.get("id") or self.name)

    @property
    def version(self) -> str:
        return self.contract.version


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


def _tuple_of_ints(value: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        return default
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError):
        return default
    return result or default


def _tuple_of_strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if str(item).strip())


def _contract_from_metadata(manifest: Mapping[str, Any], config: Mapping[str, Any], metadata: Mapping[str, Any]) -> ModelContract:
    return ModelContract(
        version=str(_first_known(manifest.get("version"), config.get("version"), "legacy")),
        architecture=str(_first_known(manifest.get("architecture"), config.get("architecture"), "ML1DGISAXSSlotModel")),
        max_points=int(_first_known(manifest.get("max_points"), config.get("max_points"), metadata.get("max_points"), 1000)),
        max_slots=int(_first_known(manifest.get("max_slots"), config.get("max_slots"), metadata.get("max_slots"), 4)),
        num_types=int(_first_known(manifest.get("num_types"), config.get("num_types"), metadata.get("num_types"), 4)),
        supported_k=_tuple_of_ints(manifest.get("supported_k"), (1, 2, 3, 4)),
        required_inputs=_tuple_of_strings(manifest.get("required_inputs")),
        required_outputs=_tuple_of_strings(manifest.get("required_outputs")),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    config = _read_json(model_dir / "model_config.json") or _read_json(model_dir / "config.json")
    metadata = _read_json(model_dir / "dataset_metadata.json") or _read_json(model_dir / "metadata.json")
    manifest = _read_json(model_dir / "manifest.json")
    training_status = _read_json(model_dir / "training_status.json")
    history_summary = _summarize_history(_read_json(model_dir / "history.json"))
    name = model_dir.name if model_dir.is_dir() else artifact_path.stem
    contract = _contract_from_metadata(manifest, config, metadata)
    display_name = _display_name(name, artifact_type, config, metadata, history_summary)
    if contract.version != "legacy":
        display_name += f" | v={contract.version}"
    state = training_status.get("state")
    if state:
        display_name += f" | {state}"
    return ModelInfo(
        name=name,
        display_name=display_name,
        model_dir=model_dir,
        artifact_path=artifact_path,
        artifact_type=artifact_type,
        config=config,
        metadata=metadata,
        history_summary=history_summary,
        manifest=manifest,
        training_status=training_status,
        contract=contract,
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


def validate_model_info(info: ModelInfo, verify_checksum: bool = True) -> None:
    if not info.artifact_path.exists():
        raise ModelRegistryError(f"Model artifact is missing: {info.artifact_path}")
    if info.contract.max_points != 1000:
        raise ModelCompatibilityError(
            f"{info.display_name} expects max_points={info.contract.max_points}; this GUI adapter requires 1000."
        )
    if info.contract.max_slots != 4 or info.contract.num_types != 4:
        raise ModelCompatibilityError(
            f"{info.display_name} declares max_slots={info.contract.max_slots}, num_types={info.contract.num_types}; expected 4/4."
        )
    expected = str(info.manifest.get("sha256") or "").strip().lower()
    if verify_checksum and expected and info.artifact_path.is_file():
        actual = file_sha256(info.artifact_path)
        if actual.lower() != expected:
            raise ModelCompatibilityError(
                f"Checksum mismatch for {info.artifact_path.name}: expected {expected}, got {actual}."
            )


def _tensor_names(tensors: Any) -> set[str]:
    if isinstance(tensors, Mapping):
        values = tensors.values()
    elif isinstance(tensors, (list, tuple)):
        values = tensors
    else:
        values = [tensors]
    names = set()
    for tensor in values:
        name = str(getattr(tensor, "name", "")).split(":", 1)[0]
        if name:
            names.add(name.split("/")[-1])
    return names


def validate_loaded_model(model: Any, info: ModelInfo) -> None:
    required_inputs = set(info.contract.required_inputs)
    if required_inputs:
        actual_inputs = set(getattr(model, "input_names", ()) or ()) or _tensor_names(getattr(model, "inputs", ()))
        missing = required_inputs - actual_inputs
        if missing:
            raise ModelCompatibilityError(
                f"Model {info.model_id} is missing required inputs: {', '.join(sorted(missing))}. "
                f"Available: {', '.join(sorted(actual_inputs)) or 'unknown'}."
            )
    required_outputs = set(info.contract.required_outputs)
    if required_outputs:
        actual_outputs = set(getattr(model, "output_names", ()) or ()) or _tensor_names(getattr(model, "outputs", ()))
        missing = required_outputs - actual_outputs
        if missing:
            raise ModelCompatibilityError(
                f"Model {info.model_id} is missing required outputs: {', '.join(sorted(missing))}. "
                f"Available: {', '.join(sorted(actual_outputs)) or 'unknown'}."
            )


class ModelRegistry:
    """Discover models once and lazily cache successfully loaded artifacts."""

    def __init__(self, base_dirs: Iterable[Path]) -> None:
        self.base_dirs = tuple(Path(path) for path in base_dirs)
        self._lock = RLock()
        self._models: Dict[str, ModelInfo] = {}
        self._loaded: Dict[str, Any] = {}

    def refresh(self) -> tuple[ModelInfo, ...]:
        with self._lock:
            models = discover_ai_fitting_models(self.base_dirs)
            self._models = {model.model_id: model for model in models}
            return tuple(models)

    def models(self) -> tuple[ModelInfo, ...]:
        with self._lock:
            if not self._models:
                return self.refresh()
            return tuple(self._models.values())

    def get(self, model_id: str) -> ModelInfo:
        with self._lock:
            if not self._models:
                self.refresh()
            try:
                return self._models[model_id]
            except KeyError as exc:
                raise ModelRegistryError(
                    f"Unknown AI fitting model {model_id!r}; available: {', '.join(self._models) or 'none'}"
                ) from exc

    def load(self, model_id: str, allow_unsafe_lambda: bool = True):
        with self._lock:
            if model_id in self._loaded:
                return self._loaded[model_id]
            info = self.get(model_id)
            validate_model_info(info)
            model, artifact = load_tensorflow_model_compatible(
                info.model_dir,
                custom_objects=None,
                allow_unsafe_lambda=allow_unsafe_lambda,
            )
            validate_loaded_model(model, info)
            loaded = (model, artifact, info)
            self._loaded[model_id] = loaded
            return loaded

    def clear_loaded(self) -> None:
        with self._lock:
            self._loaded.clear()


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
