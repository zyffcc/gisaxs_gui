"""不导入 TensorFlow 的模型 artifact 与 manifest 校验。"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

from .errors import TensorFlowModelError


@dataclass(frozen=True)
class TensorFlowArtifact:
    path: Path
    artifact_type: str
    model_root: Path
    manifest: dict[str, Any] = field(default_factory=dict)


def _read_manifest(model_root: Path) -> dict[str, Any]:
    path = model_root / "manifest.json"
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise TensorFlowModelError(f"Invalid model manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TensorFlowModelError(f"Model manifest must contain a JSON object: {path}")
    return value


def discover_tensorflow_artifacts(path: Path) -> tuple[TensorFlowArtifact, ...]:
    candidate = Path(path).expanduser()
    found: list[tuple[Path, str, Path]] = []
    if candidate.is_file() and candidate.suffix.lower() == ".keras":
        found.append((candidate, "keras", candidate.parent))
    elif candidate.is_dir():
        if (candidate / "model.keras").is_file():
            found.append((candidate / "model.keras", "keras", candidate))
        for keras_file in sorted(candidate.glob("*.keras")):
            item = (keras_file, "keras", candidate)
            if item not in found:
                found.append(item)
        saved_subdir = candidate / "saved_model"
        if (saved_subdir / "saved_model.pb").is_file():
            found.append((saved_subdir, "saved_model", candidate))
        if (candidate / "saved_model.pb").is_file():
            found.append((candidate, "saved_model", candidate))
    artifacts = []
    for artifact_path, artifact_type, model_root in found:
        artifacts.append(
            TensorFlowArtifact(
                path=artifact_path.resolve(),
                artifact_type=artifact_type,
                model_root=model_root.resolve(),
                manifest=_read_manifest(model_root),
            )
        )
    return tuple(artifacts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_model_manifest(artifact: TensorFlowArtifact) -> None:
    if not artifact.path.exists():
        raise TensorFlowModelError(f"TensorFlow model artifact is missing: {artifact.path}")
    manifest = artifact.manifest
    expected = str(manifest.get("sha256") or "").strip().lower()
    if expected and artifact.path.is_file():
        actual = _sha256(artifact.path)
        if actual.lower() != expected:
            raise TensorFlowModelError(
                f"Checksum mismatch for {artifact.path.name}: expected {expected}, got {actual}."
            )
    required_inputs = manifest.get("required_inputs", ())
    required_outputs = manifest.get("required_outputs", ())
    for label, value in (("required_inputs", required_inputs), ("required_outputs", required_outputs)):
        if value is not None and not isinstance(value, (list, tuple)):
            raise TensorFlowModelError(f"Manifest field {label} must be a list.")


def resolve_tensorflow_artifact(path: Path) -> TensorFlowArtifact:
    artifacts = discover_tensorflow_artifacts(path)
    if not artifacts:
        raise TensorFlowModelError(
            f"No .keras or SavedModel artifact was found under {Path(path).expanduser()}."
        )
    artifact = artifacts[0]
    validate_model_manifest(artifact)
    return artifact
