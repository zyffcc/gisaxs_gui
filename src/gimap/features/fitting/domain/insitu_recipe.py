"""Immutable, serializable processing recipe for in-situ fitting series."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from types import MappingProxyType
from typing import Literal, Mapping


RecipeSource = Literal["single_analysis", "insitu_edit", "saved_recipe"]
TrackingMode = Literal["fixed", "detect_each_frame", "previous_success"]
FitInitialization = Literal["recipe_values", "previous_success", "ai_each_frame"]
RefinementMode = Literal["plot_only", "every_frame", "every_n", "quality_drop"]
FailurePolicy = Literal["fallback_recipe", "continue", "stop"]


def _freeze_json_value(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze_json_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _plain_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json_value(item) for item in value]
    return value


def _freeze_json_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    """Validate a mapping at the boundary and detach it from mutable UI state."""
    try:
        normalized = json.loads(
            json.dumps(_plain_json_value(value), ensure_ascii=False)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Recipe values must be JSON serializable") from exc
    frozen = _freeze_json_value(normalized)
    assert isinstance(frozen, Mapping)
    return frozen


def _plain_mapping(value: Mapping[str, object]) -> dict[str, object]:
    result = _plain_json_value(value)
    assert isinstance(result, dict)
    return result


@dataclass(frozen=True)
class InSituTrackingPolicy:
    """How geometry may follow a changing series without changing its definition."""

    center: TrackingMode = "fixed"
    yoneda: TrackingMode = "fixed"

    def __post_init__(self) -> None:
        allowed = {"fixed", "detect_each_frame", "previous_success"}
        if self.center not in allowed or self.yoneda not in allowed:
            raise ValueError("Unsupported in-situ tracking mode")

    def to_dict(self) -> dict[str, str]:
        return {"center": self.center, "yoneda": self.yoneda}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "InSituTrackingPolicy":
        return cls(
            center=str(value.get("center", "fixed")),  # type: ignore[arg-type]
            yoneda=str(value.get("yoneda", "fixed")),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class InSituFittingPolicy:
    """How fitting is initialized, refined and recovered for each frame."""

    initialization: FitInitialization = "previous_success"
    refinement: RefinementMode = "plot_only"
    refine_every_n: int = 1
    failure: FailurePolicy = "continue"

    def __post_init__(self) -> None:
        if self.initialization not in {
            "recipe_values",
            "previous_success",
            "ai_each_frame",
        }:
            raise ValueError("Unsupported in-situ fitting initialization")
        if self.refinement not in {
            "plot_only",
            "every_frame",
            "every_n",
            "quality_drop",
        }:
            raise ValueError("Unsupported in-situ refinement mode")
        if int(self.refine_every_n) < 1:
            raise ValueError("refine_every_n must be at least one")
        if self.failure not in {"fallback_recipe", "continue", "stop"}:
            raise ValueError("Unsupported in-situ failure policy")

    def to_dict(self) -> dict[str, object]:
        return {
            "initialization": self.initialization,
            "refinement": self.refinement,
            "refine_every_n": self.refine_every_n,
            "failure": self.failure,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "InSituFittingPolicy":
        return cls(
            initialization=str(  # type: ignore[arg-type]
                value.get("initialization", "previous_success")
            ),
            refinement=str(value.get("refinement", "plot_only")),  # type: ignore[arg-type]
            refine_every_n=int(value.get("refine_every_n", 1)),
            failure=str(value.get("failure", "continue")),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class InSituProcessingRecipe:
    """Versioned scientific configuration shared by Live, Review and Batch."""

    version: int
    created_at: str
    source: RecipeSource
    experiment_setup: Mapping[str, object] = field(default_factory=dict)
    preprocessing: Mapping[str, object] = field(default_factory=dict)
    cut: Mapping[str, object] = field(default_factory=dict)
    model: Mapping[str, object] = field(default_factory=dict)
    tracking: InSituTrackingPolicy = field(default_factory=InSituTrackingPolicy)
    fitting: InSituFittingPolicy = field(default_factory=InSituFittingPolicy)
    parent_version: int | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if int(self.version) < 1:
            raise ValueError("Recipe version must be at least one")
        if self.source not in {"single_analysis", "insitu_edit", "saved_recipe"}:
            raise ValueError("Unsupported recipe source")
        if self.parent_version is not None and int(self.parent_version) >= int(self.version):
            raise ValueError("A parent recipe version must precede its child")
        for name in ("experiment_setup", "preprocessing", "cut", "model"):
            object.__setattr__(self, name, _freeze_json_mapping(getattr(self, name)))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "gimap_insitu_recipe_v1",
            "version": self.version,
            "created_at": self.created_at,
            "source": self.source,
            "parent_version": self.parent_version,
            "note": self.note,
            "experiment_setup": _plain_mapping(self.experiment_setup),
            "preprocessing": _plain_mapping(self.preprocessing),
            "cut": _plain_mapping(self.cut),
            "model": _plain_mapping(self.model),
            "tracking": self.tracking.to_dict(),
            "fitting": self.fitting.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "InSituProcessingRecipe":
        schema = str(value.get("schema", "gimap_insitu_recipe_v1"))
        if schema != "gimap_insitu_recipe_v1":
            raise ValueError(f"Unsupported in-situ recipe schema: {schema}")
        tracking = value.get("tracking", {})
        fitting = value.get("fitting", {})
        return cls(
            version=int(value["version"]),
            created_at=str(value["created_at"]),
            source=str(value["source"]),  # type: ignore[arg-type]
            parent_version=(
                None
                if value.get("parent_version") is None
                else int(value["parent_version"])
            ),
            note=str(value.get("note", "")),
            experiment_setup=dict(value.get("experiment_setup", {})),  # type: ignore[arg-type]
            preprocessing=dict(value.get("preprocessing", {})),  # type: ignore[arg-type]
            cut=dict(value.get("cut", {})),  # type: ignore[arg-type]
            model=dict(value.get("model", {})),  # type: ignore[arg-type]
            tracking=InSituTrackingPolicy.from_dict(
                tracking if isinstance(tracking, Mapping) else {}
            ),
            fitting=InSituFittingPolicy.from_dict(
                fitting if isinstance(fitting, Mapping) else {}
            ),
        )


__all__ = [
    "FailurePolicy",
    "FitInitialization",
    "InSituFittingPolicy",
    "InSituProcessingRecipe",
    "InSituTrackingPolicy",
    "RecipeSource",
    "RefinementMode",
    "TrackingMode",
]
