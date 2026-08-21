"""Use cases for explicit Single Analysis to In-situ recipe handoff."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Callable, Literal, Mapping

from ..domain.insitu_recipe import (
    InSituFittingPolicy,
    InSituProcessingRecipe,
    InSituTrackingPolicy,
)


RecipeChangeScope = Literal["future", "selected_and_future", "all"]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass(frozen=True)
class SingleAnalysisRecipeSnapshot:
    """Framework-neutral values explicitly captured from the Single page."""

    experiment_setup: Mapping[str, object]
    preprocessing: Mapping[str, object]
    cut: Mapping[str, object]
    model: Mapping[str, object]
    tracking: InSituTrackingPolicy = InSituTrackingPolicy()
    fitting: InSituFittingPolicy = InSituFittingPolicy()
    note: str = ""


@dataclass(frozen=True)
class InSituRecipeRevision:
    recipe: InSituProcessingRecipe
    scope: RecipeChangeScope
    selected_frame_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReviseInSituRecipeRequest:
    current: InSituProcessingRecipe
    scope: RecipeChangeScope = "future"
    selected_frame_ids: tuple[str, ...] = ()
    experiment_setup: Mapping[str, object] | None = None
    preprocessing: Mapping[str, object] | None = None
    cut: Mapping[str, object] | None = None
    model: Mapping[str, object] | None = None
    tracking: InSituTrackingPolicy | None = None
    fitting: InSituFittingPolicy | None = None
    note: str = ""


class CreateInSituRecipe:
    """Create a detached recipe only after an explicit user command."""

    def __init__(self, clock: Callable[[], str] = _now) -> None:
        self._clock = clock

    def execute(
        self,
        snapshot: SingleAnalysisRecipeSnapshot,
        previous: InSituProcessingRecipe | None = None,
    ) -> InSituProcessingRecipe:
        version = 1 if previous is None else previous.version + 1
        return InSituProcessingRecipe(
            version=version,
            created_at=self._clock(),
            source="single_analysis",
            experiment_setup=snapshot.experiment_setup,
            preprocessing=snapshot.preprocessing,
            cut=snapshot.cut,
            model=snapshot.model,
            tracking=snapshot.tracking,
            fitting=snapshot.fitting,
            parent_version=None if previous is None else previous.version,
            note=snapshot.note,
        )


class ReviseInSituRecipe:
    """Create a new recipe version and retain the requested application scope."""

    def __init__(self, clock: Callable[[], str] = _now) -> None:
        self._clock = clock

    def execute(self, request: ReviseInSituRecipeRequest) -> InSituRecipeRevision:
        if request.scope not in {"future", "selected_and_future", "all"}:
            raise ValueError("Unsupported recipe change scope")
        selected = tuple(str(value) for value in request.selected_frame_ids if str(value))
        if request.scope == "selected_and_future" and not selected:
            raise ValueError("selected_and_future requires at least one selected frame")

        current = request.current
        recipe = replace(
            current,
            version=current.version + 1,
            created_at=self._clock(),
            source="insitu_edit",
            parent_version=current.version,
            experiment_setup=(
                current.experiment_setup
                if request.experiment_setup is None
                else request.experiment_setup
            ),
            preprocessing=(
                current.preprocessing
                if request.preprocessing is None
                else request.preprocessing
            ),
            cut=current.cut if request.cut is None else request.cut,
            model=current.model if request.model is None else request.model,
            tracking=current.tracking if request.tracking is None else request.tracking,
            fitting=current.fitting if request.fitting is None else request.fitting,
            note=request.note or current.note,
        )
        return InSituRecipeRevision(recipe, request.scope, selected)


__all__ = [
    "CreateInSituRecipe",
    "InSituRecipeRevision",
    "RecipeChangeScope",
    "ReviseInSituRecipe",
    "ReviseInSituRecipeRequest",
    "SingleAnalysisRecipeSnapshot",
]
