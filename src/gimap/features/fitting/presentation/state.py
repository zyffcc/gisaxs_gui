"""Typed state of the Fitting ViewModel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..application.ai_models import CandidateGenerationResult
from ..application.insitu import InSituWorkflowState
from ..application import InSituProcessingRecipe
from ..application import CurveData, ManualFitResult
from .workflow_state import FittingWorkflowState, initial_workflow_state


LoadStatus = Literal["idle", "loading", "ready", "error"]
ManualFitStatus = Literal["idle", "running", "ready", "error"]
AiFitStatus = Literal["idle", "running", "ready", "cancelled", "error"]
CurveQViewMode = Literal[
    "signed", "positive", "negative", "negative_abs", "fold", "average"
]
CurveLayerMode = Literal["data", "compare", "model"]
QDisplayUnit = Literal["nm", "angstrom"]


@dataclass(frozen=True)
class CurveViewState:
    """One display contract shared by the embedded and independent curve views."""

    q_mode: CurveQViewMode = "signed"
    layer_mode: CurveLayerMode = "data"
    log_x: bool = False
    log_y: bool = False
    normalize: bool = False
    q_unit: QDisplayUnit = "nm"
    y_range: Literal["experimental", "fitting", "all"] = "all"


@dataclass(frozen=True)
class FittingState:
    curve_status: LoadStatus = "idle"
    manual_fit_status: ManualFitStatus = "idle"
    ai_fit_status: AiFitStatus = "idle"
    curve_view: CurveViewState = field(default_factory=CurveViewState)
    ai_progress: float = 0.0
    ai_progress_message: str = ""
    current_curve: CurveData | None = None
    manual_fit_result: ManualFitResult | None = None
    ai_fit_result: CandidateGenerationResult | None = None
    insitu_workflow: InSituWorkflowState = field(default_factory=InSituWorkflowState)
    insitu_recipe: InSituProcessingRecipe | None = None
    insitu_recipe_scope: str = "future"
    ai_error_code: str | None = None
    error_message: str | None = None
    status_message: str = "Ready"
    workflow: FittingWorkflowState = field(default_factory=initial_workflow_state)
