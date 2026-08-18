"""Fitting ViewModel 的 typed state。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..application.models import ScatteringFileData
from ..application.ai_models import CandidateGenerationResult
from ..application.insitu import InSituWorkflowState
from ..application import CurveData, ManualFitResult


LoadStatus = Literal["idle", "loading", "ready", "error"]
ManualFitStatus = Literal["idle", "running", "ready", "error"]
AiFitStatus = Literal["idle", "running", "ready", "cancelled", "error"]


@dataclass(frozen=True)
class FittingState:
    image_status: LoadStatus = "idle"
    curve_status: LoadStatus = "idle"
    manual_fit_status: ManualFitStatus = "idle"
    ai_fit_status: AiFitStatus = "idle"
    ai_progress: float = 0.0
    ai_progress_message: str = ""
    current_image: ScatteringFileData | None = None
    current_curve: CurveData | None = None
    manual_fit_result: ManualFitResult | None = None
    ai_fit_result: CandidateGenerationResult | None = None
    insitu_workflow: InSituWorkflowState = field(default_factory=InSituWorkflowState)
    ai_error_code: str | None = None
    error_message: str | None = None
    status_message: str = "Ready"
