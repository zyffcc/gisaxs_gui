"""Fitting ViewModel 的 typed state。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..application.models import ScatteringFileData
from ..application.ai_models import CandidateGenerationResult
from ..application.insitu import InSituWorkflowState
from ..application import InSituProcessingRecipe
from ..application import CurveData, ManualFitResult
from .workflow_state import FittingWorkflowState, initial_workflow_state


LoadStatus = Literal["idle", "loading", "ready", "error"]
ManualFitStatus = Literal["idle", "running", "ready", "error"]
AiFitStatus = Literal["idle", "running", "ready", "cancelled", "error"]
CutStatus = Literal["idle", "ready", "stale", "error"]
CurveQViewMode = Literal[
    "signed", "positive", "negative", "negative_abs", "fold", "average"
]
CurveLayerMode = Literal["data", "compare", "model"]
QDisplayUnit = Literal["nm", "angstrom"]


@dataclass(frozen=True)
class CutGeometryDraft:
    center_x: float = 0.0
    center_y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    revision: int = 0


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
class DetectorDisplayState:
    """Display-only detector state; scientific preprocessing is stored separately."""

    log_intensity: bool = True
    auto_scale: bool = True
    vmin: float | None = None
    vmax: float | None = None
    colormap: str = "viridis"
    show_cut_region: bool = True
    show_center: bool = True
    show_q_axis: bool = False
    horizontal_q_axis: Literal["qy", "qr"] = "qy"


@dataclass(frozen=True)
class FittingState:
    image_status: LoadStatus = "idle"
    curve_status: LoadStatus = "idle"
    manual_fit_status: ManualFitStatus = "idle"
    ai_fit_status: AiFitStatus = "idle"
    cut_status: CutStatus = "idle"
    cut_geometry: CutGeometryDraft = field(default_factory=CutGeometryDraft)
    curve_view: CurveViewState = field(default_factory=CurveViewState)
    detector_display: DetectorDisplayState = field(default_factory=DetectorDisplayState)
    analysis_revision: int | None = None
    cut_result_analysis_revision: int | None = None
    cut_result_geometry_revision: int | None = None
    ai_progress: float = 0.0
    ai_progress_message: str = ""
    current_image: ScatteringFileData | None = None
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
