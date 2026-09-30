"""Fitting application public API。"""

from .errors import FileOperationError
from .ai_models import (
    CandidateGenerationRequest,
    CandidateGenerationResult,
    CandidateJobError,
)
from .ai_use_cases import (
    GenerateCandidates,
    LoadCandidateResults,
    MapCandidateParameters,
    RefineCandidates,
    ReviewCandidates,
)
from .models import (
    ExportCurveFigureRequest,
    ExportFitResultRequest,
    FigureSeries,
    DiscoverInSituFramesRequest,
    ExportedFitResult,
    LoadCurveRequest,
    OperationResult,
    CURVE_SUFFIXES,
    DEFAULT_CURVE_PATTERN,
    InSituSourceFrame,
    InSituSourceKind,
)
from .insitu import (
    InSituFileFitRequest,
    InSituFileFitResult,
    InSituFileRecord,
    InSituProgress,
    InSituWorkflowCoordinator,
    InSituWorkflowRequest,
    InSituWorkflowState,
    RunInSituWorkflow,
)
from .insitu_records import ManageInSituRecords
from .insitu_recipe import (
    CreateInSituRecipe,
    InSituRecipeRevision,
    ReviseInSituRecipe,
    ReviseInSituRecipeRequest,
    SingleAnalysisRecipeSnapshot,
)
from .parameter_files import ManageFittingParameterFiles
from .ai_artifacts import ManageAiFittingArtifacts
from .logs import SaveFittingLog
from .dependencies import CheckFittingDependency
from .model_parameters import ManageFittingModelParameters
from .ai_catalog import AiFittingCatalog
from .scientific import (
    FittingAiCalculations,
    FittingCurveCalculations,
    FittingModelCalculations,
    ManualRefinementCalculations,
)
from .use_cases import (
    ExportCurveFigure,
    ExportFitResult,
    LoadCurve,
    DiscoverInSituFrames,
    RunManualFit,
)
from ..domain import (
    CurveData,
    ConstraintSet,
    InSituFittingPolicy,
    InSituProcessingRecipe,
    InSituTrackingPolicy,
    ManualFitRequest,
    ManualFitResult,
)

__all__ = [
    "ExportCurveFigure",
    "ExportCurveFigureRequest",
    "ExportFitResult",
    "FigureSeries",
    "CandidateGenerationRequest",
    "CandidateGenerationResult",
    "CandidateJobError",
    "ExportFitResultRequest",
    "DiscoverInSituFrames",
    "DiscoverInSituFramesRequest",
    "ExportedFitResult",
    "FileOperationError",
    "LoadCurve",
    "LoadCurveRequest",
    "CURVE_SUFFIXES",
    "DEFAULT_CURVE_PATTERN",
    "InSituSourceFrame",
    "InSituSourceKind",
    "GenerateCandidates",
    "LoadCandidateResults",
    "MapCandidateParameters",
    "ManageInSituRecords",
    "CreateInSituRecipe",
    "InSituRecipeRevision",
    "ReviseInSituRecipe",
    "ReviseInSituRecipeRequest",
    "SingleAnalysisRecipeSnapshot",
    "ManageFittingParameterFiles",
    "ManageFittingModelParameters",
    "AiFittingCatalog",
    "ManageAiFittingArtifacts",
    "SaveFittingLog",
    "CheckFittingDependency",
    "FittingAiCalculations",
    "FittingCurveCalculations",
    "ManualRefinementCalculations",
    "FittingModelCalculations",
    "CurveData",
    "ConstraintSet",
    "InSituFittingPolicy",
    "InSituProcessingRecipe",
    "InSituTrackingPolicy",
    "ManualFitRequest",
    "ManualFitResult",
    "OperationResult",
    "RunManualFit",
    "RefineCandidates",
    "ReviewCandidates",
    "InSituFileFitRequest",
    "InSituFileFitResult",
    "InSituFileRecord",
    "InSituProgress",
    "InSituWorkflowCoordinator",
    "InSituWorkflowRequest",
    "InSituWorkflowState",
    "RunInSituWorkflow",
]
