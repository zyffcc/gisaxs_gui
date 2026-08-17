from pathlib import Path

import numpy as np

from src.gimap.app import AppContext
from src.gimap.features.fitting.application import (
    ExportFitResult,
    LoadCurveRequest,
    LoadScatteringFileRequest,
    OperationResult,
    RunManualFit,
    MapCandidateParameters,
    ReviewCandidates,
    ScatteringFileData,
    InSituWorkflowCoordinator,
    LoadCandidateResults,
)
from src.gimap.features.fitting.application.errors import FileOperationError
from src.gimap.features.fitting.application.models import ExportedFitResult
from src.gimap.features.fitting.domain import CurveData, ManualFitRequest
from src.gimap.features.fitting.presentation import FittingViewModel
from src.gimap.integrations.state import (
    InMemorySessionRepository,
    InMemorySettingsRepository,
)


class _SuccessfulScatteringLoader:
    def execute(self, request):
        return OperationResult(
            value=ScatteringFileData(
                image=np.ones((2, 3)),
                source_path=request.path,
                source_files=(request.path,),
            )
        )


class _CurveLoader:
    def __init__(self, fail=False):
        self.fail = fail

    def execute(self, request):
        if self.fail:
            return OperationResult(
                error=FileOperationError("invalid_data", "bad curve", str(request.path))
            )
        return OperationResult(
            value=CurveData(
                q=np.array([0.1, 0.2]),
                intensity=np.array([10.0, 20.0]),
                source_path=str(request.path),
            )
        )


class _FitResultRepository:
    def export(self, request):
        return ExportedFitResult(request.path, len(request.q), "\t")


class _LinearModel:
    def parameter_names(self, shapes):
        assert shapes == ("sphere",)
        return ("scale", "BG")

    def evaluate(self, shapes, q_model, parameters):
        scale, background = parameters
        return scale * q_model + background


class _UnusedCandidateUseCase:
    def execute(self, _request, on_progress=None):
        raise AssertionError("AI use case was not expected in this test")

    def cancel(self):
        return False


class _CandidateRepository:
    def load(self, _output_dir):
        return ({"rank": 1},)


def _view_model(curve_loader=None):
    context = AppContext(
        settings=InMemorySettingsRepository(),
        session=InMemorySessionRepository(),
    )
    return FittingViewModel(
        context=context,
        load_scattering_file=_SuccessfulScatteringLoader(),
        load_curve=curve_loader or _CurveLoader(),
        export_fit_result=ExportFitResult(_FitResultRepository()),
        run_manual_fit=RunManualFit(_LinearModel()),
        generate_candidates=_UnusedCandidateUseCase(),
        refine_candidates=_UnusedCandidateUseCase(),
        review_candidates=ReviewCandidates(),
        map_candidate_parameters=MapCandidateParameters(),
        load_candidate_results=LoadCandidateResults(_CandidateRepository()),
        insitu_workflow=InSituWorkflowCoordinator(),
    )


def test_view_model_load_transitions_without_qapplication(tmp_path):
    view_model = _view_model()
    image_path = tmp_path / "image.tif"
    curve_path = tmp_path / "curve.dat"

    image_outcome = view_model.load_scattering(LoadScatteringFileRequest(image_path))
    curve_outcome = view_model.load_curve(LoadCurveRequest(curve_path))

    assert image_outcome.succeeded and curve_outcome.succeeded
    assert view_model.state.image_status == "ready"
    assert view_model.state.curve_status == "ready"
    assert view_model.state.current_image.source_path == image_path
    assert view_model.state.current_curve.source_path == str(curve_path)
    assert view_model.state.error_message is None


def test_view_model_maps_curve_error_to_typed_state(tmp_path):
    view_model = _view_model(curve_loader=_CurveLoader(fail=True))

    outcome = view_model.load_curve(LoadCurveRequest(tmp_path / "bad.dat"))

    assert not outcome.succeeded
    assert view_model.state.curve_status == "error"
    assert view_model.state.error_message == "bad curve"


def test_view_model_manual_fit_state_and_units_are_stable():
    view_model = _view_model()

    result = view_model.run_manual_fit(
        ManualFitRequest(
            q=np.array([0.1, 0.2]),
            q_source_unit="angstrom",
            shapes=("sphere",),
            parameters=(2.0, 0.5),
        )
    )

    assert view_model.state.manual_fit_status == "ready"
    assert view_model.state.manual_fit_result is result
    np.testing.assert_allclose(result.q_model, [1.0, 2.0])
    np.testing.assert_allclose(result.intensity, [2.5, 4.5])


def test_view_model_manual_fit_failure_is_display_state():
    class FailingManualFit:
        def execute(self, _request):
            raise RuntimeError("model failed")

    view_model = _view_model()
    view_model._run_manual_fit = FailingManualFit()

    result = view_model.run_manual_fit(
        ManualFitRequest(
            q=np.array([1.0]),
            q_source_unit="nm",
            shapes=("sphere",),
            parameters=(1.0, 0.0),
        )
    )

    assert result is None
    assert view_model.state.manual_fit_status == "error"
    assert view_model.state.error_message == "model failed"


def test_view_model_maps_insitu_commands_to_typed_state_without_qapplication():
    view_model = _view_model()

    view_model.start_insitu_workflow(("one.cbf", "two.cbf"))
    current = view_model.begin_next_insitu_file()
    view_model.complete_insitu_file({"chi_square": 0.5})

    assert current.paths == ("one.cbf",)
    assert view_model.state.insitu_workflow.status == "running"
    assert view_model.state.insitu_workflow.processed_count == 1
    assert view_model.state.insitu_workflow.records[0].values == {
        "chi_square": 0.5
    }

    view_model.cancel_insitu_workflow()
    assert view_model.state.insitu_workflow.status == "cancelled"
