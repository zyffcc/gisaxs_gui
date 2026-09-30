from pathlib import Path

import numpy as np

from src.gimap.app import AppContext
from src.gimap.features.fitting.application import (
    ExportFitResult,
    LoadCurveRequest,
    OperationResult,
    RunManualFit,
    MapCandidateParameters,
    ReviewCandidates,
    InSituWorkflowCoordinator,
    CreateInSituRecipe,
    ReviseInSituRecipe,
    SingleAnalysisRecipeSnapshot,
    LoadCandidateResults,
    FittingAiCalculations,
    FittingCurveCalculations,
    FittingModelCalculations,
    ManualRefinementCalculations,
)
from src.gimap.features.fitting.application.errors import FileOperationError
from src.gimap.features.fitting.application.models import ExportedFitResult
from src.gimap.features.fitting.domain import (
    CurveData,
    InSituFittingPolicy,
    InSituTrackingPolicy,
    ManualFitRequest,
)
from src.gimap.features.fitting.presentation import (
    FittingScientificViewModel,
    FittingViewModel,
)
from src.gimap.integrations.state import (
    InMemorySessionRepository,
    InMemorySettingsRepository,
    InMemoryUserPreferencesRepository,
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


class _InSituRecords:
    def __init__(self):
        self.rows = []

    def cache_directory(self):
        return Path(".gimap_cache")

    def session_path(self):
        return self.cache_directory() / "insitu_current_session.jsonl"

    def ensure_directory(self):
        return self.cache_directory()

    def reset(self):
        self.rows = []

    def append(self, record):
        self.rows.append(dict(record))

    def load(self):
        return list(self.rows)

    def export_csv(self, path, rows):
        self.rows = list(rows)
        return Path(path)


class _ParameterFiles:
    def __init__(self):
        self.values = {}

    def save_snapshot(self, path, values):
        self.values[Path(path)] = dict(values)
        return Path(path)

    def load_snapshot(self, path):
        return self.values[Path(path)]

    def export_model_parameters(self, source, destination):
        return Path(destination)

    def import_model_parameters(self, source, destination):
        return Path(destination)


class _AiArtifacts:
    def __init__(self):
        self.logs = []

    def has_output(self, _path):
        return True

    def append_log(self, output_dir, text):
        self.logs.append(text)
        return Path(output_dir) / "gui_run.log"

    def export_output(self, output_dir, parent_dir, timestamp):
        return Path(parent_dir) / f"ai_prediction_{timestamp}"


class _SaveLog:
    def execute(self, path, content):
        assert content
        return Path(path)


class _CheckDependency:
    def execute(self, name):
        return name == "numpy"


class _ScientificModel:
    def parameter_names(self, shapes):
        return tuple(f"parameter_{index}" for index, _shape in enumerate(shapes))

    def evaluate(self, shapes, q_model, parameters):
        return np.asarray(q_model) * parameters[0]

    def components(self, shapes, q_model, parameters):
        return {"shapes": tuple(shapes), "q": q_model, "parameters": parameters}

    def build_function(self, shapes):
        return lambda q, scale: np.asarray(q) * scale


def _view_model(curve_loader=None):
    context = AppContext(
        settings=InMemorySettingsRepository(),
        session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(),
    )
    return FittingViewModel(
        context=context,
        load_curve=curve_loader or _CurveLoader(),
        export_fit_result=ExportFitResult(_FitResultRepository()),
        run_manual_fit=RunManualFit(_LinearModel()),
        generate_candidates=_UnusedCandidateUseCase(),
        refine_candidates=_UnusedCandidateUseCase(),
        review_candidates=ReviewCandidates(),
        map_candidate_parameters=MapCandidateParameters(),
        load_candidate_results=LoadCandidateResults(_CandidateRepository()),
        insitu_workflow=InSituWorkflowCoordinator(),
        create_insitu_recipe=CreateInSituRecipe(),
        revise_insitu_recipe=ReviseInSituRecipe(),
        insitu_records=_InSituRecords(),
        parameter_files=_ParameterFiles(),
        ai_artifacts=_AiArtifacts(),
        save_fitting_log=_SaveLog(),
        check_dependency=_CheckDependency(),
        scientific=FittingScientificViewModel(
            curve=FittingCurveCalculations(),
            ai=FittingAiCalculations(),
            refinement=ManualRefinementCalculations(),
            model=FittingModelCalculations(_ScientificModel()),
        ),
    )


def test_view_model_insitu_record_commands_are_repository_neutral(tmp_path):
    view_model = _view_model()
    record = {"file_name": "frame.cbf", "fit_status": "ok"}

    view_model.reset_insitu_records()
    view_model.append_insitu_record(record)
    exported = view_model.export_insitu_records(
        tmp_path / "records.csv",
        view_model.load_insitu_records(),
    )

    assert view_model.insitu_session_path().name == "insitu_current_session.jsonl"
    assert exported == tmp_path / "records.csv"


def test_view_model_parameter_file_commands_are_repository_neutral(tmp_path):
    view_model = _view_model()
    target = tmp_path / "fitting.json"
    values = {"schema_version": 1, "fitting": {"points_num": 50}}

    assert view_model.save_parameter_snapshot(target, values) == target
    assert view_model.load_parameter_snapshot(target) == values
    assert view_model.export_model_parameters(
        tmp_path / "model.json", tmp_path / "export.json"
    ) == tmp_path / "export.json"


def test_view_model_ai_artifact_commands_are_repository_neutral(tmp_path):
    view_model = _view_model()
    output = tmp_path / "current_prediction"

    assert view_model.has_ai_output(output)
    assert view_model.append_ai_log(output, "Progress 1/2").name == "gui_run.log"
    assert view_model.export_ai_output(output, tmp_path, "stamp") == (
        tmp_path / "ai_prediction_stamp"
    )


def test_view_model_saves_log_through_application_command(tmp_path):
    view_model = _view_model()

    assert view_model.save_fitting_log(
        tmp_path / "fitting.log", "fit completed"
    ) == tmp_path / "fitting.log"


def test_view_model_checks_optional_dependencies_through_application_port():
    view_model = _view_model()

    assert view_model.dependency_available("numpy")
    assert not view_model.dependency_available("missing")


def test_view_model_settings_use_injected_repository_without_global_singleton():
    view_model = _view_model()

    view_model.set_setting("fitting", "detector.beam_center_x", 42.5)
    view_model.save_settings()

    assert view_model.get_setting(
        "fitting", "detector.beam_center_x", 0.0
    ) == 42.5


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

    assert view_model.insitu is not None
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


def test_view_model_maps_explicit_insitu_recipe_to_typed_state_without_qapplication():
    view_model = _view_model()

    recipe = view_model.insitu.create_recipe_from_single(
        SingleAnalysisRecipeSnapshot(
            experiment_setup={"distance_mm": 2000.0},
            preprocessing={"flip_ud": True},
            cut={"width_px": 5},
            model={"shapes": ["sphere"]},
            tracking=InSituTrackingPolicy(),
            fitting=InSituFittingPolicy(),
        )
    )
    snapshot = view_model.insitu.snapshot_recipe()

    assert recipe is view_model.state.insitu_recipe
    assert view_model.state.insitu_recipe_scope == "future"
    assert snapshot["schema"] == "gimap_insitu_recipe_v1"

    restored = _view_model()
    restored.insitu.restore_recipe(snapshot)
    assert restored.state.insitu_recipe.to_dict() == recipe.to_dict()


def test_view_model_loads_a_curve_and_tracks_the_fit_step(tmp_path):
    view_model = _view_model()
    curve_path = tmp_path / "curve.dat"

    outcome = view_model.load_curve(LoadCurveRequest(curve_path))

    assert outcome.succeeded
    assert view_model.state.curve_status == "ready"
    assert view_model.state.current_curve.source_path == str(curve_path)
    assert [step.key for step in view_model.state.workflow.steps] == ["fit"]
    assert view_model.state.workflow.step("fit").status == "available"
    view_model.begin_workflow_step("fit", "running")
    assert view_model.state.workflow.step("fit").status == "running"
    view_model.fail_workflow_step("fit", "no convergence")
    assert view_model.state.workflow.step("fit").message == "no convergence"
    view_model.complete_workflow_step("fit", "done")
    assert view_model.state.workflow.step("fit").status == "complete"


def test_view_model_curve_and_model_commands_run_without_qapplication():
    view_model = _view_model()
    np.testing.assert_allclose(
        view_model.science.curve.normalize_intensity([2.0, 4.0]),
        [0.5, 1.0],
    )
    np.testing.assert_allclose(
        view_model.science.curve.interpolate([0.0, 1.0], [0.0, 2.0], [0.5], "Linear"),
        [1.0],
    )
    assert view_model.science.model.parameter_names(["sphere"]) == ("parameter_0",)
