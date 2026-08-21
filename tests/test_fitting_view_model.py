from pathlib import Path

import numpy as np

from src.gimap.app import AppContext
from src.gimap.features.fitting.application import (
    ExportFitResult,
    LoadCurveRequest,
    LoadDetectorSettings,
    LoadScatteringFileRequest,
    OperationResult,
    RunManualFit,
    MapCandidateParameters,
    ReviewCandidates,
    ScatteringFileData,
    SaveDetectorSettings,
    InSituWorkflowCoordinator,
    CreateInSituRecipe,
    ReviseInSituRecipe,
    SingleAnalysisRecipeSnapshot,
    LoadCandidateResults,
    ComputeInSituCut,
    FittingAiCalculations,
    FittingCurveCalculations,
    FittingCutCalculations,
    FittingImageCalculations,
    FittingModelCalculations,
    FittingQSpaceCalculations,
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


class _SuccessfulScatteringLoader:
    def execute(self, request):
        return OperationResult(
            value=ScatteringFileData(
                image=np.ones((2, 3)),
                source_path=request.path,
                source_files=(request.path,),
            )
        )


class _ScatteringSequenceInspector:
    def execute(self, path):
        return {"path": Path(path)}


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


class _RemoteCache:
    def default_directory(self):
        return ".gimap_cache/remote_files"

    def display_directory(self, path):
        return str(path)

    def resolve_directory(self, path):
        return Path(path)

    def is_remote(self, path):
        return "remote" in str(path)

    def target_path(self, source, cache):
        return Path(cache) / Path(source).name

    def prepare(self, source, cache, max_gb, **_callbacks):
        return Path(cache) / Path(source).name

    def clear(self, _cache):
        return 2


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


class _QSpace:
    def create_detector(self, **geometry):
        return {"geometry": geometry}

    def axis_labels_and_extent(self, detector):
        return "qy", "qz", detector["geometry"]["extent"]


def _view_model(curve_loader=None):
    context = AppContext(
        settings=InMemorySettingsRepository(),
        session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(),
    )
    detector_settings = context.settings
    return FittingViewModel(
        context=context,
        load_scattering_file=_SuccessfulScatteringLoader(),
        inspect_scattering_sequence=_ScatteringSequenceInspector(),
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
        load_detector_settings=LoadDetectorSettings(detector_settings),
        save_detector_settings=SaveDetectorSettings(detector_settings),
        scattering_loader_factory=lambda **_options: _SuccessfulScatteringLoader(),
        remote_file_cache=_RemoteCache(),
        insitu_records=_InSituRecords(),
        parameter_files=_ParameterFiles(),
        ai_artifacts=_AiArtifacts(),
        save_fitting_log=_SaveLog(),
        check_dependency=_CheckDependency(),
        scientific=FittingScientificViewModel(
            image=FittingImageCalculations(),
            cut=FittingCutCalculations(),
            curve=FittingCurveCalculations(),
            ai=FittingAiCalculations(),
            refinement=ManualRefinementCalculations(),
            insitu_cut=ComputeInSituCut(),
            model=FittingModelCalculations(_ScientificModel()),
            q_space=FittingQSpaceCalculations(_QSpace()),
        ),
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
    assert view_model.state.workflow.step("import").status == "complete"
    assert view_model.state.workflow.step("setup").status == "available"


def test_view_model_workflow_uses_verified_transitions_and_stale_state():
    view_model = _view_model()

    for key in ("import", "setup", "center", "cut", "fit"):
        view_model.complete_workflow_step(key, f"{key} complete")

    assert [step.status for step in view_model.state.workflow.steps] == [
        "complete",
        "complete",
        "complete",
        "complete",
        "complete",
    ]

    view_model.complete_workflow_step("center", "center changed")

    assert view_model.state.workflow.step("center").status == "complete"
    assert view_model.state.workflow.step("cut").status == "stale"
    assert view_model.state.workflow.step("fit").status == "stale"

    view_model.fail_workflow_step("cut", "invalid region")
    assert view_model.state.workflow.step("cut").status == "error"
    assert view_model.state.workflow.step("cut").message == "invalid region"


def test_cut_geometry_draft_invalidates_results_without_recalculating():
    view_model = _view_model()
    for key in ("import", "setup", "center", "cut", "fit"):
        view_model.complete_workflow_step(key, f"{key} complete")

    assert view_model.state.cut_status == "ready"
    assert view_model.update_cut_geometry(
        center_x=12.0,
        center_y=34.0,
        width=20.0,
        height=8.0,
    )

    assert view_model.state.cut_geometry.revision == 1
    assert view_model.state.cut_status == "stale"
    assert view_model.state.workflow.step("center").status == "complete"
    assert view_model.state.workflow.step("cut").status == "stale"
    assert view_model.state.workflow.step("fit").status == "stale"
    assert not view_model.update_cut_geometry(
        center_x=12.0,
        center_y=34.0,
        width=20.0,
        height=8.0,
    )

    view_model.complete_workflow_step("cut", "updated")
    assert view_model.state.cut_status == "ready"
    assert view_model.state.cut_result_geometry_revision == 1


def test_view_model_background_loader_uses_injected_factory_without_qapplication(
    tmp_path,
):
    view_model = _view_model()
    progress = lambda *_args: None
    request = LoadScatteringFileRequest(tmp_path / "image.tif")

    outcome = view_model.load_scattering_background(
        request,
        prepare_path=str,
        on_progress=progress,
    )

    assert outcome.succeeded
    assert outcome.value.source_path == request.path


def test_view_model_remote_cache_commands_are_adapter_neutral():
    view_model = _view_model()

    assert view_model.storage is not None
    assert view_model.default_remote_cache_directory().endswith("remote_files")
    assert view_model.is_remote_source("/remote/frame.cbf")
    assert view_model.remote_cache_target("/remote/frame.cbf", "/cache") == (
        "/cache/frame.cbf"
    )
    assert view_model.prepare_remote_source(
        "/remote/frame.cbf", "/cache", 3.0
    ) == "/cache/frame.cbf"
    assert view_model.clear_remote_cache("/cache") == 2


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


def test_view_model_scientific_commands_run_without_qapplication():
    view_model = _view_model()
    result = view_model.science.insitu_cut.execute(
        {
            "image_data": np.arange(1.0, 17.0).reshape(4, 4),
            "vertical": 2.0,
            "parallel": 2.0,
            "center_x": 1.5,
            "center_y": 1.5,
            "cut_type": "horizontal",
            "n_points": 10,
        }
    )

    np.testing.assert_allclose(result["y_intensity"], np.linspace(9.0, 11.0, 10))
    np.testing.assert_allclose(
        view_model.science.curve.normalize_intensity([2.0, 4.0]),
        [0.5, 1.0],
    )
    assert view_model.science.model.parameter_names(["sphere"]) == (
        "parameter_0",
    )
    detector = view_model.science.q_space.create_detector(extent=[-1, 1, 0, 2])
    assert view_model.science.q_space.axis_labels_and_extent(detector) == (
        "qy",
        "qz",
        [-1, 1, 0, 2],
    )


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
