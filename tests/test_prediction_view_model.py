from pathlib import Path

import numpy as np

from src.gimap.app import AppContext
from src.gimap.features.prediction.application import (
    FilePredictionResult,
    ImagePredictionResult,
    LoadedPredictionImage,
    MultiplePredictionResult,
    PreprocessedPredictionInput,
)
from src.gimap.features.prediction.domain import (
    ModelRuntimeInfo,
    ModelSpec,
    OutputSpec,
    PredictionModule,
)
from src.gimap.features.prediction.presentation import PredictionViewModel
from src.gimap.integrations.state import (
    InMemorySessionRepository,
    InMemorySettingsRepository,
)


def _module() -> PredictionModule:
    return PredictionModule(
        id="test",
        name="Test module",
        model=ModelSpec("keras", "model.keras"),
        input_shape=(1, 2, 2, 1),
        outputs=OutputSpec(names=("image",)),
    )


class _Call:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def execute(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.error is not None:
            raise self.error
        return self.result


class _MultipleCall(_Call):
    def execute(self, request, on_progress=None):
        self.calls.append(((request,), {"on_progress": on_progress}))
        return self.result

    def cancel(self):
        self.calls.append((("cancel",), {}))


def _view_model(**overrides):
    module = _module()
    image = LoadedPredictionImage(
        image=np.ones((2, 2), dtype=np.float32),
        source_paths=(Path("frame.cbf"),),
    )
    prepared = PreprocessedPredictionInput(
        values=np.ones((1, 2, 2, 1), dtype=np.float32),
        steps=({"name": "identity"},),
    )
    prediction = ImagePredictionResult(
        outputs={"image": np.ones((2, 2), dtype=np.float32)},
        model_input=prepared.values,
        preprocess_steps=prepared.steps,
    )
    defaults = {
        "discover_modules": _Call((module,)),
        "load_module": _Call(module),
        "update_model_path": _Call(None),
        "inspect_model": _Call(
            ModelRuntimeInfo(Path("model.keras"), "fake", (1, 2, 2, 1))
        ),
        "resolve_stack": _Call((Path("frame.cbf"),)),
        "load_image": _Call(image),
        "prepare_input": _Call(prepared),
        "predict_prepared": _Call(prediction),
        "predict_image": _Call(prediction),
        "predict_file": _Call(FilePredictionResult((Path("frame.cbf"),), "succeeded", prediction)),
        "predict_multiple": _MultipleCall(
            MultiplePredictionResult(
                (FilePredictionResult((Path("frame.cbf"),), "succeeded", prediction),)
            )
        ),
    }
    defaults.update(overrides)
    context = AppContext(
        settings=InMemorySettingsRepository(),
        session=InMemorySessionRepository(),
    )
    return PredictionViewModel(context=context, **defaults), module, prepared


def test_view_model_settings_and_module_state_need_no_qapplication():
    view_model, module, _prepared = _view_model()

    view_model.save_settings({"module_name": module.name, "stack_value": "3"})
    modules = view_model.discover_modules()
    selected = view_model.select_module(module.name)

    assert view_model.load_settings()["stack_value"] == "3"
    assert modules == (module,)
    assert selected is module
    assert view_model.state.module_status == "ready"
    assert view_model.state.current_module is module


def test_view_model_image_prepare_and_prediction_transitions():
    view_model, module, prepared = _view_model()
    view_model.discover_modules()
    view_model.select_module(module.name)

    loaded = view_model.load_stack(Path("frame.cbf"), 1)
    actual_prepared = view_model.prepare_input(loaded.image)
    result = view_model.predict_prepared(
        actual_prepared.values,
        module,
        Path("model.keras"),
        actual_prepared.steps,
    )

    assert actual_prepared is prepared
    assert result is view_model.state.prediction
    assert view_model.state.image_status == "ready"
    assert view_model.state.prediction_status == "ready"


def test_view_model_converts_port_error_to_display_state():
    view_model, _module_value, _prepared = _view_model(
        discover_modules=_Call(error=OSError("module.yaml is damaged"))
    )

    assert view_model.discover_modules() == ()
    assert view_model.state.module_status == "error"
    assert view_model.state.error_message == "module.yaml is damaged"


def test_view_model_batch_state_and_cancel_are_presentation_only(tmp_path):
    multiple = _MultipleCall(
        MultiplePredictionResult(
            (FilePredictionResult((tmp_path / "one.cbf",), "succeeded"),)
        )
    )
    view_model, module, _prepared = _view_model(predict_multiple=multiple)
    from src.gimap.features.prediction.application import PredictMultipleFilesRequest

    result = view_model.predict_multiple(
        PredictMultipleFilesRequest(
            batches=((tmp_path / "one.cbf",),),
            module=module,
            model_path=tmp_path / "model.keras",
        )
    )
    view_model.cancel_multiple()

    assert result.items[0].status == "succeeded"
    assert view_model.state.batch_status == "ready"
    assert view_model.state.batch_progress == 1.0
    assert multiple.calls[-1][0] == ("cancel",)
