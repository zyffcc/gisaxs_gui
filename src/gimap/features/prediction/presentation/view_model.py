"""Prediction framework-neutral ViewModel。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from src.gimap.app import AppContext

from ..application import (
    DiscoverPredictionModules,
    DiscoverNumberedPredictionFiles,
    DiscoverPredictionFiles,
    describe_prediction_module,
    InspectPredictionModel,
    LoadPredictionImage,
    LoadPredictionModule,
    PredictFileBatch,
    PredictFileBatchRequest,
    PredictImage,
    PredictImageRequest,
    PredictMultipleFiles,
    PredictMultipleFilesRequest,
    PredictPreparedInput,
    PredictPreparedInputRequest,
    PreparePredictionInput,
    ResolvePredictionStack,
    UpdatePredictionModelPath,
    PredictionSequenceRules,
)
from .state import PredictionState
from .export_view_model import PredictionExportViewModel
from .file_view_model import PredictionFileViewModel


class PredictionViewModel:
    def __init__(
        self,
        *,
        context: AppContext,
        discover_modules: DiscoverPredictionModules,
        load_module: LoadPredictionModule,
        update_model_path: UpdatePredictionModelPath,
        inspect_model: InspectPredictionModel,
        resolve_stack: ResolvePredictionStack,
        discover_numbered_files: DiscoverNumberedPredictionFiles,
        discover_files: DiscoverPredictionFiles,
        load_image: LoadPredictionImage,
        load_mask=None,
        prepare_input: PreparePredictionInput,
        predict_prepared: PredictPreparedInput,
        predict_image: PredictImage,
        predict_file: PredictFileBatch,
        predict_multiple: PredictMultipleFiles,
        export_jsonl=None,
        export_ascii=None,
        export_array=None,
        sequence_rules: PredictionSequenceRules,
    ):
        self.context = context
        self._discover_modules = discover_modules
        self._load_module = load_module
        self._update_model_path = update_model_path
        self._inspect_model = inspect_model
        self._resolve_stack = resolve_stack
        self._load_image = load_image
        self._load_mask = load_mask
        self._prepare_input = prepare_input
        self._predict_prepared = predict_prepared
        self._predict_image = predict_image
        self._predict_file = predict_file
        self._predict_multiple = predict_multiple
        self.state = PredictionState()
        self.files = PredictionFileViewModel(
            numbered_files=discover_numbered_files, files=discover_files,
            sequence_rules=sequence_rules, on_error=self._set_file_error,
        )
        self.exports = PredictionExportViewModel(
            jsonl=export_jsonl, ascii=export_ascii, array=export_array,
            on_error=self._set_export_error,
        )

    def load_settings(self) -> dict[str, object]:
        return dict(self.context.settings.get_section("gisaxs_predict"))

    def save_settings(self, values: dict[str, object]) -> None:
        self.context.settings.update_section("gisaxs_predict", dict(values))
        self.context.settings.save()

    def discover_modules(self):
        self.state = replace(self.state, module_status="loading", error_message=None)
        try:
            modules = self._discover_modules.execute()
        except Exception as exc:
            self.state = replace(
                self.state,
                module_status="error",
                error_message=str(exc),
                status_message=str(exc),
            )
            return ()
        current = self.state.current_module
        if current is not None:
            current = next((module for module in modules if module.id == current.id), None)
        self.state = replace(
            self.state,
            module_status="ready",
            modules=modules,
            current_module=current,
            error_message=None,
            status_message=f"Discovered {len(modules)} prediction modules",
        )
        return modules

    def load_module(self, yaml_path: Path):
        try:
            module = self._load_module.execute(Path(yaml_path))
        except Exception as exc:
            self.state = replace(self.state, module_status="error", error_message=str(exc))
            return None
        self.state = replace(
            self.state,
            module_status="ready",
            current_module=module,
            error_message=None,
        )
        return module

    def select_module(self, name: str):
        module = next((item for item in self.state.modules if item.name == name), None)
        if module is not None:
            self.state = replace(self.state, current_module=module, error_message=None)
        return module

    def module_display_values(self, module):
        return describe_prediction_module(module)

    def update_model_path(self, module, model_path: Path) -> bool:
        try:
            self._update_model_path.execute(module, Path(model_path))
        except Exception as exc:
            self.state = replace(self.state, error_message=str(exc), status_message=str(exc))
            return False
        return True

    def inspect_model(self, model_path: Path, *, allow_unsafe_lambda: bool = False):
        self.state = replace(self.state, model_status="loading", error_message=None)
        try:
            runtime = self._inspect_model.execute(
                Path(model_path), allow_unsafe_lambda=allow_unsafe_lambda
            )
        except Exception as exc:
            self.state = replace(
                self.state,
                model_status="error",
                error_message=str(exc),
                status_message=str(exc),
            )
            return None
        self.state = replace(
            self.state,
            model_status="ready",
            model_path=Path(model_path),
            model_runtime=runtime,
            error_message=None,
            status_message="Prediction model is ready",
        )
        return runtime

    def load_stack(self, start_path: Path, count: int):
        self.state = replace(self.state, image_status="loading", error_message=None)
        try:
            paths = self._resolve_stack.execute(Path(start_path), count)
            loaded = self._load_image.execute(paths)
        except Exception as exc:
            self.state = replace(
                self.state,
                image_status="error",
                error_message=str(exc),
                status_message=str(exc),
            )
            return None
        self.state = replace(
            self.state,
            image_status="ready",
            current_image=loaded,
            error_message=None,
            status_message=f"Loaded {len(loaded.source_paths)} image(s)",
        )
        return loaded

    def __getattr__(self, name):
        files = self.__dict__.get("files")
        if files is not None and hasattr(files, name):
            return getattr(files, name)
        raise AttributeError(name)

    def _set_file_error(self, message):
        self.state = replace(self.state, image_status="error", error_message=message)

    def load_paths(self, paths):
        try:
            loaded = self._load_image.execute(tuple(Path(path) for path in paths))
        except Exception as exc:
            self.state = replace(self.state, image_status="error", error_message=str(exc))
            return None
        self.state = replace(self.state, image_status="ready", current_image=loaded)
        return loaded

    def load_mask(self, path: Path):
        if self._load_mask is None:
            self.state = replace(self.state, error_message="Prediction mask loading is unavailable")
            return None
        try:
            return self._load_mask.execute(Path(path))
        except Exception as exc:
            self.state = replace(self.state, error_message=str(exc), status_message=str(exc))
            return None

    def prepare_input(self, image, module=None):
        selected = module or self.state.current_module
        if selected is None:
            return None
        try:
            return self._prepare_input.execute(image, selected)
        except Exception as exc:
            self.state = replace(self.state, prediction_status="error", error_message=str(exc))
            return None

    def predict_prepared(self, values, module, model_path, steps=()):
        try:
            result = self._predict_prepared.execute(
                PredictPreparedInputRequest(values, module, Path(model_path), tuple(steps))
            )
        except Exception as exc:
            self.state = replace(self.state, prediction_status="error", error_message=str(exc))
            return None
        self.state = replace(
            self.state,
            prediction_status="ready",
            prediction=result,
            error_message=None,
        )
        return result

    def predict_current_image(self, model_path: Path):
        module = self.state.current_module
        loaded = self.state.current_image
        if module is None or loaded is None:
            return None
        self.state = replace(self.state, prediction_status="running", error_message=None)
        try:
            result = self._predict_image.execute(
                PredictImageRequest(loaded.image, module, Path(model_path))
            )
        except Exception as exc:
            self.state = replace(self.state, prediction_status="error", error_message=str(exc))
            return None
        self.state = replace(self.state, prediction_status="ready", prediction=result)
        return result

    def predict_file_batch(self, paths, module, model_path):
        return self._predict_file.execute(
            PredictFileBatchRequest(
                tuple(Path(path) for path in paths), module, Path(model_path)
            )
        )

    def predict_multiple(self, request: PredictMultipleFilesRequest, on_progress=None):
        self.state = replace(self.state, batch_status="running", batch_progress=0.0)

        def update(progress):
            self.state = replace(self.state, batch_progress=progress.fraction)
            if on_progress is not None:
                on_progress(progress)

        result = self._predict_multiple.execute(request, on_progress=update)
        self.state = replace(
            self.state,
            batch_status="cancelled" if result.cancelled else "ready",
            batch_results=result.items,
            batch_progress=(len(result.items) / len(request.batches) if request.batches else 1.0),
        )
        return result

    def _set_export_error(self, message):
        self.state = replace(self.state, error_message=message, status_message=message)

    def export_jsonl(self, items, export_path: Path, timestamp: str):
        return self.exports.export_jsonl(items, export_path, timestamp)

    def export_ascii(self, items, export_path: Path, timestamp: str):
        return self.exports.export_ascii(items, export_path, timestamp)

    def export_array(self, request):
        return self.exports.export_array(request)

    def cancel_multiple(self) -> None:
        self._predict_multiple.cancel()
