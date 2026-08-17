"""Prediction feature composition root。"""

from __future__ import annotations

from pathlib import Path

from src.gimap.app import AppContext
from src.gimap.integrations.tensorflow import TensorFlowPredictor

from .application import (
    DiscoverPredictionModules,
    InspectPredictionModel,
    LoadPredictionImage,
    LoadPredictionModule,
    PredictFileBatch,
    PredictImage,
    PredictMultipleFiles,
    PredictPreparedInput,
    PreparePredictionInput,
    ResolvePredictionStack,
    UpdatePredictionModelPath,
)
from .infrastructure import (
    FabioPredictionImageRepository,
    LocalPredictionFileCatalog,
    ModuleEntryPreprocessor,
    YamlModuleRepository,
)
from .presentation import PredictionViewModel


def create_prediction_view_model(
    context: AppContext, modules_root: Path | None = None
) -> PredictionViewModel:
    if context.jobs is None:
        raise ValueError("PredictionViewModel requires AppContext.jobs")
    root = modules_root or Path(__file__).resolve().parents[4] / "modules"
    modules = YamlModuleRepository(root)
    images = FabioPredictionImageRepository()
    preprocessor = ModuleEntryPreprocessor()
    predictor = TensorFlowPredictor(runner=context.jobs)
    load_image = LoadPredictionImage(images)
    predict_image = PredictImage(preprocessor, predictor)
    predict_file = PredictFileBatch(load_image, predict_image)
    return PredictionViewModel(
        context=context,
        discover_modules=DiscoverPredictionModules(modules),
        load_module=LoadPredictionModule(modules),
        update_model_path=UpdatePredictionModelPath(modules),
        inspect_model=InspectPredictionModel(predictor),
        resolve_stack=ResolvePredictionStack(LocalPredictionFileCatalog()),
        load_image=load_image,
        prepare_input=PreparePredictionInput(preprocessor),
        predict_prepared=PredictPreparedInput(predictor),
        predict_image=predict_image,
        predict_file=predict_file,
        predict_multiple=PredictMultipleFiles(predict_file),
    )
