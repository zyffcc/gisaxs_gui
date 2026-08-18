"""Prediction feature composition root。"""

from __future__ import annotations

from pathlib import Path

from src.gimap.app import AppContext
from src.gimap.integrations.tensorflow import TensorFlowPredictor

from .application import (
    DiscoverNumberedPredictionFiles,
    DiscoverPredictionFiles,
    DiscoverPredictionModules,
    ExportPredictionAscii,
    ExportPredictionJsonl,
    ExportPredictionArray,
    InspectPredictionModel,
    LoadPredictionImage,
    LoadPredictionMask,
    LoadPredictionModule,
    PredictFileBatch,
    PredictImage,
    PredictMultipleFiles,
    PredictPreparedInput,
    PreparePredictionInput,
    ResolvePredictionStack,
    UpdatePredictionModelPath,
    PredictionSequenceRules,
)
from .infrastructure import (
    FabioPredictionImageRepository,
    LocalPredictionFileCatalog,
    LocalPredictionExportRepository,
    ModuleEntryPreprocessor,
    NumpyPredictionMaskRepository,
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
    files = LocalPredictionFileCatalog()
    images = FabioPredictionImageRepository()
    masks = NumpyPredictionMaskRepository()
    preprocessor = ModuleEntryPreprocessor()
    predictor = TensorFlowPredictor(runner=context.jobs)
    exports = LocalPredictionExportRepository()
    load_image = LoadPredictionImage(images)
    predict_image = PredictImage(preprocessor, predictor)
    predict_file = PredictFileBatch(load_image, predict_image)
    return PredictionViewModel(
        context=context,
        discover_modules=DiscoverPredictionModules(modules),
        load_module=LoadPredictionModule(modules),
        update_model_path=UpdatePredictionModelPath(modules),
        inspect_model=InspectPredictionModel(predictor),
        resolve_stack=ResolvePredictionStack(files),
        discover_numbered_files=DiscoverNumberedPredictionFiles(files),
        discover_files=DiscoverPredictionFiles(files),
        load_image=load_image,
        load_mask=LoadPredictionMask(masks),
        prepare_input=PreparePredictionInput(preprocessor),
        predict_prepared=PredictPreparedInput(predictor),
        predict_image=predict_image,
        predict_file=predict_file,
        predict_multiple=PredictMultipleFiles(predict_file),
        export_jsonl=ExportPredictionJsonl(exports),
        export_ascii=ExportPredictionAscii(exports),
        export_array=ExportPredictionArray(exports),
        sequence_rules=PredictionSequenceRules(),
    )
