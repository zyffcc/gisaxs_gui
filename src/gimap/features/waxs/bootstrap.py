"""WAXS feature composition root。"""

from src.gimap.app import AppContext

from .application import (
    ComputeWaxsQMaps,
    CutWaxsImage,
    EstimateWaxsDisplayLimits,
    ExportWaxsCurve,
    ExportWaxsImage,
    IntegrateWaxsImage,
    LoadWaxsImage,
    GetWaxsWorkingDirectory,
    NormalizeWaxsPath,
    PrepareWaxsDisplay,
    RunWaxsBatch,
    ValidateWaxsDirectory,
)
from .infrastructure import (
    CalibrationWaxsImageRepository,
    JobRunnerWaxsBatchAdapter,
    LocalWaxsExportAdapter,
    LocalWaxsPathAdapter,
)
from .presentation import WaxsViewModel


def create_waxs_view_model(context: AppContext) -> WaxsViewModel:
    if context.jobs is None:
        raise ValueError("WaxsViewModel requires AppContext.jobs")
    exporter = LocalWaxsExportAdapter()
    paths = LocalWaxsPathAdapter()
    return WaxsViewModel(
        load_image=LoadWaxsImage(CalibrationWaxsImageRepository()),
        integrate_image=IntegrateWaxsImage(),
        run_batch=RunWaxsBatch(JobRunnerWaxsBatchAdapter(context.jobs)),
        export_curve=ExportWaxsCurve(exporter),
        export_image=ExportWaxsImage(exporter),
        compute_q_maps=ComputeWaxsQMaps(),
        cut_image=CutWaxsImage(),
        prepare_display=PrepareWaxsDisplay(),
        estimate_display_limits=EstimateWaxsDisplayLimits(),
        normalize_path=NormalizeWaxsPath(paths),
        get_working_directory=GetWaxsWorkingDirectory(paths),
        validate_directory=ValidateWaxsDirectory(paths),
    )
