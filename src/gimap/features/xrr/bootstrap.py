"""Composition root for the XRR Tools window."""

from src.gimap.app import AppContext

from .application import (
    ExportXrrCurve,
    InspectXrrSeries,
    LoadLastCalibrationGeometry,
    RunXrrExtraction,
)
from .infrastructure import (
    JobRunnerXrrExtractionAdapter,
    LocalXrrCurveExportAdapter,
    LocalXrrRecordAdapter,
    LocalXrrSeriesRepository,
    PreferencesXrrFolderAdapter,
    SettingsXrrGeometryAdapter,
)
from .presentation.view_model import XrrViewModel


def create_xrr_view_model(context: AppContext) -> XrrViewModel:
    if context.jobs is None:
        raise ValueError("XRR extraction requires AppContext.jobs")
    settings = getattr(context, "settings", None)
    preferences = getattr(context, "preferences", None)
    return XrrViewModel(
        inspect_series=InspectXrrSeries(LocalXrrSeriesRepository()),
        run_extraction=RunXrrExtraction(JobRunnerXrrExtractionAdapter(context.jobs)),
        export_curve=ExportXrrCurve(LocalXrrCurveExportAdapter(), LocalXrrRecordAdapter()),
        last_calibration=(
            LoadLastCalibrationGeometry(SettingsXrrGeometryAdapter(settings))
            if settings is not None
            else None
        ),
        input_folders=PreferencesXrrFolderAdapter(preferences) if preferences is not None else None,
    )


__all__ = ["create_xrr_view_model"]
