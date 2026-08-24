"""Composition root for the XRR Tools window."""

from src.gimap.app import AppContext

from .application import ExportXrrCurve, InspectXrrSeries, RunXrrExtraction
from .infrastructure import (
    JobRunnerXrrExtractionAdapter,
    LocalXrrCurveExportAdapter,
    LocalXrrSeriesRepository,
)
from .presentation.view_model import XrrViewModel


def create_xrr_view_model(context: AppContext) -> XrrViewModel:
    if context.jobs is None:
        raise ValueError("XRR extraction requires AppContext.jobs")
    return XrrViewModel(
        inspect_series=InspectXrrSeries(LocalXrrSeriesRepository()),
        run_extraction=RunXrrExtraction(JobRunnerXrrExtractionAdapter(context.jobs)),
        export_curve=ExportXrrCurve(LocalXrrCurveExportAdapter()),
    )


__all__ = ["create_xrr_view_model"]
