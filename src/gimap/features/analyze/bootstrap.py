"""Analyze composition root."""

from __future__ import annotations

from pathlib import Path

from src.gimap.app import AppContext

from .application import AnalyzeFrame, ExportAnalysis, SaveInstrumentProfile
from .infrastructure.adapters import CsvCurveWriter, DetectorIoFrameSource, MatplotlibFigureWriter
from .infrastructure.frame_workers import ProcessFrameWorkers, system_resources
from .presentation import AnalyzeViewModel


def create_analyze_view_model(app_context: AppContext) -> AnalyzeViewModel:
    frames = DetectorIoFrameSource()
    profiles = getattr(app_context, "instrument_profiles", None)
    view_model = AnalyzeViewModel(
        analyze_frame=AnalyzeFrame(frames, profiles),
        export_analysis=ExportAnalysis(CsvCurveWriter()),
        save_profile=SaveInstrumentProfile(profiles),
        frames=frames,
        profiles=profiles,
        read_setting=app_context.settings.get,
        settings=app_context.settings,
        figures=MatplotlibFigureWriter(),
    )
    view_model.frame_workers = ProcessFrameWorkers
    view_model.system_resources = system_resources
    data_dir = getattr(app_context, "data_dir", None)
    if data_dir is not None:
        view_model.last_setup_path = Path(data_dir) / "last_analyze_setup.json"
    return view_model


__all__ = ["create_analyze_view_model"]
