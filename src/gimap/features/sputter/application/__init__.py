"""Sputter use cases and their ports (and the result types the page shows)."""

from ..domain import (FILM_NOT_DETERMINED, GRADES, LITERATURE_GRADE, LITERATURE_TEMPLATES, PHYSICAL_TEMPLATES,
                      SKETCH_TEMPLATES, VERDICT_COLOURS, VERDICT_MARKS, Particle, Series, Sketcher, StructureResult,
                      WindowSketch, beam_off_count, beam_off_record, checks_html, deposition_end, film_onset, is_film,
                      literature_available, modulation_period, onset_of, percolation, percolation_errors,
                      percolation_last_time, percolation_sentence, percolation_short, reliability_report,
                      series_extent, smoothed_rows, storyboard_indices, window_sketch)
from .environment import (DOWNLOAD_SIZE, INSTALL_MESSAGES, INSTALL_STAGES, INSTALL_STEP, STOP_NOW,
                          EnvironmentInstaller, EnvironmentStatus, InstallError, InstallProgress, install_message)
from .progress import (EXCLUDED_FRAMES, FRAMES_READ, STAGE_TEXT, Excluded, Progress, ProgressTracker,
                       overall_fraction, parse_progress)
from .scan import (FILLING_HINTS, GROUPS, HINT_TEXT, NO_FRAMES, SOURCE_TEXT, SUMMARY_HINT, Prefill, ScanInfo, Source,
                   add_preset, energy_kev, folder_key, make_preset, matching_preset, prefill, remember)
from .use_cases import (METALS, NO_TABLE, PROBLEMS, STEP_ANALYZE, STEP_ANALYZE_ONLY, STEP_CHECK, STOPPED, TOOLKIT_FAILED,
                        TOOLKIT_MISSING, VARIANTS, AnalyzeRequest, PredictStep, ResultReader, ScanInspector,
                        SputterService, Stopped, ToolkitFailed, ToolkitMissing, ToolkitRunner, deposition_end_ok,
                        results_folder, scan_label)

__all__ = [
    "AnalyzeRequest", "DOWNLOAD_SIZE", "EXCLUDED_FRAMES", "EnvironmentInstaller", "EnvironmentStatus", "Excluded",
    "FILLING_HINTS", "FILM_NOT_DETERMINED", "FRAMES_READ", "GRADES", "GROUPS", "HINT_TEXT", "INSTALL_MESSAGES", "INSTALL_STAGES",
    "INSTALL_STEP", "InstallError", "InstallProgress", "LITERATURE_GRADE", "LITERATURE_TEMPLATES", "METALS", "NO_FRAMES", "NO_TABLE", "PHYSICAL_TEMPLATES", "PROBLEMS", "Particle", "PredictStep", "Prefill",
    "Progress", "ProgressTracker", "ResultReader", "SKETCH_TEMPLATES", "SOURCE_TEXT", "STAGE_TEXT", "STEP_ANALYZE",
    "STEP_ANALYZE_ONLY", "STEP_CHECK", "STOPPED", "STOP_NOW", "SUMMARY_HINT", "ScanInfo", "ScanInspector", "Series",
    "Sketcher", "Source",
    "SputterService", "Stopped", "StructureResult", "TOOLKIT_FAILED", "TOOLKIT_MISSING", "ToolkitFailed",
    "ToolkitMissing", "ToolkitRunner", "VARIANTS", "VERDICT_COLOURS", "VERDICT_MARKS",
    "WindowSketch", "add_preset", "beam_off_count", "beam_off_record", "checks_html", "deposition_end",
    "deposition_end_ok", "energy_kev", "film_onset",
    "folder_key", "install_message", "is_film",
    "literature_available", "make_preset", "matching_preset", "modulation_period", "onset_of", "overall_fraction",
    "parse_progress", "percolation", "percolation_errors", "percolation_last_time", "percolation_sentence", "percolation_short", "prefill", "reliability_report",
    "remember", "results_folder", "scan_label", "series_extent", "smoothed_rows",
    "storyboard_indices", "window_sketch",
]
