"""Sputter use cases and their ports (and the result types the page shows)."""

from ..domain import (NO_SOLUTION, ClientEntry, ClientQuantity, ClientResult, ClientWindow, WindowPatterns)
from ..domain import (FIELD_DISTANCES, SIDE_DISTANCES, Grain, PorousSpace, RealSpace, arrangement, clusters,
                      layer_particles, pore_density, porous_space, real_space, side_grains, spanning, top_grains,
                      visible)
from .client import (CLIENT_FAILED, CLIENT_INCOMPLETE, CLIENT_MISSING, CLIENT_NO_MAIN, CLIENT_PROBLEMS, CLIENT_STOPPED,
                     CLIENT_TEMPLATES, ENGINE_FOLDER, FILM_SUBSTRATES, LABEL_PREFIX, MASK_UNREADABLE, MODES, SUBSTRATES,
                     ClientFailed, ClientMissing, ClientPackage, ClientReader, ClientRequest, ClientRun, ClientRunner,
                     ClientService, ClientStarted)
from .client_progress import (STATIC_STEP_TEXT, STATIC_STEP_WEIGHTS, STEP_OF, STEP_REUSED, STEP_TEXT, STEP_WEIGHTS,
                              ClientProgress)
from .client_static import (REFERENCE_SUFFIX, SAMPLE_EXISTS, SERIES_ELSEWHERE, SERIES_NOT_EMPTY, SERIES_OTHER_MASK,
                            STATIC_FAILED, STATIC_MODES, STATIC_PROBLEMS, STATIC_STOPPED, STATIC_TEMPLATES,
                            STATIC_UNSUPPORTED, StaticDefaults, StaticRequest, StaticRunner, StaticService,
                            frames_found, join_frames, split_frames)
from ..domain import (FRINGE_STATUS_NAMES, FRINGE_TEMPLATES, FringeGroup, FringeHeights, fringe_paragraph,
                      FILM_NOT_DETERMINED, GRADES, LITERATURE_GRADE, LITERATURE_TEMPLATES, PHYSICAL_TEMPLATES,
                      SKETCH_TEMPLATES, VERDICT_COLOURS, VERDICT_MARKS, WINDOW_TEMPLATES, Particle, Series, Sketcher,
                      SputterGeometry, StructureResult, WindowFrames, WindowSketch, beam_off_count, beam_off_record,
                      checks_html, deposition_end, film_onset, film_onset_established, is_film,
                      literature_available, modulation_period, onset_of, percolation, percolation_errors,
                      percolation_last_time, percolation_sentence, percolation_short, reliability_report,
                      series_extent, smoothed_rows, storyboard_indices, window_sketch)
from .environment import (DOWNLOAD_SIZE, INSTALL_MESSAGES, INSTALL_STAGES, INSTALL_STEP, STOP_NOW,
                          EnvironmentInstaller, EnvironmentStatus, InstallError, InstallProgress, install_message)
from .fringes import (FALLBACK_FOLDER, FRINGES_FAILED, FRINGES_MISSING, FringeRunner, FringeService, FringeStore,
                      FringesFailed)
from .progress import (EXCLUDED_FRAMES, FRAMES_READ, STAGE_TEXT, Excluded, Progress, ProgressTracker,
                       overall_fraction, parse_progress)
from .scan import (CHECK_SOURCES, FILLING_HINTS, GROUPS, HINT_TEXT, NO_FRAMES, SOURCE_TEXT, Prefill, ScanInfo, Source,
                   add_preset, energy_kev, folder_key, make_preset, matching_preset, prefill, remember)
from .use_cases import (METALS, NO_TABLE, PROBLEMS, STEP_ANALYZE, STEP_ANALYZE_ONLY, STEP_CHECK, STOPPED, TOOLKIT_FAILED,
                        TOOLKIT_MISSING, VARIANTS, AnalyzeRequest, PredictStep, ResultReader, ScanInspector,
                        SputterService, Stopped, ToolkitFailed, ToolkitMissing, ToolkitRunner, WindowReader,
                        deposition_end_ok, results_folder, scan_label)

__all__ = [
    "CLIENT_FAILED", "CLIENT_INCOMPLETE", "CLIENT_MISSING", "CLIENT_NO_MAIN", "CLIENT_PROBLEMS", "CLIENT_STOPPED",
    "CLIENT_TEMPLATES", "ClientEntry", "ClientFailed", "ClientMissing", "ClientPackage", "ClientProgress",
    "ClientQuantity", "ClientReader", "ClientRequest", "ClientResult", "ClientRun", "ClientRunner", "ClientService",
    "ClientStarted", "ClientWindow", "ENGINE_FOLDER", "FILM_SUBSTRATES", "LABEL_PREFIX", "MASK_UNREADABLE", "MODES",
    "NO_SOLUTION", "STEP_OF", "STEP_REUSED", "STEP_TEXT", "STEP_WEIGHTS", "SUBSTRATES", "WindowPatterns",
    "FIELD_DISTANCES", "Grain", "PorousSpace", "RealSpace", "SIDE_DISTANCES", "arrangement", "clusters",
    "layer_particles", "pore_density", "porous_space", "real_space", "side_grains", "spanning", "top_grains",
    "visible",
    "REFERENCE_SUFFIX", "SAMPLE_EXISTS", "SERIES_ELSEWHERE", "SERIES_NOT_EMPTY", "SERIES_OTHER_MASK", "STATIC_FAILED",
    "STATIC_MODES", "STATIC_PROBLEMS", "STATIC_STEP_TEXT", "STATIC_STEP_WEIGHTS", "STATIC_STOPPED",
    "STATIC_TEMPLATES", "STATIC_UNSUPPORTED", "StaticDefaults", "StaticRequest", "StaticRunner", "StaticService",
    "frames_found", "join_frames", "split_frames",
    "FALLBACK_FOLDER", "FRINGES_FAILED", "FRINGES_MISSING", "FRINGE_STATUS_NAMES", "FRINGE_TEMPLATES", "FringeGroup",
    "FringeHeights", "FringeRunner", "FringeService", "FringeStore", "FringesFailed", "fringe_paragraph",
    "AnalyzeRequest", "CHECK_SOURCES", "DOWNLOAD_SIZE", "EXCLUDED_FRAMES", "EnvironmentInstaller", "EnvironmentStatus", "Excluded",
    "FILLING_HINTS", "FILM_NOT_DETERMINED", "FRAMES_READ", "GRADES", "GROUPS", "HINT_TEXT", "INSTALL_MESSAGES", "INSTALL_STAGES",
    "INSTALL_STEP", "InstallError", "InstallProgress", "LITERATURE_GRADE", "LITERATURE_TEMPLATES", "METALS", "NO_FRAMES", "NO_TABLE", "PHYSICAL_TEMPLATES", "PROBLEMS", "Particle", "PredictStep", "Prefill",
    "Progress", "ProgressTracker", "ResultReader", "SKETCH_TEMPLATES", "SOURCE_TEXT", "STAGE_TEXT", "STEP_ANALYZE",
    "STEP_ANALYZE_ONLY", "STEP_CHECK", "STOPPED", "STOP_NOW", "ScanInfo", "ScanInspector", "Series",
    "Sketcher", "Source", "SputterGeometry",
    "SputterService", "Stopped", "StructureResult", "TOOLKIT_FAILED", "TOOLKIT_MISSING", "ToolkitFailed",
    "ToolkitMissing", "ToolkitRunner", "VARIANTS", "VERDICT_COLOURS", "VERDICT_MARKS", "WINDOW_TEMPLATES",
    "WindowFrames", "WindowReader", "WindowSketch", "add_preset", "beam_off_count", "beam_off_record", "checks_html",
    "deposition_end", "deposition_end_ok", "energy_kev", "film_onset", "film_onset_established",
    "folder_key", "install_message", "is_film",
    "literature_available", "make_preset", "matching_preset", "modulation_period", "onset_of", "overall_fraction",
    "parse_progress", "percolation", "percolation_errors", "percolation_last_time", "percolation_sentence", "percolation_short", "prefill", "reliability_report",
    "remember", "results_folder", "scan_label", "series_extent", "smoothed_rows",
    "storyboard_indices", "window_sketch",
]
