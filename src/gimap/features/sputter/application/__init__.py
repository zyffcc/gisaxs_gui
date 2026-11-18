"""Sputter use cases and their ports (and the result types the page shows)."""

from ..domain import (NO_SOLUTION, ClientEntry, ClientQuantity, ClientResult, ClientWindow, WindowPatterns)
from ..domain import (FIELD_DISTANCES, SIDE_DISTANCES, Grain, PorousSpace, RealSpace, arrangement, clusters,
                      layer_particles, pore_density, porous_space, real_space, side_grains, spanning, top_grains,
                      visible)
from .client import (CLIENT_FAILED, CLIENT_FALLBACK, CLIENT_INCOMPLETE, CLIENT_MISSING, CLIENT_NO_MAIN, CLIENT_PROBLEMS,
                     EXPORT_FAILED, EXPORT_FOLDER, EXPORT_NOTHING, EXPORT_OLD, CLIENT_STOPPED, CLIENT_TEMPLATES, ENGINE_FOLDER, FILM_SUBSTRATES, LABEL_PREFIX, MASK_UNREADABLE,
                     MODES, QCM_FACTOR, SUBSTRATES, ClientFailed, ClientMissing, ClientPackage, ClientReader,
                     ClientRequest, ClientRun, ClientRunner, ClientService, ClientStarted, ResultTexts)
from .client_progress import (STATIC_STEP_TEXT, STATIC_STEP_WEIGHTS, STEP_OF, STEP_REUSED, STEP_TEXT, STEP_WEIGHTS,
                              ClientProgress)
from .client_live import (FRAME_LABEL, LIVE_PREFIX, LIVE_STEPS, SLOW_DISK_S_PER_FRAME, SLOW_DISK_SHARE,
                          STATIC_LIVE_STEPS, LiveEvent, LiveRun, disk_slow, live_event, parse_live)
from .live_steps import STEP_TITLES, STEPS_RECORD, StepNode, StepTree
from .step_details import (DETAIL_KINDS, JOINT_KEYS, RANGES_SHOWN, StepDetail, detector_image, deviations,
                           frame_ranges, joint_iterations, joint_solution)
from .storage import (DEFAULT_LIMIT_GB, GB, PLACE_LOCAL, PLACE_USER, ImportCache, over_limit, total_size,
                      unused_for)
from .client_static import (REFERENCE_SUFFIX, SAMPLE_EXISTS, SERIES_ELSEWHERE, SERIES_NOT_EMPTY, SERIES_OTHER_MASK,
                            STATIC_FAILED, STATIC_MODES, STATIC_PROBLEMS, STATIC_STOPPED, STATIC_TEMPLATES,
                            STATIC_UNSUPPORTED, StaticDefaults, StaticRequest, StaticRunner, StaticService,
                            frames_found, join_frames, split_frames)
from .scan import (CHECK_SOURCES, FILLING_HINTS, GROUPS, HINT_TEXT, NO_FRAMES, SOURCE_TEXT, Prefill, ScanInfo, Source,
                   add_preset, energy_kev, folder_key, make_preset, matching_preset, prefill, remember)
from .use_cases import (FOLDER_VALUES, METALS, PROBLEMS, ScanInspector, SputterService, Stopped, scan_label,
                        scan_results_folder)

__all__ = [
    "CLIENT_FAILED", "EXPORT_FAILED", "EXPORT_FOLDER", "EXPORT_NOTHING", "EXPORT_OLD", "CLIENT_FALLBACK", "CLIENT_INCOMPLETE", "CLIENT_MISSING", "CLIENT_NO_MAIN", "CLIENT_PROBLEMS",
    "CLIENT_STOPPED", "CLIENT_TEMPLATES", "ClientEntry", "ClientFailed", "ClientMissing", "ClientPackage",
    "ClientProgress", "ClientQuantity", "ClientReader", "ClientRequest", "ClientResult", "ClientRun", "ClientRunner",
    "ClientService", "ClientStarted", "ClientWindow", "ENGINE_FOLDER", "FILM_SUBSTRATES", "LABEL_PREFIX",
    "MASK_UNREADABLE", "MODES", "QCM_FACTOR", "ResultTexts",
    "FRAME_LABEL", "LIVE_PREFIX", "LIVE_STEPS", "STATIC_LIVE_STEPS", "LiveEvent", "LiveRun", "live_event", "parse_live",
    "SLOW_DISK_SHARE", "SLOW_DISK_S_PER_FRAME", "disk_slow",
    "STEP_TITLES", "STEPS_RECORD", "StepNode", "StepTree", "DETAIL_KINDS", "JOINT_KEYS", "RANGES_SHOWN", "StepDetail",
    "detector_image", "deviations", "frame_ranges", "joint_iterations", "joint_solution",
    "DEFAULT_LIMIT_GB", "GB", "PLACE_LOCAL", "PLACE_USER", "ImportCache", "over_limit", "total_size", "unused_for",
    "NO_SOLUTION", "STEP_OF", "STEP_REUSED", "STEP_TEXT", "STEP_WEIGHTS", "SUBSTRATES", "WindowPatterns",
    "FIELD_DISTANCES", "Grain", "PorousSpace", "RealSpace", "SIDE_DISTANCES", "arrangement", "clusters",
    "layer_particles", "pore_density", "porous_space", "real_space", "side_grains", "spanning", "top_grains",
    "visible",
    "REFERENCE_SUFFIX", "SAMPLE_EXISTS", "SERIES_ELSEWHERE", "SERIES_NOT_EMPTY", "SERIES_OTHER_MASK", "STATIC_FAILED",
    "STATIC_MODES", "STATIC_PROBLEMS", "STATIC_STEP_TEXT", "STATIC_STEP_WEIGHTS", "STATIC_STOPPED",
    "STATIC_TEMPLATES", "STATIC_UNSUPPORTED", "StaticDefaults", "StaticRequest", "StaticRunner", "StaticService",
    "frames_found", "join_frames", "split_frames",
    "CHECK_SOURCES", "FILLING_HINTS", "GROUPS", "HINT_TEXT", "NO_FRAMES", "SOURCE_TEXT", "Prefill", "ScanInfo",
    "Source", "add_preset", "energy_kev", "folder_key", "make_preset", "matching_preset", "prefill", "remember",
    "FOLDER_VALUES", "METALS", "PROBLEMS", "ScanInspector", "SputterService", "Stopped", "scan_label",
    "scan_results_folder",
]
