"""Pure description of an Analyze result of a sputter series and of its real-space pictures."""

from .client import (ANALYSIS_FILE, BAD_ARRAYS, BAD_MAIN, CLIENT_RESULT_TEMPLATES, LISTED_ONLY, MAIN_FILE, MAIN_FOLDER,
                     NO_ARRAYS, NO_MAIN, NO_SOLUTION, POROUS_KEYS, STATIC_KIND, ClientEntry, ClientQuantity, ClientResult,
                     ClientWindow, WindowPatterns, client_result, window_key, window_patterns)
from .process import (COMPONENT_NAMES, COMPONENTS, DESCRIPTORS, STAGE_NAMES, STAGES, ProcessAnalysis, ProcessStage,
                      SystemRange, components_listed, entry_components, process_analysis)
from .real_space import (FIELD_DISTANCES, SIDE_DISTANCES, SKETCH_KEYS, Grain, PorousSpace, RealSpace, arrangement,
                         clusters, drawn_coverage, layer_particles, pore_density, porous_space, real_space,
                         side_grains, spanning, top_grains, visible, height_map, profile, cap_height, cut, grown,
                         series_field, series_ceiling)

__all__ = ["ANALYSIS_FILE", "BAD_ARRAYS", "BAD_MAIN", "CLIENT_RESULT_TEMPLATES", "ClientEntry", "ClientQuantity",
           "ClientResult", "ClientWindow", "LISTED_ONLY", "MAIN_FILE", "MAIN_FOLDER", "NO_ARRAYS", "NO_MAIN", "NO_SOLUTION",
           "POROUS_KEYS", "STATIC_KIND", "WindowPatterns", "client_result", "window_key", "window_patterns",
           "FIELD_DISTANCES", "Grain", "PorousSpace", "RealSpace", "SIDE_DISTANCES", "SKETCH_KEYS", "arrangement",
           "clusters", "drawn_coverage", "layer_particles", "pore_density", "porous_space", "real_space",
           "side_grains", "spanning", "top_grains", "height_map", "profile", "cap_height", "visible", "cut", "grown",
           "series_field", "series_ceiling", "COMPONENT_NAMES", "COMPONENTS", "DESCRIPTORS", "STAGE_NAMES", "STAGES", "ProcessAnalysis",
           "ProcessStage", "SystemRange", "components_listed", "entry_components", "process_analysis"]
