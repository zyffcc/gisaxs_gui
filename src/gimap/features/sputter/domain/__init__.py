"""Pure description of a sputter size/structure series, its reliability and real-space pictures."""

from .client import (ANALYSIS_FILE, BAD_ARRAYS, BAD_MAIN, CLIENT_RESULT_TEMPLATES, MAIN_FILE, MAIN_FOLDER, NO_ARRAYS,
                     NO_MAIN, NO_SOLUTION, POROUS_KEYS, STATIC_KIND, ClientEntry, ClientQuantity, ClientResult,
                     ClientWindow, WindowPatterns, client_result, window_key, window_patterns)
from .fringes import (FRINGE_FOLDER, HEIGHTS_FILE, STATUS_NAMES as FRINGE_STATUS_NAMES, SUMMARY_FILE,
                      TEMPLATES as FRINGE_TEMPLATES, FringeGroup, FringeHeights, fringe_paragraph, parse_heights)
from .literature import (LITERATURE_GRADE, TEMPLATES as LITERATURE_TEMPLATES, VERDICT_COLOURS, VERDICT_MARKS,
                         checks_html, literature_available, literature_rows, onset_of, percolation,
                         percolation_errors, percolation_last_time, percolation_sentence, percolation_short)
from .physical import (FILM_NOT_DETERMINED, TEMPLATES as PHYSICAL_TEMPLATES, beam_off_count, beam_off_record,
                       deposition_end, film_onset, film_onset_established, modulation_period)
from .real_space import (FIELD_DISTANCES, SIDE_DISTANCES, SKETCH_KEYS, Grain, PorousSpace, RealSpace, arrangement,
                         clusters, drawn_coverage, layer_particles, pore_density, porous_space, real_space,
                         side_grains, spanning, top_grains, visible)
from .sketch import (TEMPLATES as SKETCH_TEMPLATES, Particle, Sketcher, WindowSketch, series_extent, smoothed_rows,
                     storyboard_indices, window_sketch)
from .structure import GRADES, Series, StructureResult, english, flag, is_film, number, parse_row, reliability_report
from .windows import (FRAMES_MISSING, NO_MEMBERS, NO_SCAN, NO_WINDOWS, SCAN_MISSING, UNKNOWN_WINDOW, WINDOW_TEMPLATES,
                      SputterGeometry, WindowFrames, geometry_of, window_frames)

__all__ = ["ANALYSIS_FILE", "BAD_ARRAYS", "BAD_MAIN", "CLIENT_RESULT_TEMPLATES", "ClientEntry", "ClientQuantity",
           "ClientResult", "ClientWindow", "MAIN_FILE", "MAIN_FOLDER", "NO_ARRAYS", "NO_MAIN", "NO_SOLUTION",
           "POROUS_KEYS", "STATIC_KIND", "WindowPatterns", "client_result", "window_key", "window_patterns",
           "FIELD_DISTANCES", "Grain", "PorousSpace", "RealSpace", "SIDE_DISTANCES", "SKETCH_KEYS", "arrangement",
           "clusters", "drawn_coverage", "layer_particles", "pore_density", "porous_space", "real_space",
           "side_grains", "spanning", "top_grains", "visible",
           "FRINGE_FOLDER", "FRINGE_STATUS_NAMES", "FRINGE_TEMPLATES", "FringeGroup", "FringeHeights", "HEIGHTS_FILE",
           "SUMMARY_FILE", "fringe_paragraph", "parse_heights",
           "FRAMES_MISSING", "NO_MEMBERS", "NO_SCAN", "NO_WINDOWS", "SCAN_MISSING", "SputterGeometry", "UNKNOWN_WINDOW",
           "WINDOW_TEMPLATES", "WindowFrames", "geometry_of", "window_frames",
           "FILM_NOT_DETERMINED", "GRADES", "LITERATURE_GRADE", "LITERATURE_TEMPLATES", "PHYSICAL_TEMPLATES", "Particle",
           "SKETCH_TEMPLATES", "beam_off_count", "beam_off_record", "modulation_period",
           "Series", "Sketcher", "StructureResult", "VERDICT_COLOURS", "VERDICT_MARKS", "WindowSketch", "checks_html",
           "deposition_end", "english", "film_onset", "film_onset_established", "flag", "is_film", "literature_available", "literature_rows",
           "number", "onset_of", "parse_row", "percolation", "percolation_errors", "percolation_last_time",
           "percolation_sentence",
           "percolation_short",
           "reliability_report",
           "series_extent", "smoothed_rows", "storyboard_indices", "window_sketch"]
