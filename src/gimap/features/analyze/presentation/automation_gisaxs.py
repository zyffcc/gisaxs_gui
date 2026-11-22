"""Detector anchors, measured pixels and reciprocal-space boundaries of the active GISAXS cuts."""


def _pixel_span(region, axis):
    span = region.get("pixel_" + axis) if "gisaxs_cut" in region else region.get(axis)
    if not span:
        return None
    start, stop = int(span[0]), int(span[1])
    return [start, max(start, stop - 1)]


def gisaxs_cut_status(reduction, halves, symmetry):
    markers = reduction.markers
    yoneda = markers.get("yoneda")
    horizontal, vertical = reduction.curve("horizontal"), reduction.curve("vertical")
    h = horizontal.region if horizontal is not None else {}
    v = vertical.region if vertical is not None else {}
    status = {
        "horizontal_rows": list(map(float, markers["horizontal_band"])),
        "horizontal_pixel_rows": _pixel_span(h, "rows"),
        "horizontal_source": markers.get("horizontal_source"),
        "yoneda_alpha_f_deg": None if yoneda is None else float(yoneda.alpha_f_deg),
        "yoneda_row": None if yoneda is None else float(yoneda.row),
        "horizon_row": markers.get("horizon_row"),
        "vertical_columns": list(map(float, markers["vertical_band"])),
        "vertical_source": markers.get("vertical_source"),
        "vertical_pixel_columns": _pixel_span(v, "columns"),
        "halves": halves,
        "symmetry": symmetry,
    }
    if "gisaxs_cut" in h:
        status["horizontal_qz_edges_A_inv"] = h["qz_edges"]
    if "gisaxs_cut" in v:
        status["vertical_qy_edges_A_inv"] = v["qy_edges"]
    return status
