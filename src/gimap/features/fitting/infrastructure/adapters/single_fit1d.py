"""The single-page fit1d handler; imported only by the isolated JobRunner process."""

from __future__ import annotations

import time

from .fit1d_method import _clean, load_kit, solution_row
from ...application.workflow_v5 import FIT1D_SHAPES, validate_options


def run_single_fit1d(payload, report, is_cancelled):
    options = validate_options(payload["options"])
    if is_cancelled():
        raise RuntimeError("Fitting cancelled")
    report(0, 1, "Loading fit1d", {})
    core = load_kit()
    arguments = dict(q=payload["q"], intensity=payload["intensity"], sigma=payload.get("sigma"),
                     q_unit="nm^-1", composition=",".join(FIT1D_SHAPES[t] for t in options["components"]),
                     mode=options["fit1d_mode"], policy=options["fit1d_policy"], model=options["fit1d_model"],
                     qz=float(options["fit1d_qz"] or 0.0) if options["fit1d_mode"] == "gisaxs" else 0.0,
                     sigma_floor=float(options["fit1d_sigma_floor"]))
    started = time.perf_counter()
    report(0, 1, "Fitting the particles in Model with fit1d", {})
    result = core.fit_curve(arguments)
    if is_cancelled():
        raise RuntimeError("Fitting cancelled")
    # Its own effective errors are also the errors of the displayed curve and of the saved score.
    item = dict(q=result["curves"]["q_nm_inv"], observed=result["curves"]["intensity"],
                sigma=result["curves"]["sigma"], sigma_estimated=False,
                side="positive", sign=1)
    row = solution_row(result, item, options, time.perf_counter() - started)
    result["warnings"] = row["warnings"]
    result["measurement_sigma"] = payload.get("sigma")
    result["settings"] = options
    report(1, 1, "fit1d finished", {})
    return _clean(dict(native=result, mapping_row=row))


__all__ = ["run_single_fit1d"]
