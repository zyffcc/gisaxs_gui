"""The rows of the pipeline's report: the peaks (with how each was fitted), the peak search, the rings."""

from __future__ import annotations

from typing import Optional

from ..domain import caveat
from .tools import clean

FULL = 15
"""Significant digits kept for the values the report prints (q, d, FWHM, SNR, size; f, σ_f, f_iso, f − f_iso): each
printed number is rounded once, from the value itself (four significant digits first made a size of 29.348 nm
29.35 in the record and '29.4' in the table, next to the 29.3 of the step)."""


def peak_rows(results) -> list[dict]:
    search = results.peak_searches.get("radial")
    if search is None:
        return []
    rows = []
    for peak in search.peaks:
        sector = min(results.sector_rows, key=lambda row: abs(row.q - peak.q), default=None)
        if sector is not None and abs(sector.q - peak.q) > 0.5 * peak.fwhm:
            sector = None
        size = min(results.sizes, key=lambda item: abs(item.q - peak.q), default=None)
        if size is not None and abs(size.q - peak.q) > 0.5 * peak.fwhm:
            size = None
        row = {
            "q": peak.q, "d_A": peak.d, "fwhm": peak.fwhm, "snr": peak.snr, "flags": list(peak.flags),
            "caveat": caveat(peak),
            "orientation": None if sector is None else sector.preference,
            "ratio_out_in": None if sector is None else sector.ratio,
            "size_nm": None if size is None or size.size is None else size.size / 10.0,
            "size_is_lower_bound": None if size is None else size.lower_bound,
            "size_note": None if size is None else size.reason,
        }
        row = clean(row)
        row.update(clean({"q": peak.q, "d_A": peak.d, "fwhm": peak.fwhm, "snr": peak.snr,
                          "size_nm": None if size is None or size.size is None else size.size / 10.0}, digits=FULL))
        fit = peak_fit(search, peak)
        if fit is not None:
            row["fit"] = fit
        rows.append(row)
    return rows


def peak_fit(search, peak) -> Optional[dict]:
    """How one peak was fitted and the points it was fitted to (the Results' fit details; not for the AI)."""
    if not peak.window or search.data_y.size == 0:
        return None
    low, high = peak.window
    inside = (search.baseline_x >= low) & (search.baseline_x <= high)
    record = clean({
        "model": "Gaussian on a local linear background", "window": [low, high], "points": int(inside.sum()),
        "height": peak.height, "background": peak.background, "slope": peak.slope, "area": peak.area,
        "q_err": peak.q_err, "d_err_A": peak.d_err, "fwhm_err": peak.fwhm_err, "reduced_chi2": peak.reduced_chi2,
    })
    record.update(x=search.baseline_x[inside].tolist(), y=search.data_y[inside].tolist(),
                  sigma=search.data_sigma[inside].tolist())
    return record


def peak_search(results) -> Optional[dict]:
    search = results.peak_searches.get("radial")
    if search is None:
        return None
    # ``points``: the I(q) points searched (not bins: sparse bins are joined into one point); ``radial_bins``: the
    # bins of the reduction, as <stem>_analysis.json gives them (``binning``), when the status says.
    record = {"q_range": list(search.x_range), "points": search.points, "bin_width": search.step,
              "background_window": search.background_window, "min_snr": search.min_snr,
              "min_relative_height": search.min_relative_height,
              "candidates": search.candidates, "max_peaks": search.max_peaks, "truncated": search.truncated}
    bins = ((getattr(results, "status", None) or {}).get("binning") or {}).get("radial_bins")
    if bins:
        record = {**record, "radial_bins": bins}
    return clean(record)


def ring_row(ring) -> dict:
    low, high = ring.q_window
    row = clean({
        "q": 0.5 * (low + high), "q_window": ring.q_window, "coverage": ring.coverage,
        "weighted_coverage": ring.weighted_coverage, "herman": ring.herman, "herman_err": ring.herman_err,
        "herman_isotropic": ring.herman_isotropic, "herman_excess": ring.herman_excess,
        "texture_margin": ring.texture_margin, "herman_excess_without_horizon": ring.horizon_excess,
        "texture": ring.texture, "reason": ring.reason,
        "maxima": ring.maxima, "notes": ring.notes, "missing": ring.missing, "shadowed": ring.shadowed,
    })
    row.update(clean({"herman": ring.herman, "herman_err": ring.herman_err, "herman_isotropic": ring.herman_isotropic,
                      "herman_excess": ring.herman_excess}, digits=FULL))
    return row


__all__ = ["peak_fit", "peak_rows", "peak_search", "ring_row"]
