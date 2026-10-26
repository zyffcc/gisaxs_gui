"""The fringe analysis of one Sputter result folder and its two output files.

``fringe_heights.csv``: one row per time group — time_s (mean of its windows), t_lo_s, t_hi_s, windows, raw_frames,
first_window, last_window, n_orders, orders, method ("minima": from the minima of known order; "phase-locked fit"),
H_nm, H_lo68_nm, H_hi68_nm, H_upper_bound_nm (``below_range`` only), H_phase_nm (the accepted phase-locked fit, for
comparison), alpha_c_used_deg, alpha_c_source, minima (the minima of moving tracks: order, alpha_f +- error in deg,
sides), other_minima (minima of no moving track), status.

``fringe_summary.json``: the status of the scan, the geometry, the q_y band, alpha_c (used, bulk, fit), the clean
alpha_f ranges, the detector gaps and static rows found in the data, the tracks (accepted and left out), the growth
fit H = v (t - t0) (t0: the deposition start the fringes imply, information only) with the change of v for alpha_c
+-0.01 deg, counts, the settings, notes, and the SHA-256 of run.json, of every chunk and of
structure_physical.csv, and of this tool's own sources (``tool_sha256``).

Both are written atomically (a temporary file, then ``os.replace``), the summary last.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from . import VERSION
from .detect import detect, make_groups, phase_cuts, phase_pass
from .heights import (AC_BULK_ERROR, BASELINE_SYSTEMATIC, STATUSES, fit_alpha_c, group_heights, growth_fit,
                      growth_points, heights_at, orders_consistent, statuses, upper_bound_nm)
from .loader import MIN_WINDOW_COUNTS, SourceError, read_scan, toolkit_grid
from .minima import cell_quality, clean_ranges, gap_ranges, lower_bound
from .minima import settings as minima_settings
from .optics import ELECTRON_DENSITY, bulk_critical_angle_deg, q_inside, qz_vacuum, wavelength_nm, wavenumber
from .phase import alpha_c_scan
from .phase import settings as phase_settings
from .tracks import STATIC_SECONDS, annotate, build
from .tracks import settings as track_settings

HEIGHTS = "fringe_heights.csv"
SUMMARY = "fringe_summary.json"
COLUMNS = ("time_s", "t_lo_s", "t_hi_s", "windows", "raw_frames", "first_window", "last_window", "n_orders", "orders",
           "method", "H_nm", "H_lo68_nm", "H_hi68_nm", "H_upper_bound_nm", "H_phase_nm", "alpha_c_used_deg",
           "alpha_c_source", "minima", "other_minima", "status")
PROTECTED = ("binned", "intake")
DEFINITION = ("H = 2 pi n / (kz(alpha_i) + kz(alpha_f)), kz(a) = k sqrt(sin^2 a - sin^2 alpha_c): the height of a "
              "uniform layer (box profile) whose order-n minimum of the vertical form factor sinc(Q H / 2) is at "
              "alpha_f, with refraction inside the metal. It is the vertical size of the islands or of the layer; it "
              "equals the film thickness only once a continuous film has formed.")


def tool_sha256(folder: Optional[Path] = None) -> str:
    """SHA-256 over this package's Python sources (names and bytes, sorted): a new tool makes old outputs stale."""
    folder = Path(folder) if folder else Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(folder.glob("*.py")):
        digest.update(path.name.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def _range_record(ranges: dict, alpha_i: float, alpha_c: float, k: float) -> dict:
    def rounded(values):
        return [[round(float(a), 4), round(float(b), 4)] for a, b in values]

    return {"clean_alpha_f_deg": {str(s): rounded(r) for s, r in ranges.items()},
            "clean_qz_vacuum_nm_inv": {str(s): rounded([(qz_vacuum(a, alpha_i, k), qz_vacuum(b, alpha_i, k))
                                                        for a, b in r]) for s, r in ranges.items()},
            "clean_q_inside_nm_inv": {str(s): rounded([(q_inside(a, alpha_i, alpha_c, k),
                                                        q_inside(b, alpha_i, alpha_c, k)) for a, b in r])
                                      for s, r in ranges.items()}}


def _minima_text(group) -> tuple:
    shown, other = [], 0
    for d in sorted(group.detections, key=lambda d: d.af):
        if not d.accepted:
            other += 1
            continue
        sides = d.sides if d.sides in ("both", "disagree") else f"side {d.sides}"
        if d.sides == "disagree":
            sides += " " + "/".join(f"{v[0]:.4f}" for _s, v in sorted(d.per_side.items()))
        shown.append(f"n={d.order or '?'}@{d.af:.4f}+-{d.sd:.4f}({sides})")
    return "; ".join(shown), other


def _number(value, digits: int = 5) -> str:
    return "" if value is None or not math.isfinite(value) else f"{value:.{digits}g}"


def _alpha_c(groups, alpha_c, bulk, metal, alpha_i, k, notes, cuts: list, upper: float) -> tuple:
    """(alpha_c, its error, its source, the fit records): given, else the scan of the phase-locked fits, else the fit
    of coexisting orders, else the bulk value."""
    if alpha_c is not None:
        return float(alpha_c), 0.0, "given", {"reason": "alpha_c given"}
    records = {}
    if bulk is not None:
        records["phase_locked"] = alpha_c_scan(cuts, alpha_i, k, bulk, upper, q_inside)
    records["coexisting_orders"] = fit_alpha_c(groups, alpha_i, k, bulk)
    for key, source in (("phase_locked", "phase-locked fits"), ("coexisting_orders", "fit of coexisting orders")):
        fit = records.get(key) or {}
        if fit.get("fitted"):
            return fit["alpha_c_deg"], fit["error_deg"], source, records
    reasons = "; ".join(r["reason"] for r in records.values() if r.get("reason"))
    notes.append(f"alpha_c is the bulk value of {metal} ({bulk:.4f} deg, assumed error {AC_BULK_ERROR} deg: islands "
                 f"may be less dense; {reasons})")
    return float(bulk), AC_BULK_ERROR, f"bulk {metal}", records


def _systematic(groups, alpha_i, ac, k, growth) -> dict:
    """How much v changes when alpha_c changes by +-0.01 deg (the minima and phases as found)."""
    out = {}
    for sign, name in ((-1, "minus"), (1, "plus")):
        fit = growth_fit(growth_points(groups, heights_at(groups, alpha_i, max(ac + sign * 0.01, 0.0), 0.0, k)))
        if fit.get("fitted"):
            out[f"v_change_percent_alpha_c_{name}_0.01deg"] = 100 * (fit["v_nm_s"] / growth["v_nm_s"] - 1)
    return out


def _span_q(ranges: dict, alpha_i: float, alpha_c: float, k: float) -> float:
    """The span of Q (inside the layer) of the clean range: from its lowest to its highest alpha_f."""
    edges = [edge for r in ranges.values() for pair in r for edge in pair]
    if not edges:
        return 0.0
    return float(q_inside(max(edges), alpha_i, alpha_c, k) - q_inside(min(edges), alpha_i, alpha_c, k))


def _clean(scan, grid, lower: float) -> tuple:
    usable, gaps, ranges = {}, {}, {}
    for side in (0, 1):
        usable[side], gap = cell_quality(scan.density[side])
        gaps[side] = gap_ranges(gap, grid.cell_edges)
        ranges[side] = clean_ranges(usable[side], gap, grid.cell_edges, lower)
    return usable, gaps, ranges


def _distrust_orders(groups, tracks, consistency, notes) -> None:
    notes.append(f"heights of coexisting orders (or of the minima and the phase-locked fit) differ in "
                 f"{consistency['pairs_inconsistent']} of {consistency['pairs_checked']} pairs: the orders are not "
                 "trusted")
    for track in tracks:
        if track.order:
            track.order, track.order_source = 0, "orders contradict each other"
    for g in groups:
        for d in g.detections:
            d.order = 0


def _row(g, scan, ac, source) -> dict:
    minima, other = _minima_text(g)
    height = g.height or {}
    H, error = height.get("H_nm"), height.get("error_nm")
    return {"time_s": _number(g.time, 6), "t_lo_s": _number(g.t_lo, 6), "t_hi_s": _number(g.t_hi, 6),
            "windows": len(g.windows), "raw_frames": g.raw_frames, "first_window": scan.frames[g.windows[0]],
            "last_window": scan.frames[g.windows[-1]], "n_orders": len(height.get("orders", [])),
            "orders": " ".join(map(str, height.get("orders", []))), "method": height.get("method", ""),
            "H_nm": _number(H), "H_lo68_nm": _number(None if H is None else H - error),
            "H_hi68_nm": _number(None if H is None else H + error), "H_upper_bound_nm": _number(g.bound_nm),
            "H_phase_nm": _number(g.phase[0] if g.phase is not None else None), "alpha_c_used_deg": _number(ac),
            "alpha_c_source": source, "minima": minima, "other_minima": other, "status": g.status}


def analyse(folder, *, metal: Optional[str] = None, alpha_c: Optional[float] = None, group_seconds: float = 1.0,
            band: tuple = (0.05, 0.15), grid=None, seed: int = 0, on_line=print) -> tuple:
    """(rows, summary) of one result folder."""
    grid = grid if grid is not None else toolkit_grid()
    scan = read_scan(folder, grid, band)
    notes = list(scan.notes)
    metal = metal or scan.metal
    energy, alpha_i = float(scan.geometry["energy_ev"]), float(scan.geometry["alpha_i_deg"])
    k = wavenumber(energy)
    bulk = bulk_critical_angle_deg(metal, energy)
    if alpha_c is None and bulk is None:
        raise SourceError(f"the metal ({metal or 'unknown'}) has no tabulated electron density: give --metal "
                          f"({', '.join(ELECTRON_DENSITY)}) or --alpha-c")
    tracking_ac = float(alpha_c) if alpha_c is not None else float(bulk)
    lower, lower_reason = lower_bound(alpha_i, max(tracking_ac, bulk or 0.0))
    usable, gaps, ranges = _clean(scan, grid, lower)
    tops = [hi for r in ranges.values() for _lo, hi in r]
    top = max(tops) if tops else None
    groups = make_groups(scan, group_seconds)
    on_line(f"FRINGES\tgroups\t{len(groups)}")
    static = {side: np.zeros(len(grid.cell_af), bool) for side in (0, 1)}
    tracks = []
    if top is None:
        notes.append("no clean alpha_f range above the exclusions: nothing can be measured")
    else:
        static = detect(scan, groups, ranges, usable, alpha_i, k, np.random.default_rng(seed))
        phase_pass(scan, groups, ranges, usable, static, alpha_i, tracking_ac, k)
        annotate(groups, alpha_i, tracking_ac, k)
        tracks = build(groups, top, group_seconds, alpha_i, tracking_ac, k)
    cuts = phase_cuts(scan, groups, ranges, usable, static, alpha_i, k) if top is not None else []
    ac, ac_error, source, fit = _alpha_c(groups, alpha_c, bulk, metal, alpha_i, k, notes, cuts, lower - 0.05)
    if top is not None and abs(ac - tracking_ac) > 1e-3:
        phase_pass(scan, groups, ranges, usable, static, alpha_i, ac, k)
    consistency = orders_consistent(groups, alpha_i, ac, ac_error, k)
    if not consistency["consistent"]:
        _distrust_orders(groups, tracks, consistency, notes)
    span_q = _span_q(ranges, alpha_i, ac, k)
    group_heights(groups, alpha_i, ac, ac_error, k, span_q)
    bound = upper_bound_nm(top, alpha_i, ac, k) if top is not None else None
    statuses(groups, tracks, bound)
    growth = growth_fit(growth_points(groups))
    if growth.get("fitted"):
        growth["deposition_start_implied_s"] = growth["t0_s"]
        growth["note"] = ("a straight line through the measured groups (the growth need not be linear); t0, where it "
                          "reaches H = 0, is the deposition start the fringes imply (information only)")
        growth.update(_systematic(groups, alpha_i, ac, k, growth))
    counts = {status: sum(g.status == status for g in groups) for status in STATUSES}
    status = next((s for s in ("measured", "one_side_only", "ambiguous_order", "sides_disagree") if counts[s]),
                  "not_detected")
    if not any(t.status == "accepted" for t in tracks) and not any(g.phase is not None for g in groups):
        thinner = f" (a layer thinner than {bound:.2f} nm has its first minimum above the range)" if bound else ""
        notes.append("no minimum moves through the range like the fringe of a growing layer: H is not measured"
                     + thinner)
    static_minima = sum(d.static for g in groups for d in g.detections)
    if static_minima:
        notes.append(f"{static_minima} minima stay at the same alpha_f for {STATIC_SECONDS:.0f} s or longer "
                     "(detector or film features, not a growing layer) and are left out")
    notes.append("the Born-slab phase (minima at H Q = 2 pi n) may leave a constant offset of about 1 nm in H")
    used = scan.usable
    summary = {
        "tool": "sputter_fringes", "version": VERSION, "tool_sha256": tool_sha256(),
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"), "folder": str(Path(folder)),
        "status": status, "status_counts": counts, "groups": len(groups), "group_seconds": group_seconds,
        "metal": metal or None, "qy_band_nm_inv": list(scan.band),
        "geometry": {"energy_ev": energy, "wavelength_nm": wavelength_nm(energy), "k_nm_inv": k,
                     "alpha_i_deg": alpha_i, **{key: scan.geometry.get(key) for key in
                                                ("sdd_mm", "pixel_mm", "sample_roll_deg")}},
        "alpha_c": {"used_deg": ac, "error_deg": ac_error, "source": source, "bulk_deg": bulk,
                    "bulk_electron_density_e_per_A3": ELECTRON_DENSITY.get(metal), "tracking_deg": tracking_ac,
                    "fit": fit},
        "ranges": {"lower_bound_deg": lower, "lower_bound_reason": lower_reason,
                   "detector_gaps_deg": {str(s): [[round(a, 4), round(b, 4)] for a, b in v] for s, v in gaps.items()},
                   "static_rows_deg": {str(s): [round(float(x), 4) for x in grid.cell_af[v]]
                                       for s, v in static.items()},
                   **_range_record(ranges, alpha_i, ac, k), "top_alpha_f_deg": top, "span_q_inside_nm_inv": span_q,
                   "baseline_systematic": [list(pair) for pair in BASELINE_SYSTEMATIC],
                   "H_upper_bound_below_range_nm": bound},
        "windows": {"total": len(scan.time), "used": int(used.sum()), "beam_off_sample": int(scan.beam_off.sum()),
                    "below_count_threshold": int(((~scan.beam_off) & ~used).sum()),
                    "count_threshold": MIN_WINDOW_COUNTS},
        "phase_locked_groups": sum(g.phase is not None for g in groups),
        "tracks": [t.summary() for t in tracks],
        "tracks_left_out": {kind: sum(t.status == kind for t in tracks) for kind in ("short", "stationary")},
        "static_minima": static_minima, "orders": consistency, "growth": growth,
        "settings": {"minima": minima_settings(), "phase": phase_settings(), "tracking": track_settings()},
        "definition": DEFINITION, "sources": scan.sources, "notes": notes}
    return [_row(g, scan, ac, source) for g in groups], summary


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_outputs(out: Path, rows: list, summary: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write(out / HEIGHTS, buffer.getvalue().encode("utf-8"))
    _atomic_write(out / SUMMARY, json.dumps(summary, indent=1, default=float).encode("utf-8"))


def refuse_output(folder: Path, out: Path) -> Optional[str]:
    """Why ``out`` may not receive the outputs (the input itself, or the toolkit's folders in it), else None."""
    folder, out = Path(folder).resolve(), Path(out).resolve()
    if out == folder:
        return "the output folder is the result folder itself"
    for name in PROTECTED:
        if out == folder / name or folder / name in out.parents:
            return f"the output folder is inside the result's {name}/ folder"
    return None


def parse(argv=None):
    parser = argparse.ArgumentParser(description="Island or layer height H(t) from the moving q_z fringes.")
    parser.add_argument("--folder", required=True, help="GIMaP Sputter result folder (read only)")
    parser.add_argument("--out", required=True, help="folder for fringe_heights.csv and fringe_summary.json")
    parser.add_argument("--metal", choices=sorted(ELECTRON_DENSITY), help="default: the result's metal")
    parser.add_argument("--alpha-c", type=float, help="critical angle of the metal (deg); default: fitted or bulk")
    parser.add_argument("--group-seconds", type=float, default=1.0, help="time group length (s), default 1.0")
    parser.add_argument("--qy-band", type=float, nargs=2, default=(0.05, 0.15), metavar=("LO", "HI"),
                        help="q_y band of the vertical cuts (nm^-1), default 0.05 0.15")
    parser.add_argument("--toolkit", help="the Sputter toolkit folder (default: utils/sputter_toolkit)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the Poisson bootstrap")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse(argv)
    folder, out = Path(args.folder), Path(args.out)
    refused = refuse_output(folder, out)
    if refused:
        print(f"FRINGES_ERROR\t{refused}")
        return 2
    if not args.group_seconds > 0 or (args.alpha_c is not None and not 0 < args.alpha_c < 2):
        print("FRINGES_ERROR\tthe group length must be positive and alpha_c between 0 and 2 deg")
        return 2
    try:
        rows, summary = analyse(folder, metal=args.metal, alpha_c=args.alpha_c, group_seconds=args.group_seconds,
                                band=tuple(args.qy_band), grid=toolkit_grid(args.toolkit), seed=args.seed)
    except SourceError as exc:
        print(f"FRINGES_ERROR\t{exc}")
        return 2
    write_outputs(out, rows, summary)
    print(f"FRINGES_DONE\t{summary['status']}\t{out}")
    return 0


__all__ = ["COLUMNS", "HEIGHTS", "SUMMARY", "analyse", "main", "refuse_output", "tool_sha256", "write_outputs"]
