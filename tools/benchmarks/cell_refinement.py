"""Benchmark indexed GIWAXS cells on noisy peak lists, a detector series and public frames."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXTERNAL = ROOT / "tests/data/external"


def public_case(name, folder, visit=None):
    from src.gimap.app.headless_assistant import open_headless_session
    from src.gimap.features.analyze.application.cell_refinement import frame_cell_refinement
    from src.gimap.features.analyze.application.indexing import assign_peaks
    from src.gimap.features.assistant.application import PipelineOptions

    manifest = json.loads((EXTERNAL / "manifest.json").read_text("utf-8"))
    info = manifest[name]
    frame = EXTERNAL / info["frame"]
    p08 = name == "giwaxs_p08_mapi"
    files = ("MAPbI3_reference.cif", "PbI2_reference.cif") if p08 else ("PbTe_bulk_reference.cif",)
    options = PipelineOptions(technique="giwaxs", calibration=str(EXTERNAL / info["poni"]),
                              incidence_deg=info["incidence_deg"], phases=tuple(str(frame.parent / f) for f in files))
    session = open_headless_session(frame, saved_profiles=False, out_dir=folder)
    try:
        if "mask" in info:
            session.apply_mask(EXTERNAL / info["mask"])
        report = session.run_pipeline(options)
        if not report.get("ok") or not report.get("cell_refinement"):
            raise RuntimeError(f"{name}: {report.get('failed') or report.get('needs_attention')}")
        result = {k: report[k] for k in ("geometry", "phase_assignment", "cell_refinement")}
        if not p08:
            # The shipped Xeuss geometry puts the three rock-salt orders below the bulk reference.
            # Test the explicit 111/200/220 hypothesis documented alongside these public data.
            # Retain the direct indexing outcome and the CIF for the strain reference.
            rows = report["phase_assignment"]["assignments"]
            selected = []
            for centre in (1.57, 1.82, 2.58):
                inside = [row for row in rows if abs(row["q"] - centre) < .04]
                if len(inside) != 1:
                    raise ValueError(f"The documented Xeuss window at {centre} Å⁻¹ has {len(inside)} peaks.")
                selected.append(inside[0])
            # Anchor the trial metric to the order with the narrowest fractional width.
            anchor, order_squared = min(zip(selected, (3, 4, 8)), key=lambda pair: pair[0]["fwhm"]/pair[0]["q"])
            trial_a = 2 * math.pi * math.sqrt(order_squared) / anchor["q"]
            original = session.page.indexing.phases[0]
            trial = replace(original, cell=(trial_a, trial_a, trial_a, 90., 90., 90.))
            analysis = session.page.view_model.state.analysis
            assigned = assign_peaks(selected, [trial], geometry=analysis.geometry)
            if [row["hkl"] for row in assigned["assignments"]] != ["111", "200", "220"]:
                measured = [(r["q"], r["fwhm"], r["hkl"]) for r in assigned["assignments"]]
                raise ValueError(f"The trial metric did not index the documented Xeuss orders: {measured}.")
            result["rock_salt_hypothesis"] = frame_cell_refinement(analysis, [trial], assignment=assigned)[0]
        if visit is not None:
            visit(session, result)
        return result
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=["all", "synthetic", "series", "public"], default="all")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    result = {}
    with tempfile.TemporaryDirectory() as home:
        os.environ["GIMAP_HOME"] = home
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        if os.name == "nt":
            os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
        if args.part in ("synthetic", "all"):
            from tools.benchmarks.cell_coverage import coverage
            result["synthetic"] = coverage(args.trials)
            for case, run in result["synthetic"].items():
                print(case, {k: v["coverage"] for k, v in run["parameters"].items()})
        if args.part in ("series", "all"):
            from tools.benchmarks.cell_series import benchmark
            t, a, c, record = benchmark(home)
            fits = [row["fits"]["Tetragonal film"] for row in record["frames"]]
            result["series"] = {"time_s": t.tolist()}
            for key, truth in (("a", a), ("c", c), ("V", a*a*c)):
                measured = np.array([f["values"][key] for f in fits])
                result["series"][key] = dict(truth=truth.tolist(), fitted=measured.tolist(),
                                             sigma=[f["errors"][key] for f in fits])
                print("Series", key, "RMSE", np.sqrt(np.mean((measured-truth)**2)))
        if args.part in ("public", "all"):
            for name in ("giwaxs_p08_mapi", "giwaxs_xeuss_pbte"):
                run = result[name] = public_case(name, Path(home) / name)
                for fit in [*run["cell_refinement"], *([run["rock_salt_hypothesis"]] if "rock_salt_hypothesis" in run else [])]:
                    print(name, fit["phase"], fit["reason"] or fit["values"], fit["errors"], fit["strain"],
                          len(fit["residuals"]), fit["chi2_red"])
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
