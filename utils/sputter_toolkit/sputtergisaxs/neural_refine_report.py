"""Pure report aggregation and strict JSON I/O; no fitting or training imports."""
from pathlib import Path
import hashlib
import json
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())
    temp.replace(path)


def summarize(results, state):
    good = [r for r in results if "refined_scores" in r]
    floors = {}
    for eps in (.05, .03, .1, 0.):
        matched = {r["key"] for r in good for s in r["refined_scores"]
                   if s["relative_model_error"] == eps and s["compatible_at_5pct_asymptotic"]}
        floors[str(eps)] = dict(windows=len(matched), denominator=state["selected_bilateral_windows"],
                               matched_keys=sorted(matched))
    distances = []
    for key in sorted({r["key"] for r in good}):
        modes = [r for r in good if r["key"] == key]
        for i, a in enumerate(modes):
            for b in modes[i+1:]:
                da, db = a["network_values"]["D_nm"], b["network_values"]["D_nm"]
                ratio = max(da/db, db/da)
                if ratio < 2.: continue
                groups = np.array(a["selected_partition_group"])
                low = np.char.startswith(groups, "H0") | np.char.startswith(groups, "H1")
                ma, mb = np.array(a["network_partition_counts"]), np.array(b["network_partition_counts"])
                distance = float(np.sqrt(np.mean((np.log(np.maximum(ma[low], 1e-12)) -
                                                    np.log(np.maximum(mb[low], 1e-12)))**2)))
                distances.append(dict(key=key, modes=[a["module"]+":"+a["mode_id"], b["module"]+":"+b["mode_id"]],
                                      D_ratio=ratio, low_angle_network_log_RMS=distance,
                                      curves_within_5pct_RMS=bool(distance < np.log1p(.05))))
    return dict(status="complete", compatible=floors, candidates=len(results), refined=len(good),
                criterion_at_least_half=bool(floors["0.05"]["windows"] >= .5*state["selected_bilateral_windows"]),
                separated_D_curve_comparisons=distances,
                near_identical_pairs=int(sum(r["curves_within_5pct_RMS"] for r in distances)),
                separated_pair_denominator=len(distances),
                peak_metal_fraction_median=float(np.median([r["peak_signal_fraction"] for r in good])) if good else "No_completed_refinement",
                no_structural_recovery_claim=True, next_step="domain_localization_no_repeat_tuning")

