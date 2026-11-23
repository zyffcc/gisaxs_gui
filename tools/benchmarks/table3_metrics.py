"""Reductions for the historical validation table (revision 14)."""
from __future__ import annotations

import math

import numpy as np


LEVELS = ("model", "measured")


def result(printed, **counts):
    return {"printed": list(printed), "counts": counts}


def archive_rows(data):
    rows = data["results"]
    if len(rows) != data["summary"]["n"] or len({r["case"]["index"] for r in rows}) != len(rows):
        raise ValueError("incomplete archive or duplicate case index")
    return rows


def radii(rows, level, stage):
    errors, hits68, hits95 = [], [], []
    total = 0
    for row in rows:
        if level != "all" and level != row["case"]["level"]:
            continue
        true = sorted(row["case"]["populations"], key=lambda pop: pop["R"])
        total += len(true)
        fit = row[stage]
        if not fit["ok"] or len(fit["populations"]) != len(true):
            continue
        for t, g in zip(true, fit["populations"]):
            r = t["R"]
            errors.append(abs(g["R"] - r) / r)
            if stage != "network":
                hits68.append(g["R_68"][0] <= r <= g["R_68"][1])
                hits95.append(g["R_95"][0] <= r <= g["R_95"][1])
    if not errors or not np.isfinite(errors).all():
        raise ValueError("missing or non-finite eligible radii")
    n = len(errors)
    return dict(n=n, total=total, median=100 * np.median(errors), p90=100 * np.percentile(errors, 90),
                coverage68=100 * sum(hits68) / n, coverage95=100 * sum(hits95) / n,
                hits68=sum(hits68), hits95=sum(hits95))


def historical_choices(rows, stage, level, high=False):
    """Reconstruct only the two family-based estimands actually printed in revision 14."""
    correct = n = 0
    for row in rows:
        if level != "all" and level != row["case"]["level"]:
            continue
        fit = row[stage]
        if not fit["ok"]:
            continue
        if stage == "auto":  # the old automatic row used the leading solution
            winner = tuple(sorted(fit["families"]))
        else:
            groups = {}
            for mode in (fit, *fit["alternatives"]):
                families = tuple(sorted(mode["families"]))
                groups[families] = groups.get(families, 0.0) + mode["probability"]
            winner = max(groups, key=groups.get)
            if high and groups[winner] < 0.99:
                continue
        correct += winner == tuple(sorted(row["truth_families"]))
        n += 1
    if not n:
        raise ValueError("no eligible composition choices")
    return correct, n


def recovery(key, data):
    rows = archive_rows(data)
    counts = {"cuts": len(rows)}
    group, metric = key.split(".")
    if metric in ("composition", "high", "choices", "auto"):
        levels = LEVELS if metric == "composition" else ("all",)
        groups = [(level, False) for level in levels] if metric != "high" else []
        if metric in ("high", "choices"):
            groups.append(("all", True))
        printed = []
        for level, high in groups:
            stage = "auto" if metric == "auto" else "choose"
            k, n = historical_choices(rows, stage, level, high)
            counts[level + ("_high" if high else "")] = {"correct": k, "eligible": n}
            printed.append(format(100 * k / n, ".0f"))
            if high:
                printed.append(str(n))
        if metric == "auto":
            r = radii(rows, "all", stage)
            counts.update(radii=r["n"], true_radii=r["total"], hits68=r["hits68"])
            printed.extend([format(r["median"], ".1f"), format(r["coverage68"], ".0f")])
        return result(printed, **counts)

    levels = ("all",) if group == "third" else LEVELS
    printed = []
    for level in levels:
        r = radii(rows, level, "network" if group == "network" else "fit")
        counts[level] = {k: r[k] for k in ("n", "total", "hits68", "hits95") if group != "network" or k in ("n", "total")}
        if metric == "coverage":
            printed.append(format(r["coverage68"], ".0f"))
            if group == "first":
                printed.append(format(r["coverage95"], ".0f"))
        elif group == "network":
            printed.append(format(r["median"], ".0f"))
        elif group == "first":
            printed.extend([format(r["median"], ".2g"), format(r["p90"], ".2g")])
        else:
            printed.extend([format(r["median"], ".1f"), format(r["p90"], ".0f")])
    if group == "extended":  # table gives the range of medians, then the range of P90s
        printed = [printed[i] for i in (0, 2, 1, 3)]
    return result(printed, **counts)


def sphere_radius_bin(run):
    if __package__:
        from .recovery_cases import GEOM
    else:
        from recovery_cases import GEOM

    errors = []
    for r in archive_rows(run):
        truth = sorted(r["case"]["populations"], key=lambda p: p["R"])
        fit = r["fit"]
        if fit["ok"] and len(fit["populations"]) == len(truth):
            for t, g in zip(truth, fit["populations"]):
                if t["shape"] == "sphere" and 2.3 <= GEOM.qz_vacuum * t["R"] < 3.5:
                    errors.append(abs(g["R"] - t["R"]) / t["R"])
    if not errors:
        raise ValueError("empty sphere q_z R bin")
    return len(errors), float(np.median(errors))


def dwba(data):
    bins = [sphere_radius_bin(run) for run in data]
    return result([format(100 * median, ".0f") for _, median in bins],
                  eligible_spheres=[n for n, _ in bins])


def reduce_row(key, inputs):
    d = inputs[0]
    if key.split(".")[0] in {"first", "third", "extended", "network"}:
        return recovery(key, d)
    if key == "dwba":
        return dwba(inputs)
    if key in ("all_pixel_q", "radial"):
        if len(d["frames"]) != 4 or len({f["frame"] for f in d["frames"]}) != 4:
            raise ValueError("expected a comparison on four distinct public frames")
    if key == "all_pixel_q":
        x = max(f["max_abs_relative_all_pixels"] for f in d["frames"])
        scale = 10.0 ** math.floor(math.log10(x))
        return result([format(math.ceil(x / scale) * scale, ".0e")], frames=len(d["frames"]), maximum=x)
    if key == "radial":
        x = max(f["intensity_same_estimator"]["p95_relative"] for f in d["frames"])
        return result([format(10.0 ** math.ceil(math.log10(x)), ".0e"), "95"],
                      frames=len(d["frames"]), largest_p95=x)
    if key == "agbh":
        comparisons = d["real_frame"]["compare_gimap_convention"]
        parameters = ("beam_x_px", "beam_y_px", "distance_mm", "rot1_deg", "rot2_deg")
        max_sigma = max(abs(comparisons[k]["difference_over_combined_sigma_jackknife"]) for k in parameters)
        return result([format(max_sigma, ".1f")], parameters=len(parameters))
    if key in ("calibration", "calibration_heldout"):
        if key == "calibration":
            frames = d["main"]["frames"]
            k, n = sum(f["as_shipped"]["identified"] for f in frames), len(frames)
            expected_n = d["main"]["meta"]["cases"]
        else:
            coverage = d["coverage"]
            k, n = map(int, coverage["as_shipped"]["identified"].split("/"))
            expected_n = coverage["cases"]
        if not 0 <= k <= n or n != expected_n:
            raise ValueError("inconsistent calibration counts")
        return result([str(n), format(100 * k / n, ".0f")], identified=k, frames=n)
    if key == "independent_model":
        rows = [c for c in d["checks"] if c["reference"] == "truth" and "σ=0.2 " in c["name"]]
        return result([format(max(c["max_rel"] for c in rows), ".1e")], checks=len(rows))
    if key == "bornagain":
        rows = [c for c in d["checks"] if "sizes σ_R/R 0.2" in c["name"]]
        shapes = {c["name"].split()[0]: c for c in rows}
        return result([format(shapes[k]["max_rel"], ".1e") for k in ("sphere", "vertical_cylinder")])
    if key == "search":
        return result([format(100 * d["parameters"][level]["R"][stat], ".0f") for level in LEVELS
                       for stat in ("median_abs", "p90_abs")], cuts=d["n"])
    if key == "galaxi":
        radii = [f["best"]["values"]["R_1"]["best"] for f in d["fits"]
                 if f["variant"] == "born1" and f["source"].startswith("bornagain")]
        if not radii:
            raise ValueError("missing BornAgain fits")
        return result([format(np.median(radii), ".1f"), format(d["truth"]["R_number_mean"], ".1f")], fits=len(radii))
    if key == "height":
        bias = {c["name"]: 100 * c["summary"]["full"]["h"]["noise_free_bias"] for c in d["cases"]}
        off = max(abs(bias["a_off_" + s]) for s in ("D14", "D24", "D12"))
        au = [bias["b_avg_" + s + "_Au"] for s in ("10", "30", "40")]
        return result([format(off, ".1f"), format(bias["b_avg_10_Si"], ".0f"),
                       format(max(au), ".0f"), format(min(au), ".0f")])
    if key == "au":
        v = d["values"]["R_1"]
        sem = inputs[1]
        if not sem["citation"]:
            raise ValueError("supply the reference for the SEM diameters")
        return result([format(sem[k] / 2, ".1f") for k in ("diameter_mean_nm", "diameter_sd_nm")] +
                      [format(v["median"], ".1f"), "95", format(v["low95"], ".1f"), format(v["high95"], ".1f")])
    raise KeyError(key)
