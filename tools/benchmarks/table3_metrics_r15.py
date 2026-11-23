"""Revision-15 Table 3 reductions and checks against the unrounded archive summaries."""
from __future__ import annotations

import math

if __package__:
    from . import table3_metrics as common
    from .composition_scores import BRANCHES, composition_score
else:
    import table3_metrics as common
    from composition_scores import BRANCHES, composition_score


RADIUS_FIELDS = {
    "median": "median_abs", "p90": "p90_abs",
    "coverage68": "posterior_coverage68", "coverage95": "posterior_coverage95",
}


def _check(path, recorded, computed):
    if not math.isclose(recorded, computed, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError(f"{path}: archive summary gives {recorded}, per-cut records give {computed}")


def _radii(rows, summary, stage, level, fields):
    size = common.radii(rows, level, stage)
    path = f"parameters.{stage}/{level}.R"
    archived = summary["parameters"][f"{stage}/{level}"]["R"]
    _check(path + ".n", archived["n"], size["n"])
    for field in fields:
        name = RADIUS_FIELDS[field]
        _check(f"{path}.{name}", archived[name], size[field] / 100)
    return size


def _radius_row(group, metric, rows, summary):
    stage = "network" if group == "network" else "fit"
    if metric == "coverage":
        fields = ("coverage68", "coverage95") if group == "first" else ("coverage68",)
        formats = [".0f"] * len(fields)
    elif stage == "network":
        fields, formats = ("median",), (".0f",)
    else:
        fields = ("median", "p90")
        formats = (".2g", ".2g") if group == "first" else (".1f", ".0f")
    cells, counts = [], {"cuts": len(rows)}
    for level in (("all",) if group == "third" else common.LEVELS):
        size = _radii(rows, summary, stage, level, fields)
        counts[level] = {key: size[key] for key in ("n", "total")}
        for field, precision in zip(fields, formats):
            cells.append(format(size[field], precision))
            if field.startswith("coverage"):
                key = field.replace("coverage", "hits")
                counts[level][key] = size[key]
    if group == "extended":
        cells = [f"{a}--{b}" for a, b in zip(cells[:2], cells[2:])]
    return common.result(cells, **counts)


def _choices(rows, stage, level):
    scores = []
    for row in rows:
        if level != "all" and row["case"]["level"] != level:
            continue
        if row[stage]["ok"]:
            score = composition_score(row, stage)
            if score is None:
                raise ValueError(f"successful {stage} cut lacks pre-display composition weights")
            scores.append(score)
    return scores


def _composition_row(group, metric, rows, summary):
    stage, name = ("auto", "composition_auto") if metric == "auto" else ("choose", "composition")
    levels = common.LEVELS if metric == "composition" and group != "third" else ("all",)
    cells, counts = [], {"cuts": len(rows)}
    for level in levels:
        scores = _choices(rows, stage, level)
        archived = summary[name][level]
        path = f"{name}.{level}"
        n, k = len(scores), sum(s.correct for s in scores)
        _check(path + ".n_scored", archived["n_scored"], n)
        _check(path + ".n_correct", archived["n_correct"], k)
        counts[level] = {"correct": k, "eligible": n}
        if metric == "high":
            for branch in BRANCHES:
                high = [score for score in scores if score.branch == branch and score.weight >= .99]
                right = sum(s.correct for s in high)
                bins = archived["reliability_by_branch"][branch]["bins"]
                last, = (b for b in bins if b["low"] == .99 and b["high"] == 1.)
                _check(f"{path}.{branch}[0.99,1].n", last["n"], len(high))
                _check(f"{path}.{branch}[0.99,1].n_correct", last["n_correct"], right)
                counts[branch] = {"correct": right, "eligible": len(high)}
                cells.append(f"{right}/{len(high)}")
        else:
            if not n:
                raise ValueError(f"no eligible {stage} choices at level {level}")
            _check(path + ".exact", archived["exact"], k / n)
            cells.append(format(100 * (k / n), ".0f"))
    if metric == "auto":
        size = _radii(rows, summary, stage, "all", ("median", "coverage68"))
        cells.extend((format(size["median"], ".1f"), format(size["coverage68"], ".0f")))
        counts.update(radii=size["n"], true_radii=size["total"], hits68=size["hits68"])
    return common.result(cells, **counts)


def _dwba(pairs, runs):
    cells, counts = [], []
    reference = None
    for label, run in zip(("si_tp", "si_born", "si_eia"), runs, strict=True):
        n, median = common.sphere_radius_bin(run)
        cases = {row["case"]["index"]: row["case"] for row in run["results"]}
        if len(cases) != pairs["n_curves"] or (reference is not None and reference != cases):
            raise ValueError(f"{label}: per-cut records disagree with the paired case draws or count")
        reference = cases
        path = "sphere q+R 2.3-3.5"
        stored = pairs["runs"][label][path]["R"]
        _check(f"{label}.{path}.R.n", stored["n"], n)
        _check(f"{label}.{path}.R.median_abs", stored["median_abs"], median)
        cells.append(format(100 * stored["median_abs"], ".0f"))
        counts.append(n)
    return common.result(cells, eligible_spheres=counts)


def _dwba4(pairs):
    cells = []
    for label in ("si_l4", "si_l2"):
        choose = pairs["runs"][label]["overall"]["choose"]
        pct = 100 * choose["composition_right"]
        precision = ".0f" if math.isclose(pct, round(pct), rel_tol=0, abs_tol=1e-12) else ".1f"
        cells.extend((format(pct, precision), format(100 * choose["parameters"]["R"]["median_abs_relative"], ".1f")))
    return common.result(cells)


def reduce_row(key, inputs):
    if key == "dwba":
        return _dwba(inputs[0], inputs[1:])
    if key == "dwba4":
        return _dwba4(inputs[0])
    group, _, metric = key.partition(".")
    if group in ("first", "third", "extended", "network"):
        data = inputs[0]
        rows = common.archive_rows(data)
        if metric in ("composition", "high", "auto"):
            return _composition_row(group, metric, rows, data["summary"])
        return _radius_row(group, metric, rows, data["summary"])
    return common.reduce_row(key, inputs)
