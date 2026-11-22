'The structure at any time of an analysed scan, between (or at) the windows of the analysis (2026-11-22, user: the'

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import time
import types
from pathlib import Path

import numpy as np

from . import components as CO
from . import live as LV
from . import process as PRC
from . import registry as REG
from . import runner
from .present import _fractions, wanted

AT_TIME_FILE = "at_time.json"
SOLUTION_KEYS = ("seq", "refined", "best")
"""The solution of an analysed window the path is taken from: the joint fit, else the window's refinement, else its
best candidate."""


def _solution(row: dict):
    for key in SOLUTION_KEYS:
        found = row.get(key)
        if isinstance(found, dict) and found.get("x") is not None:
            return key, np.asarray(found["x"], float)
    return None, None


def path_points(rows: list) -> list:
    '(t, source, x) of the analysed windows with a solution, in time order.'
    points = []
    for row in rows:
        key, x = _solution(row)
        if x is not None:
            points.append((float(row["t"]), key, x))
    return sorted(points, key=lambda item: item[0])


def interpolate(points: list, t: float, onset: float) -> dict:
    "The path's parameters at ``t``: linear in u = ln(t - onset) between the neighbouring windows (in t when a"
    if not points:
        raise SystemExit("the result has no solution in any window: nothing to interpolate")
    times = [p[0] for p in points]
    if t <= times[0] or t >= times[-1]:
        near = points[0] if t <= times[0] else points[-1]
        return dict(x=near[2].copy(), before=near[0], after=near[0], weight=0.0, nearest=near[0], source=near[1],
                    extrapolated=abs(t - near[0]) > 1e-9)
    j = int(np.searchsorted(times, t, side="right"))
    (ta, ka, xa), (tb, kb, xb) = points[j - 1], points[j]
    if ta - onset > 0.0 and tb - onset > 0.0:
        w = (math.log(t - onset) - math.log(ta - onset)) / (math.log(tb - onset) - math.log(ta - onset))
    else:
        w = (t - ta) / (tb - ta)
    w = float(min(max(w, 0.0), 1.0))
    if xa.shape != xb.shape:
        raise SystemExit("the neighbouring windows have solutions of different models: nothing to interpolate")
    return dict(x=(1.0 - w) * xa + w * xb, before=ta, after=tb, weight=w, nearest=ta if w < 0.5 else tb,
                source=ka if w < 0.5 else kb, extrapolated=False)


def _window(args):
    'Build the window at t, draw the interpolated solution, refine, and write the arrays (in a runner process).'
    cfg, scan, t, starts, info, outdir, refine = args
    from . import adapter as A
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = A.get_scan(scan)
    ctx = A.Context(cfg, s, t)
    if list(ctx.pnames) != list(info["pnames"]):
        raise SystemExit("the model's parameters differ from the result's: analyse the scan again with this package")
    ok = ctx.w["mask"]
    E = np.where(ok, ctx.w["E"], 0.0)
    Es = E.sum(0)
    good = Es > 0
    measured = np.where(good, np.where(ok, ctx.w["Y"], 0.0).sum(0) / np.maximum(Es, 1e-30), np.nan)
    nf = len(ctx.names)
    lb, ub = np.asarray(ctx.f.lb, float), np.asarray(ctx.f.ub, float)

    def clip(x):
        x = np.asarray(x, float).copy()
        x[:nf] = np.clip(x[:nf], lb + 1e-9, ub - 1e-9)
        return x

    def drawn(x):
        (mu, _, _), m, p, h = ctx.render(x)
        return ctx.dev5(mu)[0]

    scored = []
    for label, x in starts:
        x = clip(x)
        try:
            scored.append((drawn(x), label, x))
        except (Exception, SystemExit):
            continue
    if not scored:
        raise SystemExit("neither the interpolated nor the nearest window's solution can be drawn at this time")
    t0 = time.time()
    entries = [("interpolated", "interpolated", next(x for d, lab, x in scored if lab == "interpolated"))
               if any(lab == "interpolated" for _, lab, _ in scored) else None]
    entries = [e for e in entries if e is not None]
    best = min(scored, key=lambda item: item[0])
    refined_from = best[1]
    if refine > 0:
        xb = best[2]
        ctx.render(xb)
        xr, _ = ctx.f.run(np.clip(xb[:nf], lb + 1e-9, ub - 1e-9), refine)
        entries.append(("refined", "refined", np.r_[xr, xb[nf:]]))
    pn, rn = info["pnames"], [r["name"] for r in cfg["reported"]]
    images, records, parts = [], [], []
    for label, kind, x in entries:
        x = np.asarray(x, float)
        try:
            mu, comps, closure, (m, p, h) = CO.split(ctx, x)
        except Exception:
            (mu, _, _), m, p, h = ctx.render(x)
            comps, closure = {}, None
        d5, _ = ctx.dev5(mu)
        rep = ctx.reported(m, p, h)
        images.append(np.where(good, (np.where(ok, mu, 0.0) * E).sum(0) / np.maximum(Es, 1e-30), np.nan))
        parts.append(dict(zip(comps, CO.images(ctx, comps))) if comps else {})
        record = dict(label=label, kind=kind, rank=None, weight=None, fit_deviation=round(float(d5), 3),
                      **wanted(pn, rn, x.tolist(), rep, h, getattr(m, "t_b", 0.0), m))
        record["growth"] = PRC.growth_point(cfg["family"], rn, rep, h)
        if comps:
            record["components"] = CO.shares(ctx, mu, comps)
            record["components_closure"] = float(f"{closure:.1e}")
        records.append(record)
    af, q = np.asarray(s.af, float), np.asarray(s.q, float)
    keys = [k for k in CO.ORDER if any(k in part for part in parts)]
    arrays = f"w_{t:g}.npz"
    np.savez_compressed(Path(outdir) / arrays, q=q.astype(np.float32), af=af.astype(np.float32),
                        measured=measured.astype(np.float32), models=np.asarray(images, np.float32),
                        labels=np.array([r["label"] for r in records]), component_keys=np.array(keys, dtype=str),
                        component_fractions=_fractions(images, parts, keys))
    pos = measured[np.isfinite(measured) & (measured > 0)]
    vmin, vmax = (np.log10(np.percentile(pos, 2)), np.log10(np.percentile(pos, 99.8))) if pos.size else (-3, 0)
    fig, axes = plt.subplots(1, 1 + len(images), figsize=(3.3 * (1 + len(images)) + 0.6, 3.9), sharey=True)
    for axis, (title, image) in zip(np.atleast_1d(axes), [("measured", measured)] +
                                    [(f"{r['label']} (fit deviation {r['fit_deviation']:.1f})", im)
                                     for r, im in zip(records, images)]):
        with np.errstate(divide="ignore", invalid="ignore"):
            axis.pcolormesh(q, af, np.log10(np.where(image > 0, image, np.nan)), vmin=vmin, vmax=vmax, cmap="viridis",
                            shading="auto")
        axis.set_xscale("log")
        axis.set_title(title, fontsize=8)
    image = f"w_{t:g}.png"
    fig.suptitle(f"{scan}   t {t:g} s (on demand)", fontsize=9, x=0.02, ha="left")
    fig.savefig(Path(outdir) / image, dpi=100, bbox_inches="tight")
    plt.close(fig)
    smooth = "refined" if any(r["label"] == "refined" for r in records) else records[0]["label"]
    pick = min(records, key=lambda r: r["fit_deviation"])["label"]
    return dict(t=t, t_rel=round(float(t - s.t_dep), 3), h_nom=round(float(ctx.h_nom_ctx), 4), image=image,
                arrays=arrays, entries=records, pick=pick, smooth=smooth, on_demand=True,
                refine=dict(nfev=refine, start=refined_from, seconds=round(time.time() - t0, 1)))


def _data_root(data: Path, label: str) -> Path:
    "The folder that holds the scan's import: ``data`` itself or a mask subfolder of it (iampred.analyze)."
    if (data / label / "scan.json").is_file():
        return data
    for sub in sorted(data.glob("mask-*")):
        if (sub / label / "scan.json").is_file():
            return sub
    raise SystemExit(f"the import of {label} is not in {data}: analyse the scan again")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--result", required=True, help="the folder of an analysis (iampred.analyze --out)")
    ap.add_argument("--t", type=float, required=True, help="time in the scan, s (as the windows' t)")
    ap.add_argument("--data", default=None, help="the data root the analysis imported the scan into")
    ap.add_argument("--refine", type=int, default=8, help="refinement evaluations per parameter (0: none)")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    result = Path(a.result).resolve()
    analysis = json.loads((result / "analysis.json").read_text(encoding="utf-8"))
    family = analysis["family"]
    folder = result / family if (result / family / "state.json.gz").is_file() else result
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    with gzip.open(folder / "state.json.gz", "rt", encoding="utf-8") as fh:
        state = json.load(fh)
    cfg = REG.load(summary["family"])
    scan = summary["scan"]
    data = _data_root(Path(a.data).resolve(), scan) if a.data else None
    if scan not in cfg["scans"]:
        from . import predict as PR
        site = runner.site_env(cfg, str(data) if data else None, None)
        cfg["scans"][scan], _ = PR._client_entry(cfg, types.SimpleNamespace(scan=scan, like=summary.get("like")),
                                                  Path(site.get("IAM_DATA", ".")))
    points = path_points(state["rows"])
    onset = float(summary.get("onset_s") or 0.0)
    path = interpolate(points, float(a.t), onset)
    starts = [("interpolated", path["x"])]
    nearest = next(x for t_, _, x in points if t_ == path["nearest"])
    if not np.allclose(nearest, path["x"]):
        starts.append(("nearest", nearest))
    main_dir = result / "main"
    main_dir.mkdir(exist_ok=True)
    LV.emit("at_time", t=float(a.t), before=path["before"], after=path["after"], weight=round(path["weight"], 4))
    print(f"{scan}: t {a.t:g} s between the windows at {path['before']:g} and {path['after']:g} s "
          f"(weight {path['weight']:.3f} on the later), refining", flush=True)
    res = runner.run(cfg, [(cfg, scan, float(a.t), starts, state["info"], str(main_dir), a.refine)], _window, 1,
                     data=str(data) if data else None)
    window = res[0]
    if window is None:
        raise SystemExit(f"{scan}: not a scan of family {cfg['family']} (no forward settings for it)")
    window["at_time"] = dict(before=path["before"], after=path["after"], weight=round(path["weight"], 4),
                             source=path["source"], beyond=path["extrapolated"],
                             axis="ln(t - onset)" if path["before"] - onset > 0 else "t")
    target = main_dir / AT_TIME_FILE
    sha = hashlib.sha256((main_dir / "main.json").read_bytes()).hexdigest()
    doc = json.loads(target.read_text(encoding="utf-8")) if target.is_file() else {}
    windows = [w for w in doc.get("windows", []) if doc.get("main_sha256") == sha and abs(float(w["t"]) - a.t) > 1e-6]
    doc = dict(main_sha256=sha, note="windows analysed on demand (iampred.at_time): 'interpolated' = the joint path "
                                     "at this time, 'refined' = refined to the frames of this time",
               windows=sorted(windows + [window], key=lambda w: w["t"]))
    target.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    best = min(window["entries"], key=lambda e: e["fit_deviation"])
    print(f"{scan}: t {a.t:g} s written to {target} (fit deviation: "
          + ", ".join(f"{e['label']} {e['fit_deviation']:.1f}" for e in window["entries"]) + f"; best {best['label']})")


if __name__ == "__main__":
    main()
