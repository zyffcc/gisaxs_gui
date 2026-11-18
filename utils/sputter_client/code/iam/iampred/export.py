'Export of an analysed scan for a paper or a talk: figures as PDF, SVG and PNG (300 dpi), the shown solution per'

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

COLUMNS = [("R_nm", "R_nm", "grain radius R (nm)"), ("H_nm", "H_nm", "grain height H (nm)"),
           ("base_nm", "t_b_nm", "thickness of the continuous metal layer under the grains t_b (nm; 0 = separate islands)"),
           ("D_nm", "D_nm", "centre distance of the grains D (nm)"), ("gap_nm", "gap_nm", "gap between grains D - 2R (nm)"),
           ("width", "sigma_lnR", "width of the log-normal radius distribution (sigma of ln R)"),
           ("coverage", "coverage", "area share covered by metal (0-1)"),
           ("amount_nm", "amount_nm", "metal amount of the solution as an equivalent film thickness (nm)"),
           ("f_islands", "f_islands", "share of the metal in resolvable islands (0-1)"),
           ("cap_share", "cap_share", "share of the optically active metal in the cap layer on the TiO2 particles (0-1)"),
           ("depth_nm", "depth_nm", "vertical correlation length of the metal network in the pores (nm)"),
           ("xi_nm", "xi_nm", "lateral correlation length of the metal network in the pores (nm)"),
           ("film_share", "film_share", "share of the layer scattering that leaves through the metal film (0-1)")]
SIZES = [("R_nm", "R"), ("H_nm", "H"), ("base_nm", "$t_b$"), ("D_nm", "D"), ("gap_nm", "D − 2R")]
COLOURS = ["#1f77b4", "#2a9d5c", "#8c564b", "#9467bd", "#e08a1e"]
FORMATS = ("pdf", "svg", "png")
TAG = dict(color="white", fontsize=7, bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.55))


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _stages(result):
    p = Path(result) / "report.csv"
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as fh:
        return {round(float(r["t"]), 3): r.get("stage", "") for r in csv.DictReader(fh)}


def _shown(win, choice):
    "The entry shown in a window: ``choice`` is 'smooth', 'pick' or a dict {round(t, 3): label} (the GUI's own)."
    if isinstance(choice, dict):
        label = choice.get(round(float(win["t"]), 3))
        if label is not None and any(e["label"] == label for e in win.get("entries", [])):
            return next(e for e in win["entries"] if e["label"] == label)
        choice = "smooth"
    label = win.get(choice) or win.get("smooth") or win.get("pick")
    return next((e for e in win.get("entries", []) if e["label"] == label), None)


def _fmt(v, nd=4):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ""
    return f"{v:.{nd}g}" if isinstance(v, float) else str(v)


def write_tables(main, stages, choice, out):
    wins = [w for w in main["windows"] if w.get("entries")]
    keys = [c for c in COLUMNS if any(e.get(c[0]) is not None for w in wins for e in w["entries"])]
    with (out / "solution.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["t_s", "t_since_onset_s", "nominal_deposit_nm", "stage", "solution", "fit_deviation"] + [c[1] for c in keys])
        for w in wins:
            e = _shown(w, choice)
            if e is None:
                continue
            wr.writerow([_fmt(w["t"]), _fmt(w.get("t_rel")), _fmt(w.get("h_nom")), stages.get(round(w["t"], 3), ""),
                         e["label"], _fmt(e.get("fit_deviation"))] + [_fmt(e.get(c[0])) for c in keys])
    with (out / "all_solutions.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["t_s", "solution", "kind", "network_weight", "fit_deviation", "shown"] + [c[1] for c in keys])
        for w in wins:
            shown = _shown(w, choice)
            for e in w["entries"]:
                wr.writerow([_fmt(w["t"]), e["label"], e["kind"], _fmt(e.get("weight")), _fmt(e.get("fit_deviation")),
                             "yes" if shown is e else ""] + [_fmt(e.get(c[0])) for c in keys])
    return wins, keys


def _style(plt):
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.6, "lines.linewidth": 1.4, "lines.markersize": 3.5,
                         "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.5})


def _save(fig, out, name, dpi):
    for f in FORMATS:
        fig.savefig(out / f"{name}.{f}", dpi=dpi if f == "png" else None, bbox_inches="tight")


def evolution_figure(main, wins, choice, out, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    rows = [(w["t"], _shown(w, choice)) for w in wins]
    rows = [(t, e) for t, e in rows if e is not None]
    t = np.array([r[0] for r in rows])
    val = lambda k: np.array([np.nan if e.get(k) is None else float(e[k]) for _, e in rows])
    has_cov = np.isfinite(val("coverage")).any()
    fig, ax = plt.subplots(2 + has_cov, 1, figsize=(3.4, 4.6 if has_cov else 3.4), sharex=True,
                           gridspec_kw=dict(height_ratios=[2.2] + [1.0] * (1 + has_cov)))
    a = ax[0]
    for (k, name), col in zip(SIZES, COLOURS):
        y = val(k)
        if np.isfinite(y).any():
            a.plot(t, y, "-o", color=col, label=name, markeredgecolor="white", markeredgewidth=0.5)
    a.set_ylabel("size (nm)")
    a.set_ylim(bottom=0)
    a.legend(ncol=3, frameon=False, loc="upper left")
    i = 1
    if has_cov:
        ax[i].plot(t, val("coverage"), "-o", color="#555555", markeredgecolor="white", markeredgewidth=0.5)
        ax[i].set_ylabel("coverage")
        ax[i].set_ylim(0, 1)
        i += 1
    ax[i].plot(t, val("fit_deviation"), "-o", color="#555555", markeredgecolor="white", markeredgewidth=0.5)
    ax[i].set_ylabel("fit deviation")
    ax[i].set_ylim(bottom=0)
    ax[i].set_xlabel("time in scan (s)")
    fig.align_ylabels(ax)
    _save(fig, out, "evolution", dpi)
    plt.close(fig)


def _pattern_windows(wins, chosen, n):
    if chosen:
        want = [float(x) for x in chosen]
        return [min(wins, key=lambda w: abs(w["t"] - x)) for x in want]
    if len(wins) <= n:
        return list(wins)
    idx = np.unique(np.round(np.linspace(0, len(wins) - 1, n)).astype(int))
    return [wins[i] for i in idx]


def patterns_figure(result, wins, choice, chosen, n, out, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    _style(plt)
    plt.rcParams["axes.grid"] = False
    picked = _pattern_windows(wins, chosen, n)
    data = []
    for w in picked:
        z = np.load(Path(result) / "main" / w["arrays"])
        e = _shown(w, choice)
        labels = [str(x) for x in z["labels"]]
        j = labels.index(e["label"]) if e is not None and e["label"] in labels else 0
        data.append((w, z["q"], z["af"], z["measured"], z["models"][j], labels[j], e))
    pos = np.concatenate([d[3][np.isfinite(d[3]) & (d[3] > 0)] for d in data]) if data else np.array([1.0])
    norm = LogNorm(vmin=np.percentile(pos, 2), vmax=np.percentile(pos, 99.8))
    k = len(data)
    fig, ax = plt.subplots(2, k, figsize=(1.75 * k + 0.6, 3.6), sharex=True, sharey=True, squeeze=False)
    for c, (w, q, af, meas, model, label, e) in enumerate(data):
        for r, img in enumerate((meas, np.where(np.isfinite(meas), model, np.nan))):
            a = ax[r, c]
            pm = a.pcolormesh(q, af, np.where(img > 0, img, np.nan), norm=norm, cmap="viridis", shading="auto", rasterized=True)
            a.set_xscale("log")
            if r == 0:
                a.set_title(f"t = {w['t']:g} s\n{w.get('h_nom', float('nan')):.1f} nm deposited")
            else:
                dev = e.get("fit_deviation") if e else None
                if dev is not None:
                    a.text(0.96, 0.05, f"dev {dev:.1f}", transform=a.transAxes, ha="right", va="bottom", **TAG)
                a.set_xlabel(r"$q_y$ (nm$^{-1}$)")
    for r, name in enumerate(("measured", "model")):
        ax[r, 0].text(0.04, 0.05, name, transform=ax[r, 0].transAxes, va="bottom", **TAG)
    for r in range(2):
        ax[r, 0].set_ylabel(r"$\alpha_f$ (deg)")
    fig.colorbar(pm, ax=ax.ravel().tolist(), fraction=0.03, pad=0.02).set_label("count rate (counts / s)")
    _save(fig, out, "patterns", dpi)
    plt.close(fig)
    return [w["t"] for w in picked]


MONOTONE_WORDS = {"lnD_mass": "the grain distance", "lnD": "the grain distance", "cov": "the coverage",
                  "lnh_mass": "the metal amount", "lnR_mass": "the grain radius", "lnH_top": "the grain height"}


def _monotone_text(family):
    try:
        cfg = _read_json(Path(__file__).resolve().parent / "families" / f"{family}.json")
        names = cfg.get("monotone", ["lnD", "cov"])
    except (OSError, ValueError):
        names = ["lnD", "cov"]
    words = [MONOTONE_WORDS.get(n, n) for n in names]
    return ", ".join(words[:-1]) + " and " + words[-1] if len(words) > 1 else (words[0] if words else "nothing")


def _shared_text(result):
    "(κ = …, Δt = … s) from the joint fit's record, '' when it is not there."
    try:
        rec = _read_json(Path(result) / "sequence_record.json")
        names, x = rec["shared_names"], rec["stages"]["seq"]["x"]
        val = dict(zip(names, x))
        return f" (κ = {math.exp(val['lnkap']):.2f}, Δt = {val['dt']:+.2f} s)"
    except (OSError, ValueError, KeyError, TypeError):
        return ""


def methods_text(result, main, analysis, request, wins, choice, shown_t):
    req = (request or {}).get("request", {})
    dev = [e.get("fit_deviation") for e in (_shown(w, choice) for w in wins) if e and e.get("fit_deviation") is not None]
    reading = analysis.get("reading") or {}
    mask = analysis.get("mask") or {}
    joint = analysis.get("joint_fit") or {}
    shown = {e["kind"] for e in (_shown(w, choice) for w in wins) if e}
    geo = []
    if req.get("wavelength_nm"):
        geo.append(f"wavelength {req['wavelength_nm']:g} nm")
    elif req.get("energy_ev"):
        geo.append(f"photon energy {req['energy_ev'] / 1000:g} keV")
    if req.get("sdd_mm"):
        geo.append(f"sample–detector distance {req['sdd_mm']:g} mm")
    if req.get("alpha_i_deg"):
        geo.append(f"incidence angle {req['alpha_i_deg']:g}°")
    rate = req.get("rate_nm_s")
    lines = [
        f"# Methods — {main.get('scan', '')}",
        "",
        "Written by `iampred.export` from the records of the analysis; every number below is taken from the result folder.",
        "",
        "## Measurement",
        "",
        f"- Metal {analysis.get('metal', main.get('metal', ''))} on {analysis.get('host') or analysis.get('substrate', '')}; "
        + (", ".join(geo) + "." if geo else "geometry as given in the analysis request."),
        f"- Deposition rate {rate:g} nm/s ({'quartz crystal monitor' if req.get('rate_is_qcm') else 'nominal'}) — the nominal deposit of each "
        "window is rate × time since the deposition started." if rate else "- Deposition rate not given.",
        f"- Frames read: {reading.get('frames_read', '?')} of {reading.get('frames_total', '?')}; detector mask "
        f"{mask.get('file', 'automatic')} with {mask.get('kept', '?')} of {mask.get('pixels', '?')} pixels kept.",
        "",
        "## Analysis",
        "",
        f"- The scan was cut into {len(wins)} time windows; in each window the frames were summed and mapped onto a fixed "
        "grid of in-plane momentum transfer q_y and exit angle α_f.",
        f"- A neural network trained on simulated patterns of the forward model (family `{main.get('family', '')}`) proposed "
        "candidate morphologies per window; each candidate was rendered back with the forward model (hemispheroidal metal "
        "grains with a log-normal radius distribution on a paracrystal lattice, an optional continuous metal layer under "
        "them, effective-medium / Parratt optics, the substrate scattering taken from the measured pattern before the "
        "deposition) and compared with the measured counts.",
    ]
    if "joint" in shown:
        lines.append(f"- The solution shown is a joint fit of the whole scan ({joint.get('iterations', '?')} iterations): "
                     "the metal amount of every window is κ × rate × (time since the start − Δt) with one factor κ and "
                     f"one shift Δt of the deposition start for the whole scan{_shared_text(result)}; "
                     f"{_monotone_text(main.get('family', ''))} may not decrease from one window to the next, and the "
                     "reported quantities are kept smooth in ln(time since the start).")
        other = sum(1 for w in wins if (_shown(w, choice) or {}).get("kind") != "joint")
        if other:
            lines.append(f"- In {other} of {len(wins)} windows another solution was chosen by hand (network candidate or "
                         "refined; all solutions are in `all_solutions.csv`).")
    else:
        lines.append("- The solution shown in each window is the candidate on the smoothest path through the scan with a "
                     "good fit (no joint fit of the whole scan was run or it did not finish).")
    lines += [
        "- Fit deviation = mean Poisson deviance per (q_y, α_f) cell, with the variance widened by a 5 % relative model "
        "error (lower is better; about 1 means consistent with counting noise plus 5 % model error).",
        f"- Fit deviation of the shown solutions: median {np.median(dev):.1f}, range {min(dev):.1f}–{max(dev):.1f} "
        f"over {len(dev)} windows." if dev else "- No fit deviation recorded.",
        f"- Patterns shown in `patterns.*`: t = {', '.join(f'{t:g}' for t in shown_t)} s.",
        "",
        "## Limits",
        "",
        f"- {analysis.get('note', 'No independent characterisation (AFM, XRR, SEM) of this sample was used.')}",
        "- The model quantities are those of the forward model's morphology (grains of one shape with a log-normal radius "
        "distribution on a continuous layer); where the fit deviation is high the pattern holds features the model does "
        "not describe, and the numbers there are the model's closest description.",
        "",
        "## Software",
        "",
        f"- Analysis package `{(request or {}).get('package_id', '')[:12] or 'unknown'}`, mode `{analysis.get('mode', '')}`, "
        f"run {(request or {}).get('written', '')}; {(request or {}).get('software_version', '')}.",
    ]
    return "\n".join(lines) + "\n"


README = """# Export of {scan}

| file | content |
|---|---|
| `solution.csv` | one row per time window: the solution shown ({choice}), its fit deviation and the quantities; units in the column names |
| `all_solutions.csv` | every solution of every window (network candidates, refined, joint); `shown` marks the one in `solution.csv` |
| `evolution.pdf / .svg / .png` | sizes and coverage against time, fit deviation below (PNG at {dpi} dpi) |
| `patterns.pdf / .svg / .png` | measured (top) and model (bottom) patterns of a few windows, one colour scale |
| `METHODS.md` | what was measured and how it was analysed, with this run's numbers |

Columns of `solution.csv`:

| column | meaning |
|---|---|
| `t_s` | time in the scan (s) |
| `t_since_onset_s` | time since the deposition started (s) |
| `nominal_deposit_nm` | rate × time since the start (nm) |
| `stage` | growth stage from the readings of the pattern |
| `solution` | which solution of the window is shown |
| `fit_deviation` | mean deviance per cell with a 5 % model-error floor (lower is better) |
{columns}
Source: `{result}`
"""


def export(result, out, choice="smooth", windows=None, n_patterns=4, dpi=300):
    result, out = Path(result), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    main = _read_json(result / "main" / "main.json")
    analysis = _read_json(result / "analysis.json") if (result / "analysis.json").exists() else {}
    request = _read_json(result / "gimap_request.json") if (result / "gimap_request.json").exists() else None
    wins, keys = write_tables(main, _stages(result), choice, out)
    if not wins:
        raise ValueError("the result has no window with a solution")
    evolution_figure(main, wins, choice, out, dpi)
    shown_t = patterns_figure(result, wins, choice, windows, n_patterns, out, dpi)
    (out / "METHODS.md").write_text(methods_text(result, main, analysis, request, wins, choice, shown_t), encoding="utf-8")
    cols = "".join(f"| `{c[1]}` | {c[2]} |\n" for c in keys)
    words = "as chosen in the GUI" if isinstance(choice, dict) else choice
    (out / "README.md").write_text(README.format(scan=main.get("scan", ""), choice=words, dpi=dpi, columns=cols,
                                                  result=result), encoding="utf-8")
    files = sorted(p.name for p in out.iterdir() if p.is_file())
    return dict(folder=str(out), files=files, windows=len(wins), patterns=shown_t)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("result")
    ap.add_argument("out")
    ap.add_argument("--choice", default="smooth", choices=["smooth", "pick"])
    ap.add_argument("--shown", default="", help="JSON list of {\"t\": s, \"label\": entry}: the solution shown per window")
    ap.add_argument("--windows", default="", help="times (s) of the windows in patterns.*, comma separated")
    ap.add_argument("--n-patterns", type=int, default=4)
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()
    chosen = [x for x in a.windows.split(",") if x.strip()]
    choice = a.choice
    if a.shown:
        choice = {round(float(x["t"]), 3): str(x["label"]) for x in _read_json(a.shown)}
    print(json.dumps(export(a.result, a.out, choice, chosen, a.n_patterns, a.dpi), ensure_ascii=False))


if __name__ == "__main__":
    main()
