'Short report of one scan from an existing OUTDIR/windows.csv (no model evaluation, no data needed).'

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from . import registry as REG
from .replot import lines_of

HERE = Path(__file__).resolve().parent
STAGES = ["nucleation", "islands", "coalescence", "film"]
STAGE_ZH = dict(nucleation="小岛（不足 1.5 nm）", islands="分立岛", coalescence="聚并", film="成膜", porous="多孔基底（不分阶段）")
STAGE_COL = dict(nucleation="#f0f0f0", islands="#deebf7", coalescence="#fee6ce", film="#e5f5e0", porous="#f0f0f0")
SRC_ZH = dict(reading="读数", model="模型值", estimate="估计")
GOOD, FAIR = 4.0, 10.0


def _fit(L):
    for k, suf in (("seq_dev5", "_seq"), ("refined_dev5", "_refined"), ("best_dev5", "")):
        if k in L:
            return float(L[k]), suf
    return float("nan"), ""


def _has(L, which):
    return (L.get("obs_fringe_status") == "fringe" and "obs_d_fringe_nm" in L) if which == "d" else \
        (L.get("obs_h_grain_status") == "fringe" and "obs_h_grain_nm" in L)


def _stages(lines, porous):
    '[(stage, source)] of the windows. From the readings of the whole scan: islands from the first grain-height'
    if porous:
        return [("porous", "substrate")] * len(lines)
    by_amount = lambda h: "nucleation" if h < 1.5 else "islands" if h < 2.6 else "coalescence" if h < 6.5 else "film"
    ig = [i for i, L in enumerate(lines) if _has(L, "g")]
    idd = [i for i, L in enumerate(lines) if _has(L, "d")]
    out = []
    for i, L in enumerate(lines):
        h = float(L["h_nom"])
        if ig and ig[0] <= i <= ig[-1]:
            out.append(("coalescence" if idd and i >= idd[0] else "islands", "reading"))
        elif ig and i > ig[-1] and idd and i >= idd[0]:
            out.append(("film", "reading"))
        elif not ig and idd and idd[0] <= i <= idd[-1]:
            out.append(("film" if h >= 6.5 else "coalescence", "reading"))
        else:
            out.append((by_amount(h), "amount"))
    return out


def rows_of(lines, main, porous):
    'Report rows: dict per usable window. main: [(column, label)] of the model quantities shown.'
    out, top = [], 0
    lines = [L for L in lines if "h_nom" in L and "best_dev5" in L]

    n_lo = sum(L.get("obs_yoneda_status") == "maximum" for L in lines); n_gr = sum(L.get("obs_yoneda_grain_status") == "maximum" for L in lines)
    yb = "" if n_lo >= 0.7 * len(lines) or n_lo >= n_gr else "_grain"
    stages = _stages(lines, porous)
    for L, (st, ss) in zip(lines, stages):
        h = float(L["h_nom"])
        fit, suf = _fit(L)
        if not porous:
            top = max(top, STAGES.index(st))
            if STAGES[top] != st:
                st, ss = STAGES[top], "earlier window"
        r = dict(t=L["t"], t_rel=L.get("t_rel"), h_nom=h, stage=st, stage_source=ss, fit=fit, joint=suf == "_seq",
                 fit_grade="good" if fit <= GOOD else "fair" if fit <= FAIR else "poor", q={})
        q = r["q"]
        if not porous:
            if L.get("obs_fringe_status") == "fringe" and "obs_d_fringe_nm" in L:
                d = float(L["obs_d_fringe_nm"]); q["d_film_nm"] = (d, 0.9 * d, 1.1 * d, "reading")
            elif st in ("coalescence", "film"):
                d = h + 1.0 if h < 20 else 1.10 * h
                q["d_film_nm"] = (d, h + 0.4, h + 2.0, "estimate") if h < 20 else (d, 1.05 * h, 1.20 * h, "estimate")
            if L.get("obs_h_grain_status") == "fringe" and "obs_h_grain_nm" in L:
                g = float(L["obs_h_grain_nm"]); q["h_grain_nm"] = (g, 0.9 * g, 1.1 * g, "reading")
            elif st in ("islands", "nucleation") and h >= 0.5:
                q["h_grain_nm"] = (h + 2.1, h + 1.5, h + 2.6, "estimate")
            elif "d_film_nm" in q:
                d = q["d_film_nm"]; q["h_grain_nm"] = (d[0] + 0.5, d[1], d[2] + 1.5, "estimate")
            if "d_film_nm" in q:
                d = q["d_film_nm"]
                q["density"] = (min(1.0, h / d[0]), min(1.0, h / d[2]), min(1.0, h / d[1]), d[3])
            if L.get(f"obs_yoneda{yb}_status") == "maximum" and f"obs_yoneda{yb}_rel" in L:
                y = float(L[f"obs_yoneda{yb}_deg"]); q["yoneda_deg"] = (y, y - 0.01, y + 0.01, "reading")
                e = float(L[f"obs_yoneda{yb}_rel"]); q["yoneda_rel"] = (e, e * (1 - 0.02 / y) ** 2, e * (1 + 0.02 / y) ** 2, "reading")
            if "obs_L_knee_nm" in L:
                k = float(L["obs_L_knee_nm"]); s_ = L.get("obs_knee_status", "knee")
                q["L_lateral_nm"] = (k, k if s_ != "at upper q limit" else np.nan, k if s_ != "at lower q limit" else np.nan, "reading")
        for c, _lab in main:
            v = L.get(c + suf, L.get(c))
            if v is None:
                continue
            alt = [float(L[k]) for k in (c, c + "_path", c + "_refined", c + "_seq", c + "_net_p16", c + "_net_p84") if k in L]
            q[c] = (float(v), min(alt), max(alt), "model" if fit <= FAIR or r["joint"] else "estimate")
        out.append(r)
    return out


def compaction(rows):
    'Interval in which the film turns optically compact, from the reading yoneda_rel: None when the reading rises by'
    R = [r for r in rows if "yoneda_rel" in r["q"]]
    if len(R) < 8:
        return None
    v = np.array([r["q"]["yoneda_rel"][0] for r in R])
    lo, hi = float(np.median(v[:3])), float(np.median(v[-max(3, len(v) // 3):]))
    if hi - lo < 0.25:
        return None
    i0 = int(np.argmax(v >= lo + 0.2 * (hi - lo))); i1 = int(np.argmax(v >= lo + 0.8 * (hi - lo)))
    i0 = max(i0 - 1, 0)
    if i1 <= i0:
        return None
    a, b = R[i0], R[i1]
    tin = {r["t"] for r in R[i0:i1 + 1]}
    fin = [r["fit"] for r in rows if r["t"] in tin]; fout = [r["fit"] for r in rows if r["t"] not in tin]
    return dict(t0=a["t"], t1=b["t"], h0=a["h_nom"], h1=b["h_nom"], af0=a["q"]["yoneda_deg"][0], af1=b["q"]["yoneda_deg"][0],
                fit_in=float(np.median(fin)), fit_out=float(np.median(fout)) if fout else float("nan"))


def _fmt(v, nd=1):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def _cell(q, key, nd=1):
    if key not in q:
        return "—"
    v, lo, hi, src = q[key]
    mark = dict(reading="", model="ᵐ", estimate="≈")[src]
    rng = f"（{_fmt(lo, nd)}–{_fmt(hi, nd)}）" if np.isfinite(lo) and np.isfinite(hi) and hi > lo * 1.02 else ("（至少）" if not np.isfinite(hi) else "（至多）" if not np.isfinite(lo) else "")
    return f"{mark}{_fmt(v, nd)}{rng}"


def write_csv(path, rows, keys):
    head = ["t", "t_rel", "h_nom", "stage", "stage_source", "fit_dev5", "fit_grade"]
    for k in keys:
        head += [k, k + "_low", k + "_high", k + "_source"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh, lineterminator="\n")
        wr.writerow(head)
        for r in rows:
            line = [r["t"], r["t_rel"], round(r["h_nom"], 4), r["stage"], r["stage_source"], round(r["fit"], 3), r["fit_grade"]]
            for k in keys:
                v = r["q"].get(k)
                line += ["", "", "", ""] if v is None else [round(v[0], 4), "" if not np.isfinite(v[1]) else round(v[1], 4),
                                                           "" if not np.isfinite(v[2]) else round(v[2], 4), v[3]]
            wr.writerow(line)


def figure(path, rows, main, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.array([r["t"] for r in rows])
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True)
    ax = ax.ravel()

    def series(key):
        return np.array([r["q"].get(key, (np.nan,) * 3)[:3] for r in rows], float), [r["q"].get(key, (0, 0, 0, ""))[3] for r in rows]

    def draw(a, key, color, label):
        v, src = series(key)
        if not np.isfinite(v[:, 0]).any():
            return
        a.fill_between(t, np.where(np.isfinite(v[:, 1]), v[:, 1], v[:, 0]), np.where(np.isfinite(v[:, 2]), v[:, 2], v[:, 0]), color=color, alpha=0.15, lw=0)
        bound = ~np.isfinite(v[:, 1]) | ~np.isfinite(v[:, 2])
        for s_, mk, mfc in (("reading", "o", color), ("model", "s", color), ("estimate", "o", "none")):
            m = np.array([x == s_ for x in src]) & ~bound
            if m.any():
                a.plot(t[m], v[m, 0], mk, color=color, mfc=mfc, ms=3.5, label=f"{label} ({s_})")
        m = np.array([x != "" for x in src]) & bound & np.isfinite(v[:, 0])
        if m.any():
            a.plot(t[m], v[m, 0], "^", color=color, mfc="none", ms=3.5, label=f"{label} (bound: q range ends)")

    for i, r in enumerate(rows):
        t0 = t[i] if i == 0 else 0.5 * (t[i - 1] + t[i]); t1 = t[i] if i == len(rows) - 1 else 0.5 * (t[i] + t[i + 1])
        for a in ax:
            a.axvspan(t0, t1, color=STAGE_COL[r["stage"]], lw=0, zorder=0)
    ax[0].plot(t, [r["h_nom"] for r in rows], "--", color="0.35", lw=1, label="nominal amount")
    draw(ax[0], "d_film_nm", "#b30000", "film thickness"); draw(ax[0], "h_grain_nm", "#08519c", "grain height")
    ax[0].set_ylabel("nm"); ax[0].set_title("thickness and height", fontsize=9); ax[0].set_yscale("log")
    cols = ["#6a51a3", "#238b45", "#d94801", "#525252"]
    draw(ax[1], "L_lateral_nm", "#000000", "lateral scale 2pi/q")
    k = 0
    for c, lab in main:
        if c.endswith("_nm"):
            draw(ax[1], c, cols[k % 4], lab); k += 1
    ax[1].set_ylabel("nm"); ax[1].set_title("lateral sizes", fontsize=9); ax[1].set_yscale("log")
    draw(ax[2], "density", "#b30000", "film density h/d")
    draw(ax[2], "yoneda_rel", "#000000", "optical compactness (alpha_max / alpha_c)^2")
    for c, lab in main:
        if not c.endswith("_nm"):
            draw(ax[2], c, cols[k % 4], lab); k += 1
    ax[2].set_title("density, coverage and other model quantities", fontsize=9); ax[2].set_xlabel("t (s)")
    f = np.array([r["fit"] for r in rows])
    ax[3].axhspan(0, GOOD, color="#c7e9c0", lw=0); ax[3].axhspan(GOOD, FAIR, color="#fff7bc", lw=0); ax[3].axhspan(FAIR, max(FAIR * 1.5, np.nanmax(f) * 1.1), color="#fcbba1", lw=0)
    ax[3].plot(t, f, "k.-", ms=4, lw=0.8); ax[3].set_ylim(0, max(FAIR * 1.5, np.nanmax(f) * 1.1))
    ax[3].set_title("fit deviation per cell (green: good, yellow: fair, red: model not usable)", fontsize=9); ax[3].set_xlabel("t (s)")
    for a in ax[:3]:
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=6, ncol=2, loc="lower right")
    seen = []
    for r in rows:
        if r["stage"] not in seen:
            seen.append(r["stage"]); ax[3].text(r["t"], ax[3].get_ylim()[1] * 0.985, " " + r["stage"], fontsize=7, va="top", rotation=90)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def markdown(path, scan, fam_zh, rows, main, summ):
    keys = [("d_film_nm", "膜厚 (nm)", 1), ("h_grain_nm", "晶粒高度 (nm)", 1), ("density", "致密度 h/d", 2), ("yoneda_rel", "光学致密度 (α/α_c)²", 2),
            ("L_lateral_nm", "横向尺度 2π/q (nm)", 0)] + [(c, lab, 2 if not c.endswith("_nm") else 1) for c, lab in main]
    keys = [k for k in keys if any(k[0] in r["q"] for r in rows)]
    txt = [f"# {scan}：形貌报告（{fam_zh}）", "",
           "开发参考：结果没有经过独立表征（AFM、XRR、SEM）验证。数字前没有记号的是直接读数，ᵐ 是模型值，≈ 是估计；括号里是范围。", ""]
    st_seq = []
    for r in rows:
        if not st_seq or st_seq[-1][0] != r["stage"]:
            st_seq.append([r["stage"], []])
        st_seq[-1][1].append(r)
    txt += ["## 生长阶段", "", "| 阶段 | 时间 t (s) | 名义沉积量 (nm) | " + " | ".join(k[1] for k in keys) + " | 拟合 |", "|---|---|---|" + "---|" * (len(keys) + 1)]
    for st, rr in st_seq:
        a, b = rr[0], rr[-1]
        cells = []
        for key, _lab, nd in keys:
            va, vb = a["q"].get(key), b["q"].get(key)
            if va is None and vb is None:
                cells.append("—"); continue
            one = lambda v: "—" if v is None else dict(reading="", model="ᵐ", estimate="≈")[v[3]] + _fmt(v[0], nd)
            cells.append(f"{one(va)} → {one(vb)}")
        fg = [r["fit_grade"] for r in rr]
        fit = f"好 {fg.count('good')}／一般 {fg.count('fair')}／差 {fg.count('poor')}"
        txt.append(f"| {STAGE_ZH[st]} | {a['t']:g}–{b['t']:g} | {a['h_nom']:.1f}–{b['h_nom']:.1f} | " + " | ".join(cells) + f" | {fit} |")
    joint = any(r.get("joint") for r in rows)
    txt += ["", "每格是该阶段第一个窗 → 最后一个窗的数值。“拟合”是该阶段里模型回图偏差为好（≤4）、一般（4–10）、差（>10）的窗数"
            + ("；模型量都来自整条扫描联合精修，差的窗照样算模型值，偏差照列。" if joint else "；差的窗里模型量只作估计。"), "",
            "## 几个时刻的数值", "", "| t (s) | 名义沉积量 (nm) | 阶段 | " + " | ".join(k[1] for k in keys) + " | 拟合偏差 |", "|---|---|---|" + "---|" * (len(keys) + 1)]
    pick = sorted({int(round(i)) for i in np.linspace(0, len(rows) - 1, min(8, len(rows)))})
    for i in pick:
        r = rows[i]
        txt.append(f"| {r['t']:g} | {r['h_nom']:.1f} | {STAGE_ZH[r['stage']]} | " + " | ".join(_cell(r["q"], k[0], k[2]) for k in keys) + f" | {r['fit']:.1f} |")
    n = len(rows); ng = sum(r["fit_grade"] == "good" for r in rows); nf = sum(r["fit_grade"] == "fair" for r in rows)
    comp = compaction(rows)
    if comp:
        txt += ["", f"**膜变致密的区间（直接读数）**：金属散射峰所在的出射角从 {comp['af0']:.2f}° 移到 {comp['af1']:.2f}°，"
                f"发生在 t = {comp['t0']:g}–{comp['t1']:g} s、名义沉积量 {comp['h0']:.1f}–{comp['h1']:.1f} nm。"
                "在这之前是基底上的分立岛（峰在基底的临界角），之后是光学上致密的一层膜（峰在金属的临界角附近）。"
                f"这一段里模型拟合偏差的中位数是 {comp['fit_in']:.1f}，其余窗是 {comp['fit_out']:.1f}。"]
    txt += ["", "![演化](report.png)", "", "## 怎么读", "",
            "- **读数**：不经过模型，直接从计数读出。膜厚来自低 q 的膜干涉条纹，晶粒高度来自晶粒尺度的同一种条纹，横向尺度是水平切线上 q²I 最大处对应的 2π/q（特征长度，不是半径也不是间距）。光学致密度是金属散射峰所在出射角与致密金属临界角之比的平方：分立岛时峰在基底的临界角（Si 上 Ag 约 0.26、Au 约 0.15），膜致密后接近 1；薄膜时它是指标，不等于密度本身。",
            ("- **模型值**：网络给出候选、正向模型画回去与实测比较后的解。整条扫描联合精修（各窗共用样品常数、金属量随时间只增不减、形貌随时间平滑变化）的解在每个窗都算模型值，"
             "不论拟合偏差大小；偏差列在旁边，偏差高的窗说明图样里有模型没描述的部分，数值是模型对它最接近的描述。" if joint else
             "- **模型值**：网络给出候选、正向模型画回去与实测比较后的解；没有整条扫描联合精修时，只在拟合偏差不超过 10 的窗里算模型值。")
            + "括号里的范围是这个窗几种解（网络区间、单窗精修、整条扫描精修）的极值。",
            "- **估计**：数据在这里定不了。膜厚和晶粒高度用 19 条扫描的经验关系（膜厚≈沉积量＋1 nm；晶粒未连片时高度≈沉积量＋2.1 nm）"
            + ("" if joint else "；拟合偏差超过 10 的窗里的模型量也归为估计") + "。估计可以用来了解量级，不要当测量值引用。",
            "- **阶段**由读数划分：只有晶粒高度条纹是分立岛；两种条纹都有是聚并；只剩膜条纹是成膜。没有读数的窗按名义沉积量划分。",
            f"- 本扫描 {n} 个窗：拟合好 {ng}、一般 {nf}、差 {n - ng - nf}。",
            "- 逐窗数值在 `report.csv`（每个量四列：值、下限、上限、来源）；全部细节在 `windows.csv`。", ""]
    Path(path).write_bytes("\n".join(txt).encode("utf-8"))


def report(d):
    d = Path(d)
    summ = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    cfg = REG.load(summ["family"])
    tab = json.loads((HERE / "batch_scans.json").read_text(encoding="utf-8"))
    kind = summ["family"].split("_v")[0]
    same = lambda d: d.get(summ["family"]) or next((v for k, v in d.items() if k.split("_v")[0] == kind), None)
    fig = same(tab.get("figure", {})) or []
    main = [(a[0], a[1].split(" (")[0]) for a in fig]
    zh = {a[0]: a[2] for a in fig}
    lines = lines_of(d / "windows.csv")
    porous = summ["family"].startswith("tio2")
    rows = rows_of(lines, main, porous)
    if not rows:
        print(f"{d}: no usable window"); return None
    keys = ([] if porous else ["d_film_nm", "h_grain_nm", "density", "yoneda_deg", "yoneda_rel", "L_lateral_nm"]) + [c for c, _ in main]
    write_csv(d / "report.csv", rows, keys)
    names = same(tab.get("families", {})) or {}
    fam_en, fam_zh = names.get("en", summ["family"]), names.get("zh", summ["family"])
    figure(d / "report.png", rows, main, f"{summ['scan']} ({fam_en}; development reference)")
    markdown(d / "REPORT.md", summ["scan"], fam_zh, rows, [(c, zh.get(c, lab)) for c, lab in main], summ)
    st = [r["stage"] for r in rows]
    print(f"{summ['scan']}: {len(rows)} windows; stages " + ", ".join(f"{s} {st.count(s)}" for s in dict.fromkeys(st))
          + f"; fit good {sum(r['fit_grade'] == 'good' for r in rows)} fair {sum(r['fit_grade'] == 'fair' for r in rows)} poor {sum(r['fit_grade'] == 'poor' for r in rows)}")
    return rows


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for d in sys.argv[1:]:
        report(d)


if __name__ == "__main__":
    main()
