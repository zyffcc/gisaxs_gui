"""Check an exported proposal network on simulated cuts it was not trained on (numpy only, one CPU thread).

Per curve and its true composition: S samples of the posterior; for every active value the rank of the truth among
them (simulation-based calibration: uniform ranks for a calibrated posterior), whether the central 68 % and 95 %
intervals hold it, and the error of the posterior median; the composition head's probability of the true composition
and its top-1 accuracy; the time per curve.

usage: python tools/ml1d/evaluate_flow.py --network PATH/NAME --data DIR --prior tools/ml1d/prior_v2.json
           [--curves 1000] [--samples 1000] [--seed 5] [--out report.json] [--figure ranks.png]
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for entry in (str(ROOT), str(Path(__file__).resolve().parent)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import simulate  # noqa: E402
from src.gimap.features.fitting.domain.posterior import encoding as E  # noqa: E402
from src.gimap.features.fitting.domain.posterior.flow import Network  # noqa: E402

BINS = 20
KINDS = tuple(E.SLOT_NAMES) + tuple(E.GLOBAL_NAMES)


def _kind(dim: int) -> str:
    return E.SLOT_NAMES[dim % E.PER_SLOT] if dim < E.GLOBAL.start else E.NAMES[dim]


def evaluate(network: Network, data: dict, prior: dict, curves: int, samples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    rows = np.arange(min(curves, data["z"].shape[0]))
    features, conditions, composition, _z, active = simulate.inputs(data, rows, prior, rng)
    truth = data["values"][rows].astype(float)
    ranks = np.full((rows.size, E.DIMS), -1)
    in68 = np.zeros((rows.size, E.DIMS), bool)
    in95 = np.zeros_like(in68)
    error = np.full((rows.size, E.DIMS), np.nan)
    true_log_prob, top1, seconds = [], [], []
    for i in range(rows.size):
        start = time.perf_counter()
        h = network.summary(features[i:i + 1], conditions[i:i + 1])
        log_probs = network.composition_log_probs(h)[0]
        z, _log_q = network.sample(network.context(h, composition[i]), int(composition[i]), samples, rng)
        seconds.append(time.perf_counter() - start)
        values = E.from_z(z, np.full(samples, data["ratio"][i]), composition[i])
        true_log_prob.append(log_probs[composition[i]])
        top1.append(int(np.argmax(log_probs)) == composition[i])
        on = active[i]
        ranks[i, on] = np.sum(values[:, on] < truth[i, on], axis=0)
        lo68, hi68, lo95, hi95 = np.quantile(values[:, on], [0.16, 0.84, 0.025, 0.975], axis=0)
        in68[i, on] = (truth[i, on] >= lo68) & (truth[i, on] <= hi68)
        in95[i, on] = (truth[i, on] >= lo95) & (truth[i, on] <= hi95)
        median = np.median(values[:, on], axis=0)
        logarithmic = E.LOGARITHMIC[on]
        error[i, on] = np.where(logarithmic, np.abs(median / truth[i, on] - 1), np.abs(median - truth[i, on]))
    k = np.array([len(E.COMPOSITIONS[c]) for c in composition])
    report = {"curves": int(rows.size), "samples": samples, "seconds_per_curve": float(np.median(seconds)),
              "composition_top1": float(np.mean(top1)), "true_composition_log_prob": float(np.mean(true_log_prob)),
              "values": {}}
    for kind in KINDS:
        dims = [d for d in range(E.DIMS) if _kind(d) == kind]
        for label, rows_k in (("all", k > 0), ("K=1", k == 1), ("K=2", k == 2), ("K=3", k == 3)):
            on = active[rows_k][:, dims]
            if not on.any():
                continue
            r = ranks[rows_k][:, dims][on]
            histogram = np.histogram(r, bins=BINS, range=(0, samples + 1))[0]
            expected = r.size / BINS
            report["values"].setdefault(kind, {})[label] = {
                "n": int(r.size), "coverage68": float(in68[rows_k][:, dims][on].mean()),
                "coverage95": float(in95[rows_k][:, dims][on].mean()),
                "rank_chi2_19dof": float(np.sum((histogram - expected) ** 2 / expected)),
                "rank_histogram": histogram.tolist(),
                "median_error": float(np.median(error[rows_k][:, dims][on])),
                "error_kind": "relative" if E.LOGARITHMIC[dims[0]] else "absolute"}
    return report


def figure(report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kinds = list(report["values"])
    fig, axes = plt.subplots(2, 5, figsize=(16, 6), constrained_layout=True)
    for ax, kind in zip(axes.flat, kinds):
        entry = report["values"][kind]["all"]
        histogram = np.asarray(entry["rank_histogram"])
        ax.bar(np.arange(BINS), histogram, width=1.0, color="#4a78b5")
        expected = entry["n"] / BINS
        ax.axhspan(expected - 2 * np.sqrt(expected), expected + 2 * np.sqrt(expected), color="0.85", zorder=0)
        ax.set_title(f"{kind}: 68 % {entry['coverage68']:.2f}, 95 % {entry['coverage95']:.2f}", fontsize=9)
        ax.set_xticks([])
    fig.suptitle(f"Ranks of the truth among {report['samples']} samples, {report['curves']} curves; composition "
                 f"top-1 {report['composition_top1']:.2f}")
    fig.savefig(path, dpi=110)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--network", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--prior", required=True)
    parser.add_argument("--curves", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--out")
    parser.add_argument("--figure")
    args = parser.parse_args()
    network = Network.load(args.network)
    data, _manifest = simulate.load(args.data)
    report = evaluate(network, data, simulate.load_prior(args.prior), args.curves, args.samples, args.seed)
    report["network"] = Path(args.network).name
    report["network_sha256"] = network.meta["sha256"]
    text = json.dumps(report, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "values"}))
    for kind, entry in report["values"].items():
        e = entry["all"]
        print(f"{kind:14s} n {e['n']:6d}  68% {e['coverage68']:.3f}  95% {e['coverage95']:.3f}  chi2 "
              f"{e['rank_chi2_19dof']:6.1f}  median {e['error_kind']} error {e['median_error']:.3f}")
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    if args.figure:
        figure(report, Path(args.figure))
    return 0


if __name__ == "__main__":
    sys.exit(main())
