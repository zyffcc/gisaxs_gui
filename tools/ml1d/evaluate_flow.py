"""Check an exported proposal network on simulated cuts it was not trained on (numpy only, one CPU thread).

Per curve: S samples of the posterior; for every active parameter the rank of the truth among them (simulation-based
calibration: uniform ranks for a calibrated posterior), whether the central 68 % and 95 % intervals hold it, and the
error of the posterior median; the time per curve (context and S samples).

usage: python tools/ml1d/evaluate_flow.py --network PATH/NAME --data DIR --prior tools/ml1d/prior_k1.json
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
from src.gimap.features.fitting.domain.posterior import encoding  # noqa: E402
from src.gimap.features.fitting.domain.posterior.flow import Network  # noqa: E402

BINS = 20


def evaluate(network: Network, data: dict, prior: dict, curves: int, samples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    rows = np.arange(min(curves, data["z"].shape[0]))
    features, conditions, _z, active = simulate.inputs(data, rows, prior, rng)
    truth = data["values"][rows].astype(float)
    dims = len(encoding.NAMES)
    ranks = np.full((rows.size, dims), -1)
    inside68 = np.zeros((rows.size, dims), bool)
    inside95 = np.zeros_like(inside68)
    median = np.zeros((rows.size, dims))
    seconds = []
    for i in range(rows.size):
        start = time.perf_counter()
        context = network.context(features[i:i + 1], conditions[i:i + 1])
        z, _log_q = network.sample(context, samples, rng)
        seconds.append(time.perf_counter() - start)
        values, _ = encoding.from_z(z, np.full(samples, data["ratio"][i]))
        ranks[i] = np.sum(values < truth[i], axis=0)
        low68, high68, low95, high95 = np.quantile(values, [0.16, 0.84, 0.025, 0.975], axis=0)
        inside68[i] = (truth[i] >= low68) & (truth[i] <= high68)
        inside95[i] = (truth[i] >= low95) & (truth[i] <= high95)
        median[i] = np.median(values, axis=0)
    report = {"curves": int(rows.size), "samples": samples, "seconds_per_curve": float(np.median(seconds)),
              "parameters": {}}
    for d, name in enumerate(encoding.NAMES):
        use = active[:, d]
        histogram = np.histogram(ranks[use, d], bins=BINS, range=(0, samples + 1))[0]
        expected = use.sum() / BINS
        chi2 = float(np.sum((histogram - expected) ** 2 / expected))
        error = median[use, d] - truth[use, d]
        entry = {"curves": int(use.sum()), "coverage68": float(inside68[use, d].mean()),
                 "coverage95": float(inside95[use, d].mean()), "rank_chi2_19dof": chi2,
                 "rank_histogram": histogram.tolist(), "median_abs_error": float(np.median(np.abs(error)))}
        if name.startswith("ln_"):
            entry["median_relative_error"] = float(np.median(np.abs(np.expm1(error))))
        report["parameters"][name] = entry
    # D itself (2R · D/2R) and per family
    use = active[:, 2]
    d_error = (median[use, 0] + median[use, 2]) - (truth[use, 0] + truth[use, 2])
    report["D_median_relative_error"] = float(np.median(np.abs(np.expm1(d_error))))
    for f, family in enumerate(encoding.FAMILIES):
        sel = data["family"][rows] == f
        report[f"R_median_relative_error_{family}"] = float(np.median(np.abs(np.expm1(median[sel, 0] - truth[sel, 0]))))
    return report


def figure(report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(report["parameters"])
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5), constrained_layout=True)
    for ax, name in zip(axes.flat, names):
        entry = report["parameters"][name]
        histogram = np.asarray(entry["rank_histogram"])
        ax.bar(np.arange(BINS), histogram, width=1.0, color="#4a78b5")
        expected = entry["curves"] / BINS
        band = 2 * np.sqrt(expected)
        ax.axhspan(expected - band, expected + band, color="0.85", zorder=0)
        ax.set_title(f"{name}: 68 % {entry['coverage68']:.2f}, 95 % {entry['coverage95']:.2f}", fontsize=9)
        ax.set_xticks([])
    fig.suptitle(f"Ranks of the truth among {report['samples']} samples, {report['curves']} curves")
    fig.savefig(path, dpi=130)


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
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    if args.figure:
        figure(report, Path(args.figure))
    return 0


if __name__ == "__main__":
    sys.exit(main())
