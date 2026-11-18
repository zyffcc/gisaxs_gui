"""Write the model card of an exported proposal network from its record and its checks.

usage: python tools/ml1d/model_card.py --network PATH/NAME --evaluation report.json [--benchmark summary.json ...]
           --out PATH/MODEL_CARD.md
The record (NAME.json) gives the architecture, the prior, the data and the training; the evaluation is
evaluate_flow.py's report on simulated cuts not trained on; each benchmark summary is posterior_recovery.py's.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _table(rows, header) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


def card(network: dict, evaluation: dict, benchmarks: list[tuple[str, dict]]) -> str:
    training = network.get("training", {})
    history = training.get("history") or []
    last = history[-1] if history else {}
    prior = network.get("prior", {})
    data = network.get("data", {})
    arch = network.get("architecture", {})
    lines = [
        f"# Model card: {Path(training.get('out', 'flow')).name or 'flow'} (GIMaP's proposal network of 1D cuts)", "",
        "**What it is.** An amortised posterior p(composition, values | curve) for horizontal GISAXS cuts: a 1D CNN "
        "summary, a head over the 258 compositions of one to three populations (sphere, upright or random cylinder, "
        "each with or without a paracrystal) and a conditional rational-quadratic spline flow over their values "
        "(Physics and Methods, section 9.12). GIMaP uses it only to propose: the values, intervals and probabilities "
        "of a solution come from the model and the curve (importance sampling and least squares).", "",
        f"**Files.** `{Path(training.get('out', 'flow')).name}.npz` (float32 arrays, SHA-256 "
        f"`{network.get('sha256')}`), `.json` (architecture, prior, data, training), this card. Encoding version "
        f"{network.get('encoding')}; format `{network.get('format')}`.", "",
        "**Architecture.** " + ", ".join(f"{k} {v}" for k, v in arch.items()) + ".", "",
        f"**Training data.** {data.get('count')} cuts simulated with cut_model from `{data.get('prior_file')}` "
        f"(seed {data.get('seed')}, code {data.get('commit')}): {prior.get('description', '')}. Every batch draws "
        "the populations' shares, the beam tail, the background, the model error, the measured points, Poisson "
        "counts and gaps anew.", "",
        f"**Training.** Neural posterior estimation (−log q of the values + cross-entropy of the head), AdamW, batch "
        f"{training.get('batch')}, learning rate {training.get('lr')} with warm-up and cosine decay, "
        f"{len(history)} epochs, early stopping on {training.get('validation')} held-out cuts; best validation loss "
        f"{training.get('best_validation')}; torch {training.get('torch')}. Last epoch: "
        f"{json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in last.items()})}.", "",
        "## Calibration on simulated cuts not trained on", "",
        f"{evaluation.get('curves')} cuts, {evaluation.get('samples')} samples each, the true composition given; "
        f"composition head top-1 {evaluation.get('composition_top1'):.3f}; "
        f"{1000 * evaluation.get('seconds_per_curve', 0):.0f} ms per curve on one thread.", "",
    ]
    rows = []
    for kind, entry in evaluation.get("values", {}).items():
        e = entry.get("all", {})
        rows.append((kind, e.get("n"), f"{e.get('coverage68', 0):.3f}", f"{e.get('coverage95', 0):.3f}",
                     f"{e.get('rank_chi2_19dof', 0):.1f}", f"{e.get('median_error', 0):.3f} ({e.get('error_kind')})"))
    lines += _table(rows, ("value", "n", "68 % coverage", "95 % coverage", "rank χ² (19 dof)", "median error"))
    for name, summary in benchmarks:
        lines += ["", f"## Benchmark: {name}", ""]
        rows = []
        for stage in ("network/all", "fit/model", "fit/measured", "fit/all"):
            row = summary["parameters"].get(stage)
            if not row:
                continue
            rows.append((stage, *(f"{row[k]['median_abs']:.4f} / {row[k]['p90_abs']:.3f}"
                                  + (f" · {row[k]['posterior_coverage68']:.2f}" if row[k].get("posterior_coverage68")
                                     is not None else "") for k in ("R", "sR", "D", "s_nn"))))
        lines += _table(rows, ("stage", "R (median / P90 · 68 % coverage)", "σR/R", "D", "σnn/D"))
        lines += ["", f"Composition chosen: {json.dumps(summary.get('composition'))}; median seconds: "
                      f"{json.dumps(summary.get('median_seconds'))}."]
    lines += ["", "## Where it does not apply", "",
              "- cuts spanning less than a factor 10 in q; points below q_max/300 are left out;",
              "- more than three populations, other families, random cylinders larger than R q_max = 50 or "
              "h q_max = 200, beam tails wider than 3 q_min;",
              "- the network itself knows Gaussian sizes by number and the LMA; other laws and the DA are weighted "
              "from its proposals; the four-term DWBA is not in the model;",
              "- its own probabilities are those of the simulated prior; GIMaP reports the importance-weighted ones."]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--network", required=True)
    ap.add_argument("--evaluation", required=True)
    ap.add_argument("--benchmark", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    network = json.loads(Path(a.network).with_suffix(".json").read_text(encoding="utf-8"))
    evaluation = json.loads(Path(a.evaluation).read_text(encoding="utf-8"))
    benchmarks = [(Path(p).stem.replace("_summary", ""), json.loads(Path(p).read_text(encoding="utf-8")))
                  for p in a.benchmark]
    Path(a.out).write_text(card(network, evaluation, benchmarks), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
