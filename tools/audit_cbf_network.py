"""Compare automatic and numerically discovered conditions on identical inputs."""

import os

for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
):
    os.environ[name] = "1"
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.gimap.features.fitting.application.workflow_v5 import validate_options
from src.gimap.features.fitting.infrastructure.adapters.workflow_v5 import (
    WorkflowEngine,
    write_side,
)

OUT = ROOT / "validation/cause_audit_20260921"


def main():
    data = json.loads((OUT / "inputs.json").read_text())
    bank = []
    for name in ("range", "mixture", "targeted"):
        p = OUT / f"{name}.json"
        if p.exists():
            bank += json.loads(p.read_text())
    engine = WorkflowEngine()
    records = []
    for side in ("positive", "negative"):
        d = data[side]
        item = dict(
            side=side,
            sign=1 if side == "positive" else -1,
            q=np.array(d["q"]),
            observed=np.array(d["y"]),
            sigma=np.array(d["sigma"]),
            normalizer=max(d["y"]),
            sigma_estimated=True,
        )
        best = min(
            (
                r
                for r in bank
                if r["task"]["side"] == side
                and r["task"]["arm"] == "train_exact"
                and r["task"].get("resolution", True)
            ),
            key=lambda r: r["logrmse"],
        )
        for source in ("automatic", "fit_derived_conditions"):
            for numerical in (False, True):
                options = validate_options(dict(numerical=numerical))
                if source == "fit_derived_conditions":
                    options.update(
                        components=best["task"]["types"],
                        sigma_res=float(np.clip(best["sigma_res"], 0.007, 0.013)),
                        nu_res=float(np.clip(best["nu_res"], 5, 10)),
                    )
                tick = time.perf_counter()
                result, artifact, raw = engine.fit_side(
                    item, options, lambda *a: None, lambda: False
                )
                folder = OUT / "network" / f"{side}_{source}_{numerical}"
                rows = write_side(folder, item, result, artifact, raw)
                valid = raw["mask"][0] > 0
                curves = raw["curves"][0][:, valid] * item["normalizer"]
                errors = np.sqrt(
                    np.mean(np.log(np.maximum(curves, 1e-30) / item["observed"]) ** 2, axis=1)
                )
                record = dict(
                    side=side,
                    conditions=source,
                    numerical=numerical,
                    rank1_logrmse=rows[0]["best_log_rmse"],
                    best_bank_logrmse=float(errors.min()),
                    candidate_count=len(errors),
                    seconds=time.perf_counter() - tick,
                    options=options,
                    condition_reference_logrmse=best["logrmse"]
                    if source == "fit_derived_conditions"
                    else None,
                    note="Fit-derived conditions are a diagnostic prior, not known experimental ground truth.",
                )
                records.append(record)
                (OUT / "network_checks.json").write_text(
                    json.dumps(records, indent=2), encoding="utf-8"
                )
                print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
