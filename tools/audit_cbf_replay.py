"""Can the frozen network recover an exactly representable, noise-free curve?"""

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
from tools.audit_cbf_causes import Model, OUT
from src.gimap.features.fitting.application.workflow_v5 import validate_options
from src.gimap.features.fitting.infrastructure.adapters.workflow_v5 import WorkflowEngine


def main():
    data = json.loads((OUT / "inputs.json").read_text())
    records = []
    bank = []
    for phase in ("range", "mixture", "targeted"):
        bank += json.loads((OUT / f"{phase}.json").read_text())
    engine = WorkflowEngine()
    for side in ("positive", "negative"):
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
        t = best["task"]
        m = Model(data[side], t["arm"], t["types"], t["gates"])
        truth = m.predict(best["z"])
        item = dict(
            side=side,
            sign=1,
            q=m.q,
            observed=truth,
            sigma=np.array(data[side]["sigma"]),
            normalizer=m.norm,
            sigma_estimated=True,
        )
        for numerical in (False, True):
            options = validate_options(
                dict(
                    components=t["types"],
                    sigma_res=float(np.clip(best["sigma_res"], 0.007, 0.013)),
                    nu_res=float(np.clip(best["nu_res"], 5, 10)),
                    normalizer=m.norm,
                    numerical=numerical,
                )
            )
            tick = time.perf_counter()
            result, artifact, raw = engine.fit_side(item, options, lambda *a: None, lambda: False)
            valid = raw["mask"][0] > 0
            pred = raw["curves"][0][:, valid] * m.norm
            errors = np.sqrt(np.mean(np.log(np.maximum(pred, 1e-30) / truth) ** 2, axis=1))
            record = dict(
                side=side,
                numerical=numerical,
                rank1_logrmse=result["solutions"][0]["positive_observation_logrmse"],
                best_bank_logrmse=float(errors.min()),
                candidate_count=len(errors),
                seconds=time.perf_counter() - tick,
                options=options,
                synthetic_teacher_task=t,
                note="Diagnostic replay from the exact frozen forward within its allowed ranges. Known composition/resolution are supplied; this is not experimental ground truth or an independent population benchmark.",
            )
            records.append(record)
            (OUT / "replay_checks.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
