"""Double quadrature orders at every fitted parameter vector, without refitting."""

import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import json
from functools import partial
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_cbf_causes import Model, OUT
from tools.audit_gauss_forward import component


def main():
    data = json.loads((OUT / "inputs.json").read_text())
    results = []
    for fit in json.loads((OUT / "gauss48.json").read_text()):
        task = fit["task"]
        m = Model(
            data[task["side"]], task["arm"], task["types"], task["gates"], quadrature="gauss48"
        )
        previous = m.predict(fit["z"])
        checks = []
        for order, angles in ((96, 192), (192, 384)):
            m.component = partial(component, order=order, orientation_order=angles)
            m.cache.clear()
            current = m.predict(fit["z"])
            checks.append(
                dict(
                    order=order,
                    orientation_order=angles,
                    change_logrmse=float(np.sqrt(np.mean(np.log(current / previous) ** 2))),
                    max_relative_change=float(np.max(abs(current / previous - 1))),
                    observed_logrmse=float(np.sqrt(np.mean(np.log(current / m.y) ** 2))),
                )
            )
            previous = current
        results.append(dict(task=task, baseline_logrmse=fit["logrmse"], checks=checks))
        print(json.dumps(results[-1]), flush=True)
    (OUT / "convergence_checks.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
