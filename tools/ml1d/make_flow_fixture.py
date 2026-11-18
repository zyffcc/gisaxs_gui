"""Write the small reference network of tests/test_ml1d_flow.py: random weights in PyTorch, exported, with PyTorch's
summaries, composition probabilities, log-densities and samples, so the test checks the numpy flow without PyTorch.

usage: python tools/ml1d/make_flow_fixture.py   (writes tests/data/ml1d/flow_fixture.npz, .json, _reference.npz)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for entry in (str(ROOT), str(Path(__file__).resolve().parent)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import torch  # noqa: E402

from flow_torch import Flow  # noqa: E402
from src.gimap.features.fitting.domain.posterior import encoding  # noqa: E402

SMALL = {"channels": [4, 4], "kernel": 3, "dense": 16, "summary": 8, "context": 8, "layers": 4, "hidden": 16,
         "blocks": 2, "bins": 4, "bound": 3.0, "head": 8}


def main() -> int:
    torch.manual_seed(0)
    flow = Flow(SMALL, seed=3).double()
    with torch.no_grad():
        for parameter in flow.parameters():
            parameter.copy_((0.4 * torch.randn_like(parameter)).float().double())  # as exported (float32)
    rng = np.random.default_rng(0)
    rows = 6
    counts = rng.poisson(1e3 * np.exp(-np.linspace(0, 6, encoding.POINTS)), (rows, encoding.POINTS)).astype(float)
    mask = rng.uniform(size=counts.shape) > 0.05
    features = encoding.features(counts, np.sqrt(np.maximum(counts, 1)), mask).astype(np.float64)
    conditions = encoding.conditions([10, 30, 100, 300, 50, 20], [0, 0, 0.3, 0.8, 0, 0.5]).astype(np.float64)
    composition = np.array([0, 5, 17, 100, 257, 40], dtype=np.int64)
    z = rng.normal(0, 2.0, (rows, encoding.DIMS))
    z[0, 1] = 3.5  # beyond the bound (identity there)
    u = rng.normal(0, 1.5, (rows, encoding.DIMS))
    active = encoding.active(composition)
    f, c, comp = torch.as_tensor(features), torch.as_tensor(conditions), torch.as_tensor(composition)
    with torch.no_grad():
        h = flow.encoder(f, c)
        log_probs = torch.log_softmax(flow.head(h), dim=1)
        context = flow.flow_context(h, comp)
        nll, _ce = flow.losses(f, c, comp, torch.as_tensor(z))
        sample, log_det = flow.inverse_u(torch.as_tensor(u), context, torch.as_tensor(active))
    folder = ROOT / "tests" / "data" / "ml1d"
    folder.mkdir(parents=True, exist_ok=True)
    arrays, meta = flow.export({"purpose": "reference network of tests/test_ml1d_flow.py (random weights)"})
    np.savez(folder / "flow_fixture.npz", **arrays)
    meta["sha256"] = hashlib.sha256((folder / "flow_fixture.npz").read_bytes()).hexdigest()
    (folder / "flow_fixture.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    np.savez(folder / "flow_fixture_reference.npz", features=features, conditions=conditions, composition=composition,
             z=z, u=u, summary=h.numpy(), composition_log_probs=log_probs.numpy(), context=context.numpy(),
             log_prob=-nll.numpy(), sample=sample.numpy(), log_det=log_det.numpy())
    print("wrote", folder)
    return 0


if __name__ == "__main__":
    sys.exit(main())
