"""Train the 1D proposal network (PyTorch) on shards of tools/ml1d/simulate.py and export it for numpy.

Maximum likelihood of the true parameters z under the flow, given the curve (neural posterior estimation); fresh
noise for every batch (``simulate.inputs``); the last ``--validation`` rows are held out with one fixed noise draw, and
the state of the lowest validation loss is exported.

usage: python tools/ml1d/train_flow.py --data DIR --prior tools/ml1d/prior_k1.json --out PATH/NAME
           [--epochs 30] [--batch 512] [--lr 1e-3] [--seed 0] [--validation 10000] [--threads 4]
           [--device cpu] [--architecture '{"layers": 8}']
  writes PATH/NAME.npz (arrays), PATH/NAME.json (architecture, data manifest, prior, history, SHA-256 of the npz)
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for entry in (str(ROOT), str(Path(__file__).resolve().parent)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import torch  # noqa: E402

import simulate  # noqa: E402
from flow_torch import ARCHITECTURE, Flow  # noqa: E402


def tensors(arrays, device):
    return [torch.as_tensor(np.asarray(a, np.float32), device=device) for a in arrays]


def validation_loss(flow, held, device, batch: int) -> float:
    flow.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, held[0].shape[0], batch):
            f, c, z = tensors([a[start:start + batch] for a in held[:3]], device)
            total += float(-flow.log_prob(z, f, c).sum())
            count += z.shape[0]
    flow.train()
    return total / count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--prior", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation", type=int, default=10000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--architecture", default="{}")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    prior = simulate.load_prior(args.prior)
    data, manifest = simulate.load(args.data)
    rows = data["z"].shape[0]
    train_rows = np.arange(rows - args.validation)
    held = simulate.inputs(data, np.arange(rows - args.validation, rows), prior, np.random.default_rng(args.seed + 1))
    arch = {**ARCHITECTURE, **json.loads(args.architecture)}
    flow = Flow(arch, seed=args.seed).to(args.device)
    optimiser = torch.optim.Adam(flow.parameters(), lr=args.lr)
    steps = args.epochs * (train_rows.size // args.batch)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda step: 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(step / steps, 1.0))))
    history, best, best_state = [], math.inf, None
    start = time.perf_counter()
    for epoch in range(args.epochs):
        order = rng.permutation(train_rows)
        losses = []
        for b in range(train_rows.size // args.batch):
            features, conditions, z, _active = simulate.inputs(data, order[b * args.batch:(b + 1) * args.batch],
                                                               prior, rng)
            f, c, zt = tensors([features, conditions, z], args.device)
            loss = -flow.log_prob(zt, f, c).mean()
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
            optimiser.step()
            schedule.step()
            losses.append(float(loss))
        held_loss = validation_loss(flow, held, args.device, 2048)
        history.append({"epoch": epoch + 1, "train": float(np.mean(losses)), "validation": held_loss,
                        "seconds": round(time.perf_counter() - start, 1)})
        print(json.dumps(history[-1]), flush=True)
        if held_loss < best:
            best, best_state = held_loss, copy.deepcopy(flow.state_dict())
    flow.load_state_dict(best_state)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data_record = {k: manifest[k] for k in ("prior_file", "count", "seed", "shard_size", "commit", "shards")}
    arrays, meta = flow.cpu().export({
        "prior": prior, "data": data_record, "training": {**vars(args), "best_validation": best,
                                                          "torch": torch.__version__, "history": history}})
    np.savez(out.with_suffix(".npz"), **arrays)
    meta["sha256"] = hashlib.sha256(out.with_suffix(".npz").read_bytes()).hexdigest()
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"best validation {best:.4f}; wrote {out.with_suffix('.npz')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
