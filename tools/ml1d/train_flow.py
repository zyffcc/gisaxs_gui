"""Train the 1D proposal network (PyTorch) on shards of tools/ml1d/simulate.py and export it for numpy.

Loss per curve: −log q(z_active | curve, composition) of the flow plus ``--head-weight`` × the cross-entropy of the
composition head (neural posterior estimation). Every batch gets fresh shares, tail, background, model error, points,
noise and gaps (``simulate.inputs``, in ``--loaders`` worker processes); the last ``--validation`` rows are held out
with one fixed draw, and the state of the lowest validation loss is exported.

usage: python tools/ml1d/train_flow.py --data DIR --prior tools/ml1d/prior_v2.json --out PATH/NAME
           [--epochs 40] [--batch 1024] [--lr 5e-4] [--seed 0] [--validation 20000] [--threads 4] [--loaders 0]
           [--device cpu|cuda] [--architecture '{"layers": 8}'] [--patience 8] [--hours 0]
  writes PATH/NAME.npz (arrays), PATH/NAME.json (architecture, data manifest, prior, history, SHA-256 of the npz) and
  PATH/NAME.pt (the last state, to continue with --resume)
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

DATA = {}
"""The training rows, set before the loader's workers start (they share it)."""


class Batches(torch.utils.data.IterableDataset):
    """Batches of one epoch: a shuffled order split among the workers, each with its own noise stream."""

    def __init__(self, rows: np.ndarray, batch: int, prior: dict, seed: int):
        super().__init__()
        self.rows, self.batch, self.prior, self.seed = rows, batch, prior, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker, workers = (info.id, info.num_workers) if info else (0, 1)
        order = np.random.default_rng(self.seed).permutation(self.rows)
        rng = np.random.default_rng([self.seed, worker])
        for b in range(worker, order.size // self.batch, workers):
            features, conditions, composition, z, _active = simulate.inputs(
                DATA, order[b * self.batch:(b + 1) * self.batch], self.prior, rng)
            yield (torch.from_numpy(features), torch.from_numpy(conditions), torch.from_numpy(composition),
                   torch.from_numpy(z))


def evaluate(flow, held, device, batch: int, head_weight: float) -> dict:
    flow.eval()
    nll = ce = correct = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, held[0].shape[0], batch):
            f, c, comp, z = (torch.as_tensor(a[start:start + batch]).to(device) for a in held[:4])
            a, b = flow.losses(f, c, comp.long(), z)
            nll += float(a.sum())
            ce += float(b.sum())
            correct += float((flow.head(flow.encoder(f, c)).argmax(dim=1) == comp).sum())
            count += z.shape[0]
    flow.train()
    return {"nll": nll / count, "ce": ce / count, "loss": (nll + head_weight * ce) / count,
            "composition_top1": correct / count}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--prior", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation", type=int, default=20000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--loaders", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--architecture", default="{}")
    parser.add_argument("--head-weight", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--hours", type=float, default=0.0, help="stop after this many hours (0: no limit)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    prior = simulate.load_prior(args.prior)
    data, manifest = simulate.load(args.data)
    DATA.update(data)
    rows = data["z"].shape[0]
    train_rows = np.arange(rows - args.validation)
    held = simulate.inputs(data, np.arange(rows - args.validation, rows), prior, np.random.default_rng(args.seed + 1))
    held = (held[0], held[1], held[2].astype(np.int64), held[3])
    arch = {**ARCHITECTURE, **json.loads(args.architecture)}
    flow = Flow(arch, seed=args.seed).to(args.device)
    optimiser = torch.optim.AdamW(flow.parameters(), lr=args.lr, weight_decay=1e-5)
    steps = args.epochs * (train_rows.size // args.batch)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda step: min(1.0, (step + 1) / 1000) * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(
            step / steps, 1.0)))))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    history, best, best_state, first_epoch = [], math.inf, None, 0
    if args.resume and out.with_suffix(".pt").exists():
        saved = torch.load(out.with_suffix(".pt"), map_location=args.device)
        flow.load_state_dict(saved["flow"])
        optimiser.load_state_dict(saved["optimiser"])
        schedule.load_state_dict(saved["schedule"])
        history, best, best_state, first_epoch = saved["history"], saved["best"], saved["best_state"], saved["epoch"]
    print(f"{rows} rows, {sum(p.numel() for p in flow.parameters())} parameters, device {args.device}", flush=True)
    start = time.perf_counter()
    stale = 0
    for epoch in range(first_epoch, args.epochs):
        loader = torch.utils.data.DataLoader(Batches(train_rows, args.batch, prior, args.seed * 1000 + epoch),
                                             batch_size=None, num_workers=args.loaders,
                                             persistent_workers=False, prefetch_factor=4 if args.loaders else None)
        sums = np.zeros(2)
        batches = 0
        for features, conditions, composition, z in loader:
            features, conditions = features.to(args.device, non_blocking=True), conditions.to(args.device)
            composition, z = composition.long().to(args.device), z.to(args.device)
            nll, ce = flow.losses(features, conditions, composition, z)
            loss = nll.mean() + args.head_weight * ce.mean()
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
            optimiser.step()
            schedule.step()
            sums += [float(nll.mean()), float(ce.mean())]
            batches += 1
        held_scores = evaluate(flow, held, args.device, 4096, args.head_weight)
        history.append({"epoch": epoch + 1, "train_nll": sums[0] / batches, "train_ce": sums[1] / batches,
                        **{f"validation_{k}": v for k, v in held_scores.items()},
                        "seconds": round(time.perf_counter() - start, 1)})
        print(json.dumps(history[-1]), flush=True)
        if held_scores["loss"] < best:
            best, best_state, stale = held_scores["loss"], copy.deepcopy(flow.state_dict()), 0
        else:
            stale += 1
        torch.save({"flow": flow.state_dict(), "optimiser": optimiser.state_dict(), "schedule": schedule.state_dict(),
                    "history": history, "best": best, "best_state": best_state, "epoch": epoch + 1},
                   out.with_suffix(".pt"))
        if stale >= args.patience or (args.hours and time.perf_counter() - start > 3600 * args.hours):
            break
    flow.load_state_dict(best_state)
    data_record = {k: manifest[k] for k in ("prior_file", "count", "seed", "shard_size", "commit", "shards")}
    arrays, meta = flow.cpu().export({
        "prior": prior, "data": data_record,
        "training": {**{k: v for k, v in vars(args).items() if k != "resume"}, "best_validation": best,
                     "torch": torch.__version__, "history": history}})
    np.savez(out.with_suffix(".npz"), **arrays)
    meta["sha256"] = hashlib.sha256(out.with_suffix(".npz").read_bytes()).hexdigest()
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"best validation {best:.4f}; wrote {out.with_suffix('.npz')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
