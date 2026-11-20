"""Export the best state so far of a training run (its NAME.pt) as a network for numpy, while it still trains.

usage: python tools/ml1d/export_checkpoint.py --checkpoint PATH/NAME.pt --data DIR --prior tools/ml1d/prior_v2.json
           --out PATH/OTHER [--architecture '{}'] [--seed 0]
  writes PATH/OTHER.npz and .json as train_flow.py does at its end (the history so far in the record)
"""
from __future__ import annotations

import argparse
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

import simulate  # noqa: E402
from flow_torch import ARCHITECTURE, Flow  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help="the training data folder (its manifest only is read)")
    parser.add_argument("--prior", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--architecture", default="{}")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu")
    flow = Flow({**ARCHITECTURE, **json.loads(args.architecture)}, seed=args.seed)
    flow.load_state_dict(saved["best_state"])
    manifest = json.loads((Path(args.data) / "manifest.json").read_text(encoding="utf-8"))
    data_record = {k: manifest[k] for k in ("prior_file", "count", "seed", "shard_size", "commit", "shards")}
    arrays, meta = flow.export({"prior": simulate.load_prior(args.prior), "data": data_record,
                                "training": {"checkpoint": Path(args.checkpoint).name, "epochs": saved["epoch"],
                                             "best_validation": saved["best"], "history": saved["history"],
                                             "torch": torch.__version__, "partial": True}})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out.with_suffix(".npz"), **arrays)
    meta["sha256"] = hashlib.sha256(out.with_suffix(".npz").read_bytes()).hexdigest()
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"epoch {saved['epoch']}, best validation {saved['best']:.4f}: wrote {out.with_suffix('.npz')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
