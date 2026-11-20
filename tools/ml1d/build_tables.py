"""Build the size-average tables of cut_model (src/gimap/features/fitting/domain/cut_model/data/*.npz).

The tables are a cache of cut_model's own quadrature (``tables.compute``), written with the parameters they were
built with; ``tables.build`` uses a file only when they match, else it computes the table on first use. Rebuild after
changing ``tables.PARAMETERS`` or the size laws, and run tests/test_cut_model.py. The moments of spheres with two
paths (``tables.compute_moments``, ``<law>_sphere_moments.npz``) are built the same way.
usage: python tools/ml1d/build_tables.py [--check] [--moments]
  --check     compare every file with a fresh build instead of writing (stored in single precision)
  --moments   only the moments of spheres with two paths (the other files and their sums are kept)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.gimap.features.fitting.domain.cut_model import tables  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--moments", action="store_true")
    args = parser.parse_args()
    sums = {}
    for law in tables.LAWS:
        start = time.perf_counter()
        if args.check:
            fresh, stored = tables.compute_moments(law), tables._load_moments(law)
            if stored is None:
                print(f"{law} moments: no matching file")
                return 1
            worst = max(float(np.max(np.abs(a["m"] - b["m"])) / np.max(np.abs(a["m"])))
                        for a, b in zip(fresh["segments"], stored["segments"]))
            print(f"{law} moments: largest difference {worst:.1e} of the table's largest value")
            if worst > 1e-6:
                return 1
        else:
            target = tables.save_moments(law)
            sums[target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
            print(f"{target.name}: {target.stat().st_size / 1e6:.2f} MB, {time.perf_counter() - start:.1f} s")
        for kind in () if args.moments else tables.KINDS:
            start = time.perf_counter()
            if args.check:
                fresh, stored = tables.compute(law, kind), tables._load(law, kind)
                if stored is None:
                    print(f"{law} {kind}: no matching file")
                    return 1
                worst = max(float(np.max(np.abs(a[key] - b[key])) / np.max(np.abs(a[key])))
                            for a, b in zip(fresh["segments"], stored["segments"]) for key in ("e", "b", "a"))
                print(f"{law} {kind}: largest difference {worst:.1e} of each table's largest value")
                if worst > 1e-6:
                    return 1
                continue
            target = tables.save(law, kind)
            sums[target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
            print(f"{target.name}: {target.stat().st_size / 1e6:.2f} MB, {time.perf_counter() - start:.1f} s")
    if sums:
        record = tables.DATA / "SHA256.json"
        if args.moments and record.is_file():
            sums = {**json.loads(record.read_text(encoding="utf-8")), **sums}
        record.write_text(json.dumps(sums, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
