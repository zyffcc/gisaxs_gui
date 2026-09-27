"""Controlled, restartable offline diagnosis; never edits GUI/model weights."""

import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares, nnls
from scipy.special import xlogy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.gimap.features.fitting.infrastructure.adapters.experimental_fit import component_function

OUT = ROOT / "validation/cause_audit_20260921"
ARMS = ("train_exact", "scale_free", "amplitude_free", "resolution_wide", "particle_wide")


def prepare():
    OUT.mkdir(exist_ok=True, parents=True)
    verified = json.loads(
        (ROOT / "validation/preprocess_mask_20260921/single/VERIFIED.json").read_text()
    )
    source = Path(verified["fit_output"])
    raw = (source / "top20_candidates.json").read_bytes()
    rows = json.loads(raw)
    inputs = {}
    for side in ("positive", "negative"):
        bank = [r for r in rows if r["side"] == side]
        r = bank[0]
        y, sg = np.array(r["observed"]), np.array(r["sigma"])
        var = np.maximum(sg**2 - (0.1 * y) ** 2, 1e-30)
        count = np.rint(np.where(y > 0, y / var, 1 / np.sqrt(var)))
        inputs[side] = dict(
            q=np.abs(r["native_q"]).tolist(),
            y=y.tolist(),
            sigma=sg.tolist(),
            count=count.tolist(),
            seeds=bank,
        )
    (OUT / "inputs.json").write_text(json.dumps(inputs), encoding="utf-8")
    (OUT / "provenance.json").write_text(
        json.dumps(
            dict(
                source=str(source),
                sha256=hashlib.sha256(raw).hexdigest(),
                arms=ARMS,
                objective="Natural log residual on all positive native observations, unless explicitly marked poisson; full reference grid fixed in every arm",
                mask=verified["observations"],
                geometry=verified["geometry"],
            ),
            indent=2,
        ),
        encoding="utf-8",
    )


class Model:
    def __init__(self, data, arm, types, gates, resolution=True, quadrature="legacy"):
        self.q, self.y, self.count = (np.array(data[k], float) for k in ("q", "y", "count"))
        self.norm = float(self.y.max())
        self.arm, self.types, self.gates = arm, types, gates
        self.resolution = resolution
        self.free = arm not in ("train_exact", "scale_free")
        self.broad = arm == "particle_wide"
        self.names, self.lo, self.hi = [], [], []
        self.cache = {}
        self.component = component_function()
        if quadrature.startswith("gauss"):
            from functools import partial
            from tools.audit_gauss_forward import component

            self.component = partial(
                component,
                order=48 if quadrature == "gauss48" else 24,
                orientation_order=96 if quadrature == "gauss48" else 64,
            )
        for j, (typ, gate) in enumerate(zip(types, gates)):
            self.add(f"r{j}", np.log(0.2 if self.broad else 1), np.log(200 if self.broad else 100))
            self.add(f"sr{j}", 0.005 if self.broad else 0.02, 1.5 if self.broad else 0.9)
            if gate:
                self.add(f"eta{j}", 0, 1)
                self.add(f"sd{j}", 0.01 if self.broad else 0.05, 1.5 if self.broad else 0.9)
            if typ == 2:
                self.add(
                    f"h{j}", np.log(0.2 if self.broad else 2), np.log(1000 if self.broad else 500)
                )
                self.add(f"sh{j}", 0.005 if self.broad else 0.02, 1.5 if self.broad else 0.9)
        wide = arm in ("resolution_wide", "particle_wide")
        self.add("rs", np.log(0.001 if wide else 0.007), np.log(0.1 if wide else 0.013))
        self.add("nu", 1 if wide else 5, 20 if wide else 10)
        if self.free:
            for j in range(len(types)):
                self.add(f"a{j}", np.log(self.norm * 1e-14), np.log(self.norm * 1e6))
            self.add("bg", np.log(self.norm * 1e-14), np.log(self.norm * 10))
            self.add("rc", np.log(self.norm * 1e-14), np.log(self.norm * 1e12))
        else:
            for j in range(len(types) - 1):
                self.add(f"w{j}", -20, 20)
            self.add("bg", np.log(1e-6), np.log(1e-2))
            self.add("rc", np.log(10), np.log(1000))
            if arm == "scale_free":
                self.add("scale", np.log(self.norm * 1e-10), np.log(self.norm * 1e6))
        self.lo, self.hi = np.array(self.lo), np.array(self.hi)

    def add(self, name, lo, hi):
        self.names.append(name)
        self.lo.append(lo)
        self.hi.append(hi)

    def decode(self, z):
        a = dict(zip(self.names, z))
        parts = []
        for j, (typ, gate) in enumerate(zip(self.types, self.gates)):
            r = np.exp(a[f"r{j}"])
            lower = max(0.5 if self.broad else 3, 2 * r * 1.001)
            upper = 1000 if self.broad else 500
            d = lower * (upper / lower) ** a.get(f"eta{j}", 0)
            parts.append(
                dict(
                    type=typ,
                    R=r,
                    sigma_R=a[f"sr{j}"],
                    D=d,
                    sigma_D=a.get(f"sd{j}", 0.1),
                    h=np.exp(a.get(f"h{j}", np.log(10))),
                    sigma_h=a.get(f"sh{j}", 0.2),
                    structure=bool(gate),
                )
            )
        return a, parts

    def basis(self, z):
        a, parts = self.decode(z)
        columns = []
        for p in parts:
            key = tuple(p.values())
            if key not in self.cache:
                if len(self.cache) > 128:
                    self.cache.clear()
                self.cache[key] = self.component(
                    self.q,
                    p["type"],
                    p["R"],
                    p["sigma_R"],
                    p["h"],
                    p["sigma_h"],
                    p["D"],
                    p["sigma_D"],
                    p["structure"],
                )
            columns.append(self.cache[key])
        resolution = 1 / (1 + (self.q / np.exp(a["rs"])) ** a["nu"])
        if not self.resolution:
            resolution = np.zeros_like(resolution)
        return np.stack(columns, 1), resolution, a, parts

    def predict(self, z):
        forms, resolution, a, _ = self.basis(z)
        if self.free:
            amps = np.exp([a[f"a{j}"] for j in range(len(self.types))])
            return np.maximum(forms @ amps + np.exp(a["bg"]) + np.exp(a["rc"]) * resolution, 1e-30)
        logits = np.array([a[f"w{j}"] for j in range(len(self.types) - 1)] + [0.0])
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        particle = forms @ weights
        background = np.exp(a["bg"]) * np.median(particle)
        resamp = np.exp(a["rc"]) * particle[:5].max() / max(resolution[:5].max(), 1e-30)
        scale = np.exp(a["scale"]) if self.arm == "scale_free" else self.norm
        return np.maximum(scale * (particle + background + resamp * resolution), 1e-30)

    def initial(self, data, start):
        rng = np.random.default_rng(719 + start)
        values = dict(zip(self.names, (self.lo + self.hi) / 2))
        for j, typ in enumerate(self.types):
            seed = next(r for r in data["seeds"] if r["components"][0]["type_id"] == typ)
            p = seed["components"][0]["params"]
            r = p["R"] if start == 0 else (1.15, 2, 4, 12, 45)[(start - 1) % 5] * (1 + 0.35 * j)
            if j and start == 0:
                r *= 1 + 0.4 * j
            if len(self.types) > 1 and start == 1:
                r = p["R"] if j == 0 else min(95, 35 * j)
            values[f"r{j}"] = np.log(r)
            values[f"sr{j}"] = p["sigma_R"] if start == 0 else rng.uniform(0.05, 0.6)
            lower = max(0.5 if self.broad else 3, 2 * r * 1.001)
            upper = 1000 if self.broad else 500
            d = (
                max(lower * 1.0001, p["D"])
                if start == 0
                else min(upper * 0.99, lower * rng.uniform(1.03, 3))
            )
            values[f"eta{j}"] = np.log(d / lower) / np.log(upper / lower)
            values[f"sd{j}"] = p["sigma_D"] if start == 0 else rng.uniform(0.06, 0.4)
            values[f"h{j}"] = (
                np.log(p.get("h", 10)) if start == 0 else rng.uniform(np.log(3), np.log(200))
            )
            values[f"sh{j}"] = p.get("sigma_h", 0.2) if start == 0 else rng.uniform(0.05, 0.6)
        values["rs"] = np.log(0.014 if self.arm in ("resolution_wide", "particle_wide") else 0.01)
        values["nu"] = 2.5 if self.arm in ("resolution_wide", "particle_wide") else 6
        values["bg"], values["rc"] = np.log(0.001), np.log(100)
        if self.arm == "scale_free":
            values["scale"] = np.log(500)
        z = np.clip([values[n] for n in self.names], self.lo + 1e-8, self.hi - 1e-8)
        if self.free:
            forms, res, _, _ = self.basis(z)
            basis = np.column_stack([forms, np.ones_like(res), res])
            weights = np.maximum(self.y, 1)
            design = basis / weights[:, None]
            sc = np.maximum(np.linalg.norm(design, axis=0), 1e-30)
            co = nnls(design / sc, self.y / weights)[0] / sc
            for name, value in zip([f"a{j}" for j in range(len(self.types))] + ["bg", "rc"], co):
                z[self.names.index(name)] = np.log(max(value, self.norm * 1e-12))
        return np.clip(z, self.lo + 1e-8, self.hi - 1e-8)


def solve(task):
    data = json.loads((OUT / "inputs.json").read_text())[task["side"]]
    m = Model(
        data,
        task["arm"],
        task["types"],
        task["gates"],
        task.get("resolution", True),
        task.get("quadrature", "legacy"),
    )
    z = np.array(task["z"]) if "z" in task else m.initial(data, task["start"])
    select = np.ones(len(m.y), bool)
    if task.get("split") is not None:
        select = np.arange(len(m.y)) % 2 == task["split"]
    if task.get("cold_start"):
        # Shape initialization uses declared constants, never held-out observations.
        constants = dict(
            r0=np.log(1.5),
            sr0=0.2,
            eta0=np.log(4.5 / 3.003) / np.log(500 / 3.003),
            sd0=0.12,
            h0=np.log(10),
            sh0=0.2,
            rs=np.log(0.02),
            nu=2.5,
        )
        for key, value in constants.items():
            if key in m.names:
                z[m.names.index(key)] = value
        forms, shape, _, _ = m.basis(z)
        basis = np.column_stack([forms, np.ones_like(shape), shape])
        design = basis[select] / np.maximum(m.y[select, None], 1)
        sc = np.maximum(np.linalg.norm(design, axis=0), 1e-30)
        co = nnls(design / sc, m.y[select] / np.maximum(m.y[select], 1))[0] / sc
        for key, value in zip(["a0", "bg", "rc"], co):
            z[m.names.index(key)] = np.log(max(value, m.norm * 1e-12))
    objective = task.get("objective", "log")

    def residual(z):
        pred = m.predict(z)
        if objective == "poisson":
            observed = m.y * m.count
            mean = pred * m.count
            dev = np.maximum(2 * (mean - observed + xlogy(observed, observed / mean)), 0)
            out = np.sign(mean - observed) * np.sqrt(dev)
        else:
            out = np.log(pred / np.maximum(m.y, 1e-30))
        return out[select]

    tick = time.perf_counter()
    opt = least_squares(
        residual,
        np.clip(z, m.lo + 1e-9, m.hi - 1e-9),
        bounds=(m.lo, m.hi),
        max_nfev=task.get("budget", 450),
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-8,
        x_scale="jac",
    )
    pred = m.predict(opt.x)
    r = np.log(pred / m.y)
    a, parts = m.decode(opt.x)
    result = dict(
        task=task,
        z=opt.x.tolist(),
        names=m.names,
        parameters=parts,
        globals={
            k: float(v)
            for k, v in a.items()
            if not any(
                k == f"{name}{j}"
                for j in range(len(m.types))
                for name in ("r", "sr", "eta", "sd", "h", "sh")
            )
        },
        sigma_res=float(np.exp(a["rs"])),
        nu_res=float(a["nu"]),
        logrmse=float(np.sqrt(np.mean(r * r))),
        train_logrmse=float(np.sqrt(np.mean(r[select] ** 2))),
        heldout_logrmse=float(np.sqrt(np.mean(r[~select] ** 2))) if not select.all() else None,
        nfev=int(opt.nfev),
        converged=bool(opt.success),
        seconds=time.perf_counter() - tick,
        at_bounds=[
            n
            for i, n in enumerate(m.names)
            if min(opt.x[i] - m.lo[i], m.hi[i] - opt.x[i]) < 1e-4 * (m.hi[i] - m.lo[i])
        ],
    )
    return result


def run(tasks, name, workers):
    path = OUT / f"{name}.json"
    results = json.loads(path.read_text()) if path.exists() else []
    seen = {json.dumps(r["task"], sort_keys=True) for r in results}
    tasks = [t for t in tasks if json.dumps(t, sort_keys=True) not in seen]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(solve, t): t for t in tasks}
        for fut in as_completed(pending):
            result = fut.result()
            results.append(result)
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(results, indent=2), encoding="utf-8")
            temp.replace(path)
            print(
                json.dumps(
                    dict(
                        done=len(results),
                        task=result["task"],
                        logrmse=result["logrmse"],
                        seconds=round(result["seconds"], 2),
                    )
                ),
                flush=True,
            )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["range", "mixture", "resoff", "validate", "targeted", "gauss", "gauss48"],
        default="range",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not (OUT / "inputs.json").exists():
        prepare()
    if args.phase == "range":
        tasks = [
            dict(side=side, arm=arm, types=[typ], gates=[gate], start=start)
            for side in ("positive", "negative")
            for arm in ARMS
            for typ in (1, 2, 3)
            for gate in (True, False)
            for start in range(6)
        ]
    elif args.phase == "mixture":
        combos = list(itertools.combinations_with_replacement((1, 2, 3), 2)) + [
            (1, 2, 3),
            (2, 2, 2),
            (1, 1, 2, 3),
            (1, 3, 3, 3),
            (2, 2, 2, 2),
        ]
        tasks = [
            dict(
                side=side,
                arm=arm,
                types=list(types),
                gates=[True] * len(types),
                start=start,
                budget=600,
            )
            for side in ("positive", "negative")
            for arm in ("train_exact", "resolution_wide")
            for types in combos
            for start in range(4)
        ]
    elif args.phase == "resoff":
        tasks = [
            dict(
                side=side,
                arm="train_exact",
                types=list(types),
                gates=[gate] * len(types),
                resolution=False,
                start=start,
                budget=600,
            )
            for side in ("positive", "negative")
            for types in [(1,), (2,), (3,), (1, 3, 3, 3), (1, 1, 2, 3)]
            for gate in (True, False)
            for start in range(3)
        ]
    elif args.phase == "targeted":
        tasks = []
        for side in ("positive", "negative"):
            for types in [(2, 3), (1, 3, 3, 3)]:
                for start in (0, 1):
                    for arm in ("train_exact", "resolution_wide"):
                        tasks.append(
                            dict(
                                side=side,
                                arm=arm,
                                types=list(types),
                                gates=[True] * len(types),
                                start=start,
                                budget=350,
                            )
                        )
            for start in (0, 1):
                tasks.append(
                    dict(
                        side=side,
                        arm="train_exact",
                        types=[1, 3, 3, 3],
                        gates=[True] * 4,
                        resolution=False,
                        start=start,
                        budget=350,
                    )
                )
    elif args.phase in ("gauss", "gauss48"):
        bank = json.loads((OUT / "range.json").read_text())
        tasks = []
        for side in ("positive", "negative"):
            for arm in ("amplitude_free", "resolution_wide"):
                best = min(
                    (r for r in bank if r["task"]["side"] == side and r["task"]["arm"] == arm),
                    key=lambda r: r["logrmse"],
                )
                if args.phase == "gauss48":
                    p = OUT / "gauss.json"
                    if p.exists():
                        candidates = [
                            r
                            for r in json.loads(p.read_text())
                            if r["task"]["side"] == side
                            and r["task"]["arm"] == arm
                            and "split" not in r["task"]
                        ]
                        if candidates:
                            best = min(candidates, key=lambda r: r["logrmse"])
                base = {
                    k: v for k, v in best["task"].items() if k not in ("z", "quadrature", "budget")
                }
                tasks.append(dict(**base, z=best["z"], quadrature=args.phase, budget=600))
            for split in (0, 1):
                tasks.append(
                    dict(
                        side=side,
                        arm="resolution_wide",
                        types=[2],
                        gates=[True],
                        start=2,
                        quadrature=args.phase,
                        budget=600,
                        split=split,
                        cold_start=True,
                    )
                )
    else:
        records = []
        for phase in ("range", "mixture", "resoff", "targeted"):
            p = OUT / f"{phase}.json"
            if p.exists():
                records += json.loads(p.read_text())
        tasks = []
        for side in ("positive", "negative"):
            for arm in ARMS:
                bank = [r for r in records if r["task"]["side"] == side and r["task"]["arm"] == arm]
                best = min(bank, key=lambda r: r["logrmse"])
                for split in (0, 1):
                    tasks.append(
                        dict(
                            **{k: v for k, v in best["task"].items() if k != "budget"},
                            z=best["z"],
                            split=split,
                            budget=1000,
                        )
                    )
                if arm in ("resolution_wide", "particle_wide"):
                    tasks.append(
                        dict(
                            **{k: v for k, v in best["task"].items() if k != "budget"},
                            z=best["z"],
                            objective="poisson",
                            budget=1000,
                        )
                    )
    run(tasks, args.phase, args.workers)


if __name__ == "__main__":
    main()
