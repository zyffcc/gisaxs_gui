"""Simulated cuts for training the 1D proposal network: parameters from a prior, particle terms from cut_model.

Writes shards of noise-free unit terms; the scale, beam-tail amplitude, background, counting noise and gaps are drawn
anew whenever a curve is used (``augment``), so one shard serves many epochs. Everything is dimensionless (q̃ = q/q_max,
``posterior.encoding``).

usage: python tools/ml1d/simulate.py --prior tools/ml1d/prior_k1.json --count 200000 --seed 1 --out DIR
           [--shard 10000] [--workers 4]
  writes DIR/shard_NNNN.npz and DIR/manifest.json (the prior, seeds, commit and the SHA-256 of every shard)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gimap.features.fitting.domain.cut_model import model  # noqa: E402
from src.gimap.features.fitting.domain.cut_model.geometry import Geometry  # noqa: E402
from src.gimap.features.fitting.domain.posterior import encoding  # noqa: E402

FLOOR = 1e-30


def load_prior(path) -> dict:
    prior = json.loads(Path(path).read_text(encoding="utf-8"))
    families = prior["families"]
    unknown = set(families) - set(encoding.FAMILIES)
    if unknown:
        raise ValueError(f"Families the network does not know: {sorted(unknown)}")
    return prior


def _log_uniform(rng, low, high, size):
    return np.exp(rng.uniform(np.log(low), np.log(high), size))


def draw(prior: dict, count: int, rng: np.random.Generator) -> dict:
    """Parameters and conditions of ``count`` cuts; values uniform in the box of ``encoding.bounds``."""
    names = list(prior["families"])
    chances = np.array([prior["families"][n] for n in names], float)
    family = np.array(names)[rng.choice(len(names), count, p=chances / chances.sum())]
    ratio = _log_uniform(rng, *prior["ratio"], count)
    lo, hi = encoding.bounds(ratio)
    values = lo + rng.uniform(size=lo.shape) * (hi - lo)
    structure = rng.uniform(size=count) < prior["structure_probability"]
    born = (rng.uniform(size=count) < prior["born_probability"]) & (family == "sphere")
    qz = np.where(born, rng.uniform(*prior["born_qz"], count), 0.0)
    resolved = rng.uniform(size=count) < prior["resolution_probability"]
    q_resolution = np.where(resolved, _log_uniform(rng, *prior["resolution"], count), 0.0)
    return {"family": family, "ratio": ratio, "values": values, "structure": structure, "qz": qz,
            "q_resolution": q_resolution}


def terms(params: dict, law: str, coupling: str) -> tuple[np.ndarray, np.ndarray]:
    """(particle term T, beam-tail shape g) of every row on its grid (B, POINTS), from cut_model."""
    rows = params["ratio"].size
    term = np.empty((rows, encoding.POINTS))
    tail = np.empty_like(term)
    for i in range(rows):
        q = encoding.grid(params["ratio"][i])
        v = params["values"][i]
        radius = float(np.exp(v[0]))
        distance = 2.0 * radius * float(np.exp(v[2])) if params["structure"][i] else 0.0
        population = model.Population(params["family"][i], radius, float(v[1]), law=law, distance=distance,
                                      distance_spread=float(v[3]) if distance else 0.0, coupling=coupling)
        settings = model.Settings(q_resolution=float(params["q_resolution"][i]),
                                  geometry=Geometry.born_at(float(params["qz"][i])))
        term[i] = model.unit_terms([population], settings, q)[0]
        tail[i] = model.tail(q, float(np.exp(v[4])), float(v[5]))
    return np.maximum(term, FLOOR), tail


def shard(prior: dict, count: int, seed) -> dict:
    rng = np.random.default_rng(seed)
    params = draw(prior, count, rng)
    term, tail = terms(params, prior["law"], prior["coupling"])
    family = np.array([encoding.FAMILIES.index(f) for f in params["family"]], np.int8)
    return {"values": params["values"].astype(np.float32), "z": encoding.to_z(params["values"], params["ratio"]),
            "family": family, "structure": params["structure"], "ratio": params["ratio"].astype(np.float32),
            "qz": params["qz"].astype(np.float32), "q_resolution": params["q_resolution"].astype(np.float32),
            "term": term.astype(np.float32), "tail": tail.astype(np.float32)}


def augment(data: dict, prior: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(counts, σ, mask) of curves made from a batch of shard rows (``term``, ``tail``): a beam tail of
    ``tail_relative`` × T at q̃_min, a background of ``background_relative`` × T at q̃ = 1, an optional smooth
    multiplicative model error, scaled to ``peak_counts`` at the highest point, Poisson counts, gaps."""
    term = np.asarray(data["term"], float)
    tail = np.asarray(data["tail"], float)
    rows, points = term.shape
    amplitude = _log_uniform(rng, *prior["tail_relative"], rows) * term[:, 0] / np.maximum(tail[:, 0], FLOOR)
    background = _log_uniform(rng, *prior["background_relative"], rows) * term[:, -1]
    clean = term + amplitude[:, None] * tail + background[:, None]
    error = float(prior.get("model_error", 0.0))
    if error > 0:
        clean *= np.exp(error * smooth_noise(rng, rows, points))
    peak = _log_uniform(rng, *prior["peak_counts"], rows)
    expected = clean * (peak / clean.max(axis=1))[:, None]
    counts = rng.poisson(expected).astype(float)
    return counts, np.sqrt(np.maximum(counts, 1.0)), gaps(rng, rows, points, prior)


def smooth_noise(rng, rows: int, points: int, modes: int = 4) -> np.ndarray:
    """Smooth random curves of unit RMS along the grid: a few cosines of random phase, amplitudes falling as 1/m."""
    t = np.linspace(0.0, np.pi, points)
    m = np.arange(1, modes + 1)
    coeff = rng.normal(size=(rows, modes)) / m
    phase = rng.uniform(0, 2 * np.pi, (rows, modes))
    curves = np.einsum("rm,rmp->rp", coeff, np.cos(m[None, :, None] * t[None, None, :] + phase[:, :, None]))
    return curves / np.sqrt(np.mean(curves**2, axis=1, keepdims=True) + 1e-12)


def gaps(rng, rows: int, points: int, prior: dict) -> np.ndarray:
    """Masks of measured points: with ``gap_probability`` one or two gaps of together at most ``gap_fraction``,
    with ``edge_probability`` the lowest points lost behind a beam stop (at most ``edge_fraction``)."""
    mask = np.ones((rows, points), bool)
    for r in np.flatnonzero(rng.uniform(size=rows) < prior["gap_probability"]):
        total = int(rng.uniform(*prior["gap_fraction"]) * points)
        pieces = rng.integers(1, 3)
        for length in np.diff(np.sort(np.r_[0, rng.integers(0, total + 1, pieces - 1), total])):
            if length:
                start = rng.integers(0, points - length)
                mask[r, start:start + length] = False
    for r in np.flatnonzero(rng.uniform(size=rows) < prior["edge_probability"]):
        mask[r, :int(rng.uniform(0, prior["edge_fraction"]) * points)] = False
    return mask


def inputs(data: dict, rows, prior: dict, rng: np.random.Generator):
    """(features, conditions, z, active) of shard rows as the network sees them, with fresh noise; inactive
    coordinates of z (D and σnn/D without a structure factor) are standard normal noise."""
    part = {key: value[rows] for key, value in data.items()}
    counts, sigma, mask = augment(part, prior, rng)
    family = np.array(encoding.FAMILIES)[part["family"]]
    conditions = encoding.conditions(family, part["structure"], part["ratio"], part["qz"], part["q_resolution"])
    active = encoding.active(part["structure"])
    z = np.where(active, part["z"], rng.standard_normal(part["z"].shape))
    return encoding.features(counts, sigma, mask), conditions, z.astype(np.float32), active


def load(folder) -> tuple[dict, dict]:
    """(rows of every shard of ``folder`` joined, manifest); refuses a shard whose SHA-256 is not the manifest's."""
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    parts = []
    for name, sha in sorted(manifest["shards"].items()):
        if _sha256(folder / name) != sha:
            raise ValueError(f"{name} differs from the manifest")
        with np.load(folder / name) as data:
            parts.append({key: data[key] for key in data.files})
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}, manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write(args) -> tuple[str, str, float]:
    prior, count, seed, path = args
    start = time.perf_counter()
    np.savez(path, **shard(prior, count, seed))
    return path.name, _sha256(path), time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prior", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    prior = load_prior(args.prior)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sizes = [min(args.shard, args.count - start) for start in range(0, args.count, args.shard)]
    seeds = np.random.SeedSequence(args.seed).spawn(len(sizes))
    jobs = [(prior, n, s, out / f"shard_{i:04d}.npz") for i, (n, s) in enumerate(zip(sizes, seeds))]
    shards = {}
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            for name, sha, seconds in pool.map(_write, jobs):
                shards[name] = sha
                print(f"{name}: {seconds:.0f} s", flush=True)
    else:
        for job in jobs:
            name, sha, seconds = _write(job)
            shards[name] = sha
            print(f"{name}: {seconds:.0f} s", flush=True)
    manifest = {"prior": prior, "prior_file": Path(args.prior).name, "count": args.count, "seed": args.seed,
                "shard_size": args.shard, "commit": _commit(), "encoding_names": list(encoding.NAMES),
                "points": encoding.POINTS, "shards": shards}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
