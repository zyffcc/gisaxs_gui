"""Simulated cuts for training the 1D proposal network: compositions and parameters from a prior, every population's
term from cut_model.

Writes shards of noise-free unit terms (each population's T on its cut's grid, with the cut's resolution, and the beam
tail's shape); the shares of the populations, the tail amplitude, the background, the model error, the measured
points, the counting noise and the gaps are drawn anew whenever a curve is used (``augment``), so one shard serves
many epochs. Everything is dimensionless (q̃ = q/q_max, ``posterior.encoding``).

usage: python tools/ml1d/simulate.py --prior tools/ml1d/prior_v1.json --count 200000 --seed 1 --out DIR
           [--shard 10000] [--workers 4] [--first-shard 0] [--shards N]
  writes DIR/shard_NNNN.npz and DIR/manifest_NNNN.json (the prior, seed, commit and SHA-256 of the shard); with
  --first-shard/--shards only those shards (one array job per range); ``manifest`` joins them
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

from src.gimap.features.fitting.domain.cut_model import model, resolution  # noqa: E402
from src.gimap.features.fitting.domain.cut_model.geometry import Geometry  # noqa: E402
from src.gimap.features.fitting.domain.posterior import encoding as E  # noqa: E402

FLOOR = 1e-30
SUB = 8
"""Points per bin of the network's grid at which the terms are evaluated and then averaged (as a measured bin is)."""
_INDEX = {slots: i for i, slots in enumerate(E.COMPOSITIONS)}


def load_prior(path) -> dict:
    prior = json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = set(prior["family_weights"]) - set(E.FAMILIES)
    if unknown:
        raise ValueError(f"Families the network does not know: {sorted(unknown)}")
    if len(prior["k_weights"]) != E.SLOTS:
        raise ValueError(f"k_weights needs {E.SLOTS} entries")
    return prior


def _log_uniform(rng, low, high, size):
    return np.exp(rng.uniform(np.log(low), np.log(high), size))


def draw_compositions(prior: dict, count: int, rng) -> np.ndarray:
    k = rng.choice(np.arange(1, E.SLOTS + 1), count, p=np.asarray(prior["k_weights"]) / np.sum(prior["k_weights"]))
    names = list(prior["family_weights"])
    chances = np.array([prior["family_weights"][n] for n in names], float)
    family = rng.choice(len(names), (count, E.SLOTS), p=chances / chances.sum())
    structure = rng.uniform(size=(count, E.SLOTS)) < prior["structure_probability"]
    out = np.empty(count, int)
    for i in range(count):
        slots = tuple(E.TYPES.index((names[family[i, s]], bool(structure[i, s]))) for s in range(k[i]))
        out[i] = _INDEX[slots]
    return out


def draw(prior: dict, count: int, rng: np.random.Generator) -> dict:
    """Compositions, conditions and values (uniform in the unit box of ``encoding``) of ``count`` cuts."""
    composition = draw_compositions(prior, count, rng)
    ratio = _log_uniform(rng, *prior["ratio"], count)
    values = E.from_unit(rng.uniform(size=(count, E.DIMS)), ratio, composition)
    born = rng.uniform(size=count) < prior["born_probability"]
    qz = np.where(born, rng.uniform(*prior["born_qz"], count), 0.0)
    return {"composition": composition, "ratio": ratio, "qz": qz, "values": values}


def populations(values_row, composition: int, law: str, coupling: str) -> list:
    """cut_model populations of one row (dimensionless lengths)."""
    out = []
    for slot, (family, structure) in enumerate(E.composition_types(composition)):
        v = values_row[slot * E.PER_SLOT:(slot + 1) * E.PER_SLOT]
        radius = float(v[0])
        cylinder = family == "cylinder"
        out.append(model.Population(
            family, radius, float(v[1]), law=law, height=2.0 * radius * float(v[2]) if cylinder else 10.0,
            height_spread=float(v[3]) if cylinder else 0.0,
            distance=2.0 * radius * float(v[4]) if structure else 0.0,
            distance_spread=float(v[5]) if structure else 0.0, coupling=coupling))
    return out


def fine_grid(ratio: float) -> np.ndarray:
    """``SUB`` points per bin of the network's grid, evenly in ln q̃ across each bin (POINTS · SUB,)."""
    centres = E.grid(ratio)
    step = np.log(centres[1] / centres[0])
    return (centres[:, None] * np.exp(((np.arange(SUB) + 0.5) / SUB - 0.5) * step)[None, :]).ravel()


def evaluation_grid(fine: np.ndarray, sigma: float) -> np.ndarray:
    """The fine grid, extended by ``resolution.SPAN`` σ on both sides (at σ/4, down to q̃ → 0 for the fold)."""
    if sigma <= 0:
        return fine
    reach = resolution.SPAN * sigma
    low = max(fine[0] - reach, 0.125 * sigma)
    lower = np.linspace(low, fine[0], int(min(256, np.ceil((fine[0] - low) / (0.25 * sigma)))) + 2)[:-1]
    upper = np.linspace(fine[-1], fine[-1] + reach, int(min(64, np.ceil(reach / (0.25 * sigma)))) + 2)[1:]
    return np.concatenate([lower, fine, upper])


def terms(params: dict, law: str, coupling: str) -> tuple[np.ndarray, np.ndarray]:
    """(T of every slot (B, SLOTS, POINTS), zero where unused; beam-tail shape (B, POINTS)) from cut_model: each
    population on ``fine_grid`` (extended for the resolution), convolved with the cut's resolution
    (``resolution.tabulated``), averaged over each bin."""
    rows = params["ratio"].size
    term = np.zeros((rows, E.SLOTS, E.POINTS))
    tail = np.empty((rows, E.POINTS))
    g = E.GLOBAL.start
    for i in range(rows):
        fine = fine_grid(params["ratio"][i])
        v = params["values"][i]
        sigma = float(v[g + 2])
        grid = evaluation_grid(fine, sigma)
        settings = model.Settings(geometry=Geometry.born_at(float(params["qz"][i])))
        pops = populations(v, int(params["composition"][i]), law, coupling)
        smeared = resolution.tabulated(fine, grid, model.unit_terms(pops, settings, grid), sigma)
        for slot, t in enumerate(smeared):
            term[i, slot] = np.maximum(t.reshape(E.POINTS, SUB).mean(axis=1), FLOOR)
        tail[i] = model.tail(fine, float(v[g]), float(v[g + 1])).reshape(E.POINTS, SUB).mean(axis=1)
    return term, tail


def shard(prior: dict, count: int, seed) -> dict:
    rng = np.random.default_rng(seed)
    params = draw(prior, count, rng)
    term, tail = terms(params, prior["law"], prior["coupling"])
    return {"values": params["values"].astype(np.float32),
            "z": E.to_z(params["values"], params["ratio"], params["composition"]).astype(np.float32),
            "composition": params["composition"].astype(np.int16), "ratio": params["ratio"].astype(np.float32),
            "qz": params["qz"].astype(np.float32), "term": term.astype(np.float32), "tail": tail.astype(np.float32)}


def _layout(rng, ratio, prior, rows):
    """Measured points per bin (B, POINTS): N points, linearly or logarithmically spaced over the window."""
    count = np.round(_log_uniform(rng, *prior["points"], rows)).astype(int)
    log_spaced = rng.uniform(size=rows) < prior["log_spacing_probability"]
    centres = -np.log(ratio)[:, None] * (1.0 - np.linspace(0.0, 1.0, E.POINTS))[None, :]  # ln of E.grid, every row
    edges = np.concatenate([centres[:, :1] - 0.5 * (centres[:, 1:2] - centres[:, :1]),
                            0.5 * (centres[:, 1:] + centres[:, :-1]),
                            centres[:, -1:] + 0.5 * (centres[:, -1:] - centres[:, -2:-1])], axis=1)
    start = np.where(log_spaced, centres[:, 0], np.exp(centres[:, 0]))
    stop = np.where(log_spaced, centres[:, -1], np.exp(centres[:, -1]))
    step = (stop - start) / (count - 1)
    axis = np.where(log_spaced[:, None], edges, np.exp(edges))
    below = np.clip(np.ceil((axis - start[:, None]) / step[:, None] - 1e-9), 0, count[:, None])
    return np.diff(below, axis=1)


def augment(data: dict, prior: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(I, σ, mask) on the grid of a batch of shard rows: populations at peak shares 10^U[−decades, 0], the model
    error, a beam tail of ``tail_relative`` × the particles at q̃_min, a background of ``background_relative`` × their
    peak, measured at ``points`` points (each bin the mean of its points), Poisson counts with ``peak_counts`` at
    the highest point, gaps and a lost low-q edge."""
    term = np.asarray(data["term"], float)
    tail = np.asarray(data["tail"], float)
    values = np.asarray(data["values"], float)
    rows = term.shape[0]
    used = np.any(term > 0, axis=2)
    shares = np.where(used, 10.0 ** rng.uniform(-prior["share_decades"], 0.0, (rows, E.SLOTS)), 0.0)
    particles = np.einsum("bs,bsp->bp", shares / np.maximum(term.max(axis=2), FLOOR), term)
    amplitude = _log_uniform(rng, *prior["tail_relative"], rows) * particles[:, 0] / np.maximum(tail[:, 0], FLOOR)
    background = _log_uniform(rng, *prior["background_relative"], rows) * particles.max(axis=1)
    clean = particles + amplitude[:, None] * tail + background[:, None]
    tau = values[:, E.GLOBAL.start + 3]
    basis = E.error_basis(np.linspace(0.0, 1.0, E.POINTS))
    clean *= np.exp((rng.normal(size=(rows, E.ERROR_BASIS)) * tau[:, None]) @ basis)
    per_bin = _layout(rng, np.asarray(data["ratio"], float), prior, rows)
    expected = clean * (_log_uniform(rng, *prior["peak_counts"], rows) / clean.max(axis=1))[:, None]
    counts = rng.poisson(expected * per_bin).astype(float)
    n = np.maximum(per_bin, 1.0)
    mask = (per_bin > 0) & gaps(rng, rows, E.POINTS, prior)
    return counts / n, np.sqrt(np.maximum(counts, 1.0)) / n, mask


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
    """(features, conditions, composition, z, active) of shard rows as the network sees them, with fresh noise."""
    part = {key: value[rows] for key, value in data.items()}
    intensity, sigma, mask = augment(part, prior, rng)
    composition = part["composition"].astype(int)
    return (E.features(intensity, sigma, mask), E.conditions(part["ratio"], part["qz"]), composition,
            part["z"].astype(np.float32), E.active(composition))


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


def join(folder) -> dict:
    """manifest.json of every shard's manifest_NNNN.json in ``folder`` (one prior and seed)."""
    folder = Path(folder)
    parts = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("manifest_*.json"))]
    if not parts or any(p["prior"] != parts[0]["prior"] or p["seed"] != parts[0]["seed"] for p in parts):
        raise ValueError("The shard manifests are missing or of different priors or seeds")
    joined = {k: parts[0][k] for k in ("prior", "prior_file", "seed", "shard_size", "commit", "names", "points")}
    joined["shards"] = {k: v for p in parts for k, v in p["shards"].items()}
    joined["count"] = sum(p["count"] for p in parts)
    (folder / "manifest.json").write_text(json.dumps(joined, indent=1), encoding="utf-8")
    return joined


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prior")
    parser.add_argument("--count", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--first-shard", type=int, default=0)
    parser.add_argument("--shards", type=int)
    parser.add_argument("--join", action="store_true", help="only write manifest.json from the shard manifests")
    args = parser.parse_args()
    out = Path(args.out)
    if args.join:
        print(json.dumps({k: v for k, v in join(out).items() if k != "shards"}))
        return 0
    prior = load_prior(args.prior)
    out.mkdir(parents=True, exist_ok=True)
    sizes = [min(args.shard, args.count - start) for start in range(0, args.count, args.shard)]
    seeds = np.random.SeedSequence(args.seed).spawn(len(sizes))
    last = len(sizes) if args.shards is None else min(len(sizes), args.first_shard + args.shards)
    jobs = [(prior, sizes[i], seeds[i], out / f"shard_{i:04d}.npz") for i in range(args.first_shard, last)]
    record = {"prior": prior, "prior_file": Path(args.prior).name, "seed": args.seed, "shard_size": args.shard,
              "commit": _commit(), "names": list(E.NAMES), "points": E.POINTS, "shards": {}, "count": 0}
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            done = list(pool.map(_write, jobs))
    else:
        done = [_write(job) for job in jobs]
    for (name, sha, seconds), job in zip(done, jobs):
        record["shards"][name] = sha
        record["count"] += job[1]
        print(f"{name}: {seconds:.0f} s", flush=True)
    (out / f"manifest_{args.first_shard:04d}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    if args.shards is None:
        join(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
