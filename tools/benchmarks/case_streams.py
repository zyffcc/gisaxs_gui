"""Random streams for independent recovery sets and defaults for replaying archived draws."""
from __future__ import annotations

import hashlib

import numpy as np


# (number of cuts, root seed). The archived third set used the first generator.
SETS = {
    "fit1d": (200, 20261116), "extended": (600, 20261201), "fresh": (300, 20261116),
    "legacy-first": (200, 20261116), "legacy-extended": (600, 20261201),
    "legacy-third": (300, 20261122),
}


def case_streams(set_name: str, seed0: int | None, index: int):
    """Return the parameter RNG, Poisson seed and stream metadata for one random-access case."""
    if set_name not in SETS:
        raise ValueError(f"Unknown recovery set {set_name!r}")
    if seed0 is None:
        seed0 = SETS[set_name][1]
    if index < 0 or seed0 < 0:
        raise ValueError("Case index and root seed must be nonnegative")
    if set_name.startswith("legacy-"):
        seed = seed0 + index
        return np.random.default_rng(seed), seed, {}

    # Address the named set and its i-th child directly, as if .spawn() had
    # been called in order. This remains stable across workers and restarts.
    digest = hashlib.sha256(set_name.encode("utf-8")).digest()
    set_key = tuple(int.from_bytes(digest[i:i + 4], "little") for i in range(0, len(digest), 4))
    case = np.random.SeedSequence(seed0, spawn_key=(*set_key, index))
    parameters, noise = case.spawn(2)
    noise_seed = int.from_bytes(noise.generate_state(4).astype("<u4").tobytes(), "little")
    return np.random.default_rng(parameters), noise_seed, {"set_name": set_name, "root_seed": seed0}
