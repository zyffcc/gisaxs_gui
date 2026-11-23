"""The extended recovery generator, including replay of the archived second set.

Same ranges, levels of physics, curve and counts as fit1d_recovery.draw/curve (the independent truth model
yoneda_truth.py), and in addition three populations (a third size scale 1.8–3 times from the second) and the decoupling
approximation (DA) in half of the cases. Compositions cycle through COMPOSITIONS; ``index`` and ``seed0`` fix a case.
The archived draws used seed0 20261201 and shared seeds with the first and third sets.
"""
from __future__ import annotations

import numpy as np

if __package__:
    from . import recovery_cases as fr
    from .case_streams import case_streams
else:
    import recovery_cases as fr
    from case_streams import case_streams

COMPOSITIONS = {"sphere": ("sphere",), "vc": ("vc",), "sphere,sphere": ("sphere", "sphere"),
                "sphere,vc": ("sphere", "vc"), "sphere,sphere,vc": ("sphere", "sphere", "vc"),
                "sphere,vc,vc": ("vc", "vc", "sphere")}
SEED0 = 20261201


def draw(index: int, seed0: int = SEED0, *, set_name="extended") -> dict:
    if set_name not in ("extended", "legacy-extended"):
        raise ValueError(f"Use recovery_cases.draw for {set_name!r}")
    rng, noise_seed, metadata = case_streams(set_name, seed0, index)
    names = list(COMPOSITIONS)
    comp = names[index % len(names)]
    level = "model" if (index // len(names)) % 2 == 0 else "measured"
    coupling = "da" if (index // (2 * len(names))) % 2 else "lma"
    pops = []
    for k, shape in enumerate(COMPOSITIONS[comp]):
        if k == 0:
            radius = float(np.exp(rng.uniform(np.log(2.0), np.log(8.0))))
        else:  # another size scale, 1.8–3 times above the largest so far (within 10 nm) or below the smallest
            ratio = float(np.exp(rng.uniform(np.log(1.8), np.log(3.0))))
            largest, smallest = max(p["R"] for p in pops), min(p["R"] for p in pops)
            radius = largest * ratio if largest * ratio <= 10.0 else smallest / ratio
            radius = float(max(radius, 1.2))
        pops.append(dict(shape=shape, R=radius, sR=float(rng.uniform(0.1, 0.35)),
                         D=float(2 * radius * rng.uniform(1.3, 2.5)), s_nn=float(rng.uniform(0.1, 0.3)),
                         share=1.0 if k == 0 else float(rng.uniform(0.3, 1.0))))
    peak = float(10 ** rng.uniform(3, 5))
    return dict(index=index, composition=comp, level=level, coupling=coupling, populations=pops, peak_counts=peak,
                background=float(10 ** rng.uniform(-3, -1.5)), seed=noise_seed, **metadata)


curve = fr.curve


__all__ = ["COMPOSITIONS", "SEED0", "curve", "draw"]
