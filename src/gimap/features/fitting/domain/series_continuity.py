"""A continuity prior over the modes returned by Predict for neighbouring curves."""

from __future__ import annotations

import math

import numpy as np
from scipy.special import logsumexp

DEFAULT_SCALE = 0.1
COMPOSITION_PENALTY = 0.01
_COMPOSITION_COST = -2 * DEFAULT_SCALE**2 * math.log(COMPOSITION_PENALTY)


def composition_penalty(scale: float) -> float:
    """Relative weight for changing composition at equal matched coordinates."""
    return math.exp(-0.5 * (math.sqrt(_COMPOSITION_COST) / scale)**2)


def _populations(model):
    groups = {}
    for component in sorted(model.components, key=lambda c: c.value("R")):
        kind = component.family, component.structure
        groups.setdefault(kind, []).append(component)
    return groups


def _separation(first, second):
    """Squared Euclidean distance in matching positive log coordinates, and whether composition changed."""
    changed = first.keys() != second.keys()
    distance2 = 0.0
    for kind in sorted(first.keys() & second.keys()):
        here, there = first[kind], second[kind]
        changed |= len(here) != len(there)
        for a, b in zip(here, there):
            for key in ("R", "D", "sigma_R"):
                if key in a.keys() and key in b.keys():
                    v, w = a.value(key), b.value(key)
                    if v > 0 and w > 0:
                        distance2 += (math.log(v) - math.log(w))**2
    return distance2, changed


def solution_distance(model, previous) -> float:
    """The greedy rule compares matching compositions. Incompatible candidates tie behind these."""
    distance2, changed = _separation(_populations(model), _populations(previous))
    return math.inf if changed else math.sqrt(distance2)


def smooth_modes(frames, scale: float):
    """Return zero-based Viterbi indices and the normalized marginal weights for all modes.

    Each frame supplies its candidates in evidence rank order. Symmetric pair weights and per-frame
    emission weights define the probability of a whole path. Forward/backward messages are evaluated
    in log space, without normalizing transitions separately for each predecessor.
    """
    if not (scale > 0):
        raise ValueError("The continuity scale must be positive")
    if not frames:
        return [], []
    emissions = []
    populations = []
    for solutions in frames:
        probabilities = np.asarray([solution.probability for solution in solutions], dtype=float)
        if (not np.isfinite(probabilities).all() or (probabilities < 0).any() or not (probabilities > 0).any()):
            raise ValueError("Predict returned no valid set of mode probabilities for this frame")
        with np.errstate(divide="ignore"):
            emissions.append(np.log(probabilities))
        populations.append([_populations(solution.model) for solution in solutions])

    transitions, parents = [], []
    forward = [emissions[0]]
    scores = emissions[0]
    for t in range(1, len(frames)):
        transition = np.empty((len(frames[t - 1]), len(frames[t])))
        for i, before in enumerate(populations[t - 1]):
            for j, after in enumerate(populations[t]):
                distance2, changed = _separation(before, after)
                if changed:
                    distance2 += _COMPOSITION_COST
                transition[i, j] = -0.5 * (math.sqrt(distance2) / scale)**2
        transitions.append(transition)
        summed = emissions[t] + logsumexp(forward[-1][:, None] + transition, axis=0)
        forward.append(summed - summed.max())
        paths = scores[:, None] + transition
        parents.append(paths.argmax(axis=0))
        scores = emissions[t] + paths.max(axis=0)
        scores -= scores.max()

    path = [int(scores.argmax())]
    for parent in reversed(parents):
        path.append(int(parent[path[-1]]))
    path.reverse()
    backward = np.zeros_like(scores)
    marginals = []
    for t in reversed(range(len(frames))):
        weights = forward[t] + backward
        marginals.append(tuple(np.exp(weights - logsumexp(weights)).tolist()))
        if t:
            backward = logsumexp(transitions[t - 1] + emissions[t] + backward, axis=1)
            backward -= backward.max()
    return path, list(reversed(marginals))
