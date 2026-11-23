"""Score Predict's probability (or BIC weight) of the radius-ordered composition, including structure."""
from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import NormalDist

METHOD = "retained_composition_weights_v1"
BRANCHES = ("evidence", "bic")
EDGES = (0.0, 0.5, 0.8, 0.99, 1.0)
LIST_THRESHOLD = 0.05  # the historical benchmark's list of substantial modes


def archive_weights(solutions, candidates):
    """Read Predict's own pre-display totals. Candidates without a positive-weight mode retain zero."""
    branch = solutions[0].extras["compared_by"]
    if branch not in BRANCHES:
        raise ValueError(f"Unknown comparison branch {branch!r}")
    field = "composition_bic_weight" if branch == "bic" else "composition_probability"
    weights = {str(index): 0.0 for index in candidates}
    mode_totals = {}
    modes = []
    for solution in solutions:  # every positive-weight mode, in descending order of individual weight
        diagnostics = solution.extras["diagnostics"]
        if diagnostics["compared_by"] != branch:
            raise ValueError("Inconsistent comparison branches on the same cut")
        index = int(solution.composition)
        weights[str(index)] = float(diagnostics[field])
        mode_totals[index] = mode_totals.get(index, 0.0) + float(solution.probability)
        modes.append(dict(composition_index=index, probability=float(solution.probability)))
    if (not math.isclose(math.fsum(weights.values()), 1.0, abs_tol=1e-12, rel_tol=0)
            or any(not math.isclose(total, weights[str(i)], rel_tol=1e-12) for i, total in mode_totals.items())):
        raise ValueError("Incomplete retained modes for Predict's composition weights")
    return dict(compared_by=branch, composition_weights=weights, retained_modes=modes)


@dataclass(frozen=True)
class Score:
    winner: int
    weight: float
    branch: str
    correct: bool
    listed: bool


def composition_score(row: dict, stage: str) -> Score | None:
    """Unscorable runs include older archives with only the leading and display-filtered solutions."""
    result = row.get(stage, {})
    if not result.get("ok") or "composition_weights" not in result:
        return None
    weights = {int(key): value for key, value in result["composition_weights"].items()}
    if (result["compared_by"] not in BRANCHES or any(not 0 <= p <= 1 + 1e-14 for p in weights.values())
            or not math.isclose(math.fsum(weights.values()), 1.0, abs_tol=1e-12, rel_tol=0)):
        raise ValueError("Invalid archived composition weights or comparison branch")
    # For exactly equal totals use the smaller encoding index, independent of serialization order.
    winner = max(weights, key=lambda i: (weights[i], -i))
    truth = row["truth_composition_index"]
    listed = any(mode["composition_index"] == truth and mode["probability"] >= LIST_THRESHOLD
                 for mode in result["retained_modes"])
    return Score(winner, weights[winner], result["compared_by"], winner == truth, listed)


def _reliability(scores):
    """Calibration of the winning weight. Every successful comparison contributes one cut."""
    n = len(scores)
    n_correct = sum(score.correct for score in scores)
    rate = n_correct / n if n else None
    bins = []
    z = NormalDist().inv_cdf(.84)
    for lo, hi in zip(EDGES, EDGES[1:]):
        selected = [s for s in scores if s.weight >= lo and (s.weight < hi or hi == 1.0)]
        count = len(selected)
        right = sum(s.correct for s in selected)
        interval = None
        if count:
            p = right / count
            centre = p + z * z / (2 * count)
            delta = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count**2))
            interval = [(centre - delta) / (1 + z*z/count), (centre + delta) / (1 + z*z/count)]
        bins.append(dict(low=lo, high=hi, n=count, n_correct=right, correct=right / count if count else None,
                         mean_stated=sum(s.weight for s in selected) / count if count else None, wilson68=interval))
    return dict(n=n, n_correct=n_correct, bins=bins,
                brier=sum((s.weight - s.correct)**2 for s in scores) / n if n else None,
                constant_brier=rate * (1.0 - rate) if n else None)


def summarize_compositions(rows, stage):
    """Preserve denominators when a stage fails or a historical record lacks pre-filter weights."""
    attempted = [row for row in rows if stage in row]
    ok = [row for row in attempted if row[stage]["ok"]]
    scores = [s for row in ok if (s := composition_score(row, stage)) is not None]
    candidates = [row["truth_composition_index"] in row[stage]["candidates"] for row in attempted
                  if "candidates" in row[stage] and "truth_composition_index" in row]
    n = len(scores)
    right = sum(s.correct for s in scores)
    return dict(n=len(attempted), n_failed=len(attempted) - len(ok), n_missing_weights=len(ok) - n,
                n_scored=n, n_correct=right, exact=right / n if n else None,
                true_in_equivalent_set=sum(s.listed for s in scores) / n if n else None,
                n_candidates_checked=len(candidates), n_truth_in_candidates=sum(candidates),
                true_in_candidates=sum(candidates) / len(candidates) if candidates else None,
                reliability_by_branch={b: _reliability([s for s in scores if s.branch == b]) for b in BRANCHES})
