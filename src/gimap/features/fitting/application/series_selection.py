"""Selection of a Predict mode and the evidence retained with each series frame."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Optional

from ..domain.fit_model import FitModel
from ..domain.series_continuity import COMPOSITION_PENALTY, DEFAULT_SCALE, composition_penalty, smooth_modes, solution_distance

CLOSE_ENOUGH = 0.2


@dataclass(frozen=True)
class SolutionChoice:
    rank: int
    probability: float
    top_probability: float
    rule: str
    reference_frame: Optional[int] = None
    smoothed_probability: Optional[float] = None
    probabilities: tuple = ()
    smoothed_probabilities: tuple = ()
    intervals: dict = field(default_factory=dict)
    ranked_modes: tuple = ()
    """Mode labels and population coordinates in the same rank order as the probability vectors."""

    def interval(self, path):
        """Predict's (least squares, median, lower/upper 68%, lower/upper 95%, error), if available."""
        owner, key = path
        if owner == "globals":
            name = {"res_width": "tail_width", "res_exponent": "tail_exponent"}.get(key, key)
        else:
            name = f"{key}_{owner + 1}"
        return self.intervals.get(name, ())

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**{**data, "probabilities": tuple(data.get("probabilities") or ()),
                      "smoothed_probabilities": tuple(data.get("smoothed_probabilities") or ()),
                      "ranked_modes": tuple(data.get("ranked_modes") or ()),
                      "intervals": {k: tuple(v) for k, v in (data.get("intervals") or {}).items()}})


@dataclass(frozen=True)
class PredictAttempt:
    fit_pass: str
    chain_break: str
    choice: Optional[SolutionChoice]
    kept: bool
    error: str = ""

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        choice = data.get("choice")
        return cls(**{**data, "choice": SolutionChoice.from_dict(choice) if choice else None})


def ranked(solutions):
    if not solutions:
        raise RuntimeError("Predict parameters found no solution for this frame")
    ordered = tuple(sorted(solutions, key=lambda solution: float(solution.probability), reverse=True))
    if ordered[0].probability <= 0 or any(not math.isfinite(s.probability) or s.probability < 0 for s in ordered):
        raise ValueError("Predict returned no valid set of mode probabilities for this frame")
    return ordered


def _mode_coordinates(solution):
    populations = []
    for c in solution.model.components:
        population = {"family": c.family, "structure": c.structure, "R_nm": c.value("R"),
                      "sigma_R_over_R": c.value("sigma_R")}
        if "D" in c.keys():
            population["D_nm"] = c.value("D")
        populations.append(population)
    return {"name": str(getattr(solution, "name", "")), "populations": populations}


def mode_choice(ordered, index, rule, *, reference_frame=None, marginals=()):
    solution = ordered[index]
    probabilities = tuple(float(item.probability) for item in ordered)
    return SolutionChoice(index + 1, probabilities[index], probabilities[0], rule, reference_frame,
                          marginals[index] if marginals else None, probabilities, tuple(marginals),
                          dict(getattr(solution, "intervals", {})), tuple(_mode_coordinates(item) for item in ordered))


def closest_index(ordered, previous, threshold):
    """Index in emission rank order. Ties in distance retain the most probable candidate."""
    if previous is None:
        return 0
    floor = threshold * ordered[0].probability
    return min((i for i, item in enumerate(ordered) if item.probability >= floor),
               key=lambda i: solution_distance(ordered[i].model, previous))


@dataclass(frozen=True)
class PredictedStart:
    model: FitModel
    index: int
    reproduction: dict


def predicted_start(solution):
    """The model to refine, with its original job index when its Predict run can be replayed."""
    recipe = getattr(solution, "reproduction", None)
    return PredictedStart(solution.model, solution.solution_index, recipe) if recipe else solution.model


class FramePredictor:
    """Adapts a job yielding modes to fit_frame's callback, keeping its selection and candidates."""

    def __init__(self, solutions, previous=None):
        self.solutions = solutions
        self.previous = previous
        self.ordered = ()
        self.choice = None

    def __call__(self, data, start):
        self.ordered = ranked(self.solutions(data, start))
        model = self.previous.result.model if self.previous is not None else None
        index = closest_index(self.ordered, model, CLOSE_ENOUGH)
        self.choice = mode_choice(self.ordered, index, "closest" if model is not None else "most probable",
                                  reference_frame=self.previous.index + 1 if model is not None else None)
        return predicted_start(self.ordered[index])


def smoothed_fits(frames, candidates, curves, settings, boundaries, stop):
    """Apply the HMM to completed runs of neighbours. Locally fit only newly selected modes.

    A row without a prediction or an explicit stage restart separates runs. The caller can discard
    this entire update on cancellation, keeping the previous complete set of results.
    """
    from .series_fit import fit_frame

    updates = {}
    segment = []

    def finish_segment():
        modes = [candidates[i] for i in segment]
        path, marginals = smooth_modes(modes, settings.smooth_scale)
        for i, ordered, index, weights in zip(segment, modes, path, marginals):
            if stop():
                return
            frame = frames[i]
            if frame.selection is None or frame.selection.rank != index + 1:
                selected = ordered[index]
                frame = fit_frame(i, frame.path, curves[i], selected.model, settings, stop,
                                  lambda _data, _start: predicted_start(selected))
            choice = mode_choice(ordered, index, "HMM", marginals=weights)
            updates[i] = replace(frame, start_from="predict", fit_pass="forward", selection=choice, predict_choice="HMM")
        segment.clear()

    for i in sorted(frames):
        if stop():
            return {}
        if not candidates.get(i) or i in boundaries:
            finish_segment()
        if candidates.get(i):
            segment.append(i)
    finish_segment()
    return {} if stop() else updates


__all__ = ["CLOSE_ENOUGH", "DEFAULT_SCALE", "COMPOSITION_PENALTY", "FramePredictor", "PredictAttempt",
           "SolutionChoice", "closest_index", "composition_penalty", "mode_choice", "ranked", "smoothed_fits"]
