"""The selection policy of a series, and the Predict decisions exported for each frame."""

from __future__ import annotations

import json

from .series_chain import CHAIN, CHI2_JUMP, CHI2_JUMP_LEAST, PREDICT_FACTOR
from .series_selection import CLOSE_ENOUGH, COMPOSITION_PENALTY, DEFAULT_SCALE, composition_penalty


def series_policy(settings) -> dict:
    chained = settings.start == CHAIN and settings.method != "predict"
    breaks = ({"failed": True, "not_converged": True, "new_stage": True,
               "chi2_ratio_above": CHI2_JUMP, "chi2_reduced_above": CHI2_JUMP_LEAST} if chained else None)
    if not settings.selection_recorded and (chained or settings.method == "predict"):
        selection = {"rule": "unrecorded"}
    elif settings.method == "predict" and settings.smooth:
        selection = {
            "rule": "HMM", "scale": settings.smooth_scale, "composition_penalty": composition_penalty(settings.smooth_scale),
            "reference_scale": DEFAULT_SCALE, "composition_penalty_at_reference": COMPOSITION_PENALTY,
            "path": "Viterbi", "marginals": "forward-backward", "reset_at_stage": settings.start == "stages",
        }
    elif chained or (settings.method == "predict" and settings.follow and settings.start != "same"):
        selection = {"rule": "closest", "min_relative_probability": 1 / PREDICT_FACTOR if chained else CLOSE_ENOUGH,
                     "without_reference": "most probable"}
    else:
        selection = {"rule": "most probable" if settings.method == "predict" else "unused"}
    if selection["rule"] in ("closest", "HMM"):
        selection["distance"] = "Euclidean"
        selection["coordinates"] = ["ln R", "ln D", "ln(sigma_R/R)"]
        selection["population_matching"] = "ascending R within family and paracrystal status"
    return {"chain_breaks": breaks, "selection": selection}


def frame_decisions(frame) -> dict:
    decisions = {}
    if frame.selection is not None:
        decisions["selected_solution"] = frame.selection.to_dict()
    if frame.predict_attempts:
        decisions["predict_attempts"] = [attempt.to_dict() for attempt in frame.predict_attempts]
    return decisions


PREDICT_COLUMNS = ("predict_rank", "predict_probability", "predict_max_probability", "predict_rule",
                   "predict_reference_frame", "predict_smoothed_probability", "predict_fit_kept", "predict_attempts")


def decision_cells(frame) -> list:
    """An unused restart is labelled as such. Both passes remain available in the attempts column."""
    choice = frame.selection or next((a.choice for a in reversed(frame.predict_attempts) if a.choice), None)
    attempts = [attempt.to_dict() for attempt in frame.predict_attempts]
    cells = [""] * (len(PREDICT_COLUMNS) - 1)
    if choice is not None:
        cells = [choice.rank, choice.probability, choice.top_probability, choice.rule,
                 choice.reference_frame or "", "" if choice.smoothed_probability is None else choice.smoothed_probability,
                 frame.selection is not None]
    return cells + [json.dumps(attempts, ensure_ascii=False, allow_nan=False) if attempts else ""]
