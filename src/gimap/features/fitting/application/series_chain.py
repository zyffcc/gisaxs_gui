"""The chained fit of an In-situ series: every frame starts from the result of the frame before it (as SasView's
chained batch fit does), Predict parameters is asked only where that chain breaks, and an optional backward pass
starts every frame from the result of the frame after it and keeps whichever of the two fits is better.

* **Where the chain breaks** (``chain_break``): the frame's fit failed or did not converge, a new stage of the series
  begins, a value of the fit is flagged (at an edge of the beam tail's range, not determined by the curve), the fit is
  poor (χ²ᵣ above ``POOR_CHI2``), or χ²ᵣ jumped: more than ``CHI2_JUMP`` times the frame before's and above
  ``CHI2_JUMP_LEAST``. There the frame is fitted again by Predict parameters (its method unchanged: the page's
  ``predictor``, with the particles of the start model), and the better of the two fits is kept (``better``).
* **Better** of two fits of one frame: one fitted to the end and converged before one that is not, then the lower
  χ²ᵣ; on equal terms the first.
* **Flags** of a frame (``frame_flags``): what a reader of the table must know before trusting its values — failed,
  not converged, stopped, a poor fit, values at a bound of their range, the beam tail at an edge, a value the curve
  does not determine, where the chain broke, the pass that gave the result kept.

Fits of a series as NXcanSAS (``series_nxcansas``): for every fitted frame the points fitted and the model at their q
(``fit_nxcansas.fit_curves``), in one file, each entry with the frame's record.
"""

from __future__ import annotations

import math
import re
from dataclasses import replace
from datetime import datetime
from typing import Callable, Optional, Sequence

from ..domain import beam_tail
from ..domain.fit_links import parameter_name
from ..domain.fit_limits import manual_chain_start
from ..domain.fit_model import FitModel
from .series_fit import POOR_CHI2, FrameFit, SeriesSettings, fit_frame, flagged_fit, poor_fit, prepare_frame

CHAIN = "chain"
"""The start choice of a chained fit (``SeriesSettings.start``)."""
CHI2_JUMP = 3.0
CHI2_JUMP_LEAST = 2.0
STARTS_FROM = ("previous", "single", "next", "predict")
"""Where a frame's kept fit started: the result of the frame before, the model of Single analysis, the result of the
frame after (the backward pass), Predict parameters (the chain broke)."""
FORWARD, BACKWARD = "forward", "backward"
FAILED_BREAK = "the fit failed ({why})"
NOT_CONVERGED_BREAK = "the fit did not converge"
STAGE_BREAK = "a new stage begins"
FLAGGED_BREAK = "flagged: {values}"
POOR_BREAK = "a poor fit (χ²ᵣ {chi2:.3g} above {limit:g})"
JUMP_BREAK = "χ²ᵣ jumped from {before:.3g} to {after:.3g}"
BREAKS = (FAILED_BREAK, NOT_CONVERGED_BREAK, STAGE_BREAK, FLAGGED_BREAK, POOR_BREAK, JUMP_BREAK)
"""The reasons a chain breaks (English templates: the record keeps them, the page translates them)."""
FLAG_KINDS = ("failed", "not converged", "stopped", "poor fit", "at a bound", "beam tail", "undetermined",
              "chain broke", "backward")


def _converged(frame: Optional[FrameFit]) -> bool:
    return frame is not None and frame.fitted and frame.result.converged


def chain_break(frame: FrameFit, previous: Optional[FrameFit], *, new_stage: bool = False) -> str:
    """Why the chain breaks at ``frame`` (module docstring; English), or ``""`` when it holds. A frame stopped by the
    person is no break: nothing is fitted again."""
    if not frame.ok:
        return FAILED_BREAK.format(why=frame.error or "no fit")
    if not frame.fitted:
        return ""
    if not frame.result.converged:
        return NOT_CONVERGED_BREAK
    if new_stage:
        return STAGE_BREAK
    if flagged_fit(frame):
        from .series_fit import flag_names

        return FLAGGED_BREAK.format(values=", ".join(flag_names(frame)))
    if poor_fit(frame):
        return POOR_BREAK.format(chi2=frame.result.chi2_reduced, limit=POOR_CHI2)
    if _converged(previous):
        before, after = previous.result.chi2_reduced, frame.result.chi2_reduced
        if math.isfinite(before) and math.isfinite(after) and after > CHI2_JUMP * before and after > CHI2_JUMP_LEAST:
            return JUMP_BREAK.format(before=before, after=after)
    return ""


def better(first: FrameFit, second: Optional[FrameFit]) -> FrameFit:
    """The better of two fits of one frame (module docstring)."""
    if second is None:
        return first

    def rank(frame: FrameFit) -> tuple:
        chi2 = frame.result.chi2_reduced if frame.ok else math.inf
        return (0 if _converged(frame) else 1 if frame.fitted else 2, chi2 if math.isfinite(chi2) else math.inf)

    return second if rank(second) < rank(first) else first


def chained_fit(index: int, path: str, curve, start: FitModel, single: FitModel, settings: SeriesSettings,
                previous: Optional[FrameFit], *, new_stage: bool = False, stop: Optional[Callable[[], bool]] = None,
                predict: Optional[Callable] = None, fit_pass: str = FORWARD) -> FrameFit:
    """One frame of a chained series: refined from ``start`` (the result of ``previous``, or ``single``); where the
    chain breaks (``chain_break``) and ``predict`` is given, fitted again by Predict parameters from the particles of
    ``single``, and the better of the two kept. The frame says where its kept fit started, why the chain broke and in
    which pass it was fitted (``FrameFit.start_from``, ``chain``, ``fit_pass``)."""
    origin = ("next" if fit_pass == BACKWARD else "previous") if (
        previous is not None and previous.ok and start is previous.result.model) else "single"
    frame = replace(fit_frame(index, path, curve, start, settings, stop), start_from=origin, fit_pass=fit_pass)
    why = chain_break(frame, previous, new_stage=new_stage)
    if not why or predict is None or (stop is not None and stop()) or not frame.points:
        return replace(frame, chain=why)
    again = fit_frame(index, path, curve, single, replace(settings, method="predict"), stop, predict)
    if stop and stop():
        return replace(frame, chain=why)
    if again.fitted:
        proposal = again.result.model
        if (proposal.weighting != single.weighting or proposal.resolution_mode != single.resolution_mode
                or [c.law for c in proposal.components] != [c.law for c in single.components]):
            # Compare χ² on the same points, with the chosen weights and smearing. Predict remains a separate fit.
            again = fit_frame(index, path, curve, manual_chain_start(proposal, single),
                              replace(settings, method="local"), stop)
    kept = better(frame, replace(again, start_from="predict", fit_pass=fit_pass))
    return replace(kept, chain=why)


def frame_flags(frame: FrameFit) -> list[tuple[str, str]]:
    """``(kind, detail)`` of everything to know before trusting a frame's values (module docstring; ``FLAG_KINDS``)."""
    flags: list[tuple[str, str]] = []
    if not frame.ok:
        return [("failed", frame.error)]
    result = frame.result
    if not frame.fitted:
        flags.append(("stopped", ""))
    elif not result.converged:
        flags.append(("not converged", str(result.message or "")))
    if poor_fit(frame):
        flags.append(("poor fit", f"chi2 {result.chi2_reduced:.3g}"))
    tail = beam_tail.tail_flags(result)
    tail_paths = {path for path, _why in tail}
    bounds = [parameter_name(path) for path in result.at_bounds if path not in tail_paths]
    if bounds:
        flags.append(("at a bound", ", ".join(bounds)))
    if tail:
        flags.append(("beam tail", ", ".join(f"{parameter_name(path)} ({why})" for path, why in tail)))
    loose = [parameter_name(path) for path in beam_tail.undetermined(result, finite_only=True) if path not in tail_paths]
    if loose:
        flags.append(("undetermined", ", ".join(loose)))
    if getattr(frame, "chain", ""):
        flags.append(("chain broke", frame.chain + (" -> Predict parameters" if frame.start_from == "predict" else "")))
    if getattr(frame, "fit_pass", "") == BACKWARD:
        flags.append(("backward", "kept from the backward pass"))
    return flags


def flags_text(frame: FrameFit) -> str:
    """The flags of a frame for a table cell (English: ``kind: detail; …``)."""
    return "; ".join(f"{kind}: {detail}" if detail else kind for kind, detail in frame_flags(frame))


_STAMP = re.compile(r"(?P<moment>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?)(?:[.,](?P<fraction>\d+))?\s*"
                    r"(?P<zone>Z|[+-]\d{2}(?::?\d{2})?)?")


def observation_time(observation) -> Optional[float]:
    """When a fit input's frame was measured, in seconds (``frame_time_s``, else its ``timestamp`` as POSIX seconds;
    detectors write “+0200”, “Z”, a space for the “T” and fractions of any length); ``None`` when it does not say."""
    if not isinstance(observation, dict):
        return None
    try:
        value = float(observation.get("frame_time_s"))
        if math.isfinite(value):
            return value
    except (TypeError, ValueError):
        pass
    found = _STAMP.fullmatch(str(observation.get("timestamp") or "").strip())
    if found is None:
        return None
    zone = found["zone"] or ""
    zone = "+00:00" if zone == "Z" else (zone if len(zone) != 5 else f"{zone[:3]}:{zone[3:]}")
    zone = f"{zone}:00" if len(zone) == 3 else zone
    fraction = f".{found['fraction'][:6].ljust(6, '0')}" if found["fraction"] else ""
    try:
        return datetime.fromisoformat(found["moment"].replace(" ", "T") + fraction + zone).timestamp()
    except ValueError:
        return None


def backward_order(indices: Sequence[int]) -> list[int]:
    """The frames of the backward pass: from the last but one to the first (the last starts it as it is)."""
    ordered = sorted(indices)
    return ordered[-2::-1] if len(ordered) > 1 else []


def series_nxcansas(frames: Sequence[FrameFit], curves: dict, settings: SeriesSettings, name_of: Callable[[FrameFit], str],
                    record_of: Callable[[FrameFit], dict]) -> list:
    """NXcanSAS curves of a series' fits: for each fitted frame with its curve in ``curves`` (index → ``Curve``), the
    points fitted and the model (``fit_nxcansas.fit_curves``), named by ``name_of`` and with the record ``record_of``."""
    from .fit_nxcansas import fit_curves

    out = []
    for frame in sorted(frames, key=lambda item: item.index):
        curve = curves.get(frame.index)
        if not frame.fitted or curve is None:
            continue
        try:
            data = prepare_frame(curve, settings, frame.result.model).data
        except ValueError:
            continue
        out.extend(fit_curves(frame.result.model, data, record_of(frame), name_of(frame)))
    return out


__all__ = ["BACKWARD", "BREAKS", "CHAIN", "CHI2_JUMP", "CHI2_JUMP_LEAST", "FLAG_KINDS", "FORWARD", "STARTS_FROM",
           "backward_order", "better", "chain_break", "chained_fit", "flags_text", "frame_flags", "observation_time",
           "series_nxcansas"]
