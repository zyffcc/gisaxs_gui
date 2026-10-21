"""What Batch Export says about the fits of its frames and about its end, in the interface language.

A fit takes one curve of the signed GISAXS cut — the mean of its halves, or one half (``BatchChoices.fit_halves``).
Here are the sentences of the dialog for each choice, the reason a frame was not fitted (the fit's own English
goes into the table and the record of the batch; the page shows it translated), what a trial fit was made of,
and how a batch ended: a batch whose fits failed, whose frames failed or that was stopped does not end as a
success — the status line, the notice and the panel say which, how far it got, and why each frame failed
(short: the folder by its last two names, a frame's reason without the file's path).
"""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path

from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.shared.signed_cut import MISSING_HALF

from ..application import NO_INTENSITY
from .texts import message_text

HALVES_TEXTS = {"mean": "Mean of both halves", "positive": "q > 0 half", "negative": "q < 0 half"}
"""The three choices as the dialog's combo names them (``views/batch_export_view.HALVES``)."""
HALVES_FITTED = {
    "mean": "A fit takes the mean of both halves of each cut.",
    "positive": "A fit takes the q > 0 half of each cut.",
    "negative": "A fit takes the q < 0 half of each cut.",
}
"""Under the combo of a batch that fits a model and writes no curve for Fitting: which halves the fit takes."""
HALVES_KEPT = ("The file written for Fitting keeps both halves, with signed qy, so the choice can still be changed "
               "in Fitting.")
HALVES_NOTES = {halves: f"{fitted} {HALVES_KEPT}" for halves, fitted in HALVES_FITTED.items()}
"""Under the combo when the curve for Fitting is written: also that its file holds both halves whatever is chosen
(one entry of the Chinese table each, as a whole)."""
HALVES_UNUSED = "Used when the curve for Fitting is saved (Data, above) or a particle model is fitted (below)."
"""Under the combo of a batch that neither writes fit inputs nor fits a model."""
MODEL_LINES = {
    "mean": "Each frame: the model is fitted to the mean of both halves of the horizontal cut (Halves of the cut, "
            "above); up to half a minute per frame.",
    "positive": "Each frame: the model is fitted to the q > 0 half of the horizontal cut (Halves of the cut, "
                "above); up to half a minute per frame.",
    "negative": "Each frame: the model is fitted to the q < 0 half of the horizontal cut (Halves of the cut, "
                "above); up to half a minute per frame.",
}
"""What the particle-model fit of the batch is fitted to."""
SENT_TO_FITTING = ("Each frame: the curve for Fitting (q in Å⁻¹, I, σ, pixels) → {file}. The folder then opens in "
                   "Fitting ▸ In-situ series.")
"""Send Series to Fitting: what is written, and what happens to it."""
SENT_GIWAXS = ("Fitting ▸ In-situ series fits particle form factors, made for the horizontal cut of a GISAXS frame: they "
               "do not describe diffraction peaks. To follow the peaks of this series use Batch Export ▸ Peaks of the "
               "ring regions.")
"""Send Series to Fitting of a GIWAXS series: what Fitting will fit its I(q) with, and where its peaks are fitted."""

QUICK_FIT_REASONS = (
    "Only {points} points on the q > 0 half: this fit needs at least {minimum}",
    "Only {points} points on the q < 0 half: this fit needs at least {minimum}",
    "Only {points} points in the mean of both halves: this fit needs at least {minimum}",
    "Only {points} points in the curve: this fit needs at least {minimum}",
)
"""The reasons of the quick physical fit that carry numbers (Fitting's ``TOO_FEW_FOR_QUICK_FIT``, written out
here: a feature does not import another; a test keeps the two alike). Its reasons without a number — a half
without points, a σ that cannot weight its point — are entries of the table as they are."""
REASONS_SHOWN = 3
"""At most this many different reasons are said at the end of a batch (the table has each frame's)."""
TRY_OTHER_HALVES = "{reason} in this frame: choose the mean of both halves or the other half (Halves of the cut, above)."
"""A trial on the frame on screen whose chosen half has no points, or no counts: what to do about it."""
ONE_HALF_REASONS = frozenset(MISSING_HALF.values()) | {NO_INTENSITY["positive"], NO_INTENSITY["negative"]}
"""The reasons that are about the chosen half alone: the other half, or the mean, may still be fitted."""


def _pattern(template: str) -> re.Pattern:
    parts = re.split(r"\{(\w+)\}", template)
    return re.compile("".join(re.escape(part) if index % 2 == 0 else f"(?P<{part}>.+?)"
                              for index, part in enumerate(parts)), re.DOTALL)


_PATTERNS = tuple((template, _pattern(template)) for template in QUICK_FIT_REASONS)


def halves_text(halves: str) -> str:
    """``mean`` / ``positive`` / ``negative`` as the combo says it (anything else: as it came)."""
    return tr(HALVES_TEXTS[halves]) if halves in HALVES_TEXTS else str(halves)


def fit_reason_text(reason) -> str:
    """Why a frame was not fitted (the fit's English), in the interface language; other text as it came."""
    reason = str(reason)
    for template, pattern in _PATTERNS:
        found = pattern.fullmatch(reason)
        if found is not None:
            return trf(template, **found.groupdict())
    return message_text(reason)


def try_failed_text(reason) -> str:
    """Why the trial fit of the frame on screen gave nothing; for a half without points or without counts also
    the way out."""
    if str(reason) in ONE_HALF_REASONS:
        return trf(TRY_OTHER_HALVES, reason=fit_reason_text(reason))
    return fit_reason_text(reason)


def made_of_text(row: dict) -> str:
    """What a solution of the model fit was fitted on (``preparation`` of the quick fit's rows): “Fitted:
    Mean of both halves, 635 points (2 neighbouring points merged into one)”; ``""`` when the fit does not say."""
    made_of = row.get("preparation") or {}
    if not made_of.get("points"):
        return ""
    values = dict(halves=halves_text(made_of.get("halves") or ""), points=int(made_of["points"]))
    merged = int(made_of.get("merged") or 1)
    if merged > 1:
        return trf("Fitted: {halves}, {points} points ({merged} neighbouring points merged into one)", merged=merged, **values)
    return trf("Fitted: {halves}, {points} points", **values) if values["halves"] else trf("Fitted: {points} points", **values)


def _fit_words(text: str) -> str:
    """The latest fit in a few words, in the interface language: a reason as a whole; of the peaks of a frame
    (“Ring A: q 0.9973 Å⁻¹, FWHM 0.0110; Ring B: no significant peak …”) each peak's own words."""
    whole = fit_reason_text(text)
    if whole != text:
        return whole
    parts = []
    for part in str(text).split("; "):
        name, colon, words = part.partition(": ")
        parts.append(f"{name}{colon}{fit_reason_text(words)}" if colon else part)
    return "; ".join(parts)


def latest_fit_text(run) -> str:
    """The panel's line about the latest fit of the batch — the frame folded last and its fit in a few words, a
    reason in the interface language; ``""`` before the first fit."""
    if not run.last_fit:
        return ""
    return trf("Latest fit ({frame}): {fit}", frame=run.frames[-1] if run.frames else "", fit=_fit_words(run.last_fit))


def fits_failed(run) -> int:
    """How many frames of the batch were to be fitted with a model and were not."""
    return sum(1 for frame in getattr(run, "fit_frames", ()) if frame.get("state") != "ok")


def fit_end_text(run) -> str:
    """The model fits of a batch in one sentence — “model fit: 2 of 3 frames fitted — No points on the q > 0
    half” — or ``""`` for a batch without a model fit. Several reasons: each with its number of frames. Under
    the mean, the frames whose cut had one half only (fitted on that half: their row of the table names it)
    are counted too, so that an odd value in the trend is no surprise."""
    frames = list(getattr(run, "fit_frames", ()) or ())
    if not frames:
        return ""
    reasons = Counter(str(frame.get("reason") or "") for frame in frames if frame.get("state") != "ok")
    values = dict(fitted=len(frames) - sum(reasons.values()), frames=len(frames))
    said = []
    if len(reasons) == 1:
        said.append(fit_reason_text(next(iter(reasons))))
    elif reasons:
        said.append(_joined(trf("{n} frame(s): {reason}", n=count, reason=fit_reason_text(reason))
                            for reason, count in reasons.most_common(REASONS_SHOWN))
                    + (" …" if len(reasons) > REASONS_SHOWN else ""))
    chosen = getattr(getattr(run, "choices", None), "fit_halves", "mean")
    alone = sum(1 for frame in frames
                if chosen == "mean" and frame.get("state") == "ok" and frame.get("halves") in ("positive", "negative"))
    if alone:
        said.append(trf("{n} frame(s) on one half alone", n=alone))
    if not said:
        return trf("model fit: {fitted} of {frames} frames fitted", **values)
    return trf("model fit: {fitted} of {frames} frames fitted — {why}", why=_joined(said), **values)


def _joined(parts) -> str:
    """Clauses one after the other, with the semicolon of the interface language between them."""
    parts = [part for part in parts if part]
    text = parts[0] if parts else ""
    for part in parts[1:]:
        text = trf("{first}; {second}", first=text, second=part)
    return text


def with_fits(text: str, run) -> str:
    """``text`` with the batch's model fits said after it (``fit_end_text``), when it had any."""
    return _joined([text, fit_end_text(run)])


def fit_toast_text(run, *, stopped: bool = False) -> str:
    """The notice of a batch that ended with frames not fitted: short (the status line has the folder), and it
    begins like the panel's title — a batch the person stopped is not “done”."""
    return _joined([tr("Batch Export stopped") if stopped else tr("Batch Export done"), fit_end_text(run)])


FAILED_SHOWN = 6
"""The panel lists at most this many failed frames with their reasons (README.txt has every one)."""


def short_folder(path) -> str:
    """A folder by its last two names (``…/run_07/gimap_analysis``): what tells one target from another on a
    line. Where it is shown, the whole path is the tooltip."""
    parts = Path(path).parts
    return "…/" + "/".join(parts[-2:]) if len(parts) > 3 else str(path)


def failure_reason(request, error, folder=None) -> tuple[str, str]:
    """``(why, file)`` of a frame that failed, for ``BatchRun.fail``: the reason without the folder of the data
    in it (the reader's own words end in the whole path) and without the file's name in front, which is said
    beside it. ``folder``: the batch's folder — a file of it that could not be written is named from there."""
    text = str(error)
    if folder is not None:
        text = text.replace(str(folder) + os.sep, "").replace(str(folder), Path(folder).name)
    if request is None:
        return text, ""
    source = Path(request.path)
    for path in dict.fromkeys([source, *(Path(item[0]) for item in request.summed_frames)]):
        text = text.replace(str(path), path.name)
    text = text.replace(str(source.parent) + os.sep, "")
    return (text[len(source.name) + 1:] if text.startswith(source.name + " ") else text), source.name


def failed_line(item) -> str:
    """One frame of a batch that failed (an entry of ``BatchRun.failed``) with its reason, in the interface
    language: “frame: reason”. The sentences about a file that cannot be read are known with the file's name in
    them; where the language puts the name inside the sentence and the frame is that file, the sentence alone
    says both."""
    frame, why, file = str(item.get("frame", "")), str(item.get("why", "")), str(item.get("file", ""))
    reason = fit_reason_text(why)
    if file:
        whole = f"{file} {why}"
        said = message_text(whole)
        if said != whole and said.startswith(file):
            reason = said[len(file):].lstrip()
        elif said != whole:
            if frame == file:
                return said
            reason = said
    return trf("{frame}: {reason}", frame=frame, reason=reason)


def failed_lines(run) -> str:
    """The frames of a batch that failed, one line each with the reason (``""``: none failed)."""
    failed = list(getattr(run, "failed", ()) or ())
    lines = [failed_line(item) for item in failed[:FAILED_SHOWN]]
    if len(failed) > FAILED_SHOWN:
        lines.append(trf("… and {n} more: README.txt in the folder has every one", n=len(failed) - FAILED_SHOWN))
    return "\n".join(lines)


def written_text(run) -> str:
    """What a batch that has ended wrote, in one line: how many files, and in which folders."""
    files, destination = list(getattr(run, "files", ()) or ()), Path(run.destination)
    if not files:
        return ""
    folders: Counter = Counter()
    for path in files:
        try:
            parts = Path(path).relative_to(destination).parts
        except ValueError:
            continue
        folders[parts[0] + "/" if len(parts) > 1 else ""] += 1
    where = [f"{folder} {count}" for folder, count in sorted(folders.items()) if folder]
    if folders.get(""):
        where.append(trf("tables and records {n}", n=folders[""]))
    return trf("{n} files written: {where}", n=len(files), where=", ".join(where))


def _frames_asked(run) -> int:
    return int(getattr(run, "requested", 0) or len(run.frames) + len(getattr(run, "failed", ()) or ()))


def end_title(run, *, stopped: bool = False) -> str:
    """The panel's title when a batch has ended: “done” only for a batch that did what was asked. Short: the
    title shares its line with the time and two buttons."""
    failed = len(getattr(run, "failed", ()) or ())
    if getattr(run, "write_error", ""):
        return tr("Batch Export stopped: the folder takes no more files")
    if stopped:
        return tr("Batch Export stopped")
    if failed:
        return trf("{n} of {total} frames failed", n=failed, total=_frames_asked(run))
    return tr("Batch Export done")


def end_toast_text(run, *, stopped: bool = False, tables_error: str = "") -> str:
    """The notice of a batch that did not end as a success: short (the status line has the folder), it says
    what happened and how far the batch got, then its model fits."""
    done, total, failed = len(run.frames), _frames_asked(run), len(getattr(run, "failed", ()) or ())
    if getattr(run, "write_error", ""):
        head = trf("Batch Export stopped: the folder takes no more files ({error})", error=run.write_error)
    elif tables_error:
        head = trf("Batch Export: the tables of all frames could not be written ({error})", error=tables_error)
    elif getattr(run, "stopped", False):
        head = trf("Batch Export stopped: {done} of {total} frames exported", done=done, total=total)
    elif failed:
        head = trf("Batch Export ended: {n} of {total} frames failed", n=failed, total=total)
    else:
        head = tr("Batch Export stopped") if stopped else tr("Batch Export done")
    return _joined([head, fit_end_text(run)])


def batch_end_text(run, *, seconds: float, tables_error: str = "") -> tuple[str, str]:
    """``(text, level)`` of the status line when a batch has ended: the frames exported of those asked for, a
    stop, how many failed (the panel lists them with their reasons), and its model fits. ``ok`` only when
    every frame asked for was exported and every fit asked for gave a solution."""
    failed = len(getattr(run, "failed", ()) or ())
    values = dict(done=len(run.frames), total=_frames_asked(run), folder=short_folder(run.destination),
                  seconds=f"{seconds:.0f}")
    level = "warning" if failed or fits_failed(run) or getattr(run, "stopped", False) else "ok"
    if getattr(run, "write_error", ""):
        text, level = trf("Stopped: {folder} takes no more files ({error}); {done} of {total} frames were exported",
                          error=run.write_error, **values), "error"
    elif tables_error:
        text, level = trf("Exported {done}/{total} frames to {folder}, but the tables of all frames could not be "
                          "written: {error}", error=tables_error, **values), "error"
    elif getattr(run, "stopped", False):
        text = trf("Stopped: {done} of {total} frames exported to {folder} in {seconds} s ({left} not started)",
                   left=len(run.not_started), **values)
    else:
        text = trf("Exported {done}/{total} frames to {folder} in {seconds} s", **values)
    if failed:
        text = _joined([text, tr("{n} failed").format(n=failed)])
    return with_fits(text, run), level


__all__ = [
    "HALVES_FITTED", "HALVES_KEPT", "HALVES_NOTES", "HALVES_TEXTS", "HALVES_UNUSED", "MODEL_LINES", "ONE_HALF_REASONS",
    "FAILED_SHOWN", "QUICK_FIT_REASONS", "SENT_GIWAXS", "SENT_TO_FITTING", "TRY_OTHER_HALVES", "batch_end_text", "end_title",
    "end_toast_text", "failed_line", "failed_lines", "failure_reason", "fit_end_text", "fit_reason_text",
    "fit_toast_text", "fits_failed", "halves_text", "latest_fit_text", "made_of_text", "short_folder", "try_failed_text",
    "with_fits", "written_text",
]
