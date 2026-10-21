"""What a batch says of itself: its state while it runs (``BatchRun``) and, when it has ended, its record
``<batch>_batch.json`` (``batch_record``) and the ``README.txt`` of its folder (``readme_text``).

Both describe the batch as it went, not as it was asked for: the files that were written (every one of them
in ``"files"``; a kind that was chosen and has no file is named as such), the frames that failed with their
reasons, a batch that was stopped and how far it got, where the cut of every frame was taken, sums of
different sizes, and what became of an earlier export in the same folder.

Keys of the record (``RECORD_FORMAT``): ``title``, ``format``, ``started``, ``finished``, ``seconds``,
``choices``, ``requested``, ``frames``, ``frame_numbers``, ``failed`` (``[{frame, why}]``), ``stopped``,
``not_started``, ``summary`` (``{requested, exported, failed, fits_failed, stopped, seconds, files}``), ``files``,
``cuts`` (one entry per frame), ``series_cut`` and ``series_cut_text`` (what was held for every frame),
``sums`` (frames summed into each frame, when any), ``sources``, ``earlier_export``,
``settings_of_first_frame`` — and, as before, ``model_fit`` / ``fit_failed``, ``series_correction``, ``pictures``.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Collection, Mapping, Optional, Sequence

from src.gimap.shared.export_text import export_text
from src.gimap.shared.signed_cut import fit_halves

from ..domain import GISAXS
from .batch_export import BATCH_TITLE, FOLDERS, BatchChoices, CurveTables, output_names
from .batch_fit import FIT_MODEL, FIT_NONE, HALVES_WORDS, FitStarts, FitTable, model_fit_record
from .batch_folder import RECORD_FORMAT

FAILED_LISTED = 30
"""The README lists at most this many failed frames with their reasons (the record has every one)."""


@dataclass
class BatchRun:
    """One batch: what to write, where, and what the frames done so far gave."""

    choices: BatchChoices
    destination: Path
    stem: str
    series: Any = None
    series_state: dict = field(default_factory=dict)
    tables: CurveTables = field(default_factory=CurveTables)
    fit_table: Optional[FitTable] = None
    starts: Optional[FitStarts] = None
    frames: list = field(default_factory=list)
    failures: list = field(default_factory=list)
    """``name (reason)`` of every frame that failed (``failed``: the two apart)."""
    first_metadata: Optional[dict] = None
    first_frame: str = ""
    curve_titles: dict = field(default_factory=dict)
    started: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M"))
    live_key: Optional[str] = None
    """The curve whose rows are shown live as the frames are done."""
    last_fit: str = ""
    """The latest fit in a few words (shown while the batch runs)."""
    fit_frames: list = field(default_factory=list)
    """The model fit of every frame done (``model_fit_frame``): fitted on which halves and points, or why not."""
    fit_previews: list = field(default_factory=list)
    """Frame metadata and best-fit previews for Series Results; temporary curve-cache keys are never exported."""
    preview_cache: Any = None
    """Bounded curve-array cache; neither its files nor its keys enter the exported batch record."""
    preview_lock: Any = field(default_factory=RLock, repr=False)
    """Coordinate a worker publishing a preview with Clear/dispose releasing the same batch."""
    display: Optional[dict] = None
    """How the pictures are drawn (``FrameJob.display``)."""
    requested: int = 0
    """The frames the batch was to export (every n-th of the listed ones)."""
    stopped: bool = False
    """The person stopped it before every frame was started."""
    not_started: list = field(default_factory=list)
    """The frames a stop left out."""
    failed: list = field(default_factory=list)
    """``{"frame", "why"[, "file"]}`` of every frame that failed (``why`` without the file's name in front)."""
    written: list = field(default_factory=list)
    """Every file written for the frames done so far."""
    files: list = field(default_factory=list)
    """Every file of the batch, once it has ended (the tables, ``README.txt`` and the record included)."""
    cuts: list = field(default_factory=list)
    """``{"frame", …}``: where the automatic cuts of every frame done were taken (``frame_cuts``)."""
    sums: list = field(default_factory=list)
    """How many frames were summed into each frame done."""
    numbers: list = field(default_factory=list)
    """The place of each frame done among the listed frames, from 1."""
    fit_keys: list = field(default_factory=list)
    """The curve keys of the peaks fitted (for the names of their files)."""
    series_cut: Optional[dict] = None
    """What was held the same for every frame (the view model's ``series_cut()``); ``None``: nothing."""
    series_cut_text: str = ""
    """The same in one English sentence (``series_cut_text()``)."""
    sources: list = field(default_factory=list)
    """``{"name", "summary", "files", "frames", "complete", "warnings"}`` of every listed source of several
    files (module files of a detector, a master with its data files)."""
    earlier: dict = field(default_factory=dict)
    """What became of an earlier export in the folder: ``{"action": replaced | kept, "when", "removed",
    "complete", "record"}``."""
    write_error: str = ""
    """Why the folder stopped taking files (the batch was ended for it)."""
    seconds: float = 0.0

    def fail(self, frame: str, why: str, file: str = "") -> None:
        """A frame that gave nothing: its name, the reason in English, and the file it was to be read from."""
        self.failed.append({"frame": str(frame), "why": str(why), **({"file": str(file)} if file else {})})
        self.failures.append(f"{frame} ({why})")


_KINDS = ("tables", "per_frame", "fit_input", "q_map", "cake", "detector_image", "q_map_image", "frames")
_MEANING = {
    "tables": "one table per curve: x in the first column, then one column per frame (mean intensity per pixel; "
              "blank = no data); its header says how the frames were put on one x",
    "per_frame": "one file per frame and curve: x, I, sigma (standard error) and the pixels in each point; "
                 "a _analysis.json per frame records the settings",
    "fit_input": "per frame: q (1/A), I, sigma, pixels — the curve Fitting reads (Fitting > In-situ series)",
    "q_map": "per frame: the intensity on a regular q_par-qz grid (qy-qz for GISAXS); first row = x axis, "
             "first column = qz",
    "cake": "per frame: the intensity unwrapped onto chi (rows) and q (columns)",
    "detector_image": "per frame: a picture of the detector image with a colour bar",
    "q_map_image": "per frame: a picture of the q map with a colour bar",
    "frames": "per frame: the detector data in another format, values as read (modules stitched, frames summed)",
    "fit": "the fit of every frame, one row per frame (values, errors, reduced chi2, and a note when a fit failed)",
    "fit_plot": "the fitted values against the frame number",
    "fit_curves": "per frame and fitted peak: q (1/A), the data and the fit",
    "model_curves": "per frame: |q| in 1/nm (q in 1/A = q in 1/nm / 10), the points fitted, their sigma and the model",
    "record": "the choices, every frame with its cuts, the frames that failed, every file written and the full "
              "settings of the first frame",
}
NAMES = ('Names: <frame> is the name of the frame\'s source (a set of module files: its scan, without "_mNN"). '
         '"_frameNNNN" is the index of the frame in a source that holds several, counted from 0; headers, tables '
         'and titles count from 1 ("frame 4 of 12" is _frame0003). "_sumK": K frames summed.')


def written_kinds(files: Sequence[Path], destination: Path, names: Mapping[str, str]) -> set[str]:
    """Which kinds of file (the keys of ``output_names``) are among ``files``."""
    by_folder = {"per_frame": FOLDERS["per_frame"], "fit_input": FOLDERS["fit_input"], "frames": FOLDERS["frames"],
                 "fit_curves": FOLDERS["fit_curves"]}
    kinds: set[str] = set()
    for path in files:
        try:
            parts = Path(path).relative_to(destination).parts
        except ValueError:
            continue
        name = parts[-1]
        if len(parts) == 1:
            if name in (names["fit"], names["fit_plot"], names["record"], names["readme"]):
                kinds.add(next(key for key in ("fit", "fit_plot", "record", "readme") if names[key] == name))
            elif "_frames." in name:
                kinds.add("tables")
            continue
        folder = parts[0]
        kinds.update(key for key, value in by_folder.items() if value == folder)
        if folder == FOLDERS["q_map"]:
            kinds.add("cake" if "_cake." in name else "q_map")
        elif folder == FOLDERS["detector_image"]:
            kinds.add("q_map_image" if "_qmap." in name else "detector_image")
    return kinds


def _span(pairs: Sequence[Sequence[float]], digits: Optional[int] = None) -> str:
    """``a-b`` when every frame has the same pair, else ``a-b to c-d`` (the lowest and the highest)."""
    def text(pair) -> str:
        return "-".join(f"{value:.{digits}f}" if digits is not None else f"{int(value)}" for value in pair)

    unique = sorted({tuple(pair) for pair in pairs})
    return text(unique[0]) if len(unique) == 1 else f"{text(unique[0])} to {text(unique[-1])}"


def cut_lines(run: BatchRun, record_name: str) -> list[str]:
    """Where the cuts were taken, for the README: what was held for every frame (the sentence of the dialog);
    and, where the frames were not all cut at the same place, between which rows or q — every frame's own cut
    is in the record."""
    lines = [f"Cuts: {export_text(run.series_cut_text)}."] if run.series_cut_text else []
    rows = [cut["horizontal_rows"] for cut in run.cuts if cut.get("horizontal_rows")]
    windows = [cut["chi_q_window"] for cut in run.cuts if cut.get("chi_q_window")]
    where = f'every frame\'s is under "cuts" in {record_name}'
    for found, name, digits, unit in ((rows, "Horizontal cut: detector rows", None, ""), (windows, "I(chi) ring: q", 3, " 1/A")):
        if not found:
            continue
        if len({tuple(pair) for pair in found}) > 1:
            lines.append(f"{name} {_span(found, digits)}{unit}, not the same in every frame ({where}).")
        elif not run.series_cut_text:
            lines.append(f"{name} {_span(found, digits)}{unit} in every frame.")
    return lines


def sum_lines(run: BatchRun) -> list[str]:
    """Frames that are sums: said, and louder when the sums are not all of the same size."""
    if not run.sums or max(run.sums) <= 1:
        return []
    sizes = Counter(run.sums)
    usual = max(sizes, key=lambda size: (sizes[size], size))
    line = f"Sums: every frame here is the sum of {usual} frames of the source; its intensities are sums over them."
    odd = [f"{frame} ({size})" for frame, size in zip(run.frames, run.sums) if size != usual]
    if odd:
        line += (f" Not of {usual} frames: {', '.join(odd[:6])}{' ...' if len(odd) > 6 else ''} - the number in brackets is "
                 "how many were summed; the intensities differ by that ratio.")
    return [line]


def result_lines(run: BatchRun, names: Mapping[str, str]) -> list[str]:
    """How the batch went: frames exported, a stop, the frames that failed and why, fits that gave nothing."""
    requested = run.requested or len(run.frames) + len(run.failed)
    lines = [f"Result: {len(run.frames)} of {requested} frames exported"
             + (f" in {run.seconds:.0f} s." if run.seconds else ".")]
    if run.write_error:
        lines.append(f"The batch was ended because the folder took no more files: {run.write_error}.")
    elif run.stopped:
        lines.append(f"Stopped by the user: {len(run.not_started)} frames were not started.")
    if run.failed:
        lines.append(f"{len(run.failed)} frames failed:")
        lines += [f"  {item['frame']}: {item['why']}" for item in run.failed[:FAILED_LISTED]]
        if len(run.failed) > FAILED_LISTED:
            lines.append(f"  ... and {len(run.failed) - FAILED_LISTED} more: \"failed\" in {names['record']} has every one.")
    unfitted = [frame for frame in run.fit_frames if frame.get("state") != "ok"]
    if unfitted:
        lines.append(f"{len(unfitted)} of {len(run.fit_frames)} frames were not fitted: their reasons are in the note "
                     f"column of {names['fit']}.")
    return lines


def earlier_lines(run: BatchRun, names: Mapping[str, str]) -> list[str]:
    earlier = run.earlier or {}
    when = f" ({earlier['when']})" if earlier.get("when") else ""
    if earlier.get("action") == "replaced":
        line = (f"An earlier export of the same data in this folder{when} was replaced: {int(earlier.get('removed', 0))} "
                "of its files were removed first.")
        if not earlier.get("complete", True):
            line += (" It was written by an earlier version of GIMaP, whose record lists only its tables: files of "
                     f"single frames from it may still be here; \"files\" in {names['record']} lists the files of this export.")
        return [line]
    if earlier.get("action") == "kept":
        return [f"This folder held an earlier export of the same data{when}. Files of it that this export did not write "
                f"again are still here; \"files\" in {names['record']} lists the files of this export."]
    return []


def readme_text(choices: BatchChoices, *, batch: str, frame: str, frames: int, curves: Sequence[tuple[str, str]],
                started: str = "", halves: Optional[str] = None, written: Optional[Collection[str]] = None,
                run: Optional[BatchRun] = None, fit_key: str = "<region>") -> str:
    """What every file of the batch folder is (written as ``README.txt``).

    ``halves``: the halves a fit takes of the cut, for a batch of GISAXS frames (``None``: curves without
    halves); said when the batch writes fit inputs or fits a model — and of each of the two only what this
    batch wrote: the table of fits, the fit-input files. ``written``: the kinds of file there are (``None``:
    every one chosen); a kind that was chosen and has no file is listed apart. ``run``: how the batch went."""
    names = output_names(choices, batch=batch, frame="<frame>", curve="<curve>", fit_key=fit_key)
    chosen = [key for key in _KINDS if getattr(choices, key)]
    if choices.fit != FIT_NONE:
        chosen += ["fit", "fit_plot"] + (["fit_curves"] if choices.fit_curves else [])
    there = [key for key in chosen if written is None or key in written]
    missing = [key for key in chosen if key not in there]
    there.append("record")
    meaning = dict(_MEANING)
    if choices.fit == FIT_MODEL:
        meaning["fit_curves"] = meaning["model_curves"]
    lines = [f"{BATCH_TITLE} - {batch}" + (f" - {started}" if started else ""), f"{frames} frames."]
    if run is not None:
        lines += result_lines(run, names) + earlier_lines(run, names)
    lines += ["", "Files (<frame> is the name of each frame, <curve> the curve key):", ""]
    width = max(len(names[key]) for key in there)
    for key in there:
        lines.append(f"{names[key]:<{width}}  {export_text(meaning[key])}")
    if missing:
        lines += ["", "Chosen, but nothing of it could be written (no frame gave it):"]
        lines += [f"  {names[key]}" for key in missing]
    if curves and (choices.tables or choices.per_frame):
        # Compare reads these named curves back; keep their titles exactly.
        lines += ["", "Curves:"] + [f"  {key}: {title}" for key, title in curves]
    lines += ["", "Units: q in 1/A (Angstrom^-1), angles in degrees, intensity = mean counts per pixel.", NAMES]
    if run is not None:
        lines += cut_lines(run, names["record"]) + sum_lines(run)
        for source in run.sources:
            # Reader warnings can quote source filenames; retain their exact spelling.
            lines.append(f"Source {source.get('name', '')}: {export_text(source.get('summary', ''))}"
                         + "".join(f"; {warning}" for warning in source.get("warnings") or ()))
    if halves is not None and (choices.fit_input or choices.fit == FIT_MODEL):
        said = [f"Halves of the horizontal cut: a fit takes the {HALVES_WORDS[fit_halves(halves)]}."]
        if choices.fit == FIT_MODEL:
            said.append(f"The columns halves and points of {names['fit']} say what each frame was fitted on.")
        if choices.fit_input:
            said.append("The fit-input files hold both halves, with signed qy; the choice is the \"halves\" of their "
                        "\"# observation:\" line.")
        lines += ["", " ".join(said)]
    lines.append("")
    return "\n".join(lines)


def batch_record(run: BatchRun, files: Sequence[Path]) -> dict[str, Any]:
    """``<batch>_batch.json``: the batch as it went (the keys: module docstring). ``files``: every file written,
    the record itself and the README included."""
    choices, destination = run.choices, run.destination
    requested = run.requested or len(run.frames) + len(run.failed)
    unfitted = [frame for frame in run.fit_frames if frame.get("state") != "ok"]
    listed = list(dict.fromkeys(
        path.relative_to(destination).as_posix() if path.is_relative_to(destination) else str(path) for path in files))
    record: dict[str, Any] = {
        "title": BATCH_TITLE,
        "format": RECORD_FORMAT,
        "started": run.started,
        "finished": time.strftime("%Y-%m-%d %H:%M"),
        "seconds": round(float(run.seconds), 1),
        "choices": choices.to_dict(),
        "requested": requested,
        "frames": list(run.frames),
        "frame_numbers": list(run.numbers) if all(number is not None for number in run.numbers) else [],
        "failed": [dict(item) for item in run.failed],
        "stopped": bool(run.stopped),
        "not_started": list(run.not_started),
        "summary": {
            "requested": requested, "exported": len(run.frames), "failed": len(run.failed),
            "fits_failed": len(unfitted), "stopped": bool(run.stopped), "seconds": round(float(run.seconds), 1),
            "files": len(listed),
        },
        "files": listed,
        "cuts": [dict(cut) for cut in run.cuts],
        "series_cut": run.series_cut,
        "series_cut_text": run.series_cut_text,
        "settings_of_first_frame": run.first_metadata or {},
    }
    if run.write_error:
        record["write_error"] = run.write_error
    if run.sums and max(run.sums) > 1:
        record["sums"] = [{"frame": frame, "frames_summed": int(size)} for frame, size in zip(run.frames, run.sums)]
    if run.sources:
        record["sources"] = [dict(source) for source in run.sources]
    if run.earlier:
        record["earlier_export"] = dict(run.earlier)
    if run.fit_frames:  # what every frame's model fit was made of, and the frames it failed on with the reason
        record["model_fit"] = model_fit_record(choices.fit_halves, run.fit_frames)
        record["fit_failed"] = [{"frame": frame["frame"], "why": frame["reason"]} for frame in unfitted]
    if run.series is not None and not run.series.is_identity:
        record["series_correction"] = asdict(run.series)
    if run.display is not None and (choices.detector_image or choices.q_map_image):
        record["pictures"] = {
            "colour_limits": "the same for every frame" if choices.image_scale == "screen" else "each frame its own (1-99.7 %)",
            **run.display,
        }
    return record


def is_gisaxs(run: BatchRun) -> bool:
    return (run.first_metadata or {}).get("measurement") == GISAXS


__all__ = [
    "BatchRun", "FAILED_LISTED", "NAMES", "batch_record", "cut_lines", "earlier_lines", "is_gisaxs", "readme_text",
    "result_lines", "sum_lines", "written_kinds",
]
