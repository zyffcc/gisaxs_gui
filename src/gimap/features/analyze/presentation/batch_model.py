"""Batch export and settings files in the Analyze view model (no QWidget).

``frame_job`` describes one listed frame; ``run_job`` reduces it and writes what ``BatchChoices``
asks for (thread-safe; the same ``run_frame_job`` runs in worker processes when several frames are
reduced at once, see ``open_workers``); ``fold_outcome`` adds its outcome to the batch in frame order
(tables, fits, the first frame's record, every file it wrote, its cuts) and ``finish_batch`` writes the
tables, ``README.txt`` and the JSON record of the batch as it went (``application/batch_report.py``).
``batch_frame`` does both for one frame. The choices and the destination of the last batch are remembered
(settings section ``analyze_batch``) together with the data they were chosen for (``batch_origin``): the
folder and “every n-th” are not carried to other data unasked. ``current_settings`` / ``apply_settings``
turn the whole set-up into a settings file and back.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Optional

from ..application import (
    FIT_MODEL,
    FIT_NONE,
    FIT_PEAKS,
    FOLDERS,
    HALVES_WORDS,
    NO_FITTER,
    FitStarts,
    FitTable,
    AnalysisRequest,
    AnalyzeSettings,
    BatchChoices,
    BatchRun,
    FrameJob,
    FrameOutcome,
    FrameTools,
    batch_record,
    fit_curve_rows,
    fit_halves,
    fit_model,
    fit_model_curve,
    fit_peak_inputs,
    fit_peaks,
    is_gisaxs,
    model_fit_curve,
    model_fit_frame,
    model_row,
    model_preview,
    failed_preview,
    other_records,
    output_names,
    peak_row,
    peak_preview,
    readme_text,
    run_frame_job,
    written_kinds,
)

BATCH_SECTION = "analyze_batch"
MISSING_NAMED = 6
"""A table names at most this many frames that are missing from it (the README has every one)."""


def _trend_panels(table: FitTable, kind: str) -> list:
    """The fitted values against frame: position, width and area of the peaks, or the model's sizes."""
    if kind == FIT_PEAKS:  # with the errors of the fit as bars
        return [(label, table.trend(suffix, error)) for label, suffix, error in (
            ("q (1/A)", " q (1/A)", " q err"), ("FWHM (1/A)", " FWHM (1/A)", " FWHM err"), ("area", " area", " area err"),
        )]
    panels = [("chi2", table.trend("chi2"))]
    for label, suffix in (("R (nm)", " R"), ("D (nm)", " D"), ("h (nm)", " h")):
        panels.append((label, table.trend(suffix)))
    return [panel for panel in panels if panel[1]]


class BatchModelMixin:
    """Needs ``state``, ``settings``, ``_analyze_frame``, ``_export``, ``_profiles`` and ``_save_frame_images``."""

    model_fitter = None
    """The quick particle-model fit of Fitting (injected by the application; ``None``: not available)."""
    frame_workers = None
    """``frame_workers(count, profiles)`` → worker processes for ``FrameJob``s (injected by the bootstrap;
    ``None``: every frame in the application)."""
    system_resources = None
    """``system_resources()`` → ``(physical cores, free memory in bytes or None)`` (injected)."""
    preview_cache_factory = None
    """Bounded RAM and private temporary overflow for actual fitted points (injected by bootstrap)."""

    # -- remembered choices ------------------------------------------------------------

    def batch_preferences(self) -> tuple[BatchChoices, str, bool]:
        """``(choices, destination, subfolder)`` of the last batch."""
        values: dict = {}
        if self.settings is not None:
            try:
                values = dict(self.settings.get_section(BATCH_SECTION) or {})
            except Exception:
                values = {}
        return (
            BatchChoices.from_dict(values.get("choices")),
            str(values.get("destination") or ""),
            bool(values.get("subfolder", True)),
        )

    def batch_origin(self) -> tuple[str, str]:
        """``(data folder, name)`` of the data the last batch was made of (``("", "")``: not known)."""
        values: dict = {}
        if self.settings is not None:
            try:
                values = dict(self.settings.get_section(BATCH_SECTION) or {})
            except Exception:
                values = {}
        return str(values.get("data_folder") or ""), str(values.get("data_name") or "")

    def remember_batch(self, choices: BatchChoices, destination: Path, subfolder: bool, *,
                       data_folder: Optional[Path] = None, data_name: str = "") -> None:
        """Keep the choices of a batch for the next one; ``data_folder`` / ``data_name``: the data it was made
        of (``batch_origin``; not given: what was kept stays)."""
        if self.settings is None:
            return
        values = {"choices": choices.to_dict(), "destination": str(destination), "subfolder": bool(subfolder)}
        if data_folder is not None:
            values.update(data_folder=str(data_folder), data_name=str(data_name))
        self.settings.update_section(BATCH_SECTION, values)
        self.settings.save()

    def remember_halves(self, halves: str) -> None:
        """The halves chosen in Send Series to Fitting, its one choice: the next dialog starts with them when the
        Cuts step does not say which. The other choices of the last batch, and its folder, stay as they are."""
        choices, destination, subfolder = self.batch_preferences()
        halves = fit_halves(halves)
        if halves != choices.fit_halves:
            self.remember_batch(replace(choices, fit_halves=halves), destination, subfolder)

    # -- one frame of a batch ------------------------------------------------------------

    def start_batch(self, destination: Path, choices: BatchChoices, stem: str, series=None,
                    live_key: Optional[str] = None) -> "BatchRun":
        """The state of one batch (its frames are then run with ``run_job`` and folded in order)."""
        run = BatchRun(choices=choices, destination=Path(destination), stem=stem, series=series, live_key=live_key)
        if choices.fit != FIT_NONE:
            run.fit_table = FitTable(choices.fit)
            run.starts = FitStarts(choices.fit_start)
            if self.preview_cache_factory is not None:
                run.preview_cache = self.preview_cache_factory()
        return run

    def frame_job(self, index: int, request: AnalysisRequest, run: "BatchRun", number: Optional[int] = None) -> FrameJob:
        """One frame of the batch as a job (the series factor of the first frame once it is known). ``number``:
        its place among the listed frames, from 1."""
        return FrameJob(
            index=int(index), request=request, choices=run.choices, destination=run.destination,
            series=run.series, series_factor=run.series_state.get("factor"), live_key=run.live_key,
            display=run.display, number=number,
        )

    def needs_first_frame_alone(self, run: "BatchRun") -> bool:
        """A series normalised to its first frame: that frame gives the factor of all the others."""
        series = run.series
        return bool(series is not None and series.normalize and not series.per_frame and "factor" not in run.series_state)

    def frame_tools(self) -> FrameTools:
        return FrameTools(self._analyze_frame, self._export, self.figures)

    def run_job(self, job: FrameJob) -> FrameOutcome:
        """Thread-safe: reduce and write one frame in the application (``run_frame_job``)."""
        return run_frame_job(self.frame_tools(), job)

    def open_workers(self, count: int):
        """``count`` worker processes for the frames of a batch, or ``None`` (then one at a time here)."""
        if self.frame_workers is None or count < 2:
            return None
        profiles = self._profiles.load_all() if self._profiles is not None else []
        return self.frame_workers(int(count), list(profiles))

    def resources(self) -> tuple[int, Optional[int]]:
        """``(cores, free memory in bytes)`` for choosing how many frames run at once."""
        if self.system_resources is None:
            return 1, None
        try:
            return self.system_resources()
        except Exception:  # noqa: BLE001 - a guess only
            return 1, None

    def batch_frame(self, request: AnalysisRequest, run: "BatchRun") -> list[Path]:
        """Thread-safe for one batch at a time: reduce, write and (optionally) fit one frame."""
        outcome = self.run_job(self.frame_job(len(run.frames) + len(run.failures), request, run))
        return self.fold_outcome(run, outcome)

    def fold_outcome(self, run: "BatchRun", outcome: FrameOutcome) -> list[Path]:
        """Add one frame's outcome to the batch — in frame order: the tables, the fit (whose start may be
        the previous frame's result), the first frame's record. Returns every file of the frame."""
        info = outcome.series_info
        if run.series is not None and "normalization_factor" in info and not run.series.per_frame:
            run.series_state.setdefault("factor", info["normalization_factor"])
        written = list(outcome.written)
        label = outcome.label
        if run.choices.tables and outcome.curves:
            run.tables.add(label, outcome.curves)
            for curve in outcome.curves:
                run.curve_titles.setdefault(curve.key, curve.title)
        if run.fit_table is not None:
            written.extend(self._fit_outcome(outcome, run))
        run.frames.append(label)
        run.numbers.append(outcome.number)
        run.sums.append(int(outcome.summed or 1))
        run.cuts.append({"frame": label, **(outcome.cuts or {})})
        run.written.extend(written)  # every file of the batch goes into its record
        if run.first_metadata is None:
            run.first_metadata = outcome.metadata
            run.first_frame = outcome.stem
        return written

    def _fit_outcome(self, outcome: FrameOutcome, run: "BatchRun") -> list[Path]:
        choices, written, label = run.choices, [], outcome.label
        stem, text = outcome.stem, choices.text_format
        if choices.fit == FIT_PEAKS:
            fits = fit_peak_inputs(outcome.peak_inputs, choices.fit_profile, run.starts)
            values = peak_row(label, fits)
            run.fit_table.add(values, outcome.number)
            self._remember_batch_preview(run, peak_preview(outcome, fits, values))
            run.fit_keys = run.fit_keys or [target.key for target, _fit in fits]
            run.last_fit = "; ".join(
                f"{target.name}: q {fit.center:.4f} Å⁻¹, FWHM {fit.fwhm:.4f}" if fit.ok else f"{target.name}: {fit.message}"
                for target, fit in fits[:3]
            )
            if choices.fit_curves:
                for key, header, rows in fit_curve_rows(None, fits):
                    path = run.destination / FOLDERS["fit_curves"] / f"{stem}_{key}_fit.{text}"
                    written.append(self._export.writer.write_table(
                        path, [f"{label}: {key}, the data and the fitted {choices.fit_profile} on a straight background"],
                        header, rows,
                    ))
            return written
        rows, error, halves = [], "", choices.fit_halves
        if self.model_fitter is None:
            error = NO_FITTER
        else:
            try:
                rows = fit_model_curve(outcome.model_curve, self.model_fitter, choices.fit_model, run.starts, halves=halves)
            except (ValueError, RuntimeError) as exc:
                error = str(exc)
        values = model_row(label, rows, error=error, halves=halves)
        run.fit_table.add(values, outcome.number)
        self._remember_batch_preview(run, model_preview(outcome, rows, values, error=error))
        run.fit_frames.append(model_fit_frame(label, rows, error=error, halves=halves))  # a failed fit is counted
        run.last_fit = f"{rows[0].get('combination', '')}, χ² {float(rows[0].get('best_chi2_weighted', float('nan'))):.3g}" if rows else error
        if rows and choices.fit_curves:  # the points that were fitted (the mean or the one half) and the model
            path = run.destination / FOLDERS["fit_curves"] / f"{stem}_model_fit.{text}"
            written.append(self._export.writer.write_table(path, *model_fit_curve(label, rows, halves=halves)))
        return written

    def _remember_batch_preview(self, run, preview) -> None:
        with run.preview_lock:
            cache = run.preview_cache
            if cache is not None and cache.closed:
                return  # Clear/dispose while a worker finishes: export continues, the preview does not reappear
            index = len(run.fit_previews)
            run.fit_previews.append(preview)
            if cache is not None:
                for moved in cache.remember(index, preview.curves):
                    run.fit_previews[moved] = replace(run.fit_previews[moved], curves=(), cache_key=moved)

    def failed_batch_preview(self, run, label, number, ref, reason) -> None:
        if run.fit_table is not None:
            self._remember_batch_preview(run, failed_preview(label, number, ref, reason))

    def batch_preview_curves(self, run, preview):
        """Thread-safe: actual fitted points from RAM, or the owned temporary cache for an older frame."""
        if preview.cache_key is not None and run.preview_cache is not None:
            return run.preview_cache.read(preview.cache_key)
        return preview.curves

    def release_batch_previews(self, run) -> None:
        if run is not None:
            with run.preview_lock:
                if run.preview_cache is not None:
                    run.preview_cache.close()
                run.fit_previews.clear()

    def try_fit(self, analysis, choices: BatchChoices):
        """Thread-safe: the fit of the frame on screen with these choices (peaks: ``[(target, PeakFit)]``;
        model: the solutions), so the settings can be checked before the batch."""
        if analysis is None or analysis.reduction is None:
            raise ValueError("Open a frame with a geometry first.")
        if choices.fit == FIT_PEAKS:
            fits = fit_peaks(analysis, choices.fit_profile, maps=self._analyze_frame.maps_of(analysis))
            if not fits:
                raise ValueError("No region with a q window to fit: pick rings in Cuts (Ring) first.")
            return fits
        if self.model_fitter is None:
            raise ValueError("The particle-model fit is not available in this session.")
        return fit_model(analysis, self.model_fitter, choices.fit_model, halves=choices.fit_halves)

    def _table_notes(self, run: "BatchRun") -> list[str]:
        """What a table of every frame must say besides its numbers: a batch that was stopped, the frames that
        failed (their columns are missing), sums of different sizes (their intensities are not alike)."""
        notes = []
        requested = run.requested or len(run.frames) + len(run.failed)
        if run.stopped or run.write_error:
            notes.append(f"the batch ended early: {len(run.frames)} of {requested} frames are here")
        if run.failed:
            names = [item["frame"] for item in run.failed]
            notes.append(f"{len(names)} frames failed and are missing here: " + ", ".join(names[:MISSING_NAMED])
                         + (" ... (README.txt has every one)" if len(names) > MISSING_NAMED else ""))
        if run.sums and max(run.sums) > 1:
            sizes = Counter(run.sums)
            usual = max(sizes, key=lambda size: (sizes[size], size))
            odd = [f"{frame} ({size})" for frame, size in zip(run.frames, run.sums) if size != usual]
            notes.append(f"every column is the sum of {usual} frames: its intensities are sums over them"
                         + (f"; other sums: {', '.join(odd[:MISSING_NAMED])}{' ...' if len(odd) > MISSING_NAMED else ''}"
                            if odd else ""))
        return notes

    def finish_batch(self, run: "BatchRun") -> list[Path]:
        """Write the curve tables, the fit table and its trend plot, ``README.txt`` and ``<stem>_batch.json`` —
        the batch as it went (``application/batch_report.py``). Returns what was written here."""
        choices, destination, stem = run.choices, run.destination, run.stem
        writer = self._export.writer
        written = (run.tables.write(writer, destination, stem, choices.text_format, self._table_notes(run))
                   if run.tables.keys else [])
        fit_key = run.fit_keys[0] if run.fit_keys else None
        names = output_names(choices, batch=stem, frame=run.first_frame or "frame",
                             curve=next(iter(run.curve_titles), "region1"), fit_key=fit_key)
        # The halves a fit takes of the cut: of a batch of GISAXS frames only (an I(q) has none).
        halves = choices.fit_halves if is_gisaxs(run) else None
        if run.fit_table is not None and run.fit_table.rows:
            comments = [
                f"start values: {choices.fit_start}; "
                + (f"peak shape: {choices.fit_profile} on a straight background" if choices.fit == FIT_PEAKS
                   else f"model: {choices.fit_model} (q of the fit in 1/nm, sizes in nm)"),
            ]
            if choices.fit == FIT_MODEL:
                chosen = fit_halves(choices.fit_halves)
                comments.append(
                    f"halves of the cut: {HALVES_WORDS[chosen]} ({chosen}); the columns halves "
                    "and points say what each frame was fitted on (a frame without a fit: its reason in note)")
            written.append(run.fit_table.write(writer, destination / names["fit"], comments))
            panels = _trend_panels(run.fit_table, choices.fit)
            if panels and self.figures is not None:
                try:
                    written.append(self.figures.write_panels(
                        panels, destination / names["fit_plot"], x_label="frame (its place among the listed frames)"))
                except (ValueError, OSError):
                    pass  # no frame gave a value: the README says that the plot is missing
        record_path, readme_path = destination / names["record"], destination / names["readme"]
        files = [*run.written, *written, readme_path, record_path]
        kinds = written_kinds(files, destination, names)
        readme = readme_text(choices, batch=stem, frame=run.first_frame or "frame", frames=len(run.frames),
                             curves=list(run.curve_titles.items()), started=run.started, halves=halves, written=kinds,
                             run=run, fit_key=fit_key or "<region>")
        others = other_records(destination, stem)
        if others:  # exports of other data in the same folder: this README is about one of them
            readme += f"This folder also holds other exports: {', '.join(others)} (each lists its own files).\n"
        readme_path.write_text(readme, encoding="utf-8")
        written.append(readme_path)
        written.append(writer.write_record(record_path, batch_record(run, files)))
        run.files = files
        return written

    # -- settings files ------------------------------------------------------------------

    def current_settings(self, export: Optional[BatchChoices] = None) -> AnalyzeSettings:
        """The set-up in use; the profile is the one chosen or, if automatic, the one matched."""
        state = self.state
        name = state.profile_name
        profile = self._profiles.find(name) if name and self._profiles is not None else None
        analysis = state.analysis
        if profile is None and analysis is not None and analysis.resolution.profile is not None:
            profile = analysis.resolution.profile
            name = profile.name
        return AnalyzeSettings(
            mode=state.mode,
            profile_name=name,
            profile=profile,
            incidence_deg=state.incidence_deg,
            beam_center=state.beam_center,
            beam_center_shape=self.session_center_shape() if hasattr(self, "session_center_shape") else None,
            sum_count=self.sum_count,
            corrections=state.corrections,
            giwaxs=state.giwaxs,
            gisaxs=state.gisaxs,
            export=export.to_dict() if export is not None else None,
        )

    def apply_settings(self, settings: AnalyzeSettings) -> list[str]:
        """Use a loaded set-up; returns what could not be taken over (missing files, a new profile)."""
        notes: list[str] = []
        name = settings.profile_name
        if name and self._profiles is not None and self._profiles.find(name) is None:
            if settings.profile is not None:
                self._profiles.save(replace(settings.profile, name=name))
                notes.append(f"Instrument profile “{name}” added from the settings file.")
            else:
                notes.append(f"Instrument profile “{name}” is not here: the geometry is matched automatically.")
                name = None
        corrections = settings.corrections
        for label, path in (("Mask file", corrections.mask_path), ("Background frame", corrections.background_path)):
            if path and not Path(path).exists():
                notes.append(f"{label} not found, left out: {path}")
                corrections = replace(corrections, **({"mask_path": None} if label == "Mask file" else {"background_path": None}))
        self.state.mode = settings.mode
        self.state.profile_name = name
        self.state.incidence_deg = settings.incidence_deg
        if settings.beam_center is None:
            self.clear_beam_center()
        else:
            self.set_beam_center(*settings.beam_center, shape=settings.beam_center_shape)
        self.state.corrections = corrections
        self.state.giwaxs = settings.giwaxs
        self.state.gisaxs = settings.gisaxs
        self.set_sum_count(settings.sum_count)
        return notes


__all__ = ["BATCH_SECTION", "BatchModelMixin", "BatchRun"]
