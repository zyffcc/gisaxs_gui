"""Batch export and settings files in the Analyze view model (no QWidget).

``frame_job`` describes one listed frame; ``run_job`` reduces it and writes what ``BatchChoices``
asks for (thread-safe; the same ``run_frame_job`` runs in worker processes when several frames are
reduced at once, see ``open_workers``); ``fold_outcome`` adds its outcome to the batch in frame order
(tables, fits, the first frame's record) and ``finish_batch`` writes the tables and a JSON record
of the batch. ``batch_frame`` does both for one frame. The choices and the destination of the last batch are remembered (settings
section ``analyze_batch``). ``current_settings`` / ``apply_settings`` turn the whole set-up into a
settings file and back.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

from ..application import (
    BATCH_TITLE,
    FIT_NONE,
    FIT_PEAKS,
    FOLDERS,
    FitStarts,
    FitTable,
    AnalysisRequest,
    AnalyzeSettings,
    BatchChoices,
    CurveTables,
    FrameJob,
    FrameOutcome,
    FrameTools,
    fit_curve_rows,
    fit_model,
    fit_model_curve,
    fit_peak_inputs,
    fit_peaks,
    model_row,
    output_names,
    peak_row,
    readme_text,
    run_frame_job,
)

BATCH_SECTION = "analyze_batch"


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
    first_metadata: Optional[dict] = None
    first_frame: str = ""
    curve_titles: dict = field(default_factory=dict)
    started: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M"))
    live_key: Optional[str] = None
    """The curve whose rows are shown live as the frames are done."""
    last_fit: str = ""
    """The latest fit in a few words (shown while the batch runs)."""
    display: Optional[dict] = None
    """How the pictures are drawn (``FrameJob.display``)."""


def _trend_panels(table: FitTable, kind: str) -> list:
    """The fitted values against frame: position, width and area of the peaks, or the model's sizes."""
    if kind == FIT_PEAKS:
        return [(label, table.trend(suffix)) for label, suffix in (
            ("q (1/A)", " q (1/A)"), ("FWHM (1/A)", " FWHM (1/A)"), ("area", " area"),
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

    def remember_batch(self, choices: BatchChoices, destination: Path, subfolder: bool) -> None:
        if self.settings is None:
            return
        self.settings.update_section(
            BATCH_SECTION, {"choices": choices.to_dict(), "destination": str(destination), "subfolder": bool(subfolder)}
        )
        self.settings.save()

    # -- one frame of a batch ------------------------------------------------------------

    def start_batch(self, destination: Path, choices: BatchChoices, stem: str, series=None,
                    live_key: Optional[str] = None) -> "BatchRun":
        """The state of one batch (its frames are then run with ``run_job`` and folded in order)."""
        run = BatchRun(choices=choices, destination=Path(destination), stem=stem, series=series, live_key=live_key)
        if choices.fit != FIT_NONE:
            run.fit_table = FitTable(choices.fit)
            run.starts = FitStarts(choices.fit_start)
        return run

    def frame_job(self, index: int, request: AnalysisRequest, run: "BatchRun") -> FrameJob:
        """One frame of the batch as a job (the series factor of the first frame once it is known)."""
        return FrameJob(
            index=int(index), request=request, choices=run.choices, destination=run.destination,
            series=run.series, series_factor=run.series_state.get("factor"), live_key=run.live_key,
            display=run.display,
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
        if run.first_metadata is None:
            run.first_metadata = outcome.metadata
            run.first_frame = outcome.stem
        return written

    def _fit_outcome(self, outcome: FrameOutcome, run: "BatchRun") -> list[Path]:
        choices, written, label = run.choices, [], outcome.label
        stem, text = outcome.stem, choices.text_format
        if choices.fit == FIT_PEAKS:
            fits = fit_peak_inputs(outcome.peak_inputs, choices.fit_profile, run.starts)
            run.fit_table.add(peak_row(label, fits))
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
        rows, error = [], ""
        if self.model_fitter is None:
            error = "no particle-model fit in this session"
        else:
            try:
                rows = fit_model_curve(outcome.model_curve, self.model_fitter, choices.fit_model, run.starts)
            except (ValueError, RuntimeError) as exc:
                error = str(exc)
        run.fit_table.add(model_row(label, rows, error=error))
        run.last_fit = f"{rows[0].get('combination', '')}, χ² {float(rows[0].get('best_chi2_weighted', float('nan'))):.3g}" if rows else error
        if rows and choices.fit_curves:
            best = rows[0]
            fit_x, fit_y = best.get("display_q") or (), best.get("display_fit") or ()
            path = run.destination / FOLDERS["fit_curves"] / f"{stem}_model_fit.{text}"
            written.append(self._export.writer.write_table(
                path, [f"{label}: the best model ({best.get('combination', '')}) on its own q grid"],
                ["q (1/nm)", "fit"], [[abs(float(a)), float(b)] for a, b in zip(fit_x, fit_y)],
            ))
        return written

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
        return fit_model(analysis, self.model_fitter, choices.fit_model)

    def finish_batch(self, run: "BatchRun") -> list[Path]:
        """Write the curve tables, the fit table and its trend plot, ``README.txt`` and ``<stem>_batch.json``."""
        choices, destination, stem = run.choices, run.destination, run.stem
        writer = self._export.writer
        written = run.tables.write(writer, destination, stem, choices.text_format) if run.tables.keys else []
        names = output_names(choices, batch=stem, frame=run.first_frame or "frame",
                             curve=next(iter(run.curve_titles), "region1"))
        if run.fit_table is not None and run.fit_table.rows:
            written.append(run.fit_table.write(writer, destination / names["fit"], [
                f"start values: {choices.fit_start}; "
                + (f"peak shape: {choices.fit_profile} on a straight background" if choices.fit == FIT_PEAKS
                   else f"model: {choices.fit_model} (q of the fit in 1/nm, sizes in nm)"),
            ]))
            panels = _trend_panels(run.fit_table, choices.fit)
            if panels and self.figures is not None:
                try:
                    written.append(self.figures.write_panels(panels, destination / names["fit_plot"], x_label="frame"))
                except (ValueError, OSError):
                    pass
        record: dict[str, Any] = {
            "title": BATCH_TITLE,
            "choices": choices.to_dict(),
            "frames": list(run.frames),
            "failed": list(run.failures),
            "files": [str(path.relative_to(destination)) if path.is_relative_to(destination) else str(path)
                      for path in written],
            "settings_of_first_frame": run.first_metadata or {},
        }
        if run.series is not None and not run.series.is_identity:
            record["series_correction"] = asdict(run.series)
        if run.display is not None and (choices.detector_image or choices.q_map_image):
            record["pictures"] = {
                "colour_limits": "the same for every frame" if choices.image_scale == "screen" else "each frame its own (1-99.7 %)",
                **run.display,
            }
        written.append(writer.write_record(destination / names["record"], record))
        readme = readme_text(choices, batch=stem, frame=run.first_frame or "frame", frames=len(run.frames),
                             curves=list(run.curve_titles.items()), started=run.started)
        (destination / names["readme"]).write_text(readme, encoding="utf-8")
        written.append(destination / names["readme"])
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
        self.state.beam_center = settings.beam_center
        self.state.corrections = corrections
        self.state.giwaxs = settings.giwaxs
        self.state.gisaxs = settings.gisaxs
        self.set_sum_count(settings.sum_count)
        return notes


__all__ = ["BATCH_SECTION", "BatchModelMixin", "BatchRun"]
