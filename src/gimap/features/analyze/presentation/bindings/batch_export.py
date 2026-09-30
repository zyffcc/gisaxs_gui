"""Batch Export: every listed frame with the current settings, to a folder, in a few clicks.

Open files (or a folder) → Batch Export… → Export. The dialog remembers the outputs, the curves and
the folder of the last batch; a settings file (Load / Save Settings…) brings back a whole set-up —
geometry, masks, corrections, cut regions and export choices — for the next raw data. Running the
batch (several frames at once, live in the Series tab, Pause and Stop) is ``bindings/batch_run.py``;
the tables of every frame are written at the end, with a JSON record of the batch.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Optional

from PyQt5.QtCore import QSignalBlocker, QTimer
from PyQt5.QtWidgets import QDialog, QFileDialog

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.components.detector_view import display_array, display_levels
from src.gimap.app.presentation.components.levels import to_intensity
from src.gimap.app.presentation.i18n import tr

from ...application import (
    AUTO,
    FIT_PEAKS,
    FOLDERS,
    GISAXS,
    SCALE_SCREEN,
    BatchChoices,
    export_stem,
    peak_targets,
    batch_stem,
    settings_from_record,
    settings_record,
)
from ..batch_dialog import BatchExportDialog

SETTINGS_FILTER = "GIMaP Analyze settings (*.json)"


class BatchExportMixin:
    """Needs ``view_model``, ``tasks``, the progress widgets, ``run_analysis`` and the options widgets."""

    # -- the dialog --------------------------------------------------------------------

    def batch_export_dialog(self, *, for_fitting: bool = False, then: Optional[Callable[[Path], None]] = None) -> None:
        requests = self.view_model.batch_requests()
        if not requests:
            self._status(tr("Open files or a folder first."), "warning")
            return
        if self.batch_running():
            self._status(tr("A batch is already running."), "warning")
            return
        if self._series_queue:
            self._status(tr("A series map is being built; Batch Export can start when it is done."), "warning")
            return
        choices, destination, subfolder = self.view_model.batch_preferences()
        if not destination:
            destination = str(self.view_model.default_export_dir() or self._last_folder or Path.home())
        if not choices.writes_anything:
            choices = BatchChoices()
        files = self.view_model.state.files
        stem = batch_stem(files)
        dialog = BatchExportDialog(
            frames=len(requests), files=len(files), stem=stem, settings_text=self.settings_summary(),
            choices=choices, destination=Path(destination), subfolder=subfolder,
            series=self.view_model.series_options, curve_options=self.batch_curve_options(),
            gisaxs=self.view_model.is_gisaxs(self.view_model.state.analysis), for_fitting=for_fitting,
            load_settings=self._load_settings_for_dialog, save_settings=lambda chosen: self.save_settings(choices=chosen),
            frame_stem=export_stem(analysis) if (analysis := self.view_model.state.analysis) is not None else "",
            fit_targets=[target.name for target in peak_targets(analysis)] if analysis is not None else [],
            model_available=self.view_model.model_fitter is not None, try_fit=self.try_batch_fit,
            speeds=self.batch_speed_options(len(requests)), screen_scale=self.batch_display_text(), parent=self,
        )
        self._batch_dialog = dialog
        try:
            accepted = dialog.exec_() == QDialog.Accepted
        finally:
            self._batch_dialog = None
        if not accepted:
            return
        chosen, series = dialog.choices(), dialog.series()
        if not for_fitting:
            self.view_model.remember_batch(chosen, dialog.destination(), dialog.subfolder())
        self.view_model.remember_series_options(series)
        self.run_batch(dialog.target(), chosen, series=series, stem=stem, then=then)

    def batch_from_folder(self) -> None:
        """Choose a folder of raw frames; Batch Export opens once its first frame is analysed."""
        folder = QFileDialog.getExistingDirectory(self, tr("Batch Export: a folder of raw frames"), self._last_folder)
        if not folder:
            return
        self._remember(last_folder=folder)
        before = len(self.view_model.state.files)
        added = self.add_paths([folder])
        if not added and not before:
            return
        self._batch_pending = True
        analysis = self.view_model.state.analysis
        if not added and analysis is not None:
            self._batch_when_ready(analysis)
        else:
            self._status(tr("Opening the first frame; Batch Export follows …"))

    def _batch_when_ready(self, analysis) -> None:
        """After ``batch_from_folder``: open the dialog once a frame has a geometry (its curves are the batch's)."""
        if not self._batch_pending:
            return
        if analysis.reduction is None:
            self._status(tr("Set the geometry first (Geometry step); Batch Export opens after."), "warning")
            return
        self._batch_pending = False
        QTimer.singleShot(0, self.batch_export_dialog)

    # -- how the pictures are drawn -------------------------------------------------------------

    def batch_display(self, choices: BatchChoices) -> dict:
        """Log or linear and the colour map as on screen; with “the limits on screen” also the limits of the
        detector image and of the q map (intensity units; the one never shown: from the frame on screen)."""
        view = self.detector_view
        log = view.log_check.isChecked()
        display = {"log": log, "colormap": view.colormap_combo.currentText(), "detector": None, "qmap": None}
        if choices.image_scale != SCALE_SCREEN:
            return display
        for context in ("detector", "qmap"):
            state = view.levels.state_for(context)
            levels = state.fixed if not state.auto and state.fixed else state.shown
            if levels is None:
                levels = self._levels_of_frame(context, log)
            display[context] = None if levels is None else [float(levels[0]), float(levels[1])]
        return display

    def _levels_of_frame(self, context: str, log: bool):
        analysis = self.view_model.state.analysis
        if analysis is None:
            return None
        if context == "detector":
            shown = display_array(analysis.data, analysis.valid, log)
        else:
            rsm = analysis.reduction.reciprocal_space_map if analysis.reduction is not None else None
            if rsm is None:
                return None
            shown = display_array(rsm.image, None, log)
        levels = display_levels(shown)
        return None if levels is None else to_intensity(levels, log)

    def batch_display_text(self) -> str:
        display = self.batch_display(BatchChoices(image_scale=SCALE_SCREEN))
        parts = [tr("log") if display["log"] else tr("linear"), display["colormap"]]
        for context, name in (("detector", tr("detector")), ("qmap", tr("q map"))):
            if display[context] is not None:
                low, high = display[context]
                parts.append(f"{name} {low:.4g} … {high:.4g}")
        return " · ".join(parts)

    def set_model_fitter(self, fitter) -> None:
        """The quick particle-model fit of Fitting (the application passes it in), for batch fits of GISAXS."""
        self.view_model.model_fitter = fitter

    def refresh_batch_entry(self) -> None:
        """Batch Export in the command bar, the Data step and the Series tab once more than one frame is listed.

        The first time it appears, a note says what it is for (set up one frame, export them all).
        """
        frames = self.view_model.listed_frames()
        many = frames > 1
        was_shown = not self.batch_export_button.isHidden()
        self.batch_export_button.setProperty("gimapWanted", many)
        self.batch_export_button.setVisible(many)
        self.data_batch_button.setVisible(many)
        self.data_batch_button.setText(tr("Batch Export {count} Frames…").format(count=frames) if many else tr("Batch Export…"))
        self.series_batch_button.setEnabled(many)
        if many and not was_shown and not getattr(self, "_batch_announced", False):
            self._batch_announced = True
            show_toast(
                self.window(),
                tr("{count} frames listed: set up one of them (geometry, mask, cuts), then Batch Export does them all.").format(count=frames),
                level="info", action=(tr("Batch Export…"), lambda: self.batch_export_dialog()),
            )

    def send_series_dialog(self) -> None:
        """Export the Fitting input of every listed frame, then open the series in Fitting."""
        if self._send_series_to_fitting is not None:
            self.batch_export_dialog(for_fitting=True, then=self._send_series_to_fitting)

    def batch_curve_options(self) -> list[tuple[str, str]]:
        """``(key, title)`` of the curves of the frame on screen (the batch has the same ones)."""
        analysis = self.view_model.state.analysis
        if analysis is None or analysis.reduction is None:
            return []
        return [(curve.key, curve.title) for curve in analysis.reduction.curves if not curve.is_empty]

    def settings_summary(self) -> str:
        """The set-up in one line: mode, profile, αi, cuts, masks, corrections."""
        state = self.view_model.state
        settings = self.view_model.current_settings()
        mode = {AUTO: tr("GISAXS or GIWAXS by angle")}.get(state.mode, state.mode.upper())
        parts = [mode]
        parts.append(tr("profile “{name}”").format(name=settings.profile_name) if settings.profile_name else tr("geometry matched automatically"))
        if state.incidence_deg is not None:
            parts.append(f"αi {state.incidence_deg:g}°")
        regions = len(state.giwaxs.regions)
        if regions and state.mode != GISAXS:
            parts.append(tr("{n} cut regions").format(n=regions))
        corrections = state.corrections
        shapes = len(corrections.mask_shapes) + (1 if corrections.mask_path else 0)
        if shapes:
            parts.append(tr("{n} masks").format(n=shapes))
        if corrections.mirror_fill:
            parts.append(tr("mirror filling"))
        if state.mode != GISAXS and (corrections.solid_angle or corrections.polarization is not None
                                     or corrections.film_thickness_nm is not None):
            parts.append(tr("intensity corrections"))
        if corrections.background_path:
            parts.append(tr("background {name}").format(name=Path(corrections.background_path).name))
        if self.view_model.sum_count > 1:
            parts.append(tr("sum of {n} frames").format(n=self.view_model.sum_count))
        return " · ".join(parts)

    # -- settings files ------------------------------------------------------------------

    def save_settings(self, path: str | Path | None = None, *, choices: Optional[BatchChoices] = None) -> Optional[Path]:
        if path is None:
            folder = self.view_model.default_export_dir() or Path(self._last_folder or ".")
            path, _ = QFileDialog.getSaveFileName(self, tr("Save Settings"), str(folder / "analyze_settings.json"), SETTINGS_FILTER)
            if not path:
                return None
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            record = settings_record(self.view_model.current_settings(choices))
            path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            self._status(tr("Could not save the settings: {error}").format(error=exc), "error")
            return None
        self.notify_written(tr("Saved {name}").format(name=path.name), path.parent)
        return path

    def load_settings(self, path: str | Path | None = None) -> bool:
        """Use a settings file (asks for one when ``path`` is ``None``); ``True`` when it was applied."""
        return self._read_settings(path) is not None

    def _read_settings(self, path: str | Path | None = None):
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, tr("Load Settings"), self._last_folder, SETTINGS_FILTER)
            if not path:
                return None
        path = Path(path)
        try:
            settings = settings_from_record(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            self._status(tr("Could not read the settings: {error}").format(error=exc), "error")
            return None
        notes = self.view_model.apply_settings(settings)
        self._sync_settings_widgets()
        self.run_analysis()
        text = tr("Settings loaded from {name}").format(name=path.name)
        self._status(text + ("; " + "; ".join(notes) if notes else ""), "warning" if notes else "ok")
        return settings, (BatchChoices.from_dict(settings.export) if settings.export is not None else None)

    def _load_settings_for_dialog(self):
        loaded = self._read_settings()
        if loaded is None:
            return None
        self.tasks.wait(120)  # the frame on screen with the new settings: its curves are the batch's
        self._app_process_events()
        return self.settings_summary(), self.batch_curve_options(), loaded[1]

    def _app_process_events(self) -> None:
        from PyQt5.QtWidgets import QApplication

        QApplication.processEvents()

    def _sync_settings_widgets(self) -> None:
        """Every control shows the state again (after a settings file was applied)."""
        state = self.view_model.state
        corrections, giwaxs = state.corrections, state.giwaxs
        self._refresh_profiles()
        values = [
            (self.sum_spin, self.view_model.sum_count), (self.gap_guard_spin, corrections.gap_guard_px),
            (self.in_plane_spin, giwaxs.in_plane_half_width_deg), (self.out_of_plane_spin, giwaxs.out_of_plane_half_width_deg),
            (self.bins_spin, giwaxs.bins or 0), (self.background_frame_spin, corrections.background_frame),
            (self.background_scale_spin, corrections.background_scale),
            (self.incidence_spin, state.incidence_deg if state.incidence_deg is not None else self.incidence_spin.minimum()),
        ]
        for spin, value in values:
            with QSignalBlocker(spin):
                spin.setValue(value)
        checks = [
            (self.bad_pixels_check, corrections.bad_pixels), (self.mirror_fill_check, corrections.mirror_fill),
            (self.minimum_check, corrections.minimum is not None), (self.maximum_check, corrections.maximum is not None),
            (self.solid_angle_check, corrections.solid_angle), (self.polarization_check, corrections.polarization is not None),
            (self.film_check, corrections.film_thickness_nm is not None and corrections.attenuation_length_um is not None),
            (self.sector_check, giwaxs.sector is not None), (self.box_check, giwaxs.box is not None),
        ]
        for check, value in checks:
            with QSignalBlocker(check):
                check.setChecked(bool(value))
        for spin, value in ((self.minimum_spin, corrections.minimum), (self.maximum_spin, corrections.maximum),
                            (self.polarization_spin, corrections.polarization),
                            (self.film_thickness_spin, corrections.film_thickness_nm),
                            (self.attenuation_spin, corrections.attenuation_length_um)):
            if value is not None:
                with QSignalBlocker(spin):
                    spin.setValue(value)
        if giwaxs.sector is not None:
            sector = giwaxs.sector
            for spin, value in ((self.sector_chi_min, sector.chi_min_deg), (self.sector_chi_max, sector.chi_max_deg),
                                (self.sector_q_min, sector.q_min or 0.0), (self.sector_q_max, sector.q_max or 0.0)):
                with QSignalBlocker(spin):
                    spin.setValue(value)
        if giwaxs.box is not None:
            box = giwaxs.box
            for spin, value in zip((self.box_par_min, self.box_par_max, self.box_qz_min, self.box_qz_max), (*box.q_parallel, *box.qz)):
                with QSignalBlocker(spin):
                    spin.setValue(value)
        self.polarization_spin.setEnabled(corrections.polarization is not None)
        film = corrections.film_thickness_nm is not None and corrections.attenuation_length_um is not None
        self.film_thickness_spin.setEnabled(film)
        self.attenuation_spin.setEnabled(film)
        self.sector_grid.setEnabled(giwaxs.sector is not None)
        self.box_grid.setEnabled(giwaxs.box is not None)
        self.x_axis_control.setCurrentIndex(max(0, self.x_axis_control.findData(giwaxs.x_axis)))
        with QSignalBlocker(self.mode_combo):
            self.mode_combo.setCurrentIndex(max(0, self.mode_combo.findData(state.mode)))
        background = corrections.background_path
        self.background_label.setText(Path(background).name if background else tr("No background"))
        self.background_label.setToolTip(background or "")
        self._refresh_mask_list()
        self._remember()

    # -- running -------------------------------------------------------------------------

    def apply_to_all(self, destination: Optional[Path] = None, series=None, *, save_images: bool = False,
                     then: Optional[Callable[[Path], None]] = None) -> None:
        """Every listed frame as the former Export All Listed Files wrote it: every curve, JSON, Fitting input."""
        target = destination or self.view_model.default_export_dir()
        if target is None:
            return
        choices = BatchChoices(
            tables=False, per_frame=True, fit_input=True, detector_image=save_images, q_map_image=save_images,
        )
        self.run_batch(Path(target), choices, series=series, stem=batch_stem(self.view_model.state.files), then=then)

    # -- trying a fit before the batch ---------------------------------------------------------

    def try_batch_fit(self, choices: BatchChoices, show: Callable[[str, str], None]) -> None:
        """Fit the frame on screen with the dialog's fit choices; ``show(text, level)`` gets the outcome."""
        analysis = self.view_model.state.analysis
        show(tr("Fitting the frame on screen …"), "info")
        self.tasks.submit(
            "batch-try-fit",
            lambda: self.view_model.try_fit(analysis, choices),
            on_done=lambda result: show(*fit_summary(result, choices)),
            on_error=lambda message, _details: show(message, "warning"),
        )


def fit_summary(result, choices: BatchChoices) -> tuple[str, str]:
    """The outcome of a trial fit in a few lines (``(text, level)``)."""
    if choices.fit == FIT_PEAKS:
        lines, ok = [], True
        for target, fit in result:
            if not fit.ok:
                ok = False
                lines.append(tr("{name}: not fitted — {why}").format(name=target.name, why=tr(fit.message)))
                continue
            lines.append(tr("{name}: q = {q} ± {dq} Å⁻¹ (d = {d} Å), FWHM {w} Å⁻¹, area {a}, χ²ᵣ {chi}").format(
                name=target.name, q=f"{fit.center:.4f}", dq=f"{fit.center_err:.1g}",
                d=f"{2 * math.pi / fit.center:.3f}", w=f"{fit.fwhm:.4f}", a=f"{fit.area:.3g}", chi=f"{fit.chi2_red:.2f}",
            ))
        return "\n".join(lines), "ok" if ok else "warning"
    if not result:
        return tr("No solution."), "warning"
    best = result[0]
    parts = []
    for component in best.get("components") or ():
        params = component.get("params") or {}
        parts.append(", ".join(f"{key} = {float(value):.3g}" for key, value in params.items()
                               if isinstance(value, (int, float)) and key in ("R", "h", "D", "sigma_R", "pdi")))
    text = tr("Best: {model}, {params} nm, χ² = {chi}").format(
        model=best.get("combination", ""), params="; ".join(part for part in parts if part),
        chi=f"{float(best.get('best_chi2_weighted', math.nan)):.3g}",
    )
    return text, "ok"


__all__ = ["BatchExportMixin", "SETTINGS_FILTER", "fit_summary"]
