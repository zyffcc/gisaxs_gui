"""The Analyze workspace around the analysis: steps, the current file, progress and overlays.

Every action answers on the page itself: opening a file shows its name and a busy bar at once,
then the image; each step says what it found; the mask and the source of each curve can be
drawn on the image; exports say where they went, with a button to open the folder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PyQt5.QtCore import QSignalBlocker, Qt, QUrl
from PyQt5.QtGui import QDesktopServices, QKeySequence
from PyQt5.QtWidgets import QFileDialog, QMenu, QShortcut

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import tr

from ...application import GISAXS, FrameAnalysis, source_labels

MASK_COLOR = "#ef4444"
BAD_PIXEL_COLOR = "#f59e0b"
FILLED_COLOR = "#22d3ee"
"""Pixels filled from the mirror side (cyan)."""
"""Hot and dead pixels found in the frame (amber, circled so single pixels stay visible)."""


def _pixels(count: int, total: int) -> str:
    return f"{count:,} px ({100.0 * count / max(total, 1):.2g} %)".replace(",", " ")


class WorkspaceMixin:
    """Needs the widgets of ``AnalyzePageView`` and ``self.view_model`` / ``self.tasks``."""

    _find_geometry: Optional[Callable[[], None]] = None
    _run_pipeline: Optional[Callable[[], None]] = None
    _stop_pipeline: Optional[Callable[[], None]] = None

    def _connect_workspace(self) -> None:
        self._overlay = None  # "mask", "sources" or None
        self._only_source: Optional[tuple[str, int]] = None
        self._plot_keys: dict[str, list[str]] = {"top": [], "bottom": []}
        self.step_rail.stepChosen.connect(self._step_chosen)
        self.right_tabs.activated.connect(lambda index: self.show_right(self.right_tabs.itemData(index)))
        self.previous_file_button.clicked.connect(lambda: self._step_file(-1))
        self.next_file_button.clicked.connect(lambda: self._step_file(+1))
        for keys, delta in (("PgUp", -1), ("PgDown", +1)):
            shortcut = QShortcut(QKeySequence(keys), self)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(lambda delta=delta: self._step_file(delta))
        batch = QShortcut(QKeySequence("Ctrl+Shift+E"), self)
        batch.setContext(Qt.WidgetWithChildrenShortcut)
        batch.activated.connect(lambda: self.batch_export_dialog())
        self.options_button.toggled.connect(lambda on: on and self.show_step("mask"))
        self.export_curves_button.clicked.connect(self.export_current)
        self.export_image_button.clicked.connect(self.save_image)
        self.export_map_button.clicked.connect(self.save_map_data)
        self.save_map_action.triggered.connect(self.save_map_data)
        plots = QMenu(self.export_plots_button)
        plots.addAction(self.save_upper_plot_action)
        plots.addAction(self.save_lower_plot_action)
        self.export_plots_button.setMenu(plots)
        self.export_all_button.clicked.connect(lambda: self.batch_export_dialog())
        self.export_fit_button.clicked.connect(self.fit_current)
        self.export_series_button.clicked.connect(self.send_series_dialog)
        self.symmetry_button.clicked.connect(self.refine_center_x)
        self.halves_combo.activated.connect(self._halves_chosen)
        self.show_mask_button.toggled.connect(self._mask_toggled)
        self.sources_button.toggled.connect(self._sources_toggled)
        self.top_plot.curveClicked.connect(lambda index: self._curve_clicked("top", index))
        self.bottom_plot.curveClicked.connect(lambda index: self._curve_clicked("bottom", index))
        self.tasks.busy_changed.connect(self._busy_changed)
        self.find_geometry_button.clicked.connect(self._find_geometry_clicked)
        self.banner_find_button.clicked.connect(self._find_geometry_clicked)
        self.run_pipeline_button.clicked.connect(self._run_pipeline_clicked)
        has_finder = self._find_geometry is not None
        self.find_geometry_button.setVisible(has_finder)
        self.banner_find_button.setVisible(has_finder)
        self.run_pipeline_button.setVisible(self._run_pipeline is not None)
        self.show_step("data")
        self.show_right("curves")
        self._sync_halves()
        self._refresh_workspace(None)

    # -- steps -----------------------------------------------------------------------

    def show_step(self, key: str) -> None:
        if key not in self.step_pages:
            return
        self.step_rail.set_current(key)
        self.step_stack.setCurrentWidget(self.step_pages[key])

    def _step_chosen(self, key: str) -> None:
        """A step chosen by the user also brings up what it is about on the right."""
        self.show_step(key)
        if key == "results":
            self.show_right("results")
        elif key in ("geometry", "mask", "cuts"):
            self.show_right("curves")

    def show_right(self, key: str) -> None:
        page = self.right_pages.get(key)
        if page is None:
            return
        self.right_stack.setCurrentWidget(page)
        index = self.right_tabs.findData(key)
        if index >= 0 and self.right_tabs.currentIndex() != index:
            self.right_tabs.setCurrentIndex(index)

    def current_right(self) -> Optional[str]:
        current = self.right_stack.currentWidget()
        return next((key for key, page in self.right_pages.items() if page is current), None)

    def add_result_panel(self, widget) -> None:
        """A panel another component fills with results (the automatic analysis, the AI)."""
        self.results_panel_host.insertWidget(self.results_panel_host.count() - 1, widget)
        self.results_panel_empty.hide()

    def current_step(self) -> Optional[str]:
        return self.step_rail.current()

    def set_step_state(self, key: str, state: str, detail: str = "") -> None:
        self.step_rail.set_state(key, state, detail)

    def add_step_panel(self, key: str, widget) -> None:
        """Put a panel another component provides (analysis results, a report) into a step."""
        host = {"results": self.results_host, "export": self.export_extra_host}[key]
        host.addWidget(widget)
        if key == "results":
            self.results_empty.hide()

    def set_mode_choice(self, mode: str) -> None:
        """Preselect GISAXS / GIWAXS (a task chosen on the Start page) before any file is open."""
        index = self.mode_combo.findData(mode)
        if index >= 0:
            with QSignalBlocker(self.mode_combo):
                self.mode_combo.setCurrentIndex(index)
            self.view_model.set_mode(mode)

    # -- files -----------------------------------------------------------------------

    def _step_file(self, delta: int) -> None:
        count = self.file_list.count()
        if count:
            row = min(max(0, self.file_list.currentRow() + int(delta)), count - 1)
            self.file_list.setCurrentRow(row)

    def _file_loading(self, path: Path) -> None:
        """Immediately after a file is chosen: its name, and that it is being read."""
        self.file_chip.setText(path.name)
        self.file_chip.setToolTip(str(path))
        self.file_meta.setText("reading…")
        count = self.file_list.count()
        for button in (self.previous_file_button, self.next_file_button):
            button.setVisible(count > 1)  # stepping through files: only with more than one
        self.previous_file_button.setEnabled(self.file_list.currentRow() > 0)
        self.next_file_button.setEnabled(0 <= self.file_list.currentRow() < count - 1)
        self.set_step_state("data", "busy", f"Reading {path.name} …")

    def _busy_changed(self, busy: bool) -> None:
        if busy:
            self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(
            bool(busy) or bool(getattr(self, "_batch", None)) or bool(getattr(self, "_automatic_busy", False))
        )

    # -- what each step found ----------------------------------------------------------

    def _intro(self, key: str, text: str) -> None:
        """One sentence of what the step found, in the interface language when the table has it."""
        self.step_intro[key].setText(tr(text))

    def _refresh_workspace(self, analysis: Optional[FrameAnalysis]) -> None:
        files = self.file_list.count()
        if analysis is None:
            self._intro("data", 
                "Open or drop detector frames (CBF, NXS, TIFF, EDF) or a folder. A multi-module "
                "NeXus series (…_m01.nxs … _m11.nxs) opens as one stitched frame."
            )
            self.data_info_label.setText("No file yet.")
            for key in ("geometry", "mask", "cuts", "results", "export"):
                self.set_step_state(key, "pending")
            self.set_step_state("data", "pending")
            self._intro("geometry", "The geometry turns pixels into q: distance, beam centre, wavelength, αi.")
            self._intro("mask", "Detector gaps and bad pixels are left out of every curve.")
            self._intro("cuts", "Where the curves are taken from on the detector.")
            self._intro("export", 
                "Everything is written next to the data in gimap_analysis/, with a JSON record of how it was made."
            )
            self.mask_summary_label.setText("No frame yet.")
            self.cuts_info_label.setText("")
            return
        self._show_data_info(analysis, files)
        self._show_geometry_state(analysis)
        self._show_mask_state(analysis)
        self._show_cut_state(analysis)
        self._refresh_overlay()

    def _show_data_info(self, analysis: FrameAnalysis, files: int) -> None:
        rows, columns = analysis.shape
        detector = analysis.detector_name or "Detector"
        frames = analysis.frame_count
        summed = f", sum of {analysis.frame_total} frames" if analysis.summed_frames else ""
        meta = f"{detector} · {rows}×{columns} px"
        if frames > 1:
            meta += f" · frame {analysis.frame_index + 1} of {frames}{summed}"
        self.file_chip.setText(analysis.path.name)
        self.file_chip.setToolTip(str(analysis.path))
        self.file_meta.setText(meta)
        self.set_step_state("data", "ok", meta)
        metadata = analysis.metadata or {}
        lines = [f"Detector: {detector}", f"Frame: {rows} × {columns} pixels"]
        geometry = analysis.geometry
        pixel = geometry.pixel_size_x_m if geometry is not None else metadata.get("pixel_size_x_m")
        if pixel:
            lines[-1] += f", {pixel * 1e6:g} µm pixels"
        lines.append(f"Frames in the file: {frames}" + (f" (showing {analysis.frame_index + 1}{summed})" if frames > 1 else ""))
        header = []
        if metadata.get("energy_kev"):
            header.append(f"{float(metadata['energy_kev']):g} keV")
        if metadata.get("header_distance_m"):
            header.append(f"distance {float(metadata['header_distance_m']) * 1e3:g} mm")
        if metadata.get("exposure_time_s"):
            header.append(f"exposure {float(metadata['exposure_time_s']):g} s")
        if header:
            lines.append("File header: " + ", ".join(header))
        lines.append(f"Files listed: {files}")
        self.data_info_label.setText("\n".join(lines))
        self._intro("data", 
            f"{analysis.path.name}: {detector}, {rows}×{columns} px"
            + (f", {frames} frames" if frames > 1 else "") + "."
        )

    def _show_geometry_state(self, analysis: FrameAnalysis) -> None:
        geometry = analysis.geometry
        if geometry is None:
            detector = analysis.detector_name or "this detector"
            self.set_step_state("geometry", "warn", "No geometry yet")
            self._intro("geometry", 
                f"No geometry for {detector} ({analysis.shape[0]}×{analysis.shape[1]}) yet. "
                + ("Find a calibration automatically, calibrate" if self._find_geometry else "Calibrate")
                + " from an image of a standard, or enter the values once: they are saved as an "
                "instrument profile and used for every such frame."
            )
            return
        profile = analysis.resolution.profile if analysis.resolution is not None else None
        name = profile.name if profile is not None else "instrument profile"
        detail = f"{geometry.distance_m * 1e3:.1f} mm · λ {geometry.wavelength_angstrom:.4g} Å · αi {geometry.incidence_deg:g}°"
        self.set_step_state("geometry", "ok", detail)
        self._intro("geometry", f"Geometry from “{name}”: {detail}.")

    def _show_mask_state(self, analysis: FrameAnalysis) -> None:
        valid = np.asarray(analysis.valid, dtype=bool)
        total = valid.size
        raw = np.asarray(analysis.raw_valid if analysis.raw_valid is not None else analysis.valid, dtype=bool)
        bad = analysis.bad_pixels
        flagged = bad.mask & raw if bad is not None else np.zeros_like(raw)
        drawn = analysis.drawn_mask & raw if analysis.drawn_mask is not None else np.zeros_like(raw)
        extra = int((raw & ~valid & ~flagged & ~drawn).sum())
        lines = [tr("Left out: {pixels}").format(pixels=_pixels(total - int(valid.sum()), total))]
        lines.append(tr("· detector gaps and flagged pixels: {pixels}").format(pixels=_pixels(int((~raw).sum()), total)))
        corrections = analysis.corrections
        if bad is not None:
            found = bad.hot_count + bad.dead_count
            lines.append(
                tr("· found in the frame: {hot} hot and {dead} dead pixels").format(hot=bad.hot_count, dead=bad.dead_count)
                if found else tr("· no hot or dead pixels found in the frame")
            )
            if bad.line_count:
                lines.append(tr("· defective detector rows or columns: {count} pixels").format(count=bad.line_count))
        if drawn.any():
            lines.append(tr("· masks you drew or loaded: {pixels}").format(pixels=_pixels(int(drawn.sum()), total)))
        if analysis.filled_pixels is not None:
            lines.append(tr("· filled from the mirror side: {pixels} (now used)").format(
                pixels=_pixels(int(analysis.filled_pixels.sum()), total)
            ))
        if analysis.metadata and str(analysis.metadata.get("stored_dtype", "")).startswith("float"):
            lines.append(tr("· floating-point frame: negative values are kept as data; error bars from the pixel scatter"))
        if extra:
            parts = []
            if corrections is not None and corrections.gap_guard_px:
                parts.append(tr("gap guard {px} px").format(px=corrections.gap_guard_px))
            if corrections is not None and (corrections.minimum is not None or corrections.maximum is not None):
                parts.append(tr("intensity range"))
            if corrections is not None and corrections.background_path:
                parts.append(tr("background"))
            lines.append(f"· {tr(' and ').join(parts) or tr('corrections')}: {_pixels(extra, total)}")
        self.mask_summary_label.setText("\n".join(lines))
        share = 100.0 * (total - int(valid.sum())) / max(total, 1)
        self.set_step_state("mask", "ok", tr("{share} % of the pixels left out").format(share=f"{share:.2g}"))
        self._intro("mask", 
            "Pixels without data are left out of every curve before any cut; corrections below apply to the frame."
        )

    def _show_cut_state(self, analysis: FrameAnalysis) -> None:
        reduction = analysis.reduction
        gisaxs = analysis.kind == GISAXS
        self.gisaxs_cuts.setVisible(gisaxs)
        self.giwaxs_section.setVisible(not gisaxs)
        if reduction is None:
            self.set_step_state("cuts", "pending", "Needs a geometry")
            self._intro("cuts", "The cuts need a geometry (step 2).")
            return
        markers = reduction.markers
        if gisaxs:
            low, high = markers["horizontal_band"]
            left, right = markers["vertical_band"]
            yoneda = markers.get("yoneda")
            source = markers.get("horizontal_source")
            where = (
                f"at the Yoneda band (αf = {yoneda.alpha_f_deg:.3f}°)" if source == "yoneda" and yoneda
                else "set by hand" if source == "manual" else "just above the horizon (no Yoneda band found)"
            )
            text = (
                f"Horizontal cut I(qy) over rows {low:.0f}–{high:.0f}, {where}.\n"
                f"Vertical cut I(qz) over columns {left:.0f}–{right:.0f} (the beam centre)."
            )
            self.cuts_info_label.setText(text)
            self.set_step_state("cuts", "ok", "Yoneda cut" if source == "yoneda" else f"Cut {source}")
            self._intro("cuts", 
                "GISAXS: the horizontal cut at the Yoneda band gives the in-plane structure, the vertical cut the "
                "out-of-plane one. Make left and right symmetric before averaging the halves."
            )
        else:
            regions = len(self.view_model.state.giwaxs.regions)
            self.set_step_state(
                "cuts", "ok", tr("Ring, bands and {n} region(s)").format(n=regions) if regions else tr("Ring and bands")
            )
            self._intro(
                "cuts",
                "GIWAXS: every region is cut along q (upper plot) and along χ (lower plot). Add regions, "
                "then move them on the Cake view or change them below.",
            )

    # -- halves of the horizontal cut --------------------------------------------------

    def _halves_chosen(self, _index: int) -> None:
        self.view_model.set_fit_side(self.halves_combo.currentData())
        self._sync_fit_side()
        self._sync_halves()

    def _sync_halves(self) -> None:
        with QSignalBlocker(self.halves_combo):
            self.halves_combo.setCurrentIndex(max(0, self.halves_combo.findData(self.view_model.fit_side)))

    # -- overlays on the image ----------------------------------------------------------

    def _mask_toggled(self, on: bool) -> None:
        if on:
            with QSignalBlocker(self.sources_button):
                self.sources_button.setChecked(False)
            self._overlay = "mask"
        elif self._overlay == "mask":
            self._overlay = None
        self._refresh_overlay()

    def _sources_toggled(self, on: bool) -> None:
        self._only_source = None
        if on:
            with QSignalBlocker(self.show_mask_button):
                self.show_mask_button.setChecked(False)
            self._overlay = "sources"
        elif self._overlay == "sources":
            self._overlay = None
        self._refresh_overlay()

    def _curve_clicked(self, plot: str, index: int) -> None:
        """Clicking a curve shows where (only) it comes from."""
        keys = self._plot_keys.get(plot, [])
        if not 0 <= index < len(keys):
            return
        self._only_source = (plot, index)
        with QSignalBlocker(self.show_mask_button):
            self.show_mask_button.setChecked(False)
        with QSignalBlocker(self.sources_button):
            self.sources_button.setChecked(True)
        self._overlay = "sources"
        self._refresh_overlay()

    def _refresh_overlay(self) -> None:
        view = self.detector_view
        analysis = self.view_model.state.analysis
        view.hide_markers()
        if analysis is None or self._overlay is None or self.view_combo.currentIndex() != 0:
            view.hide_labels()
            return
        if self._overlay == "mask":
            labels = (~np.asarray(analysis.valid, dtype=bool)).astype(np.uint8)
            bad = analysis.bad_pixels
            if bad is not None and bad.mask.any():
                labels[bad.mask] = 2
                rows, columns = np.nonzero(bad.mask)
                view.show_markers(list(zip(columns + 0.5, rows + 0.5)), symbol="o", color=BAD_PIXEL_COLOR, size=14)
            if analysis.filled_pixels is not None:
                labels[analysis.filled_pixels] = 3
            view.show_labels(labels, [MASK_COLOR, BAD_PIXEL_COLOR, FILLED_COLOR], alpha=150)
            return
        entries = []
        for plot_name, plot in (("top", self.top_plot), ("bottom", self.bottom_plot)):
            for index, (key, color) in enumerate(zip(self._plot_keys.get(plot_name, []), plot.curve_colors())):
                if self._only_source is None or self._only_source == (plot_name, index):
                    entries.append((key, color))
        if not entries:
            view.hide_labels()
            return
        keys = [key for key, _color in entries]
        self.tasks.submit(
            "sources",
            lambda: self.view_model.source_masks(analysis, keys),
            on_done=lambda masks: self._show_sources(analysis, entries, masks),
            on_error=lambda message, _details: self._status(f"Could not draw the sources: {message}", "error"),
        )

    def _show_sources(self, analysis: FrameAnalysis, entries, masks: dict) -> None:
        if analysis is not self.view_model.state.analysis or self._overlay != "sources":
            return
        shown = [(masks[key], color) for key, color in entries if key in masks]
        if not shown:
            self.detector_view.hide_labels()
            return
        # The widest region first, so sectors stay visible on top of the full ring.
        shown.sort(key=lambda item: -int(item[0].sum()))
        self.detector_view.show_labels(source_labels([mask for mask, _ in shown]), [color for _, color in shown])

    # -- export extras ----------------------------------------------------------------------

    def save_map_data(self) -> None:
        analysis = self.view_model.state.analysis
        if analysis is None or analysis.reduction is None or analysis.reduction.reciprocal_space_map is None:
            self._status("The q map needs a frame with a geometry.", "warning")
            return
        folder = self.view_model.default_export_dir() or Path(self._last_folder or ".")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save q-Map Data", str(folder / f"{analysis.path.stem}_qmap.csv"), "CSV table (*.csv)"
        )
        if not path:
            return
        try:
            written = self.view_model.export_q_map(Path(path))
        except (ValueError, OSError) as exc:
            self._status(f"Could not save the q map: {exc}", "error")
            return
        self.notify_written(f"Saved the q map to {written.name}", written.parent)

    def notify_written(self, text: str, folder: Optional[Path]) -> None:
        """Status line and a toast with a button that opens the folder."""
        self._status(text, "ok")
        action = None
        if folder is not None:
            action = ("Open Folder", lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))))
        show_toast(self, text, level="ok", action=action)

    # -- extensions from the composition root -----------------------------------------------

    def set_automatic_analysis(
        self, *, run: Callable[[], None], find_geometry: Callable[[], None], stop: Optional[Callable[[], None]] = None,
    ) -> None:
        """Another component runs the automatic analysis; the workspace shows its buttons and progress."""
        self._run_pipeline, self._find_geometry, self._stop_pipeline = run, find_geometry, stop
        self.run_pipeline_button.show()
        self.find_geometry_button.show()
        self.banner_find_button.show()

    def automatic_started(self, message: str) -> None:
        self._automatic_busy = True
        self.run_pipeline_button.setEnabled(False)
        self.find_geometry_button.setEnabled(False)
        self.banner_find_button.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()
        self.stop_pipeline_button.setVisible(self._stop_pipeline is not None)
        self.stop_pipeline_button.setEnabled(True)
        self.stop_pipeline_button.setText(tr("Stop"))
        self.show_right("results")  # the progress of the run is at the top of the Results tab
        self.set_step_state("results", "busy", "Running…")
        self._status(message)

    def automatic_progress(self, text: str) -> None:
        self._status(text)

    def automatic_stopping(self) -> None:
        """Stop was asked for (here or on the progress panel): the run ends after the step in progress."""
        self.stop_pipeline_button.setEnabled(False)
        self.stop_pipeline_button.setText(tr("Stopping…"))
        self._status(tr("Stopping after the current step …"))

    def _stop_clicked(self) -> None:
        if self._stop_pipeline is not None:
            self._stop_pipeline()

    def automatic_finished(self, state: str, detail: str, *, show_results: bool = True) -> None:
        """``state``: ``ok`` or ``warn`` (questions only a person can answer)."""
        self._automatic_busy = False
        for button in (self.run_pipeline_button, self.find_geometry_button, self.banner_find_button):
            button.setEnabled(True)
        self.stop_pipeline_button.hide()
        self.progress_bar.setVisible(self.tasks.is_busy())
        self.set_step_state("results", state, detail)
        self._intro("results", detail)
        if show_results:
            self.show_right("results")
        if state != "ok":
            self.show_step("results")
        self._status(detail, "ok" if state == "ok" else "warning")

    def automatic_failed(self, message: str) -> None:
        self._automatic_busy = False
        for button in (self.run_pipeline_button, self.find_geometry_button, self.banner_find_button):
            button.setEnabled(True)
        self.stop_pipeline_button.hide()
        self.progress_bar.setVisible(self.tasks.is_busy())
        self.set_step_state("results", "error", message)
        self._status(f"The automatic analysis stopped: {message}", "error")

    def _find_geometry_clicked(self) -> None:
        if self._find_geometry is not None:
            self.show_step("geometry")
            self._find_geometry()

    def _run_pipeline_clicked(self) -> None:
        if self._run_pipeline is not None:
            self._run_pipeline()


__all__ = ["MASK_COLOR", "WorkspaceMixin"]
