"""The Batch Export dialog: every listed frame with the current settings, saved as chosen.

The layout is ``views/batch_export_view.py``. Every option shows the file it writes (with its
extension, following the chosen formats), so it is clear before the export what each file will be;
the same names go into the ``README.txt`` of the folder. The dialog remembers nothing itself: the
page passes the last choices in and stores what was accepted. ``for_fitting`` turns it into Send
Series to Fitting: only the Fitting input of each frame is written, then the series opens in Fitting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QFileDialog, QListWidgetItem

from src.gimap.app.presentation.i18n import tr

from ..application import (
    FIT_MODEL,
    FIT_NONE,
    FIT_PEAKS,
    FRAME_FORMATS,
    IMAGE_FORMATS,
    TEXT_FORMATS,
    BatchChoices,
    SeriesCorrection,
    output_names,
)
from .views.batch_export_view import OUTPUTS, BatchExportView

LoadSettings = Callable[[], Optional[tuple[str, list, Optional[BatchChoices]]]]
TryFit = Callable[[BatchChoices, Callable[[str, str], None]], None]


class BatchExportDialog(QDialog, BatchExportView):
    def __init__(
        self,
        *,
        frames: int,
        files: int,
        stem: str,
        settings_text: str,
        choices: BatchChoices,
        destination: Path,
        subfolder: bool,
        series: SeriesCorrection,
        curve_options: Sequence[tuple[str, str]],
        gisaxs: bool = False,
        for_fitting: bool = False,
        load_settings: Optional[LoadSettings] = None,
        save_settings: Optional[Callable[[BatchChoices], None]] = None,
        frame_stem: str = "",
        fit_targets: Sequence[str] = (),
        model_available: bool = False,
        try_fit: Optional[TryFit] = None,
        speeds: Sequence[tuple[str, str]] = (),
        screen_scale: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self.setup_batch_export()
        self._stem = stem
        self._frame_stem = frame_stem or "frame"
        self._frames = int(frames)
        self._for_fitting = for_fitting
        self._gisaxs = gisaxs
        self._load_settings = load_settings
        self._save_settings = save_settings
        self._try_fit = try_fit
        self._fit_targets = list(fit_targets)
        self.setWindowTitle(tr("Send Series to Fitting" if for_fitting else "Batch Export"))
        if frames == 1:
            what = tr("1 frame from 1 file")
        else:
            what = tr("{frames} frames from {files} files") if files > 1 else tr("{frames} frames from 1 file")
        self.frames_label.setText(
            what.format(frames=frames, files=files) + " — " + tr("each reduced with the settings below.")
        )
        self.settings_label.setText(settings_text)
        for combo, options in ((self.text_format_combo, TEXT_FORMATS), (self.image_format_combo, IMAGE_FORMATS),
                               (self.frame_format_combo, FRAME_FORMATS)):
            for key, text in options.items():
                combo.addItem(tr(text), key)
            combo.currentIndexChanged.connect(self._sync)
        for key, text in speeds:
            self.speed_combo.addItem(text, key)
        self.speed_combo.setCurrentIndex(max(0, self.speed_combo.findData(choices.speed)))
        self.speed_combo.setEnabled(self.speed_combo.count() > 1)
        self._stored_speed = choices.speed
        for key, text in (("auto", "Each frame its own limits (1–99.7 %)"),
                          ("screen", "The limits on screen, the same for every frame")):
            self.image_scale_combo.addItem(tr(text), key)
        self.image_scale_combo.setCurrentIndex(max(0, self.image_scale_combo.findData(choices.image_scale)))
        self._screen_scale = screen_scale
        self.image_scale_combo.currentIndexChanged.connect(self._sync)
        self._set_choices(choices)
        self.every_spin.valueChanged.connect(self._sync)
        for check in self.output_checks.values():
            check.toggled.connect(self._sync)
        self.output_checks["cake"].setVisible(not gisaxs)
        self.output_files["cake"].setVisible(not gisaxs)
        self._set_curves(curve_options, choices.curves)
        self.curve_list.itemChanged.connect(lambda _item: self._sync())
        self.all_curves_button.clicked.connect(lambda: self._check_all(True))
        self.no_curves_button.clicked.connect(lambda: self._check_all(False))
        # fitting: what this frame can be fitted with
        self.fit_peaks_radio.setVisible(not gisaxs)
        self.fit_model_radio.setVisible(gisaxs)
        self.fit_peaks_radio.setEnabled(bool(self._fit_targets))
        if not self._fit_targets:
            self.fit_peaks_radio.setToolTip(tr("Pick rings first (Cuts ▸ Ring): every region with a q window is one peak"))
        self.fit_model_radio.setEnabled(model_available)
        if not model_available:
            self.fit_model_radio.setToolTip(tr("The particle-model fit of Fitting is not available in this session"))
        for button in (self.fit_none_radio, self.fit_peaks_radio, self.fit_model_radio):
            button.toggled.connect(self._sync)
        for combo in (self.fit_profile_combo, self.fit_model_combo):
            combo.currentIndexChanged.connect(self._sync)
        for radio in self.fit_start_radios.values():
            radio.toggled.connect(self._sync)
        self.fit_curves_check.toggled.connect(self._sync)
        self.fit_try_button.clicked.connect(self._try)
        self.fit_try_button.setVisible(try_fit is not None)
        # destination and the rest
        self.destination_edit.setText(str(destination))
        self.destination_edit.textChanged.connect(self._sync)
        self.subfolder_check.setText(tr("In a subfolder named “{name}”").format(name=stem))
        self.subfolder_check.setChecked(subfolder)
        self.subfolder_check.toggled.connect(self._sync)
        self.browse_button.clicked.connect(self._browse)
        self.load_settings_button.setVisible(load_settings is not None)
        self.save_settings_button.setVisible(save_settings is not None)
        self.load_settings_button.clicked.connect(self._load)
        self.save_settings_button.clicked.connect(lambda: self._save_settings(self.choices()) if self._save_settings else None)
        self._set_series(series)
        self.series_toggle.toggled.connect(self._toggle_series)
        self.normalize_check.toggled.connect(self._sync_series)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        if for_fitting:
            for widget in (self.data_group, self.pictures_group, self.frames_group, self.fit_group):
                widget.hide()
        self._sync()

    # -- choices in, choices out --------------------------------------------------------

    def _set_choices(self, choices: BatchChoices) -> None:
        self.every_spin.setValue(max(1, int(choices.every)))
        for key, check in self.output_checks.items():
            check.setChecked(bool(getattr(choices, key)))
        for combo, value in ((self.text_format_combo, choices.text_format), (self.image_format_combo, choices.image_format),
                             (self.frame_format_combo, choices.frame_format), (self.fit_profile_combo, choices.fit_profile),
                             (self.fit_model_combo, choices.fit_model)):
            index = combo.findData(value)
            if index >= 0:
                combo.setCurrentIndex(index)
        kind = choices.fit
        if (kind == FIT_PEAKS and (self._gisaxs or not self._fit_targets)) or (kind == FIT_MODEL and not self._gisaxs):
            kind = FIT_NONE
        {FIT_NONE: self.fit_none_radio, FIT_PEAKS: self.fit_peaks_radio, FIT_MODEL: self.fit_model_radio}[kind].setChecked(True)
        self.fit_start_radios.get(choices.fit_start, self.fit_start_radios["previous"]).setChecked(True)
        self.fit_curves_check.setChecked(choices.fit_curves)

    def _fit_kind(self) -> str:
        if self.fit_peaks_radio.isChecked() and self.fit_peaks_radio.isEnabled():
            return FIT_PEAKS
        if self.fit_model_radio.isChecked() and self.fit_model_radio.isEnabled():
            return FIT_MODEL
        return FIT_NONE

    def choices(self) -> BatchChoices:
        if self._for_fitting:
            return BatchChoices(curves=(), tables=False, per_frame=False, fit_input=True, every=self.every_spin.value(),
                                speed=self._speed())
        chosen = self.chosen_curves()
        all_curves = len(chosen) == self.curve_list.count()
        start = next((key for key, radio in self.fit_start_radios.items() if radio.isChecked()), "previous")
        return BatchChoices(
            curves=() if all_curves else chosen, every=self.every_spin.value(),
            text_format=self.text_format_combo.currentData(), image_format=self.image_format_combo.currentData(),
            frame_format=self.frame_format_combo.currentData(), fit=self._fit_kind(),
            fit_profile=self.fit_profile_combo.currentData(), fit_start=start,
            fit_model=self.fit_model_combo.currentData(), fit_curves=self.fit_curves_check.isChecked(),
            speed=self._speed(), image_scale=self.image_scale_combo.currentData() or "auto",
            **{key: self.output_checks[key].isChecked() and not self.output_checks[key].isHidden()
               for key, _text, _tip in OUTPUTS},
        )

    def _speed(self) -> str:
        """The speed chosen; with only one offered (a short batch), the one remembered stays."""
        if self.speed_combo.count() < 2:
            return self._stored_speed
        return self.speed_combo.currentData() or BatchChoices().speed

    # -- curves ----------------------------------------------------------------------

    def _set_curves(self, options: Sequence[tuple[str, str]], wanted: Sequence[str]) -> None:
        self.curve_list.blockSignals(True)
        self.curve_list.clear()
        wanted = set(wanted)
        for key, title in options:
            item = QListWidgetItem(f"{title}   [{key}]", self.curve_list)
            item.setData(Qt.UserRole, key)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if not wanted or key in wanted else Qt.Unchecked)
        self.curve_list.blockSignals(False)

    def set_curve_options(self, options: Sequence[tuple[str, str]]) -> None:
        """New curves (after loading settings); the ticks of the curves that stay are kept."""
        self._set_curves(options, self.chosen_curves())
        self._sync()

    def _check_all(self, checked: bool) -> None:
        self.curve_list.blockSignals(True)
        for index in range(self.curve_list.count()):
            self.curve_list.item(index).setCheckState(Qt.Checked if checked else Qt.Unchecked)
        self.curve_list.blockSignals(False)
        self._sync()

    def chosen_curves(self) -> tuple[str, ...]:
        return tuple(
            self.curve_list.item(index).data(Qt.UserRole)
            for index in range(self.curve_list.count())
            if self.curve_list.item(index).checkState() == Qt.Checked
        )

    # -- the result ------------------------------------------------------------------

    def destination(self) -> Path:
        return Path(self.destination_edit.text().strip())

    def subfolder(self) -> bool:
        return self.subfolder_check.isChecked()

    def target(self) -> Path:
        """Where the files go: the destination, or its subfolder named after the data."""
        return self.destination() / self._stem if self.subfolder() else self.destination()

    def frame_count(self) -> int:
        """Frames the batch will process (every n-th of the listed ones)."""
        return len(range(0, self._frames, max(1, self.every_spin.value())))

    def series(self) -> SeriesCorrection:
        return SeriesCorrection(
            reference_q=self.reference_q_spin.value(),
            half_width=self.half_width_spin.value(),
            align_distance=self.align_check.isChecked(),
            normalize=self.normalize_check.isChecked(),
            target_intensity=self.target_spin.value(),
            per_frame=self.per_frame_radio.isChecked(),
        )

    # -- behaviour -------------------------------------------------------------------

    def _sync(self, *_args) -> None:
        choices = self.choices()
        curves = self.chosen_curves()
        names = output_names(choices, batch=self._stem, frame=self._frame_stem, curve=curves[0] if curves else "region1")
        for key, label in self.output_files.items():
            label.setText("→ " + names[key])
            label.setEnabled(self.output_checks[key].isChecked())
        kind = choices.fit
        self.fit_details.setVisible(kind != FIT_NONE)
        for widget in (self.fit_shape_caption, self.fit_profile_combo):
            widget.setVisible(kind == FIT_PEAKS)
        for widget in (self.fit_model_caption, self.fit_model_combo):
            widget.setVisible(kind == FIT_MODEL)
        if kind == FIT_PEAKS:
            self.fit_targets_label.setText(tr("Peaks: {names} (the regions with a q window; add more with Cuts ▸ Ring)").format(
                names=", ".join(self._fit_targets)))
            curve_name = names["fit_curves"]
        else:
            self.fit_targets_label.setText(tr(
                "The curve for Fitting of each frame (the chosen half of the horizontal cut); about half a minute per frame."
            ))
            curve_name = names["fit_curves"].replace(f"_{curves[0] if curves else 'region1'}_fit", "_model_fit")
        self.fit_curves_file.setText("→ " + curve_name)
        self.fit_curves_file.setEnabled(choices.fit_curves)
        self.fit_file_label.setText(tr("→ {table} (one row per frame) and {plot} (the values against frame)").format(
            table=names["fit"], plot=names["fit_plot"]))
        needs_curves = choices.tables or choices.per_frame
        has_folder = bool(self.destination_edit.text().strip())
        ready = choices.writes_anything and has_folder and (not needs_curves or bool(curves))
        self.export_button.setEnabled(ready)
        pictures = self.output_checks["detector_image"].isChecked() or self.output_checks["q_map_image"].isChecked()
        self.image_scale_combo.setEnabled(pictures)
        self.image_scale_label.setText(
            tr("On screen: {scale}").format(scale=self._screen_scale)
            if self.image_scale_combo.currentData() == "screen" and self._screen_scale else "")
        count = self.frame_count()
        self.every_label.setText(tr("frame(s): {count} of {total}").format(count=count, total=self._frames))
        if self._for_fitting:
            self.export_button.setText(tr("Export and Open in Fitting"))
        else:
            self.export_button.setText(tr("Export 1 Frame") if count == 1 else tr("Export {count} Frames").format(count=count))
        if not has_folder:
            self.target_label.setText(tr("Choose a folder."))
        elif not choices.writes_anything:
            self.target_label.setText(tr("Tick at least one thing to save."))
        elif needs_curves and not curves:
            self.target_label.setText(tr("Tick at least one curve."))
        else:
            text = tr("Files go to {folder} — README.txt there says what each file is").format(folder=self.target())
            room = max(320, self.width() - 160)
            self.target_label.setText(self.target_label.fontMetrics().elidedText(text, Qt.ElideMiddle, room))
            self.target_label.setToolTip(str(self.target()))

    def _try(self) -> None:
        if self._try_fit is not None:
            self._try_fit(self.choices(), self._show_try)

    def _show_try(self, text: str, level: str) -> None:
        self.fit_try_label.setText(text)
        self.fit_try_label.setProperty("gimapRole", {"ok": "success", "warning": "warning"}.get(level, "muted"))
        self.fit_try_label.style().unpolish(self.fit_try_label)
        self.fit_try_label.style().polish(self.fit_try_label)

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, tr("Save the Batch to"), self.destination_edit.text())
        if folder:
            self.destination_edit.setText(folder)

    def _load(self) -> None:
        if self._load_settings is None:
            return
        loaded = self._load_settings()
        if loaded is None:
            return
        settings_text, options, choices = loaded
        self.settings_label.setText(settings_text)
        if choices is not None and not self._for_fitting:
            self._set_choices(choices)
            self._set_curves(options, choices.curves)
        else:
            self.set_curve_options(options)
        self._sync()

    def _set_series(self, series: SeriesCorrection) -> None:
        self.reference_q_spin.setValue(series.reference_q)
        self.half_width_spin.setValue(series.half_width)
        self.align_check.setChecked(series.align_distance)
        self.normalize_check.setChecked(series.normalize)
        self.target_spin.setValue(series.target_intensity)
        (self.per_frame_radio if series.per_frame else self.first_frame_radio).setChecked(True)
        if not series.is_identity:
            self.series_toggle.setChecked(True)
            self._toggle_series(True)
        self._sync_series()

    def _toggle_series(self, shown: bool) -> None:
        self.series_box.setVisible(shown)
        self.series_toggle.setArrowType(Qt.DownArrow if shown else Qt.RightArrow)

    def _sync_series(self, *_args) -> None:
        enabled = self.normalize_check.isChecked()
        for widget in (self.target_spin, self.first_frame_radio, self.per_frame_radio):
            widget.setEnabled(enabled)


__all__ = ["BatchExportDialog"]
