"""The Batch Export dialog: every listed frame with the current settings, saved as chosen.

The layout is ``views/batch_export_view.py``. Every option shows the file it writes (with its
extension, following the chosen formats), so it is clear before the export what each file will be;
the same names go into the ``README.txt`` of the folder. The dialog remembers nothing itself: the
page passes the last choices in and stores what was accepted. ``for_fitting`` turns it into Send
Series to Fitting: only the Fitting input of each frame is written — into a folder of its own,
``<name>_fitting``, never over a Batch Export of the same data —, then the series opens in Fitting.

For GISAXS one combo, *Halves of the cut*, says what a fit takes of the signed horizontal cut — the mean of
both halves, or one half. It starts on the Cuts step's choice (``halves``: the page gives it as it is); when
that is “both halves on |qy|”, which is for looking and no choice of a fit, on the halves of the last batch
(``choices``), like every other option. The particle model of the batch and its trial on the frame on screen
are fitted on it, and every Fitting input says it (the file itself keeps both halves); in Send Series to
Fitting it is the choice the dialog is about.

Before Export the dialog says what the batch will be (``batch_summary.py``): the frames — sums as sums, a
last sum of fewer frames named —, where the cut of every frame is taken, the folder and what it holds
already, and a summary with the rough size and time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QDialog, QListWidgetItem

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
    fit_halves,
    output_names,
)
from .batch_summary import BatchFolderMixin, Cost
from .batch_texts import HALVES_FITTED, HALVES_NOTES, HALVES_UNUSED, MODEL_LINES, SENT_GIWAXS, SENT_TO_FITTING
from .views.batch_export_view import (
    CURVE_LIST_HEIGHT,
    FILE_MARGIN,
    FILE_MIN_WIDTH,
    LINE_MARGIN,
    OUTPUTS,
    SCREEN_SHARE,
    BatchExportView,
)

FIT_PASSES = 3
"""How often ``fit_to_screen`` measures the dialog again (once more is enough when wrapped lines were squeezed)."""
LoadSettings = Callable[[], Optional[tuple[str, list, Optional[BatchChoices]]]]
TryFit = Callable[[BatchChoices, Callable[[str, str], None]], None]


class BatchExportDialog(BatchFolderMixin, QDialog, BatchExportView):
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
        halves: Optional[str] = None,
        frames_text: str = "",
        sum_size: int = 1,
        notes: Sequence[str] = (),
        cut_text: Optional[Callable[[], str]] = None,
        pin_cuts: Optional[bool] = None,
        set_pin_cuts: Optional[Callable[[bool], None]] = None,
        cost: Optional[Cost] = None,
        fit_keys: Sequence[str] = (),
        beside: Optional[Path] = None,
        carried: bool = False,
        reading: bool = False,
        parent=None,
    ):
        """``frames``: what the batch has to do, one for each request (a sum of N frames is one);
        ``frames_text``: how to say them (default: “N frames from M files”); ``sum_size``: the N of a sum;
        ``notes``: what must be known about the frames before they are exported. ``cut_text()``: where the
        automatic cut of every frame is taken; ``pin_cuts`` (``None``: no such choice) and ``set_pin_cuts``:
        one cut for every frame, and how to change that. ``cost(choices, frames)`` → bytes per frame, bytes in
        all, seconds. ``fit_keys``: the curve keys of ``fit_targets``. ``beside``: the folder next to the data;
        ``carried``: ``destination`` is the folder of a batch of other data. ``reading``: the frame on screen is
        being read again (Export waits until ``frame_shown`` says it is there)."""
        super().__init__(parent)
        self.setup_batch_export()
        self._sized = False
        self._stem = stem
        self._frame_stem = frame_stem or "frame"
        self._frames = int(frames)
        self._for_fitting = for_fitting
        self._gisaxs = gisaxs
        self._load_settings = load_settings
        self._save_settings = save_settings
        self._try_fit = try_fit
        self._fit_targets = list(fit_targets)
        self._fit_keys = list(fit_keys)
        self._curve_titles: dict[str, str] = {}
        self.setWindowTitle(tr("Send Series to Fitting" if for_fitting else "Batch Export"))
        if frames_text:
            what = frames_text
        elif frames == 1:
            what = tr("1 frame from 1 file")
        else:
            what = (tr("{frames} frames from {files} files") if files > 1 else tr("{frames} frames from 1 file")).format(
                frames=frames, files=files)
        self.frames_label.setText(what + " — " + tr("each reduced with the settings below."))
        self.settings_label.setText(settings_text)
        for combo, options in ((self.text_format_combo, TEXT_FORMATS), (self.image_format_combo, IMAGE_FORMATS),
                               (self.frame_format_combo, FRAME_FORMATS)):
            for key, text in options.items():
                combo.addItem(tr(text), key)
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
        self._set_choices(choices)
        if gisaxs:  # the halves of the Cuts step; “both halves on |qy|” there is for looking: those of the last batch
            self.halves_combo.setCurrentIndex(max(0, self.halves_combo.findData(fit_halves(halves, choices.fit_halves))))
        self.halves_combo.currentIndexChanged.connect(self._sync)
        self.halves_group.setVisible(gisaxs or for_fitting)
        for widget in (self.halves_caption, self.halves_combo):
            widget.setVisible(gisaxs)  # an I(q) has no halves
        self.halves_note.setVisible(gisaxs or for_fitting)  # (a GIWAXS series sent: what Fitting fits it with)
        self.sent_label.setVisible(for_fitting)
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
        # the cut of every frame, the folder and the rest (``batch_summary.py``)
        self._setup_folder(destination=destination, subfolder=subfolder, beside=beside, carried=carried, cost=cost,
                           cut_text=cut_text, pin_cuts=pin_cuts, set_pin_cuts=set_pin_cuts, notes=notes, sum_size=sum_size,
                           reading=reading)
        # Restoring formats can emit currentIndexChanged: the summary needs the complete folder state first.
        for combo in (self.text_format_combo, self.image_format_combo, self.frame_format_combo, self.image_scale_combo):
            combo.currentIndexChanged.connect(self._sync)
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
            self.setMinimumHeight(0)  # one group: as tall as it is (``fit_to_screen``), not a mostly empty window
        self._sync()

    # -- choices in, choices out --------------------------------------------------------

    def _set_choices(self, choices: BatchChoices) -> None:
        self.every_spin.setValue(max(1, int(choices.every)))
        for key, check in self.output_checks.items():
            check.setChecked(bool(getattr(choices, key)))
        for combo, value in ((self.text_format_combo, choices.text_format), (self.image_format_combo, choices.image_format),
                             (self.frame_format_combo, choices.frame_format), (self.fit_profile_combo, choices.fit_profile),
                             (self.fit_model_combo, choices.fit_model), (self.halves_combo, choices.fit_halves)):
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

    def halves(self) -> str:
        """What a fit takes of the cut: ``mean``, ``positive`` or ``negative`` (the combo; for GIWAXS it is hidden
        and keeps the stored choice, which changes nothing there)."""
        return fit_halves(self.halves_combo.currentData())

    def choices(self) -> BatchChoices:
        if self._for_fitting:
            return BatchChoices(curves=(), tables=False, per_frame=False, fit_input=True, every=self.every_spin.value(),
                                speed=self._speed(), fit_halves=self.halves())
        chosen = self.chosen_curves()
        all_curves = len(chosen) == self.curve_list.count()
        start = next((key for key, radio in self.fit_start_radios.items() if radio.isChecked()), "previous")
        return BatchChoices(
            curves=() if all_curves else chosen, every=self.every_spin.value(),
            text_format=self.text_format_combo.currentData(), image_format=self.image_format_combo.currentData(),
            frame_format=self.frame_format_combo.currentData(), fit=self._fit_kind(),
            fit_profile=self.fit_profile_combo.currentData(), fit_start=start,
            fit_model=self.fit_model_combo.currentData(), fit_curves=self.fit_curves_check.isChecked(),
            fit_halves=self.halves(), speed=self._speed(), image_scale=self.image_scale_combo.currentData() or "auto",
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
        self._curve_titles = {key: title for key, title in options}
        for key, title in options:
            item = QListWidgetItem(f"{title}   [{key}]", self.curve_list)
            item.setData(Qt.UserRole, key)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if not wanted or key in wanted else Qt.Unchecked)
        self.curve_list.blockSignals(False)
        self._fit_curve_list()

    def _fit_curve_list(self) -> None:
        """As tall as its rows (one row's height when empty), at most ``CURVE_LIST_HEIGHT``: two curves do not
        take the room of eight, and push the other groups below the fold."""
        listing = self.curve_list
        rows = sum(max(0, listing.sizeHintForRow(row)) for row in range(listing.count()))
        rows = rows or listing.fontMetrics().height() + 8
        listing.setFixedHeight(min(CURVE_LIST_HEIGHT, rows + 2 * listing.frameWidth() + 2))

    # -- the first size ----------------------------------------------------------------

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().showEvent(event)
        if not self._sized:  # once: a size the person gives it afterwards stays
            self._sized = True
            self._fit_curve_list()  # the rows' height with the style sheet applied
            self._sync()  # … and the file names with its fonts
            self.fit_to_screen()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        if getattr(self, "_sized", False):  # a new width: the file names take the room it gives them
            self._sync()

    def fit_to_screen(self, room: Optional[int] = None) -> int:
        """Open as tall as the options need — Pictures, Converted detector frames and Fitting in view, not under a
        scroll bar — at most ``SCREEN_SHARE`` of the screen's free height (``room``, px: that height); the width
        stays. Returns the height given.

        What stands around the options is measured as it is laid out now; in a dialog shorter than it needs, the
        lines that wrap (an earlier export in the folder: two more lines) are squeezed too, so the height is
        measured again once they have their room (at most ``FIT_PASSES`` times)."""
        screen = self.screen() or QApplication.primaryScreen()
        if room is None:
            room = int(SCREEN_SHARE * screen.availableGeometry().height()) if screen is not None else self.height()
        self.setMinimumHeight(min(self.minimumHeight(), int(room)))  # a small screen: the dialog still fits on it
        layout = self.layout()
        if layout is not None:
            layout.activate()
        scroll, body = self.options_scroll, self.options_scroll.widget()
        if scroll.viewport().height() <= 1:  # squeezed below what stands around the options: the sum below would be short
            self.resize(self.width(), max(self.height(), self.sizeHint().height()))
            if layout is not None:
                layout.activate()
        before = height = self.height()
        for _ in range(FIT_PASSES):
            width = max(1, scroll.viewport().width())
            needed = body.heightForWidth(width) if body.hasHeightForWidth() else -1
            needed = needed if needed > 0 else body.sizeHint().height()
            wanted = max(self.minimumHeight(), min(int(room), self.height() - scroll.viewport().height() + needed))
            if wanted == height and (wanted >= int(room) or scroll.viewport().height() >= needed):
                break  # (as tall as it needs, or as the screen allows)
            height = wanted
            self.resize(self.width(), height)
            if layout is not None:
                layout.activate()
        if self.isVisible() and screen is not None and height != before:  # still centred where it opened, on screen
            free, frame = screen.availableGeometry(), self.frameGeometry()
            top = frame.top() - (height - before) // 2
            top = max(free.top(), min(top, free.bottom() + 1 - frame.height()))
            self.move(frame.left(), top)
        return height

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

    def _file_room(self) -> int:
        """The width (px) a file name may take beside its option: what the dialog leaves next to the widest one."""
        widest = max((check.sizeHint().width() for check in self.output_checks.values() if not check.isHidden()), default=0)
        return max(FILE_MIN_WIDTH, self.width() - widest - FILE_MARGIN)

    @staticmethod
    def _show_file(label, text: str, room: int) -> None:
        """``text`` (a file name) on ``label``, shortened in its middle when it is longer than ``room`` px — the
        ending says which file it is; the whole name is then the tooltip. A long name is not to widen the dialog."""
        label.setText(label.fontMetrics().elidedText(text, Qt.ElideMiddle, max(FILE_MIN_WIDTH, int(room))))
        label.setToolTip(text if label.text() != text else "")

    def _sync(self, *_args) -> None:
        choices = self.choices()
        curves = self.chosen_curves()
        names = output_names(choices, batch=self._stem, frame=self._frame_stem, curve=curves[0] if curves else "region1",
                             fit_key=self._fit_keys[0] if self._fit_keys else None)  # the files as they are written
        room = self._file_room()
        for key, label in self.output_files.items():
            self._show_file(label, "→ " + names[key], room)
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
        else:
            self.fit_targets_label.setText(tr(MODEL_LINES[choices.fit_halves]))
        curve_name = names["fit_curves"]  # of the first peak's region (one file per peak), or of the model
        # the halves: used by the model fit and said in the fit inputs; otherwise the choice changes nothing
        written = self._for_fitting or choices.fit_input  # the curve for Fitting: only then is there a file to speak of
        used = written or kind == FIT_MODEL
        for widget in (self.halves_caption, self.halves_combo):
            widget.setEnabled(used)
        note = (HALVES_NOTES if written else HALVES_FITTED)[choices.fit_halves]
        self.halves_note.setText(tr(note if used else HALVES_UNUSED) if self._gisaxs else tr(SENT_GIWAXS))
        line = self.width() - LINE_MARGIN
        self.sent_label.setText(tr(SENT_TO_FITTING).format(
            file=self.sent_label.fontMetrics().elidedText(names["fit_input"], Qt.ElideMiddle, max(FILE_MIN_WIDTH, line))))
        self.sent_label.setToolTip(names["fit_input"])
        self._show_file(self.fit_curves_file, "→ " + curve_name, line)
        self.fit_curves_file.setEnabled(choices.fit_curves)
        self.fit_file_label.setText(tr("→ {table} (one row per frame) and {plot} (the values against frame)").format(
            table=names["fit"], plot=names["fit_plot"]))
        # the folder, and what it holds already; and the frame on screen, whose cut every frame gets, has been read
        self.export_button.setEnabled(self._sync_folder(choices, curves) and not self.reading())
        pictures = self.output_checks["detector_image"].isChecked() or self.output_checks["q_map_image"].isChecked()
        self.image_scale_combo.setEnabled(pictures)
        self.image_scale_label.setText(
            tr("On screen: {scale}").format(scale=self._screen_scale)
            if self.image_scale_combo.currentData() == "screen" and self._screen_scale else "")
        self.image_scale_label.setVisible(bool(self.image_scale_label.text()))  # no empty line in Pictures
        count = self.frame_count()
        self.every_label.setText((tr("sum(s): {count} of {total}") if self._sum_size > 1 else tr("frame(s): {count} of {total}"))
                                 .format(count=count, total=self._frames))
        if self._for_fitting:
            self.export_button.setText(tr("Export and Open in Fitting"))
        elif self._sum_size > 1:  # a sum of N frames is one thing exported: not called a frame
            self.export_button.setText(tr("Export 1 Sum") if count == 1 else tr("Export {count} Sums").format(count=count))
        else:
            self.export_button.setText(tr("Export 1 Frame") if count == 1 else tr("Export {count} Frames").format(count=count))
        self._sync_summary(choices, curves)

    def _try(self) -> None:
        if self._try_fit is not None:
            self._try_fit(self.choices(), self._show_try)

    def _show_try(self, text: str, level: str) -> None:
        self.fit_try_label.setText(text)
        self.fit_try_label.setProperty("gimapRole", {"ok": "success", "warning": "warning"}.get(level, "muted"))
        self.fit_try_label.style().unpolish(self.fit_try_label)
        self.fit_try_label.style().polish(self.fit_try_label)

    def _load(self) -> None:
        if self._load_settings is None:
            return
        loaded = self._load_settings()
        if loaded is None:
            return
        settings_text, options, choices = loaded
        self.settings_label.setText(settings_text)
        self._show_cut()  # another set-up: its cuts
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
