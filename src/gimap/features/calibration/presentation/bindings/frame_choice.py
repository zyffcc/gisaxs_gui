"""Which frame of a multi-frame calibration image is calibrated.

A NeXus / HDF5 measurement of a standard may hold several frames. The dialog reads the first one, says
so ("frame 1 of 12") and shows a row under the file: another frame, or the sum of all frames (counts add;
a pixel that is invalid in any frame stays invalid). Changing the choice reads the image again, so the
preview and the calibration are always of the frames named. The row is hidden for a file with one frame.

An imported calibration shows the frames it was made on: for a sum, the import shows the first frame at
once and the sum is read on the reader thread (the imported result stays), so that a new run or a manual
refinement works on the same image.
"""

from __future__ import annotations

from PyQt5.QtCore import QSignalBlocker

from src.gimap.app.presentation.i18n import language_changed, trf

from ...application import CalibrationCancelledError
from ..workers import ImportedSumWorker

FRAME_MODE_ONE, FRAME_MODE_SUM = 0, 1
FRAME_OF = "frame {frame} of {count}"
FRAMES_SUMMED = "{count} frames summed"
FRAME_RANGE_SUMMED = "frames {first}–{last} of {count} summed"
TOTAL = "of {count}"
SUMMING = "Summing frames: {done} of {total}"
# An imported calibration made on summed frames whose sum could not be read to its end.
NOT_THE_SUM = (
    "{imported}. The calibration was made on {count} frames summed from frame {frame} on: "
    "frame {frame} alone is shown."
)


def frames_of(image) -> tuple[int, int, int]:
    """``(frames in the file, first frame read (1-based), frames summed)`` of a loaded image."""
    metadata = getattr(image, "metadata", None) or {}

    def number(key: str, default: int) -> int:
        try:
            return int(metadata.get(key, default) or default)
        except (TypeError, ValueError):
            return default

    total = max(1, number("frame_count", 1))
    summed = min(total, max(1, number("summed_frames", 1)))
    first = min(total, max(0, number("frame_index", 0)) + 1)
    return total, first, summed


def frames_text(image) -> str:
    """What the image is of a multi-frame file ('' for a file with one frame), in the interface language."""
    total, first, summed = frames_of(image)
    if total <= 1:
        return ""
    if summed <= 1:
        return trf(FRAME_OF, frame=first, count=total)
    if summed == total:
        return trf(FRAMES_SUMMED, count=total)
    return trf(FRAME_RANGE_SUMMED, first=first, last=first + summed - 1, count=total)


class FrameChoiceMixin:
    """Own the frame row of the Input section."""

    def _bind_frame_choice(self) -> None:
        self._image_request: tuple[str, str | None] | None = None
        """Path and dataset the image shown was read with (``None``: not read by this dialog)."""
        self._loading_request: tuple[str, str | None] | None = None
        self._frames_total = 1
        self.frame_mode_combo.currentIndexChanged.connect(self._frame_choice_changed)
        self.frame_spin.valueChanged.connect(self._frame_choice_changed)
        language_changed().connect(self._refresh_frame_texts)

    def _show_frame_choice(self, image) -> None:
        """The row as the image was read: its frame, or the sum (no reload is started by this)."""
        total, first, summed = frames_of(image)
        blockers = (QSignalBlocker(self.frame_mode_combo), QSignalBlocker(self.frame_spin))
        self.frame_spin.setMaximum(total)
        self.frame_spin.setValue(first)
        self.frame_mode_combo.setCurrentIndex(FRAME_MODE_SUM if summed > 1 else FRAME_MODE_ONE)
        del blockers
        self._frames_total = total
        self.frames_row.setVisible(total > 1)
        self._refresh_frame_texts()

    def _refresh_frame_texts(self, *_args) -> None:
        total = getattr(self, "_frames_total", 1)
        self.frames_total_label.setText(trf(TOTAL, count=total))
        self.frame_spin.setEnabled(
            self.frame_mode_combo.currentIndex() == FRAME_MODE_ONE and self.frame_mode_combo.isEnabled()
        )

    def _show_imported_frames(self) -> None:
        """The image of an imported calibration: read by the import, the first frame of its block."""
        self._image_request = None
        if self.image is None:
            self.frames_row.setVisible(False)
        else:
            self._show_frame_choice(self.image)

    def _read_imported_frames(self, imported_text: str) -> None:
        """After an import: say which frames the image is, and when the calibration was made on a sum
        read that sum on the reader thread. ``imported_text``: the stage line of the import."""
        image = self.image
        first, summed = self.view_model.imported_frames()
        total, _shown, shown_sum = frames_of(image) if image is not None else (1, 1, 1)
        busy = self._load_thread is not None and self._load_thread.isRunning()
        if image is None or total <= 1 or summed <= 1 or shown_sum >= min(summed, total) or busy:
            self._say_imported_frames(imported_text)
            return
        blocker = QSignalBlocker(self.frame_mode_combo)
        self.frame_mode_combo.setCurrentIndex(FRAME_MODE_SUM)  # what is being read
        del blocker
        self._set_running(True)
        self.job_status.set_state("running", trf(SUMMING, done=0, total=summed), progress=0.0)
        self._start_reader(
            ImportedSumWorker(self.view_model),
            lambda image: self._imported_sum_read(image, imported_text),
            lambda exc: self._imported_sum_failed(exc, imported_text, first, summed),
        )

    def _say_imported_frames(self, imported_text: str) -> None:
        frames = frames_text(self.image) if self.image is not None else ""
        self.stage_label.setText(imported_text + (" | " + frames if frames else ""))

    def _imported_sum_read(self, image, imported_text: str) -> None:
        """The sum has arrived: it is the image now; the imported result and its rings stay."""
        self.image = image
        self._image_request = None
        self._preview_cache.clear()
        self._show_frame_choice(image)
        self._say_imported_frames(imported_text)
        self.job_status.set_state("succeeded", self.stage_label.text(), progress=1.0)
        if self.result is not None:
            self._show_candidate(self.result.selected_candidate)
        self._set_running(False)

    def _imported_sum_failed(self, exc: Exception, imported_text: str, first: int, summed: int) -> None:
        """Stopped or unreadable: the first frame stays, and the line says what the calibration was made on."""
        if self.image is not None:
            self._show_frame_choice(self.image)
        state = "cancelled" if isinstance(exc, CalibrationCancelledError) else "failed"
        self.job_status.set_state(  # its message is the stage line
            state, trf(NOT_THE_SUM, imported=imported_text, count=summed, frame=first + 1), progress=0.0
        )
        self._set_running(False)

    def _set_running(self, running: bool) -> None:
        super()._set_running(running)
        self.frame_mode_combo.setEnabled(not running)
        self._refresh_frame_texts()

    def _frame_choice_changed(self, *_args) -> None:
        """Read the chosen frame, or the sum, of the image shown."""
        self._refresh_frame_texts()
        image = self.image
        if image is None or (self._load_thread is not None and self._load_thread.isRunning()):
            return
        path, dataset = self._image_request or (str(image.source_path), None)
        if self.frame_mode_combo.currentIndex() == FRAME_MODE_SUM:
            self.load_image(path, dataset, frame=0, sum_frames=getattr(self, "_frames_total", 1))
        else:
            self.load_image(path, dataset, frame=self.frame_spin.value() - 1)

    def _frames_read(self, done: int, total: int) -> None:
        self.job_status.set_state(
            "running", trf(SUMMING, done=done, total=total), progress=done / max(1, total)
        )

    def cancel_calibration(self) -> None:
        """Cancel also stops a sum of frames that is being read (before its next frame)."""
        if self._load_worker is not None:
            self._load_worker.cancel()
        super().cancel_calibration()


__all__ = ["FRAME_MODE_ONE", "FRAME_MODE_SUM", "FrameChoiceMixin", "frames_of", "frames_text"]
