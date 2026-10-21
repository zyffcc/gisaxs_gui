"""The Data step: what the file shown is, which of its frames is on screen, and what is wrong with it.

* **The frame** — the command bar (name and “frame 4 of 12”), the Data step's detail, card and sentence.
* **The source** — above the file list, for a NeXus / HDF5 source: what it consists of (“11 module files ·
  240 frames · stitched”), how its frames are oriented, and every warning of its layout. Warnings that mean
  data are left out put the Data step in the warning state and are said with the frame's messages. While the
  folder watch follows a source that is being written, it also says which frame is the newest complete one:
  the frames behind it are in the count already and are shown a look later. Module files that hold different
  numbers of frames are no warning meanwhile: a detector writes its modules one after the other.
* **The frame bar** — “Frame [4] of 12” above every step, for a source of several frames. The count is
  asked of the source each time (it may have grown since the frame was read).

A file name is never the reason the step grows wider than its panel, and never cut at its edge: inside a
sentence a long name or dataset path is shortened in the middle (the whole name is in the command bar, the
list and the tooltips; what is said of the source is whole in the tooltip of its label).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QSignalBlocker

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.shared.detector_io.nexus_modules import MODULES_UNEQUAL

from ...application import FrameAnalysis
from ...application.ports import source_name
from ..source_texts import (
    FRAMES_TOTAL,
    LISTED_FOLDERS,
    frame_text,
    names_text,
    orientation_text,
    short_name,
    summary_text,
    warning_texts,
)
from ..texts import detector_text

OPEN_HINT = ("Open or drop detector frames (CBF, NeXus/HDF5, TIFF, EDF) or a folder. Files that belong together open "
             "as one row: module files (…_m01.nxs … _m11.nxs) are stitched, a master is read with its data files. "
             "Every frame in them is a frame of the series.")
"""The Data step's sentence while nothing is open. A fixed sentence, no value is filled in: while no frame is
shown nothing is composed again after a switch of the interface language, so it has to be a key of the table as
it stands."""
NOTHING_FOUND = "No new detector frames ({formats}) found."
SENTENCE = "{text}."
"""A warning of a layout (the reader ends none with a full stop) among the messages of a frame."""
GROWING = "Still growing: frame {n} is the newest complete one"
"""Under what a watched source consists of, while its file(s) change at every look (``view_model.growing``)."""
FOUND_BELOW = "{folder} holds no frames itself: its subfolders {names} were opened"
FOUND_BELOW_ONE = "{folder} holds no frames itself: its subfolder {names} was opened"
"""Open Folder on a scan folder whose detectors write one level down (``lmbdp03/``, ``pilatus/``); of many
subfolders the first few are named (“a, b, c and 37 more”)."""


class DataStepMixin:
    """Needs the Data-step widgets, the command bar's file widgets, ``view_model``, ``_intro``,
    ``set_step_state`` and ``_show_file_position``."""

    def source_name(self, path) -> str:
        """What the source of ``path`` is called on screen (a set of module files: its scan, without ``_m01``)."""
        return source_name(path, self.view_model.describe(path))

    def _file_loading(self, path: Path) -> None:
        """Immediately after a file is chosen: its name, and that it is being read."""
        name = self.source_name(path)
        self.file_chip.setText(name)
        self.file_chip.setToolTip(str(path))
        self.file_meta.setText(tr("reading…"))
        self._show_file_position()  # ‹ 3 / 40 ›: only with more than one file (``bindings/file_list.py``)
        self.set_step_state("data", "busy", trf("Reading {name} …", name=name))

    def _data_empty(self) -> None:
        """No frame is shown: the sentence of the empty step, no source."""
        self._intro("data", OPEN_HINT)
        self.data_info_label.setText(tr("No file yet."))
        self._show_source(None)

    def frames_now(self, analysis: FrameAnalysis) -> int:
        """Frames of the source of ``analysis`` as it is now (never fewer than when the frame was read)."""
        return max(int(analysis.frame_count), self.view_model.frame_count(analysis.path))

    def _show_data_info(self, analysis: FrameAnalysis, files: int) -> None:
        """The command-bar meta, the Data step's detail, card and intro. The detector is the name the file gives
        it (else the instrument profile it matched); without either, nothing is said of it."""
        rows, columns = analysis.shape
        shape = f"{rows}×{columns} px"
        detector = detector_text(analysis)
        layout = self.view_model.describe(analysis.path)
        name = source_name(analysis.path, layout)
        frames, shown, total = self.frames_now(analysis), analysis.frame_index + 1, analysis.frame_total
        meta = [part for part in (detector, shape) if part]  # a name and a size: the same in every language
        if frames > 1:
            meta.append(trf("frame {i} of {n}, sum of {total} frames", i=shown, n=frames, total=total)
                        if analysis.summed_frames else frame_text(analysis.frame_index, frames))
        meta = " · ".join(meta)
        self.file_chip.setText(name)
        self.file_chip.setToolTip(str(analysis.path))
        self.file_meta.setText(meta)
        expected = self._expected(analysis.path)
        self.set_step_state("data", "warn" if warning_texts(layout, notes=False, without=expected) else "ok", meta)
        metadata = analysis.metadata or {}
        lines = [trf("Detector: {name}", name=analysis.detector_name)] if analysis.detector_name else []
        geometry = analysis.geometry
        pixel = geometry.pixel_size_x_m if geometry is not None else metadata.get("pixel_size_x_m")
        lines.append(trf("Frame: {rows} × {columns} pixels, {size} µm pixels", rows=rows, columns=columns,
                         size=f"{pixel * 1e6:g}") if pixel else
                     trf("Frame: {rows} × {columns} pixels", rows=rows, columns=columns))
        if frames <= 1:
            lines.append(trf("Frames in the file: {n}", n=frames))
        elif analysis.summed_frames:
            lines.append(trf("Frames in the file: {n} (showing {i}, sum of {total} frames)", n=frames, i=shown, total=total))
        else:
            lines.append(trf("Frames in the file: {n} (showing {i})", n=frames, i=shown))
        header = []
        if metadata.get("energy_kev"):
            header.append(f"{float(metadata['energy_kev']):g} keV")
        if metadata.get("header_distance_m"):
            header.append(trf("distance {value} mm", value=f"{float(metadata['header_distance_m']) * 1e3:g}"))
        if metadata.get("exposure_time_s"):
            header.append(trf("exposure {value} s", value=f"{float(metadata['exposure_time_s']):g}"))
        if header:
            lines.append(trf("File header: {items}", items=", ".join(header)))
        lines.append(trf("Files listed: {n}", n=files))
        listed = self.view_model.listed_frames()
        if files > 1 and listed > files:  # the total Batch Export and the Series map work on
            lines.append(trf("Frames listed: {n}", n=listed))
        self.data_info_label.setText("\n".join(lines))
        self.frame_total_label.setText(trf(FRAMES_TOTAL, n=frames))  # (again after a switch of the language)
        values = dict(name=short_name(name), detector=detector, shape=shape, frames=frames)
        if detector:
            self._intro("data", "{name}: {detector}, {shape}, {frames} frames." if frames > 1 else
                        "{name}: {detector}, {shape}.", **values)
        else:
            self._intro("data", "{name}: {shape}, {frames} frames." if frames > 1 else "{name}: {shape}.", **values)
        self._show_source(layout, analysis.path)
        shown = self.view_model.current_path  # its row says the same (frames, files, warnings) as the step
        if shown is not None and str(shown).casefold() == str(analysis.path).casefold():
            self._write_row(self.view_model.state.current_index, shown)

    def _expected(self, path) -> tuple:
        """Warnings of the source of ``path`` that are to be expected right now and are not said as problems:
        while the folder watch sees it grow, its module files hold different numbers of frames at most looks
        (they are written one after the other). Said again as soon as the source rests like that."""
        return (MODULES_UNEQUAL,) if path is not None and self.view_model.growing(path) else ()

    def _growing_note(self, path) -> Optional[str]:
        """While the folder watch follows the source of ``path`` and it is still being written: which of its
        frames is the newest complete one (``None``: it rests, or every frame counted is complete)."""
        model = self.view_model
        newest = model.followed_frame(path) if path is not None and model.growing(path) else None
        if newest is None or newest + 1 >= model.known_frame_count(path):
            return None
        return trf(GROWING, n=newest + 1)

    def _show_source(self, layout, path=None) -> None:
        """Above the file list: what the source shown consists of and how its frames are oriented, then (in
        the warning colour) what is left out of it or may be wrong. Nothing for a frame of another format.
        Long names are shortened in the sentences; the tooltips have them whole. ``path``: the source, to say
        which frame is the newest complete one while the folder watch follows it as it grows."""
        if layout is None:
            self.source_label.hide()
            self.source_warning_label.hide()
            return
        lines = [summary_text(layout)]
        if layout.kind != "no_image":
            lines.append(orientation_text(layout))
        growing = self._growing_note(path)
        if growing is not None:
            lines.append(growing)
        self.source_label.setText("\n".join(lines + warning_texts(layout, notes=True, narrow=True)))
        self.source_label.setToolTip("\n".join(lines + warning_texts(layout, notes=True)))
        self.source_label.show()
        expected = self._expected(path)
        problems = warning_texts(layout, notes=False, without=expected)
        self.source_warning_label.setText("\n".join(warning_texts(layout, notes=False, narrow=True, without=expected)))
        self.source_warning_label.setToolTip("\n".join(problems))
        self.source_warning_label.setVisible(bool(problems))

    def _source_warnings(self, path, without=()) -> list[str]:
        """What is left out of the source of ``path`` or may be wrong with it, as sentences in the interface
        language (with their full stop: they are said beside the frame's own messages). ``without``: warnings
        left out because the line says the same in other words."""
        layout = self.view_model.describe(path)
        without = (*without, *self._expected(path))
        return [trf(SENTENCE, text=text) for text in warning_texts(layout, notes=False, without=without)]

    # -- the frame bar -------------------------------------------------------------------

    def _update_frame_controls(self, analysis: Optional[FrameAnalysis]) -> None:
        """“Frame [4] of 12” above the steps, for a source of several frames; then the stepper's position."""
        count = self.frames_now(analysis) if analysis is not None else 1
        self.frame_bar.setVisible(count > 1)
        if analysis is not None:
            self._show_frame_count(count, analysis.frame_index)
        self._show_file_position()

    def _show_frame_count(self, count: int, frame_index: int) -> None:
        with QSignalBlocker(self.frame_spin):
            self.frame_spin.setMaximum(max(1, int(count)))
            self.frame_spin.setValue(int(frame_index) + 1)
        self.frame_total_label.setText(trf(FRAMES_TOTAL, n=int(count)))

    # -- what opening found ----------------------------------------------------------------

    def _found_below(self, paths, added) -> Optional[str]:
        """When a folder asked for holds no frames itself and those listed come from its subfolders: the
        sentence that says so (``None``: every frame is where it was asked for)."""
        for item in paths:
            folder = Path(item).expanduser()
            if not folder.is_dir():
                continue
            folder = folder.resolve()
            below = [path for path in added if path.parent != folder and path.parent.parent == folder]
            if below:
                names = sorted({path.parent.name for path in below})
                return trf(FOUND_BELOW if len(names) > 1 else FOUND_BELOW_ONE, folder=folder.name,
                           names=names_text(names, LISTED_FOLDERS))
        return None

    def _say_found_below(self, paths, added) -> None:
        """Say (a notice: the status line is about the frame being read) that the frames listed were found
        one level below the folder that was opened, and in which subfolders."""
        text = self._found_below(paths, added)
        if text is not None:
            show_toast(self.window(), text, level="info")


__all__ = ["DataStepMixin", "FOUND_BELOW", "FOUND_BELOW_ONE", "GROWING", "NOTHING_FOUND", "OPEN_HINT", "SENTENCE"]
