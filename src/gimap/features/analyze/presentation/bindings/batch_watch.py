"""Export of the frame on screen, figures and folder watching for Analyze (batches: ``batch_export.py``; Send to
Fitting: ``fit_handover.py``).

Folder watching. A timer looks at the folder every two seconds. The look itself (listing the folder,
measuring its files, reading the layout of a NeXus file that changed) runs on a thread of its own
(``_watch_tick``), so a large folder or a slow disk never holds the window up; what it found is applied on
the GUI thread (``_watch_found``). ``poll_watch`` does both at once (tests, and callers that want the answer).

What a look can find:

* new files: listed, the newest shown, exported when Export New Frames While Watching is on;
* frames added to a listed source: the same, frame by frame, each frame once;
* a source that grows at every look (a detector writing a frame a second for minutes): listed and followed
  while it grows. Each look brings the frames it already held at the look before; the frame on screen is the
  newest of those, not the newest of the file, which may still be being written (with a sum of N frames: the
  N shown end at it);
* a listed source whose files changed without a new frame (a module file that arrived, a file written again,
  a writer that closed its file): its row is written again, and when it is on screen its frame is read again;
* files that turned out to be one source (a master that arrived after its data files): one row;
* a source that cannot be read yet because its writer holds it: said in the status line (once, also while
  other sources bring frames), asked again at the next look. A file that is marked as being written and has
  not changed for a while was left by a writer that ended: the status line says so, as a warning, with how the
  file can be repaired;
* a set of module files with a module missing, or one that may still be arriving: listed and shown, exported
  once it is whole and has rested (``sources_model.SET_RESTS``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QSignalBlocker
from PyQt5.QtWidgets import QFileDialog

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.app.presentation.task_runner import TaskRunner
from src.gimap.shared.figures import FIGURE_FILE_FILTER

from ...application import cannot_write, export_stem
from ...application.ports import parts_missing, source_name
from ..source_texts import new_frames_text
from ..texts import left_unfinished, message_text
from .watch_series import WatchSeriesMixin

WATCH_WAITS = "Watching {folder}: {reason}"
"""A file of the watched folder cannot be read yet; ``reason`` is the reader's sentence."""
EXPORT_WAITS = "{name} is not exported yet: {reason}"
"""Export New Frames While Watching holds a source back: a part of the detector is missing."""
SET_ARRIVING = "{name} is not exported yet: its module files are still arriving ({n} so far)"
"""… or its module files may still be arriving one after the other: it is exported once no more arrive."""
SET_FEWER = "{name} is not exported yet: {n} of the {total} module files that the other scans in this folder have are here"
AUTO_EXPORT_OFF = ("Export New Frames While Watching is off: the folder of the data cannot be written ({error}) and no "
                   "other folder was chosen.")
"""The folder of the data takes no file and the question for another one was cancelled (``_auto_export_folder``)."""


class BatchWatchMixin(WatchSeriesMixin):
    """Own everything that writes results or follows new files."""

    def export_current(self) -> None:
        """Export next to the data; if that folder takes no file, into the session's folder (asked for once:
        ``write_elsewhere``, the one Send to Fitting uses too)."""
        try:
            try:
                written = self.view_model.export()
            except OSError as exc:
                written, _folder = self.write_elsewhere(lambda folder: self.view_model.export(folder), exc)
        except (ValueError, OSError) as exc:
            self._status(trf("Export failed: {error}", error=message_text(exc)), "error")
            return
        self.notify_written(exported_text(written), written[0].parent)

    def _auto_export_folder(self, destination: Path) -> Optional[Path]:
        """Where Export New Frames While Watching writes now: next to the data, else the session's folder
        (``write_elsewhere``: asked for once). ``None`` when no folder takes the files: the export is switched off
        and the status line says why — a question at every look of the watch would be no help."""

        def takes_files(folder: Path) -> Path:
            reason = cannot_write(folder)
            if reason is not None:
                raise OSError(reason)
            return folder

        try:
            return takes_files(destination)
        except OSError as exc:
            refused = exc
        looking, self._watch_looking = getattr(self, "_watch_looking", False), True  # (no look while it asks)
        try:
            folder, _kept = self.write_elsewhere(takes_files, refused)
        except OSError as exc:
            with QSignalBlocker(self.auto_export_check):  # (off for now; the choice remembered stays as it was)
                self.auto_export_check.setChecked(False)
            why = str(exc)
            self._watch_says(lambda: trf(AUTO_EXPORT_OFF, error=message_text(why)), "warning")
            show_toast(self, self.status_text(), level="warning")  # (the line goes to the frame shown next)
            return None
        finally:
            self._watch_looking = looking
        return folder

    # -- figures ---------------------------------------------------------------------

    FIGURE_FILTER = FIGURE_FILE_FILTER

    def _figure_path(self, title: str, suffix: str) -> Optional[Path]:
        analysis = self.view_model.state.analysis
        folder = self.view_model.default_export_dir() or Path(self._last_folder or ".")
        # Named as the frame's exports are: a set of module files by its scan, a frame of several by _frameNNNN.
        stem = export_stem(analysis) if analysis is not None else "figure"
        path, _ = QFileDialog.getSaveFileName(
            self, tr(title), str(folder / f"{stem}_{suffix}.png"), self.FIGURE_FILTER
        )
        return Path(path) if path else None

    def save_image(self) -> None:
        state = self.detector_view.display_state()
        if state is None:
            self._status(tr("Open a frame first."), "warning")
            return
        path = self._figure_path("Save Image", ("detector", "qmap", "cake")[self.view_combo.currentIndex()])
        if path is None:
            return
        image = state.pop("image")
        try:
            written = self.view_model.save_image_figure(path, image, **state)
        except (ValueError, OSError) as exc:
            self._status(trf("Could not save the image: {error}", error=message_text(exc)), "error")
            return
        self.notify_written(trf("Saved {name}", name=written.name), written.parent)

    def save_view_data(self) -> None:
        """The q map or the cake as a CSV table with its axes (the view shown)."""
        index = self.view_combo.currentIndex()
        if index == 1:
            self.save_map_data()
            return
        cached = getattr(self, "_cake", None)
        analysis = self.view_model.state.analysis
        if index != 2 or cached is None or cached[0] is not analysis:
            self._status(tr("Show the q map or the cake to save it as data."), "warning")
            return
        folder = self.view_model.default_export_dir() or Path(self._last_folder or ".")
        path, _ = QFileDialog.getSaveFileName(
            self, tr("Save Cake Data"), str(folder / f"{export_stem(analysis)}_cake.csv"),
            "CSV table (*.csv)")
        if not path:
            return
        try:
            written = self.view_model.export_cake(analysis, cached[1], Path(path))
        except (ValueError, OSError) as exc:
            self._status(trf("Could not save the cake: {error}", error=message_text(exc)), "error")
            return
        self.notify_written(trf("Saved {name}", name=written.name), written.parent)

    def save_plot_data(self, which: str) -> None:
        """The curves of the upper or lower plot, one CSV each (with the JSON record)."""
        keys = self._plot_keys.get("top" if which == "upper" else "bottom", [])
        destination = self.view_model.default_export_dir()
        if not keys or destination is None:
            self._status(tr("The plot shows no curve."), "warning")
            return
        folder = QFileDialog.getExistingDirectory(self, tr("Save the Curves of the Plot"), str(destination))
        if not folder:
            return
        try:
            written = self.view_model.export_curves(keys, Path(folder))
        except (ValueError, OSError) as exc:
            self._status(trf("Could not save the curves: {error}", error=message_text(exc)), "error")
            return
        self.notify_written(trf("Saved {n} files", n=len(written)), written[0].parent)

    def save_plot(self, plot, suffix: str) -> None:
        state = plot.figure_state()
        path = self._figure_path("Save Plot", suffix)
        if path is None:
            return
        curves = state.pop("curves")
        try:
            written = self.view_model.save_curves_figure(path, curves, **state)
        except (ValueError, OSError) as exc:
            self._status(trf("Could not save the plot: {error}", error=message_text(exc)), "error")
            return
        self.notify_written(trf("Saved {name}", name=written.name), written.parent)

    # (The Fitting buttons, ``fit_current``, are ``fit_handover.FitHandoverMixin``'s: Send to Fitting.)

    # -- watch -------------------------------------------------------------------------

    def _watch_clicked(self, checked: bool) -> None:
        if not checked:
            self.stop_watch()
            return
        folder = QFileDialog.getExistingDirectory(
            self, tr("Watch a folder for new frames"), self._last_folder
        )
        if folder:
            self._remember(last_folder=folder)
            self.start_watch(Path(folder))
        else:
            self.watch_button.setChecked(False)

    def start_watch(self, folder: Path) -> None:
        """Watch ``folder``: what it holds is listed (as Open Folder lists it), and when no frame is on screen
        the first of them is shown; frames written from now on are followed."""
        self._watch_series_start()
        before = len(self.view_model.state.files)
        self._watch_told = {}  # (what a watch before said is news to this one)
        self.view_model.start_watch(folder)
        self._watch_series_align_groups()
        before -= len(self.view_model.dropped)  # (rows of files that belong to another listed source left the list)
        self._rows_regrouped()
        self._append_list_items(self.view_model.state.files[before:])
        self.refresh_batch_entry()
        self.watch_button.setChecked(True)
        self._watch_timer.start()
        listed = len(self.view_model.state.files) > before
        if listed:
            self._series_list_grew(existing=True)  # files already there are listed, not queued as newly measured frames
        if listed and self.file_list.currentRow() < 0 and not self._automatic_busy:
            self.show_step("data")
            self.file_list.setCurrentRow(before)
        self._watch_says(lambda: trf("Watching {folder} for new frames …", folder=folder), "info")

    def stop_watch(self) -> None:
        self._stop_watch_series()
        self._watch_timer.stop()
        self.view_model.stop_watch()
        self.watch_button.setChecked(False)
        self._status(tr("Stopped watching."))

    def poll_watch(self) -> list[Path]:
        """Look at the watched folder now and act on what is new; returns the files added to the list."""
        added = self.view_model.poll_watch()
        self._watch_found(added)
        return added

    def _watch_tick(self) -> None:
        """The timer: look at the folder on the watch's own thread (one look at a time), then act on it here."""
        if getattr(self, "_watch_looking", False) or not self.view_model.watch.active:
            return
        runner = getattr(self, "_watch_runner", None)
        if runner is None:  # its own runner: a look every two seconds is no "busy" for the progress bar
            runner = self._watch_runner = TaskRunner(self, max_threads=1)
        self._watch_looking = True
        runner.submit("watch", self.view_model.watch.scan, on_done=self._watch_looked,
                      on_error=lambda _message, _details: setattr(self, "_watch_looking", False))

    def _watch_looked(self, scan) -> None:
        self._watch_looking = False
        if self.view_model.watch.active:  # (stopped meanwhile: what it saw is of no use)
            self._watch_found(self.view_model.take_watch(scan))

    def _watch_found(self, added) -> None:
        """Act on a look at the folder: list the new files, write the rows of the sources that changed, follow
        the newest frame, read the frame on screen again when its files changed, export the new frames, and say
        what was found or what is waited for."""
        model = self.view_model
        shown = model.current_path
        read = self._rows_regrouped()  # whether the frame on screen is read (again) by what this look found
        grown, changed = list(model.watch_grown), list(model.watch_changed)
        if added:
            self._append_list_items(added)
            # Follow the newest frame, the usual need during an in-situ run.
            self.file_list.setCurrentRow(len(model.state.files) - 1)
            read = read or not self._automatic_busy  # (the automatic analysis keeps the frame it works on)
        if grown or changed:
            self._refresh_rows([*changed, *(path for path, _count in grown)])
            self.refresh_batch_entry()
        if grown:
            self._series_list_grew()  # a map built before holds the frames of then, and says so
            if not added:
                read = self._follow(grown[-1][0]) or read
        if not read and shown is not None and str(shown).casefold() in {str(path).casefold() for path in changed}:
            self._show_again(shown)
        destination = model.default_export_dir()  # (next to the file shown)
        exporting = self.auto_export_check.isChecked() and destination is not None
        self._watch_series_sum(exporting=exporting)
        # With frame summing only complete groups of N new frames are exported.
        requests = model.watch_requests(added, wanted=exporting or self._series_map is not None)
        if self._watch_dispatch_requests(requests):
            self._say_watch(grown, exporting)

    def _watch_dispatch_requests(self, requests) -> bool:
        """Submit ready watch work, including frames waiting for the first analysis's geometry."""
        model = self.view_model
        destination = model.default_export_dir()
        exporting = self.auto_export_check.isChecked() and destination is not None
        prepared = self._watch_series_prepare(requests, exporting=exporting)
        if prepared and exporting:  # (a folder of the data that takes no file: the session's folder, or no export)
            destination = self._auto_export_folder(destination)
            if destination is None:
                return False
        for request, key, token, built in prepared:
            if not exporting and key is None:
                continue
            job = f"{'export' if exporting else 'watch-series'}:{request.path}:{request.frame_index}"
            if not exporting:
                self._watch_series_jobs.add(job)
            self.tasks.submit(
                job,
                (lambda request=request, key=key: model.analyze_and_export(request, destination, live_key=key))
                if exporting else (lambda request=request, key=key: model.series_row(request, key)),
                on_done=lambda value, key=key, token=token, built=built, job=job:
                    self._watch_series_done(value, key, token, built, job),
                on_error=lambda message, _details, name=request.path.name, token=token, job=job, exporting=exporting:
                    self._watch_series_failed(message, name, token, job, exporting=exporting),
            )
        return True

    def _follow(self, path: Path) -> bool:
        """Show the newest frame the watch handed out of a listed source that grew (the automatic analysis
        keeps its own frame); ``True`` when it is shown. With a sum of N frames the frame shown is the first of
        the N it sums, so that they end at that newest frame: the frames after it may still be being written."""
        model = self.view_model
        files = [str(listed).casefold() for listed in model.state.files]
        key = str(path).casefold()
        if key not in files or self._automatic_busy:
            return False
        newest = model.followed_frame(path)  # (the newest of the file may still be being written)
        frame = model.frame_count(path) - 1 if newest is None else max(0, newest - (model.sum_count - 1))
        return self.show_frame(files.index(key), frame)

    def _show_again(self, path: Path) -> None:
        """The files of the source on screen changed and no new frame is followed (a module file arrived, the
        file was written again, its writer closed it): its frame is read again and the Data step says what the
        source is now. The automatic analysis keeps the frame it works on: only the Data step follows then."""
        if self._automatic_busy:
            self._show_source(self.view_model.describe(path), path)
            return
        self._file_loading(path)
        self.run_analysis()

    def _watch_notes(self, path) -> list[str]:
        """Why Export New Frames While Watching does not export the source of ``path`` yet although no part of
        it is known to be missing: its module files may still be arriving (``view_model.settling``)."""
        model = self.view_model
        if not (model.watch.active and self.auto_export_check.isChecked()):
            return []
        layout, arriving = model.describe(path), model.settling(path)
        if arriving is None or parts_missing(layout):  # (a missing part is said by the layout's own warning)
            return []
        count, others = arriving
        name = source_name(path, layout)
        if others is None or count >= others:
            return [trf(SET_ARRIVING, name=name, n=count)]
        return [trf(SET_FEWER, name=name, n=count, total=others)]

    def _watch_says(self, compose, level: str) -> None:
        """Say ``compose()`` in the status line, and keep how it is made: a switch of the interface language
        says it again (``_watch_language``) while the line still shows it."""
        text = compose()
        self._status(text, level)
        self._watch_status = (compose, text, level)

    def _watch_language(self) -> None:
        """After a switch of the interface language: what the watch said last, in the new language, when the
        status line still says it (it is said once, so nothing else would bring it up to date)."""
        compose, text, level = getattr(self, "_watch_status", None) or (None, None, None)
        if compose is not None and self.view_model.watch.active and self.status_text() == text:
            self._watch_says(compose, level)

    def _say_watch(self, grown, exporting: bool) -> None:
        """The status line after a look: frames a source gained; else what is held back or waited for. That is
        said once, not at every look, and a look that brings frames meanwhile does not make it new: it is said
        again when what is held back or waited for changes, or why, or when it comes back after it was over."""
        model = self.view_model
        told = dict(getattr(self, "_watch_told", None) or {})  # "held" → the sentence; "waiting" → (file, why)
        if not (exporting and model.watch_held):
            told.pop("held", None)
        if told.get("waiting") not in {(str(path), reason) for path, reason in model.watch_waiting}:
            told.pop("waiting", None)
        if grown:
            path, count = grown[-1]
            self._watch_says(lambda: new_frames_text(source_name(path, model.describe(path)), count), "ok")
        elif exporting and model.watch_held:
            path = model.watch_held[-1]

            def held() -> str:
                notes = self._watch_notes(path)
                if notes:
                    return notes[0]
                name = source_name(path, model.describe(path))
                return trf(EXPORT_WAITS, name=name, reason=" ".join(self._source_warnings(path)))

            text = held()
            if told.get("held") != text:
                told["held"] = text
                self._watch_says(held, "info" if self._watch_notes(path) else "warning")
        elif model.watch_waiting:
            path, reason = model.watch_waiting[-1]
            folder = model.watch.folder.name
            if told.get("waiting") != (str(path), reason):
                told["waiting"] = (str(path), reason)
                # (A file its writer left unfinished is nothing to wait for: it is for the person to repair.)
                self._watch_says(lambda: trf(WATCH_WAITS, folder=folder, reason=message_text(reason)),
                                 "warning" if left_unfinished(reason) else "info")
        self._watch_told = told


def exported_text(written) -> str:
    """“Exported 3 files to …” (the export of the frame shown) in the interface language."""
    return trf("Exported {n} files to {folder}", n=len(written), folder=written[0].parent)


__all__ = ["AUTO_EXPORT_OFF", "BatchWatchMixin", "EXPORT_WAITS", "SET_ARRIVING", "SET_FEWER", "WATCH_WAITS", "exported_text"]
