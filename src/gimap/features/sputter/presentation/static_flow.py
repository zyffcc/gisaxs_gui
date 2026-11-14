"""Details ▸ Static Frames (after deposition): analyse one sample into a series (a mixin of ``SputterPage``).

Analyze Sample is one background task (``StaticService.analyze``): the analysis package's ``static`` command runs as a
subprocess, its three steps move the bar of the section's job status, Cancel ends its whole process tree and its lines
go to the section's log. When it ends, the series — every sample analysed into that folder so far — is shown on the
main screen with the new sample selected; Open Result… shows a series like any other result.

The geometry fields are filled with the values the package itself uses for static frames (Package Values fills them
again); an empty field leaves the package's value. The series folder, the mode, the geometry and the mask are
remembered as the last used ones; everything of the section is part of a project. A sample name the series already
has, with other frames or values, is replaced after a question. Analyze and Analyze Sample do not run at the same
time: each button waits for the other analysis.

Masks of the Package (here and under the mask of More options) lists the pixel masks the package brings by their
file names; choosing one fills the mask field beside it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QFileDialog, QInputDialog, QMenu, QMessageBox

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import tr

from ..application import ClientProgress, ClientResult, ClientStarted, StaticRequest, join_frames, split_frames
from .client_flow import _guarded
from .client_texts import STARTING, STOPPING_ANALYSIS
from .geometry import warning_style
from .predict_flow import Failure
from .texts import message_text, text

STATIC_KEY = "sputter_static"
STATIC_NUMBERS = ("static_nominal", "static_energy", "static_sdd", "static_beam_col", "static_beam_row",
                  "static_roll")
STATIC_GEOMETRY = ("static_energy", "static_sdd", "static_beam_col", "static_beam_row", "static_roll")
STATIC_SCOPE = "Au films measured after the deposition, one sample at a time into a series."
STATIC_SCOPE_ALPHA = ("Au films measured after the deposition at an incidence angle near {alpha}°, one sample at a "
                      "time into a series.")
STATIC_DONE = "Done: {samples} samples in the series."
STATIC_SHOWN_AGAIN = "This sample was analysed before: the series is shown ({samples} samples)."
STATIC_READY_TOAST = "Sample {name} is in the series."
STATIC_REPLACE_TITLE = "Replace the Sample?"
STATIC_REPLACE = ("The series already has a sample named {name}, analysed from other frames or values. Analysing it "
                  "again replaces that sample in the series.")
ARCHIVE_EMPTY = "{file} holds no CBF frames."
ARCHIVE_TITLE = "Folder Inside the Archive"
ARCHIVE_QUESTION = "The archive holds frames in several folders. Frames to use:"
ARCHIVE_TOP = "(frames at the top of the archive)"
NO_PACKAGE_MASKS = "The analysis package brings no masks"
TEMPLATES = (STATIC_SCOPE, STATIC_SCOPE_ALPHA, STATIC_DONE, STATIC_SHOWN_AGAIN, STATIC_READY_TOAST,
             STATIC_REPLACE_TITLE, STATIC_REPLACE, ARCHIVE_EMPTY, ARCHIVE_TITLE, ARCHIVE_QUESTION, ARCHIVE_TOP,
             NO_PACKAGE_MASKS)
"""The run-time texts of the section that need Chinese."""


def _float(value: str) -> Optional[float]:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _shown(value) -> str:
    return "" if value is None else f"{float(value):.7g}"


class SputterStaticMixin:
    """Needs the widgets of ``views/sputter_static_view.py``, ``service``, ``tasks``, ``preferences`` and the
    main screen (``show_client_result``)."""

    def _init_static(self) -> None:
        self._static_busy = False
        self._static_stop = False
        self._static_tracker: Optional[ClientProgress] = None
        self._static_status_parts = ("idle", [], 0.0)
        self._static_label = ""

    def _restore_static_last(self) -> None:
        """The series folder, the mode, the geometry and the mask as they were used last; the package's own
        geometry where nothing was."""
        last = self._stored().get("static_last")
        self._apply_static_values(last if isinstance(last, dict) else {}, defaults=True)

    def _connect_static(self) -> None:
        self.static_button.clicked.connect(self.analyze_static)
        self.static_status.cancelRequested.connect(self.stop_static)
        self.static_status.detailsRequested.connect(self._show_static_log)
        self.staticEvent.connect(self._static_event)
        self.static_frames.browseRequested.connect(lambda: self._choose_static_folder(self.static_frames))
        self.static_reference.browseRequested.connect(lambda: self._choose_static_folder(self.static_reference))
        self.static_frames_file.clicked.connect(lambda: self._choose_static_file(self.static_frames))
        self.static_reference_file.clicked.connect(lambda: self._choose_static_file(self.static_reference))
        self.static_series.browseRequested.connect(self._choose_static_series)
        self.static_mask.browseRequested.connect(lambda: self._choose_mask_into(self.static_mask))
        self.static_defaults_button.clicked.connect(lambda: self._fill_static_defaults(replace=True))
        for button, picker in ((self.static_masks_button, self.static_mask),
                               (self.package_masks_button, self.mask_picker)):
            menu = QMenu(button)
            menu.aboutToShow.connect(lambda menu=menu, picker=picker: self._fill_masks_menu(menu, picker))
            button.setMenu(menu)
        self._show_static_package()
        self._show_static_status()

    def _static_service(self):
        client = self.service.client
        return client.static if client is not None else None

    # -- the package's own values ---------------------------------------------------------------------------------
    def _show_static_package(self) -> None:
        """What the section is for — or why this package cannot do it —, the default series and the buttons."""
        static = self._static_service()
        defaults = static.defaults() if static is not None else None
        problem = defaults.problem if defaults is not None else ""
        if defaults is None:
            words = ""
        elif problem:
            words = message_text(problem)
        elif defaults.alpha_i_deg is not None:
            words = text(STATIC_SCOPE_ALPHA, alpha=_shown(defaults.alpha_i_deg))
        else:
            words = tr(STATIC_SCOPE)
        self.static_note.setText(words)
        self.static_note.setVisible(bool(words))
        warning_style(self.static_note, bool(problem))
        if static is not None:
            self.static_series.path_edit.setPlaceholderText(str(static.default_series()))
        self._sync_run_buttons()

    def _fill_static_defaults(self, *, replace: bool = False) -> None:
        """The package's own geometry in the fields that are empty (``replace``: in all of them)."""
        static = self._static_service()
        defaults = static.defaults() if static is not None else None
        if defaults is None or defaults.problem:
            return
        values = {"static_energy": None if defaults.energy_ev is None else defaults.energy_ev / 1000.0,
                  "static_sdd": defaults.sdd_mm, "static_beam_col": defaults.beam_col,
                  "static_beam_row": defaults.beam_row, "static_roll": defaults.sample_roll_deg}
        for name, value in values.items():
            field = getattr(self, name)
            if value is not None and (replace or not field.text().strip()):
                field.setText(_shown(value))
        if replace:
            self.static_mask.set_path("")

    # -- pickers --------------------------------------------------------------------------------------------------
    def _choose_static_folder(self, picker) -> None:
        start = split_frames(picker.path())[0] or str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, tr("Choose the Folder of the Frames"), start)
        if folder:
            picker.set_path(folder)

    def _choose_static_file(self, picker) -> None:
        start = split_frames(picker.path())[0] or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, tr("Choose a CBF File or a Zip Archive"), start,
                                              "Frames (*.cbf *.zip);;All files (*)")
        if path:
            self.set_static_source(picker, path)

    def set_static_source(self, picker, path: str) -> None:
        """Put a chosen file into a frames field; for a zip archive with frames in several folders, ask which."""
        inner = ""
        if str(path).lower().endswith(".zip"):
            static = self._static_service()
            try:
                folders = static.archive_folders(path) if static is not None else []
            except (OSError, ValueError, RuntimeError) as exc:
                self._notify(str(exc), error=True)
                return
            if not folders:
                self._notify(text(ARCHIVE_EMPTY, file=Path(path).name), error=True)
                return
            if len(folders) > 1:
                inner = self._ask_archive_folder(folders)
                if inner is None:
                    return
        picker.set_path(join_frames(path, inner))

    def _ask_archive_folder(self, folders: list) -> Optional[str]:
        """The folder of an archive the user picks ("" for the frames at its top; None: cancelled)."""
        names = [folder or tr(ARCHIVE_TOP) for folder in folders]
        chosen, accepted = QInputDialog.getItem(self, tr(ARCHIVE_TITLE), tr(ARCHIVE_QUESTION), names, 0, False)
        return folders[names.index(chosen)] if accepted and chosen in names else None

    def _choose_static_series(self) -> None:
        static = self._static_service()
        start = self.static_series.path() or (str(static.default_series().parent) if static is not None else "")
        folder = QFileDialog.getExistingDirectory(self, tr("Choose the Folder of the Series"), start)
        if folder:
            self.static_series.set_path(folder)

    def _choose_mask_into(self, picker) -> None:
        path, _ = QFileDialog.getOpenFileName(self, tr("Choose a Detector Mask"), picker.path(),
                                              "NumPy mask (*.npy);;All files (*)")
        if path:
            picker.set_path(path)

    def _fill_masks_menu(self, menu, picker) -> None:
        """The masks of the package as it is now, by file name (file names are data: never translated)."""
        menu.clear()
        client = self.service.client
        masks = client.package_masks() if client is not None else []
        for mask in masks:
            action = menu.addAction(mask.name.replace("&", "&&"))
            action.setToolTip(str(mask))
            action.triggered.connect(lambda _checked=False, mask=mask: picker.set_path(str(mask)))
        if not masks:
            menu.addAction(tr(NO_PACKAGE_MASKS)).setEnabled(False)

    # -- the values -----------------------------------------------------------------------------------------------
    def _static_values(self) -> dict:
        values = {name: getattr(self, name).text().strip() for name in STATIC_NUMBERS}
        values.update(frames=self.static_frames.path(), name=self.static_name.text().strip(),
                      reference=self.static_reference.path(), series=self.static_series.path(),
                      mode=self.static_mode.currentData(), mask=self.static_mask.path())
        return values

    def _apply_static_values(self, values: dict, *, defaults: bool = False) -> None:
        """Values of the section (a key that is absent leaves its field as it is); ``defaults``: the package's
        geometry afterwards in the fields that are still empty."""
        for name in STATIC_NUMBERS:
            if name in values:
                getattr(self, name).setText(str(values[name] or ""))
        for key, picker in (("frames", self.static_frames), ("reference", self.static_reference),
                            ("series", self.static_series), ("mask", self.static_mask)):
            if key in values:
                picker.set_path(str(values[key] or ""))
        if "name" in values:
            self.static_name.setText(str(values["name"] or ""))
        index = self.static_mode.findData(values.get("mode")) if "mode" in values else -1
        if index >= 0:
            self.static_mode.setCurrentIndex(index)
        if defaults:
            self._fill_static_defaults()

    def _static_request(self) -> StaticRequest:
        values = self._static_values()
        static = self._static_service()
        series = values["series"] or (str(static.default_series()) if static is not None else "")
        energy = _float(values["static_energy"])
        return StaticRequest(
            frames=values["frames"], name=values["name"], nominal_nm=_float(values["static_nominal"]),
            series_folder=Path(series), reference_frames=values["reference"], mode=values["mode"],
            energy_ev=None if energy is None else energy * 1000.0, sdd_mm=_float(values["static_sdd"]),
            beam_col=_float(values["static_beam_col"]), beam_row=_float(values["static_beam_row"]),
            sample_roll_deg=_float(values["static_roll"]), mask_file=Path(values["mask"]) if values["mask"] else None)

    def _remember_static(self) -> None:
        stored = self._stored()
        values = self._static_values()
        stored["static_last"] = {key: values[key] for key in (*STATIC_GEOMETRY, "series", "mode", "mask")}
        self._store(stored)

    # -- Analyze Sample ---------------------------------------------------------------------------------------------
    def static_analyzing(self) -> bool:
        return self._static_busy

    def _sync_run_buttons(self) -> None:
        """Analyze and Analyze Sample wait for each other; Analyze Sample needs a package that knows static frames."""
        busy = self._client_busy or self._static_busy
        static = self._static_service()
        self.analyze_button.setEnabled(not busy)
        self.static_button.setEnabled(not busy and static is not None and not static.defaults().problem)

    def _static_should_stop(self):
        return self._static_stop

    def _static_status(self, state: str, parts: list, progress: Optional[float] = 0.0) -> None:
        """``parts``: (template, values) pairs joined by " — " (composed again after a language switch)."""
        self._static_status_parts = (state, parts, progress)
        self._show_static_status()

    def _show_static_status(self) -> None:
        state, parts, progress = self._static_status_parts
        words = []
        for template, values in parts:
            shown = {key: (message_text(value) if key == "error" else tr(str(value)) if key == "stage" else value)
                     for key, value in values.items()}
            words.append(text(template, **shown))
        self.static_status.set_state(state, " — ".join(word for word in words if word), progress=progress)
        self.static_status.cancel_button.setEnabled(state == "running" and self._static_busy and not self._static_stop)

    def _show_static_log(self) -> None:
        self.show_details(True)
        self.static_section.set_expanded(True)
        self.static_log.setVisible(True)
        self.details_scroll.ensureWidgetVisible(self.static_log)

    def _confirm_static_replace(self, name: str) -> bool:
        answer = QMessageBox.question(self, tr(STATIC_REPLACE_TITLE), text(STATIC_REPLACE, name=name),
                                      QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
        return answer == QMessageBox.Yes

    def analyze_static(self) -> None:
        static = self._static_service()
        if static is None or self._static_busy or self._client_busy:
            return
        request = self._static_request()
        problems = request.problems()
        unsupported = static.defaults().problem
        if unsupported:
            problems = [unsupported]
        if problems:
            self._static_status("failed", [(problem, {}) for problem in problems], 0.0)
            self._notify(" ".join(message_text(problem) for problem in problems), error=True)
            return
        if static.replaces(request) and not self._confirm_static_replace(request.label()):
            return
        self._remember_static()
        self._static_stop = False
        self._static_busy = True
        self._static_tracker = ClientProgress.static()
        self._static_label = request.label()
        self.static_log.appendPlainText("— " + request.frames)
        self._sync_run_buttons()
        self._static_status("running", [(STARTING, {})], None)
        emit, should_stop = self.staticEvent.emit, self._static_should_stop
        self.tasks.submit(STATIC_KEY, lambda: _guarded(lambda: static.analyze(request, emit, should_stop)),
                          on_done=self._static_finished,
                          on_error=lambda message, details: self._static_finished(Failure(message, details)))

    def stop_static(self) -> None:
        if not self._static_busy or self._static_stop:
            return
        self._static_stop = True
        state, parts, progress = self._static_status_parts
        self._static_status("running", parts[:1] + [(STOPPING_ANALYSIS, {})], progress)

    def _dispose_static(self) -> None:
        """The window closes: end the package's processes, forget the task."""
        if self._static_busy:
            self._static_stop = True
        self.tasks.cancel(STATIC_KEY)

    def _static_event(self, event) -> None:
        if isinstance(event, ClientStarted):
            self.static_log.appendPlainText(f"{event.label} -> {event.folder}")
            return
        line = str(event)
        self.static_log.appendPlainText(line)
        tracker = self._static_tracker
        if tracker is None or not tracker.feed(line) or self._static_stop:
            return
        template, values = tracker.message()
        if template:
            self._static_status("running", [(template, values)], tracker.fraction)

    def _static_finished(self, outcome) -> None:
        tracker, self._static_tracker = self._static_tracker, None
        self._static_busy = False
        self._sync_run_buttons()
        if isinstance(outcome, ClientResult):
            ran = tracker is not None and bool(tracker.step)
            template = STATIC_DONE if ran else STATIC_SHOWN_AGAIN
            self._static_status("succeeded", [(template, {"samples": len(outcome.windows)})], 1.0)
            self.show_client_result(outcome, sample=self._static_label)
            self._remember_client_result(outcome.folder)
            self.show_details(False)
            folder = outcome.folder
            show_toast(self, text(STATIC_READY_TOAST, name=self._static_label), level="success",
                       action=(tr("Open Folder"), lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))))
        elif isinstance(outcome, Failure):
            if outcome.details:
                self.static_log.appendPlainText(outcome.details)
            state = "cancelled" if outcome.kind == "cancelled" else "failed"
            self._static_status(state, [("{error}", {"error": outcome.message})], 0.0)
            message = message_text(outcome.message)
            self._notify(message.splitlines()[0] if message else outcome.message, error=state == "failed")
        self._run_queued()  # an Analyze that waited for this analysis

    # -- the project and the language ------------------------------------------------------------------------------
    def static_project_state(self) -> dict:
        return self._static_values() | {"expanded": self.static_section.is_expanded()}

    def apply_static_project_state(self, state) -> None:
        """The section as a project holds it (None: empty, as an older project has none); a project without any
        geometry of static frames gets the package's."""
        state = state if isinstance(state, dict) else {}
        keys = (*STATIC_NUMBERS, "frames", "name", "reference", "series", "mask")
        self._apply_static_values({key: state.get(key, "") for key in keys} | {"mode": state.get("mode") or "quick"},
                                  defaults=not any(state.get(name) for name in STATIC_GEOMETRY))
        self.static_section.set_expanded(bool(state.get("expanded")))

    def refresh_static_language(self) -> None:
        self._show_static_package()
        self._show_static_status()


__all__ = ["STATIC_KEY", "SputterStaticMixin", "TEMPLATES"]
