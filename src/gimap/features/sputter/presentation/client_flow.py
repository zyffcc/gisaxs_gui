"""Analyze, Stop, Open Result… and the input bar (a mixin of ``SputterPage``).

Analyze is one background task (``ClientService.analyze``): the client analysis package runs as a subprocess, its
steps move the bar of the job status in the input bar, Cancel ends its whole process tree, and its lines go to the
log in Details ▸ Analyze options. It uses the folder in the picker, the bar's metal, substrate and rate, the geometry
of Details (the one line under the bar says what will be used; Edit shows the fields) and the options of Details.
Pressed while the folder's header is still read, it starts once the form is filled from it. A request that was
analysed before by the same package shows the result it has.

The bar's metal and the metal of Details are one choice (the earlier method names Cu differently). Substrate, rate,
QCM reading, onset, duration and film thickness are remembered per scan folder; the substrate, the mode and the
package folder also as the last used ones. A folder that was analysed before gets its own values back; another folder
starts without the rate, onset, duration and film thickness of the scan before it — but what the user typed and has
not analysed yet stays. The page opens on the main screen; Details is the second view.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QFileDialog

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import tr

from ..application import (FILM_SUBSTRATES, ClientFailed, ClientProgress, ClientRequest, ClientResult, ClientStarted,
                           Stopped, folder_key)
from .client_texts import (DONE, FAILED_TOAST, GEOMETRY_ALPHA, GEOMETRY_CENTRE, GEOMETRY_CENTRE_AUTO, GEOMETRY_DISTANCE,
                           GEOMETRY_ENERGY, GEOMETRY_MISSING, GEOMETRY_ROLL, GEOMETRY_TO_ENTER, GEOMETRY_WAVELENGTH,
                           NO_REPORT, OPENED_EARLIER, PACKAGE_LINE, PACKAGE_NONE, READY_TOAST, SHOWN_AGAIN, STARTING,
                           STOPPING_ANALYSIS, WAITING_TO_ANALYZE)
from .geometry import warning_style
from .predict_flow import Failure
from .texts import message_text, text

CLIENT_KEY = "sputter_client"
CLIENT_FOLDERS = "client_folders"
MAX_CLIENT_FOLDERS = 300
CLIENT_FIELDS = ("sample_rate", "onset", "duration", "film_thickness")


def _guarded(work):
    """Run ``work``; a failure comes back as a ``Failure`` (its details are lost on the way otherwise)."""
    try:
        return work()
    except Stopped as exc:
        return Failure(str(exc), "", "cancelled")
    except ClientFailed as exc:
        return Failure(str(exc), exc.details, "client")
    except (OSError, ValueError, RuntimeError) as exc:
        return Failure(str(exc) or type(exc).__name__, "", "error")
    except Exception as exc:  # noqa: BLE001 - shown to the user, with the traceback under Details
        return Failure(str(exc) or type(exc).__name__, traceback.format_exc(), "error")


def _float(value: str) -> Optional[float]:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _shown(value) -> str:
    return "" if value is None else f"{float(value):.6g}"


class SputterClientMixin:
    """Needs the widgets of the input bar and of Analyze options, ``service``, ``tasks``, ``preferences``."""

    def _init_client(self) -> None:
        self._client_busy = False
        self._client_stop = False
        self._analyze_queued = False
        self._client_tracker: Optional[ClientProgress] = None
        self._client_status_parts = ("idle", [], 0.0)
        self._client_typed = False  # a field of the bar was typed since the folder was chosen
        self._client_owner = ""  # folder_key of the scan the bar's values belong to ("": typed, not analysed yet)
        stored = self._stored()
        client = self.service.client
        if client is not None and stored.get("client_package"):
            client.choose_package(stored["client_package"])

    def _restore_client_last(self) -> None:
        """The substrate, the mode and the QCM box as they were used last."""
        last = self._stored().get("client_last")
        last = last if isinstance(last, dict) else {}
        self._apply_client_values({key: last.get(key) for key in ("substrate", "mode", "qcm")})

    def _connect_client(self) -> None:
        self.analyze_button.clicked.connect(self.analyze)
        self.analyze_status.cancelRequested.connect(self.stop_analysis)
        self.analyze_status.detailsRequested.connect(self._show_client_log)
        self.clientEvent.connect(self._client_event)
        self.view_switch.activated.connect(lambda index: self.show_details(self.view_switch.itemData(index) == "details"))
        self.geometry_edit_button.clicked.connect(self.edit_geometry)
        self.open_result_button.clicked.connect(self._choose_client_result)
        self.package_button.clicked.connect(self._choose_package)
        self.report_button.clicked.connect(self._open_report)
        self.result_folder_button.clicked.connect(self._open_result_folder)
        self.sample_metal.currentIndexChanged.connect(lambda _index: self._metal_synchronised(self.sample_metal, self.metal))
        self.metal.currentIndexChanged.connect(lambda _index: self._metal_synchronised(self.metal, self.sample_metal))
        self.substrate.currentIndexChanged.connect(self._substrate_changed)
        self.substrate.activated.connect(lambda _index: self._client_typed_now())
        self.qcm_check.clicked.connect(lambda _on: self._client_typed_now())
        for name in CLIENT_FIELDS:
            getattr(self, name).textEdited.connect(lambda _text: self._client_typed_now())
        for field in (self.energy_value, self.sdd, self.alpha, self.beam_col, self.beam_row, self.roll):
            field.textChanged.connect(lambda _text: self._show_geometry_summary())
        self.energy_kind.currentIndexChanged.connect(lambda _index: self._show_geometry_summary())
        self.scan_picker.pathChanged.connect(lambda _path: self._show_geometry_summary())
        self._metal_synchronised(self.metal, self.sample_metal)
        self._substrate_changed()
        self._show_geometry_summary()
        self._show_package()
        self._show_client_status()

    # -- the two views -------------------------------------------------------------------------------------------
    def show_details(self, shown: bool = True) -> None:
        """Details (the geometry, the options, the earlier method and its tabs) or the main screen."""
        self.view_stack.setCurrentIndex(1 if shown else 0)
        self.view_switch.blockSignals(True)
        self.view_switch.setCurrentIndex(self.view_switch.findData("details" if shown else "main"))
        self.view_switch.blockSignals(False)

    def details_shown(self) -> bool:
        return self.view_stack.currentIndex() == 1

    def edit_geometry(self) -> None:
        """Details at the geometry fields: the first one that is still empty has the cursor."""
        self.show_details(True)
        self.details_scroll.ensureWidgetVisible(self.geometry_section)
        fields = (self.energy_value, self.sdd, self.alpha)
        next((field for field in fields if not field.text().strip()), fields[0]).setFocus()

    def _show_client_log(self) -> None:
        self.show_details(True)
        self.client_log.setVisible(True)
        self.details_scroll.ensureWidgetVisible(self.client_log)

    # -- the bar -------------------------------------------------------------------------------------------------
    def _metal_synchronised(self, source, target) -> None:
        index = target.findData(source.currentData())
        if index >= 0 and index != target.currentIndex():
            target.setCurrentIndex(index)

    def _substrate_changed(self, _index: int = 0) -> None:
        film = self.substrate.currentData() in FILM_SUBSTRATES
        self.film_label.setVisible(film)
        self.film_thickness.setVisible(film)

    def _client_typed_now(self) -> None:
        self._client_typed = True
        self._client_owner = ""

    def _geometry_line(self) -> tuple:
        """(the geometry Analyze will use in one line — or what is still to enter —, whether something is)."""
        values = self._values()
        col, row = values["beam_col"], values["beam_row"]
        needed = (("xray", values["energy"]), ("sdd", values["sdd"]), ("alpha", values["alpha"]))
        missing = [GEOMETRY_MISSING[key] for key, value in needed if value is None or value <= 0]
        if (col is None) != (row is None):
            missing.append(GEOMETRY_MISSING["beam"])
        if missing:
            return text(GEOMETRY_TO_ENTER, names=tr(", ").join(tr(name) for name in missing)), True
        parts = [GEOMETRY_WAVELENGTH.format(wavelength=_shown(values["energy"]))
                 if values["energy_kind"] == "wavelength" else GEOMETRY_ENERGY.format(energy=_shown(values["energy"])),
                 GEOMETRY_DISTANCE.format(distance=_shown(values["sdd"])),
                 GEOMETRY_ALPHA.format(alpha=_shown(values["alpha"])),
                 tr(GEOMETRY_CENTRE_AUTO) if col is None else text(GEOMETRY_CENTRE, col=_shown(col), row=_shown(row)),
                 text(GEOMETRY_ROLL, roll=_shown(values["roll"]))]
        return " · ".join(parts), False

    def _show_geometry_summary(self) -> None:
        line, missing = self._geometry_line()
        self.geometry_summary.set_full_text(line)
        warning_style(self.geometry_summary, missing and bool(self.scan_picker.path()))  # needed once a scan is chosen

    def _show_package(self) -> None:
        client = self.service.client
        package = client.package() if client is not None else None
        if package is None or package.home is None:
            words = tr(PACKAGE_NONE)
        else:
            words = text(PACKAGE_LINE.get(package.source, "{folder}"), folder=str(package.home))
        problem = package.problem if package is not None else ""
        # a folder has no spaces: a zero-width space after each separator lets the line wrap inside the column
        self.package_label.setText(words.replace("\\", "\\\u200b").replace("/", "/\u200b"))
        self.package_label.setToolTip(message_text(problem) if problem else words)
        warning_style(self.package_label, package is None or bool(problem))
        self._show_client_empty()
        self._show_static_package()  # static frames run through the same package

    def _choose_package(self) -> None:
        client = self.service.client
        if client is None:
            return
        package = client.package()
        start = str(package.home) if package.home is not None else str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, tr("Choose the Folder of the Analysis Package"), start)
        if not folder:
            return
        package = client.choose_package(folder)
        stored = self._stored()
        stored["client_package"] = str(folder)
        self._store(stored)
        self._show_package()
        if package.problem:
            self._notify(message_text(package.problem), error=True)

    # -- the values ----------------------------------------------------------------------------------------------
    def _client_values(self) -> dict:
        return {"substrate": self.substrate.currentData(), "rate": _float(self.sample_rate.text()),
                "qcm": self.qcm_check.isChecked(), "mode": self.mode_box.currentData(),
                "onset": _float(self.onset.text()), "duration": _float(self.duration.text()),
                "film": _float(self.film_thickness.text())}

    def _apply_client_values(self, values: dict) -> None:
        """Values of the bar and of Analyze options (a key that is absent leaves its field as it is)."""
        for key, box in (("substrate", self.substrate), ("mode", self.mode_box)):
            index = box.findData(values.get(key)) if key in values else -1
            if index >= 0:
                box.setCurrentIndex(index)
        if "qcm" in values and values["qcm"] is not None:
            self.qcm_check.setChecked(bool(values["qcm"]))
        for key, name in (("rate", "sample_rate"), ("onset", "onset"), ("duration", "duration"),
                          ("film", "film_thickness")):
            if key in values:
                getattr(self, name).setText(_shown(_float(values[key])) if values[key] not in (None, "") else "")
        self._substrate_changed()

    def _client_request(self) -> ClientRequest:
        geometry, client = self._values(), self._client_values()
        energy, is_energy = geometry["energy"], geometry["energy_kind"] != "wavelength"
        return ClientRequest(
            scan_folder=Path(self.scan_picker.path() or ""), metal=self.sample_metal.currentData(),
            substrate=client["substrate"], rate_nm_s=client["rate"], rate_is_qcm=client["qcm"],
            alpha_i_deg=geometry["alpha"], sdd_mm=geometry["sdd"],
            energy_ev=energy * 1000. if is_energy and energy is not None else None,
            wavelength_nm=None if is_energy else energy, beam_col=geometry["beam_col"], beam_row=geometry["beam_row"],
            sample_roll_deg=geometry["roll"], mask_file=Path(geometry["mask"]) if geometry["mask"] else None,
            mode=client["mode"], onset_s=client["onset"], duration_s=client["duration"], film_nm=client["film"])

    def _remember_client(self, request: ClientRequest) -> None:
        self._remember(self._request())  # the geometry, for this folder and as the last used
        stored = self._stored()
        values = self._client_values()
        folders = dict(stored.get(CLIENT_FOLDERS) or {})
        key = folder_key(request.scan_folder)
        folders.pop(key, None)
        folders[key] = values
        while len(folders) > MAX_CLIENT_FOLDERS:
            folders.pop(next(iter(folders)))
        stored[CLIENT_FOLDERS] = folders
        stored["client_last"] = {key: values[key] for key in ("substrate", "mode", "qcm")}
        self._store(stored)
        self._client_owner = key

    def _restore_client(self, folder) -> None:
        """The bar for a folder just chosen: what its last Analyze used. A rate, onset, duration and film thickness
        belong to one scan: a folder without values of its own starts without those of another scan, and keeps what
        was typed and not analysed yet."""
        key = folder_key(folder)
        folders = self._stored().get(CLIENT_FOLDERS)
        kept = folders.get(key) if isinstance(folders, dict) else None
        if isinstance(kept, dict):
            values = {name: kept.get(name) for name in ("rate", "onset", "duration", "film")}
            values.update({name: kept[name] for name in ("substrate", "qcm") if kept.get(name) is not None})
            self._apply_client_values(values)
            self._client_owner = key
        elif self._client_owner and self._client_owner != key:
            self._apply_client_values({name: None for name in ("rate", "onset", "duration", "film")})
            self._client_owner = ""

    def set_scan_folder(self, folder) -> None:
        super().set_scan_folder(folder)
        self._restore_client(folder)
        self._client_typed = False

    def _inspected(self, info) -> None:
        if not self._client_typed:  # the frames may be one level down: that folder is the one remembered
            self._restore_client(info.folder)
        super()._inspected(info)  # fills the form, then runs an Analyze that waited for the header
        self._show_geometry_summary()

    # -- Analyze -------------------------------------------------------------------------------------------------
    def analyzing(self) -> bool:
        return self._client_busy

    def _client_should_stop(self):
        return self._client_stop

    def _client_status(self, state: str, parts: list, progress: Optional[float] = 0.0) -> None:
        """``parts``: (template, values) pairs joined by " — " (composed again after a language switch)."""
        self._client_status_parts = (state, parts, progress)
        self._show_client_status()

    def _show_client_status(self) -> None:
        state, parts, progress = self._client_status_parts
        words = []
        for template, values in parts:
            shown = {key: (message_text(value) if key == "error" else tr(str(value)) if key == "stage" else value)
                     for key, value in values.items()}
            words.append(text(template, **shown))
        self.analyze_status.set_state(state, " — ".join(word for word in words if word), progress=progress)
        self.analyze_status.cancel_button.setEnabled(state == "running" and self._client_busy and not self._client_stop)

    def _client_refused(self, message: str) -> None:
        self._client_status("failed", [("{error}", {"error": message})], 0.0)
        self._notify(message_text(message), error=True)

    def analyze(self) -> None:
        client = self.service.client
        if self._client_busy or client is None or self.static_analyzing():
            return
        picker = self.scan_picker.path()
        if picker and not self.header_ready():
            if not self._inspecting:  # a folder typed in but not looked at yet: what the bar says now is kept
                kept = self._client_values()
                self.set_scan_folder(picker)
                self._apply_client_values(kept)
                self._client_typed_now()
            self._analyze_queued = True
            self._client_status("running", [(WAITING_TO_ANALYZE, {})], None)
            return
        if picker and self.scan_info.problem:
            self._client_refused(self.scan_info.problem)
            return
        request = self._client_request()
        problems = request.problems()
        if problems:
            self._client_status("failed", [(problem, {}) for problem in problems], 0.0)
            self._notify(" ".join(tr(problem) for problem in problems), error=True)
            return
        package = client.package()
        if package.problem:
            self._client_refused(package.problem)
            self._show_package()
            return
        scan_results = self.service.results_folder(self._request())
        self._remember_client(request)
        self._client_stop = False
        self._client_busy = True
        self._client_tracker = ClientProgress()
        self.client_log.appendPlainText("— " + str(request.scan_folder))
        self._sync_run_buttons()
        self._client_status("running", [(STARTING, {})], None)
        emit, should_stop = self.clientEvent.emit, self._client_should_stop
        self.tasks.submit(CLIENT_KEY, lambda: _guarded(lambda: client.analyze(request, scan_results, emit, should_stop)),
                          on_done=self._client_finished,
                          on_error=lambda message, details: self._client_finished(Failure(message, details)))

    def _run_queued(self) -> None:
        """Analyze pressed while the header was read: now (and the earlier method's Predict likewise)."""
        super()._run_queued()
        if self._analyze_queued and not self._client_busy and not self.static_analyzing():
            self._analyze_queued = False
            self.analyze()

    def _forget_queued(self) -> None:
        """Another folder was chosen: an Analyze pressed for the earlier one does not run for it."""
        super()._forget_queued()
        if self._analyze_queued:
            self._analyze_queued = False
            if not self._client_busy:
                self._client_status("idle", [], 0.0)

    def stop_analysis(self) -> None:
        if not self._client_busy or self._client_stop:
            return
        self._client_stop = True
        state, parts, progress = self._client_status_parts
        self._client_status("running", parts[:1] + [(STOPPING_ANALYSIS, {})], progress)

    def _dispose_client(self) -> None:
        """The window closes: end the package's processes, forget the task."""
        self._analyze_queued = False
        if self._client_busy:
            self._client_stop = True
        self.tasks.cancel(CLIENT_KEY)

    def _client_event(self, event) -> None:
        if isinstance(event, ClientStarted):
            self.client_log.appendPlainText(f"{event.label} -> {event.folder}")
            return
        line = str(event)
        self.client_log.appendPlainText(line)
        tracker = self._client_tracker
        if tracker is None or not tracker.feed(line) or self._client_stop:
            return
        template, values = tracker.message()
        if template:
            self._client_status("running", [(template, values)], tracker.fraction)

    def _client_finished(self, outcome) -> None:
        tracker, self._client_tracker = self._client_tracker, None
        self._client_busy = False
        self._sync_run_buttons()
        if isinstance(outcome, ClientResult):
            ran = tracker is not None and bool(tracker.step)
            self._client_status("succeeded", [(DONE if ran else SHOWN_AGAIN, {"windows": len(outcome.windows)})], 1.0)
            self.show_client_result(outcome)
            self._remember_client_result(outcome.folder)
            self.show_details(False)
            folder = outcome.folder
            show_toast(self, text(READY_TOAST, name=outcome.scan or folder.name), level="success",
                       action=(tr("Open Folder"), lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))))
        elif isinstance(outcome, Failure):
            if outcome.details:
                self.client_log.appendPlainText(outcome.details)
            state = "cancelled" if outcome.kind == "cancelled" else "failed"
            self._client_status(state, [("{error}", {"error": outcome.message})], 0.0)
            message = message_text(outcome.message)
            self._notify(message.splitlines()[0] if message else tr(FAILED_TOAST), error=state == "failed")
        self._run_queued()

    # -- results -------------------------------------------------------------------------------------------------
    def _remember_client_result(self, folder: Optional[Path]) -> None:
        stored = self._stored()
        stored["last_client_result"] = str(folder) if folder else ""
        self._store(stored)

    def _choose_client_result(self) -> None:
        shown = self.client_result
        start = str(shown.folder) if shown is not None else str(self.service.results_root())
        folder = QFileDialog.getExistingDirectory(self, tr("Open the Folder of a Result"), start)
        if folder:
            self.open_client_result(Path(folder))

    def open_client_result(self, folder, *, quiet: bool = False) -> bool:
        """Show the Analyze result in (or under) ``folder``; a folder that only holds a result of the earlier method
        is shown in Details. False when the folder holds neither (said, unless ``quiet``)."""
        client = self.service.client
        if client is None:
            return False
        folder = Path(folder)
        try:
            result = client.open(folder)
        except (OSError, ValueError) as exc:
            if client.locate(folder) is None and self.service.variants(folder):
                self.open_results(folder, quiet=quiet)
                self.show_details(True)
                if not quiet:
                    self._notify(tr(OPENED_EARLIER))
                return True
            if not quiet:
                self._notify(message_text(exc), error=True)
            return False
        self.show_client_result(result)
        self._remember_client_result(result.folder)
        if not quiet:
            self.show_details(False)
        return True

    def _open_report(self) -> None:
        result = self.client_result
        report = result.file("report_md") if result is not None else None
        if report is None:
            self._notify(tr(NO_REPORT), error=True)
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(report)))

    def _open_result_folder(self) -> None:
        if self.client_result is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.client_result.folder)))

    # -- the project ---------------------------------------------------------------------------------------------
    def client_project_state(self) -> dict:
        values = self._client_values()
        fields = {name: getattr(self, name).text() for name in CLIENT_FIELDS}
        return {"substrate": values["substrate"], "qcm": values["qcm"], "mode": values["mode"], "fields": fields,
                "result_folder": str(self.client_result.folder) if self.client_result is not None else "",
                "details_shown": self.details_shown()}

    def apply_client_project_state(self, state) -> list:
        """The bar, the options and the result of a project (None: empty, as an older project has none)."""
        state = state if isinstance(state, dict) else {}
        self._analyze_queued = False
        fields = state.get("fields") if isinstance(state.get("fields"), dict) else {}
        for name in CLIENT_FIELDS:
            getattr(self, name).setText(str(fields.get(name, "")))
        self._apply_client_values({"substrate": state.get("substrate") or self.substrate.itemData(0),
                                   "mode": state.get("mode") or "auto", "qcm": bool(state.get("qcm"))})
        self._client_typed = True  # the project's values stay when the folder's header is read
        self._client_owner = folder_key(self.scan_picker.path()) if self.scan_picker.path() else ""
        self.show_client_result(None)
        notes = []
        folder = str(state.get("result_folder") or "")
        if folder and Path(folder).is_dir():
            try:
                self.show_client_result(self.service.client.open(Path(folder)))
            except (OSError, ValueError) as exc:
                notes.append(f"Sputter: {exc}")
        self.show_details(bool(state.get("details_shown")))
        self._show_geometry_summary()
        return notes

    def refresh_client_language(self) -> None:
        self._show_client_status()
        self._show_geometry_summary()
        self._show_package()
        self.refresh_client_panel_language()


__all__ = ["CLIENT_KEY", "SputterClientMixin"]
