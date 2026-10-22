"""The scan folder, the values of the form and where each comes from (a mixin of ``SputterPage``).

Choosing a folder reads the first frame's header in the background (``SputterService.inspect``) and fills
the form: each geometry value from what was used for this folder before, else a geometry preset of the user
that matches the scan, else (the energy) a ``Wavelength`` line of the header or the scan's NeXus file, else
the values used last (``application.scan.prefill``). Next to each field the page says where its value comes
from; a field the user edits while the header is read keeps the user's value. Predict waits for the header
of the folder in the picker (``header_ready``). The values are kept in the preferences when Predict starts
(for the folder, and as the last used ones).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QDate
from PyQt5.QtWidgets import QFileDialog

from src.gimap.app.presentation.i18n import tr

from ..application import (GROUPS, SUMMARY_HINT, AnalyzeRequest, Prefill, ScanInfo, Source, add_preset, folder_key,
                           make_preset, prefill, remember)
from .texts import message_text, source_text, text

PREFERENCES_KEY = "sputter_page"
INSPECT_KEY = "sputter_inspect"
OLD_KEYS = {"sdd": "sdd", "alpha": "alpha", "energy": "energy", "window": "window", "mask": "mask"}
"""Values the earlier page kept (``python``, ``toolkit`` and ``device`` are no longer used)."""


def _float(text_value: str) -> Optional[float]:
    try:
        return float(str(text_value).strip())
    except (TypeError, ValueError):
        return None


def _shown(value) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


class SputterGeometryMixin:
    def _connect_geometry(self) -> None:
        self.scan_picker.browseRequested.connect(self._choose_scan)
        self.scan_picker.path_edit.editingFinished.connect(self._scan_typed)
        edits = {"xray": (self.energy_value,), "sdd": (self.sdd,), "alpha": (self.alpha,),
                 "beam": (self.beam_col, self.beam_row), "roll": (self.roll,)}
        for group, widgets in edits.items():
            for widget in widgets:
                widget.textEdited.connect(lambda _text, g=group: self._entered(g))
        self.energy_kind.activated.connect(lambda _index: self._entered("xray"))
        self.metal.currentIndexChanged.connect(self._metal_changed)
        self.preset_button.clicked.connect(self._save_preset)
        self.options_section.expandedChanged.connect(lambda _open: self._remember_view())

    # -- the preferences -----------------------------------------------------------------------------------
    def _stored(self) -> dict:
        stored = self.preferences.get(PREFERENCES_KEY, {}) if self.preferences is not None else {}
        return dict(stored) if isinstance(stored, dict) else {}

    def _memory(self) -> dict:
        stored = self._stored()
        if not isinstance(stored.get("last"), dict) and any(key in stored for key in OLD_KEYS):
            last = {new: stored[old] for old, new in OLD_KEYS.items() if stored.get(old) not in (None, "")}
            if "energy" in last:
                last["energy_kind"] = "wavelength" if stored.get("energy_kind") == 1 else "energy"
            stored["last"] = last
        return stored

    def _store(self, values: dict) -> None:
        if self.preferences is not None:
            self.preferences.set(PREFERENCES_KEY, values)

    def _remember_view(self) -> None:
        stored = self._stored()
        stored["advanced_open"] = self.options_section.is_expanded()
        self._store(stored)

    def _restore(self) -> None:
        stored = self._memory()
        self.options_section.set_expanded(bool(stored.get("advanced_open", False)))
        if stored.get("results_root"):
            self.results_picker.set_path(str(stored["results_root"]))
        last = stored.get("last") if isinstance(stored.get("last"), dict) else {}
        self._apply(Prefill(values=dict(last), sources={g: Source("last") for g in GROUPS}), set())
        for group in GROUPS:
            if group not in self.sources or not any(self._field_text(group)):
                self.sources[group] = Source("missing") if group not in ("beam",) else Source("auto")
        self._show_sources()

    # -- the folder ------------------------------------------------------------------------------------------
    def _choose_scan(self) -> None:
        start = self.scan_picker.path() or str(self._stored().get("last_scan") or Path.home())
        folder = QFileDialog.getExistingDirectory(self, tr("Choose the Scan Folder"), start)
        if folder:
            self.set_scan_folder(folder)

    def _scan_typed(self) -> None:
        path = self.scan_picker.path()
        if path and Path(path) != Path(self._requested_scan or "."):
            self.set_scan_folder(path)

    def set_scan_folder(self, folder) -> None:
        """Look at ``folder`` (its first frame's header, in the background) and fill the form from it."""
        self.scan_picker.set_path(str(folder))
        self._requested_scan = str(folder)
        self.scan_info = None
        self._inspecting = True
        self._forget_queued()
        self._edited = set()
        self._summary = ("Reading the header of the first frame…", {})
        self._show_scan_summary()
        self.tasks.submit(INSPECT_KEY, lambda: self.service.inspect(Path(folder)), on_done=self._inspected,
                          on_error=lambda message, _details: self._inspect_failed(message))

    def header_ready(self) -> bool:
        """The header of the folder in the picker has been read: the form holds that folder's values."""
        info, picker = self.scan_info, self.scan_picker.path()
        return info is not None and not self._inspecting and bool(picker) and \
            folder_key(info.folder) == folder_key(picker)

    def _inspect_failed(self, message: str) -> None:
        self._inspecting = False
        self.scan_info = ScanInfo(Path(self._requested_scan or "."), problem=message)
        self._summary = ("{error}", {"error": message})
        self._show_scan_summary()
        self._run_queued()  # says why it cannot run

    def _inspected(self, info: ScanInfo) -> None:
        self._inspecting = False
        self.scan_info = info
        if Path(info.folder) != Path(self.scan_picker.path() or "."):
            self._requested_scan = str(info.folder)
            self.scan_picker.set_path(str(info.folder))  # the frames are one level down (``p2m/``)
        self._summary = None
        self._apply(prefill(info, self._memory()), self._edited)
        self._show_thickness_fields()  # back on the scan of the result shown: what it was drawn with
        self._show_scan_summary()
        self._show_sources()
        self._run_queued()

    def _summary_text(self) -> str:
        if self._summary is not None:
            template, values = self._summary
            return message_text(values["error"]) if "error" in values else text(template, **values)
        info = self.scan_info
        if info is None:
            return tr("Choose the folder of a scan: its first frame tells the detector and the time.")
        if info.problem:
            return message_text(info.problem)
        parts = [text("{count} frames", count=info.frames)]
        if info.detector:
            parts.append(info.detector + (f" (S/N {info.serial})" if info.serial else ""))
        if info.pixel_um:
            parts.append(f"{info.pixel_um:.4g} µm")
        if info.exposure_s is not None:
            parts.append(text("exposure {exposure} s", exposure=f"{info.exposure_s:.4g}") + (
                text(" every {period} s", period=f"{info.period_s:.4g}") if info.period_s else ""))
        if info.timestamp:
            parts.append(info.timestamp[:19].replace("T", " "))
        if info.energy_kev_hint:
            hint = SUMMARY_HINT.get(info.energy_hint_source or "header", SUMMARY_HINT["header"])
            parts.append(text(hint, energy=f"{info.energy_kev_hint:.4g}"))
        return " · ".join(parts)

    def _show_scan_summary(self) -> None:
        self.scan_summary.setText(self._summary_text())

    # -- the fields ------------------------------------------------------------------------------------------
    def _field_text(self, group: str) -> tuple:
        return {"xray": (self.energy_value.text(),), "sdd": (self.sdd.text(),), "alpha": (self.alpha.text(),),
                "beam": (self.beam_col.text(), self.beam_row.text()), "roll": (self.roll.text(),)}[group]

    def _apply(self, filled: Prefill, keep: set) -> None:
        values = filled.values
        for group in GROUPS:
            if group in keep or group not in filled.sources:
                continue
            self.sources[group] = filled.sources[group]
            if group == "xray":
                kind = values.get("energy_kind") or "energy"
                self.energy_kind.setCurrentIndex(max(0, self.energy_kind.findData(kind)))
                self.energy_value.setText(_shown(values.get("energy")))
            elif group == "beam":
                self.beam_col.setText(_shown(values.get("beam_col")))
                self.beam_row.setText(_shown(values.get("beam_row")))
            else:
                getattr(self, group).setText(_shown(values.get(group)))
        if values.get("metal") in ("Ag", "Au", "Cu"):
            self.metal.setCurrentIndex(max(0, self.metal.findData(values["metal"])))
        if values.get("polymer"):
            self.polymer.setCurrentText(str(values["polymer"]))
        if "mask" in values:
            self.mask_picker.set_path(str(values.get("mask") or ""))
        if isinstance(values.get("window"), (int, float)) and values["window"] > 0:
            self.bin_seconds.setValue(float(values["window"]))
        if "rate" in values:
            self.rate.setText(_shown(values.get("rate")))
            self.t0.setText(_shown(values.get("t0")))

    def _entered(self, group: str) -> None:
        self._edited.add(group)
        filled = any(self._field_text(group))
        self.sources[group] = Source("entered") if filled else Source("auto" if group == "beam" else "missing")
        self._show_sources()

    def _show_sources(self) -> None:
        for group, label in self.source_labels.items():
            source = self.sources.get(group)
            label.setText(source_text(source) if source is not None else "")
            label.setToolTip(label.text())

    def _metal_changed(self, _index: int = 0) -> None:
        self.cu_note.setVisible(self.metal.currentData() == "Cu")

    def _values(self) -> dict:
        return {"energy_kind": self.energy_kind.currentData(), "energy": _float(self.energy_value.text()),
                "sdd": _float(self.sdd.text()), "alpha": _float(self.alpha.text()),
                "beam_col": _float(self.beam_col.text()), "beam_row": _float(self.beam_row.text()),
                "roll": _float(self.roll.text()) or 0.0, "metal": self.metal.currentData(),
                "polymer": self.polymer.currentText().strip() or "polymer", "mask": self.mask_picker.path(),
                "window": self.bin_seconds.value(), "rate": _float(self.rate.text()), "t0": _float(self.t0.text())}

    def _request(self) -> AnalyzeRequest:
        v = self._values()
        energy = v["energy"]
        is_energy = v["energy_kind"] != "wavelength"
        root = self.results_picker.path()
        return AnalyzeRequest(
            scan_folder=Path(self.scan_picker.path() or ""), sdd_mm=v["sdd"], alpha_i_deg=v["alpha"],
            metal=v["metal"], energy_ev=energy * 1000. if is_energy and energy is not None else None,
            wavelength_nm=None if is_energy else energy, polymer_type=v["polymer"], beam_col=v["beam_col"],
            beam_row=v["beam_row"], sample_roll_deg=v["roll"], mask_file=Path(v["mask"]) if v["mask"] else None,
            bin_seconds=v["window"], rate_nm_s=v["rate"], t0_s=v["t0"], results_root=Path(root) if root else None)

    def _remember(self, request: AnalyzeRequest) -> None:
        stored = remember(self._memory(), request.scan_folder, self._values())
        stored["last_scan"] = str(request.scan_folder)
        stored["results_root"] = self.results_picker.path()
        stored["advanced_open"] = self.options_section.is_expanded()
        for old in ("python", "toolkit", "device", "energy_kind", *OLD_KEYS):
            stored.pop(old, None)
        self._store(stored)

    def _remember_thickness(self, scan: Optional[Path], rate: Optional[float], t0: Optional[float]) -> None:
        """A new deposition rate and t0 belong to the scan of the result they were applied to (when it is known)."""
        stored = self._memory()
        folders = dict(stored.get("folders") or {})
        if scan and folder_key(scan) in folders:
            folders[folder_key(scan)] = dict(folders[folder_key(scan)], rate=rate, t0=t0)
            stored["folders"] = folders
            self._store(stored)

    def _remember_result(self, folder: Path) -> None:
        stored = self._memory()
        stored["last_result"] = str(folder)
        self._store(stored)

    # -- presets ---------------------------------------------------------------------------------------------
    def _save_preset(self) -> None:
        info = self.scan_info
        if info is None or not info.serial:
            self._notify(tr("Choose a scan folder whose frames name the detector's serial number first: a preset "
                            "is found again by the serial number, the flat field and the date."), error=True)
            return
        from .preset_dialog import ask_preset

        day = QDate.fromString(info.date, "yyyy-MM-dd") if info.date else QDate.currentDate()
        answer = ask_preset(self, serial=info.serial, flat_field=info.flat_field, date=day)
        if answer is None:
            return
        name, date_from, date_to = answer
        stored = add_preset(self._memory(), make_preset(name, info, self._values(), date_from, date_to))
        self._store(stored)
        self._notify(text("Geometry preset {name} saved: scans of this detector from {start} to {end} are filled in "
                          "from it.", name=name, start=date_from, end=date_to))


__all__ = ["INSPECT_KEY", "PREFERENCES_KEY", "SputterGeometryMixin"]
