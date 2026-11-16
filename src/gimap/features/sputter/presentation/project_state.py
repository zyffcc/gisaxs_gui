"""Sputter form and existing result references in a GIMaP project (no detector data copied).

A project keeps the geometry fields with their sources, the scan folder, the metal, the mask, the results folder, the
open state of More options, the bar and the options of Analyze with the result folder shown (``client``) and Static
Frames (``static``). What a project of an older page holds besides (the result of the earlier method, its rate, polymer
and time window) is not read.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from ..application import GROUPS, Source
from .geometry import INSPECT_KEY

FIELDS = ("energy_value", "sdd", "alpha", "beam_col", "beam_row", "roll")


class SputterProjectMixin:
    def project_state(self) -> dict:
        return {
            "fields": {name: getattr(self, name).text() for name in FIELDS},
            "energy_kind": self.energy_kind.currentData(), "metal": self.sample_metal.currentData(),
            "scan_folder": self.scan_picker.path(), "mask": self.mask_picker.path(),
            "results_root": self.results_picker.path(),
            "sources": {key: asdict(value) for key, value in self.sources.items()},
            "advanced_open": self.options_section.is_expanded(),
            "client": self.client_project_state(),  # Analyze: the bar, its options and its result folder
            "static": self.static_project_state(),  # Static Frames: the section's fields
        }

    def _restore_project_form(self, state: dict) -> None:
        for name in FIELDS:
            getattr(self, name).setText(str(state.get("fields", {}).get(name, "")))
        self.energy_kind.setCurrentIndex(max(0, self.energy_kind.findData(state.get("energy_kind"))))
        if self.sample_metal.findData(state.get("metal")) >= 0:
            self.sample_metal.setCurrentIndex(self.sample_metal.findData(state.get("metal")))
        self.mask_picker.set_path(str(state.get("mask") or ""))
        self.results_picker.set_path(str(state.get("results_root") or ""))
        self.sources = {key: Source(**value) for key, value in state.get("sources", {}).items()
                        if key in GROUPS and isinstance(value, dict)}
        self._edited = set(GROUPS)
        self.options_section.set_expanded(bool(state.get("advanced_open")))
        self._show_sources()

    def apply_project_state(self, state: dict) -> list[str]:
        state = deepcopy(state or self._empty_project_state)
        self.tasks.cancel(INSPECT_KEY)
        self._inspecting = False
        self._forget_queued()
        self.scan_info = None
        self._requested_scan = ""
        self._summary = None
        self._pending_project_state = None
        self.scan_picker.set_path(str(state.get("scan_folder") or ""))
        self._restore_project_form(state)
        self._show_scan_summary()
        scan = self.scan_picker.path()
        if scan and Path(scan).is_dir():
            self.set_scan_folder(scan)
            self._pending_project_state = state
            self._edited = set(GROUPS)
        self.apply_static_project_state(state.get("static"))
        return self.apply_client_project_state(state.get("client"))  # after the folder: its values stay
