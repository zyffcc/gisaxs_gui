"""Sputter form and existing result references in a GIMaP project (no detector data copied)."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from src.gimap.app.presentation.i18n import tr
from ..application import GROUPS, Source
from .views.sputter_page_view import REPORT_EMPTY
from .geometry import INSPECT_KEY

FIELDS = ("energy_value", "sdd", "alpha", "beam_col", "beam_row", "roll", "rate", "t0", "t_end")


class SputterProjectMixin:
    def project_state(self) -> dict:
        return {
            "fields": {name: getattr(self, name).text() for name in FIELDS},
            "energy_kind": self.energy_kind.currentData(), "metal": self.metal.currentData(),
            "polymer": self.polymer.currentText(), "window": self.bin_seconds.value(),
            "scan_folder": self.scan_picker.path(), "mask": self.mask_picker.path(),
            "results_root": self.results_picker.path(),
            "sources": {key: asdict(value) for key, value in self.sources.items()},
            "result_folder": str(self.result.folder) if self.result is not None else "",
            "variant": self.result.variant if self.result is not None else None,
            "advanced_open": self.options_section.is_expanded(),
            "log_scale": self.log_scale.isChecked(), "trend_details": self.trend_details.isChecked(),
            "client": self.client_project_state(),  # Analyze: the bar, its options and its result folder
            "static": self.static_project_state(),  # Static Frames: the section's fields
        }

    def _restore_project_form(self, state: dict) -> None:
        for name in FIELDS:
            getattr(self, name).setText(str(state.get("fields", {}).get(name, "")))
        for name in ("energy_kind", "metal"):
            box = getattr(self, name)
            box.setCurrentIndex(max(0, box.findData(state.get(name))))
        self.polymer.setCurrentText(str(state.get("polymer") or "polymer"))
        self.bin_seconds.setValue(float(state.get("window") or 1))
        self.mask_picker.set_path(str(state.get("mask") or ""))
        self.results_picker.set_path(str(state.get("results_root") or ""))
        self.sources = {key: Source(**value) for key, value in state.get("sources", {}).items()
                        if key in GROUPS and isinstance(value, dict)}
        self._edited = set(GROUPS)
        self.options_section.set_expanded(bool(state.get("advanced_open")))
        self.log_scale.setChecked(bool(state.get("log_scale")))
        self.trend_details.setChecked(bool(state.get("trend_details")))
        self._show_sources()

    def apply_project_state(self, state: dict) -> list[str]:
        state = deepcopy(state or self._empty_project_state)
        self.tasks.cancel(INSPECT_KEY)
        self._inspecting = False
        self._forget_queued()
        self._stop_fringes()
        self.result = self.fringes = self.scan_info = None
        self._fringes_folder = self._drawn = None
        self._result_owner = self._requested_scan = ""
        self._summary = None
        self._pending_project_state = None
        self.result_label.clear()
        self.log.clear()
        self.table.clearContents()
        self.table.setRowCount(0)
        self.report.setHtml(tr(REPORT_EMPTY))
        self.sketch.set_result(None)
        self.literature.set_result(None)
        self.merge_button.setEnabled(False)
        self._fill_choices([], None)
        self.scan_picker.set_path(str(state.get("scan_folder") or ""))
        self._restore_project_form(state)
        self._draw_trend()
        self._show_scan_summary()
        notes = []
        folder = str(state.get("result_folder") or "")
        if folder and Path(folder).is_dir():
            try:
                self.show_result(self.service.open(Path(folder), state.get("variant")))
                # The form may hold an unapplied rate; preserve it separately from the saved result.
                self._restore_project_form(state)
            except (OSError, ValueError) as exc:
                notes.append(f"Sputter: {exc}")
        scan = self.scan_picker.path()
        if scan and Path(scan).is_dir():
            self.set_scan_folder(scan)
            self._pending_project_state = state
            self._edited = set(GROUPS)
        self.apply_static_project_state(state.get("static"))
        return notes + self.apply_client_project_state(state.get("client"))  # after the folder: its values stay
