"""A fixed, ordered selection of table columns, separate from folder discovery and Watch.

Sources keep their physical file and explicit column mapping. No files are split or rewritten;
the reader receives each source's token. Independent samples never acquire time continuity.
"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QSignalBlocker

from src.gimap.app.presentation.i18n import tr, trf

from ...application.curve_sources import ColumnMapping, source_record, validate_columns
from ...application.models import InSituSourceFrame

IMPORTED = "{count} imported curves"
INDEPENDENT = "Independent samples: each curve starts from the same model; stages and continuity are off."
ORDERED = "Ordered sequence: curves follow the selected file and column order."
FIXED_SELECTION = "Imported curves are a fixed selection. Import again to change files or columns; Watch is available for folders."
IMPORT_SETTING = "{count} curves: q in {unit}, no unit in their header (chosen at import)"


def source_state(source):
    """JSON-safe source identity; column indices use the reader's zero-based convention."""
    return source_record(source.token)


def source_from_state(data):
    if not isinstance(data, dict) or not isinstance(data.get("path"), str) or not data["path"]:
        raise ValueError("An imported curve has no source file.")
    path = Path(data["path"])
    physical = InSituSourceFrame.from_token(str(path))
    if physical.path != path or physical.columns is not None or physical.frame_index:
        raise ValueError("An imported curve must name a physical file; keep its column mapping separately.")
    columns = data.get("columns")
    return InSituSourceFrame(path, frame_index=data.get("frame_index", 0),
                             source_kind=data.get("source_kind", "curve"),
                             columns=None if columns is None else ColumnMapping.from_dict(columns))


def inspect_imported_sources(sources, inspect, stop):
    """Read each physical table once; a saved missing/changed column remains an explicit error."""
    tables, errors = {}, {}
    for source in sources:
        if stop.is_set():
            break
        if source.path not in tables:
            try:
                tables[source.path] = inspect(source.path)
            except (OSError, ValueError, RuntimeError) as exc:
                tables[source.path] = exc
        table = tables[source.path]
        try:
            if isinstance(table, Exception):
                raise table
            if source.columns is not None:
                validate_columns(table, source.columns)
        except (OSError, ValueError, RuntimeError) as exc:
            errors[source.token] = str(exc)
    return errors


class FitSeriesImportsMixin:
    _imported_sources = None
    _imported_independent = False
    _imported_q_unit = None
    _folder_watch_tip = None
    _folder_choices = None

    def imported_mode(self):
        return self._imported_sources is not None

    def independent_samples(self):
        return self.imported_mode() and self._imported_independent

    def source_unit_setting(self):
        return self._imported_q_unit or self.single.unit_setting()

    def open_imported_sources(self, sources, *, independent: bool, q_unit=None) -> bool:
        """Accept a fixed selection; True means background listing/previewing was started."""
        return self._open_imported_sources(sources, independent=independent, q_unit=q_unit, ask=True)

    def _open_imported_sources(self, sources, *, independent, q_unit, ask):
        if self.running:
            self._status(tr("Stop the running series first."), "warning")
            return False
        sources = tuple(sources)
        if not sources or any(not isinstance(source, InSituSourceFrame) for source in sources):
            self._status(tr("Choose at least one curve to import."), "warning")
            return False
        if q_unit not in (None, "angstrom", "nm"):
            self._status(tr("Choose a valid q unit for the imported curves."), "warning")
            return False
        if len({source.token for source in sources}) != len(sources):
            self._status(tr("The same source column was selected more than once."), "warning")
            return False
        if ask and not self._may_drop_results():
            return False
        if not self.imported_mode():
            self._folder_choices = (self._start_choice(), self.series_method(), self.follow_check.isChecked(),
                                    self.skip_odd_check.isChecked())
        self._restore_listing = None
        self._imported_sources = sources
        self._imported_independent = bool(independent)
        self._imported_q_unit = q_unit
        self.folder = str(sources[0].path.parent)  # default save location, never the selection's identity
        self.watch_check.setChecked(False)
        self._watch.stop()
        self._forget_halves()
        self._set_start("same" if independent else "previous")
        self.follow_check.setChecked(not independent)
        self.skip_odd_check.setChecked(not independent)
        self.every_spin.setValue(1)
        self._show_folder()
        return self._begin_listing(again=True)

    def _clear_imported_sources(self):
        self._imported_sources = None
        self._imported_independent = False
        self._imported_q_unit = None
        if self._folder_choices is not None:
            start, method, follow, skip_odd = self._folder_choices
            self._folder_choices = None
            blockers = [QSignalBlocker(control) for control in
                        (self.start_group, self.method_group, self.follow_check, self.skip_odd_check)]
            self._set_start(start)
            self.set_series_method(method)
            self.follow_check.setChecked(follow)
            self.skip_odd_check.setChecked(skip_odd)
            del blockers

    def _refresh_imported_controls(self):
        imported, independent = self.imported_mode(), self.independent_samples()
        if self._folder_watch_tip is None:
            self._folder_watch_tip = self.watch_check.toolTip()
        for control in (self.pattern_edit, self.subfolders_check, self.watch_check):
            control.setEnabled(not imported)
        self.watch_check.setToolTip(tr(FIXED_SELECTION) if imported else tr(self._folder_watch_tip))
        self.start_previous.setEnabled(not independent)
        self.start_stages.setEnabled(not independent)
        self.follow_check.setEnabled(self.method_fit1d.isChecked() and not independent)
        self.skip_odd_check.setEnabled(not independent)
        self.import_button.setEnabled(not self.running)
        if imported:
            self._show_imported_sources()

    def _show_imported_sources(self):
        self.folder_chip.setText(trf(IMPORTED, count=len(self._imported_sources)))
        meaning = INDEPENDENT if self.independent_samples() else ORDERED
        self.folder_chip.setToolTip(tr(meaning) + "\n" + tr(FIXED_SELECTION))

    def imported_state(self):
        if not self.imported_mode():
            return {}
        return {"source_mode": "selection", "sources": [source_state(source) for source in self._imported_sources],
                "independent": self._imported_independent, "q_unit_of_files": self._imported_q_unit,
                "skip_odd": self.skip_odd_check.isChecked()}

    def _restore_imported_sources(self, data):
        try:
            sources = tuple(source_from_state(item) for item in data.get("sources", ()))
        except (TypeError, ValueError, KeyError) as exc:
            self.close_series()
            return [trf("The imported curves could not be restored: {reason}", reason=tr(str(exc)))]
        if self._open_imported_sources(sources, independent=bool(data.get("independent", True)),
                                       q_unit=data.get("q_unit_of_files"), ask=False):
            self._restore_listing = dict(data)
            self.set_series_method(data.get("method"))
            if not self.independent_samples():
                self._set_start(data.get("start"))
                self.follow_check.setChecked(bool(data.get("follow", True)))
                self.skip_odd_check.setChecked(bool(data.get("skip_odd", True)))
            return []
        self.close_series()
        return [tr("The imported curves could not be restored.")]
