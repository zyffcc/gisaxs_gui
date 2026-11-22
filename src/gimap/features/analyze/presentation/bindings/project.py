"""The Analyze part of a GIMaP project: the frames listed, the one on screen, the whole set-up
(``settings_record``: mode, geometry, αi, masks, corrections, cuts) and the Series map, to reopen a sample as it
was left — also after a crash, from the autosave.

**The Series map** (``series_map_state``): every row as the map stacks it (x and intensity, its x label, the row's
name with the facts of its frame, the frame it is of) and what it was built from (the curve, the frame range, every
n-th). A map built online while watching, by Build Map or by a Batch Export comes back as it was, without reducing a
frame again; the settings it was built with are those of the project. The arrays are written as base64 of their
float64 bytes (little-endian), compressed with zlib: the values come back exactly. Each row is encoded once and kept
(the work is digested every second or two for unsaved changes); a map of more than ``MOST_KEPT`` values is not kept
(the note says to build it again).
"""

from __future__ import annotations

import base64
import binascii
import zlib
from pathlib import Path

import numpy as np

from src.gimap.app.presentation.i18n import tr, trf

from ...application import RowLabel, settings_from_record, settings_record

MOST_KEPT = 4_000_000
"""Values (the x and intensity of every row) a project keeps of a Series map; a larger one is built again."""
MAP_NOT_KEPT = "The Series map ({rows} frames) was too large to keep in the project: Build Map again."
MAP_UNREADABLE = "The Series map of the project could not be read: Build Map again."
_FACTS = ("name", "source", "frame_index", "frame_count", "summed", "timestamp", "exposure_s", "frame_time_s")


def _encoded(values) -> dict:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return {"shape": list(array.shape), "f8": base64.b64encode(zlib.compress(array.tobytes(), 1)).decode("ascii")}


def _decoded(data: dict) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(str(data["f8"]).encode("ascii")))
    return np.frombuffer(raw, dtype="<f8").reshape([int(size) for size in data["shape"]]).astype(float)


def _row_record(row) -> dict:
    x, intensity, x_label, label, ref = row
    facts = {key: getattr(label, key) for key in _FACTS if isinstance(label, RowLabel)}
    return {"x": _encoded(x), "y": _encoded(intensity), "x_label": str(x_label), "label": str(label), "facts": facts,
            "ref": [str(ref[0]), int(ref[1])]}


def _row_from(record: dict) -> tuple:
    label = RowLabel(str(record.get("label") or ""), **{key: value for key, value in (record.get("facts") or {}).items()
                                                        if key in _FACTS})
    path, index = record["ref"]
    return _decoded(record["x"]), _decoded(record["y"]), str(record.get("x_label") or ""), label, (Path(path), int(index))


class ProjectMixin:
    """Needs ``view_model``, ``file_list``, ``add_paths``, ``clear_files``, ``run_analysis`` and the Series map's
    state (``_series_map``, ``_series_rows``, ``_series_key``, ``_series_built``, ``_show_series_map``)."""

    def project_state(self) -> dict:
        state = self.view_model.state
        out = {
            "files": [str(path) for path in state.files],
            "current": int(self.file_list.currentRow()),
            "frame": int(getattr(state, "frame_index", 0) or 0),
            "import_selections": self.view_model.export_import_selections(),
            "settings": settings_record(self.view_model.current_settings()),
        }
        series = self.series_map_state()
        if series is not None:
            out["series_map"] = series
        return out

    def series_map_state(self):
        """The Series map shown (module docstring), or ``None`` when there is none."""
        rows = list(getattr(self, "_series_rows", ()) or ()) if getattr(self, "_series_map", None) is not None else []
        if len(rows) < 2:
            return None
        if sum(np.size(row[0]) + np.size(row[1]) for row in rows) > MOST_KEPT:
            return {"kept": False, "frames": len(rows)}
        cache, kept = getattr(self, "_series_memory", {}), {}
        records = []
        for row in rows:
            entry = cache.get(id(row))
            if entry is None or entry[0] is not row:
                entry = (row, _row_record(row))
            kept[id(row)] = entry
            records.append(entry[1])
        self._series_memory = kept  # (each row encoded once: the work is digested every second or two)
        built = getattr(self, "_series_built", {}) or {}
        return {"key": str(self._series_key), "title": str(getattr(self, "_series_curve_title", "") or ""),
                "view": str(getattr(self, "_series_view", "both")), "row": int(getattr(self, "_series_row", 0)),
                "first": int(built.get("first") or 1), "last": int(built.get("last") or 0),
                "every": int(built.get("every") or 1), "kind": built.get("kind"), "rows": records}

    def restore_series_map(self, data) -> list[str]:
        """The Series map of a project (``series_map_state``) shown again; returns notes (too large, unreadable)."""
        if not isinstance(data, dict):
            return []
        if data.get("kept") is False:
            return [trf(MAP_NOT_KEPT, rows=int(data.get("frames") or 0))]
        try:
            rows = [_row_from(record) for record in data["rows"]]
            key = str(data["key"])
        except (KeyError, ValueError, TypeError, zlib.error, binascii.Error):
            return [tr(MAP_UNREADABLE)]
        if len(rows) < 2:
            return []
        self._series_rows, self._series_key = rows, key
        self._series_curve_title = str(data.get("title") or "")
        self._series_run = None
        self._series_view = str(data.get("view") or "both")
        self._series_built = dict(self.view_model.series_built_with(), kind=data.get("kind"),
                                  every=int(data.get("every") or 1), first=int(data.get("first") or 1),
                                  last=int(data.get("last") or 0), watch_tail=())
        self._series_row = min(max(0, int(data.get("row") or 0)), len(rows) - 1)
        self._clear_stages()
        self._show_series_map(keep_view=False)
        self._find_stages()
        return []

    def apply_project_state(self, data: dict) -> list[str]:
        """The frames and set-up of a project; returns notes (files or masks no longer there)."""
        notes: list[str] = []
        self.clear_files()
        files = [str(path) for path in data.get("files") or ()]
        present = [path for path in files if Path(path).exists()]
        missing = len(files) - len(present)
        if missing:
            notes.append(tr("{count} frames of the project are no longer there").format(count=missing))
        if data.get("settings"):
            try:
                notes += self.view_model.apply_settings(settings_from_record(data["settings"]))
                self._sync_settings_widgets()
            except (ValueError, KeyError, TypeError) as exc:
                notes.append(tr("the Analyze set-up could not be read: {reason}").format(reason=exc))
        if present:
            self._import_applying = True
            try:
                self.add_paths(present, preserve_order=True)
                notes += self.view_model.apply_import_selections(data.get("import_selections") or {}, sources=files)
            finally:
                self._import_applying = False
            sources = self.view_model.state.files
            current = min(max(0, int(data.get("current", 0) or 0)), len(files) - 1)
            saved_source = files[current]
            found = self.view_model.locate_source_frame(saved_source, int(data.get("frame", 0) or 0))
            if found is None and sources:
                notes.append(trf("The previously displayed source {name} is unavailable; opened the first imported frame of another source.",
                                 name=Path(saved_source).name))
                found = next(((source, indices[0]) for source in sources
                              if len(indices := self.view_model.selected_frame_indices(source))), None)
            if found is not None:
                source, frame = found
                indices = self.view_model.selected_frame_indices(source)
                if len(indices):
                    self.show_frame(sources.index(source), frame if frame in indices else indices[0])
            self.refresh_frame_list()
            notes += self.restore_series_map(data.get("series_map"))
        return notes


__all__ = ["MAP_NOT_KEPT", "MAP_UNREADABLE", "MOST_KEPT", "ProjectMixin"]
