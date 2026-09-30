"""Tools for a frame without geometry: find, read and use calibration material.

Reading is read-only and limited to detector images, calibrations and text or
log files, never in hidden or system folders.  Readable without asking: the
open frame, everything the searches found, the tree up to two folders above
the frame, and files or folders the person named in the notes.  Other files
are read after the person agrees (confirm mode) or with a note (automatic
mode).  A geometry is only saved through ``use_geometry``, which asks first in
the confirm mode.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from ..domain import (
    GIMAP_CALIBRATION,
    LOG,
    POINT_OF_NORMAL_INCIDENCE,
    READABLE_IMAGE_SUFFIXES,
    READABLE_SUFFIXES,
    STANDARD_IMAGE,
    STANDARD_NAMES,
    SUPPORTED_STANDARDS,
    assessment,
    classify,
    compare_verdict,
    energy_kev,
    line_quality,
    module_series,
    name_preference,
    parse_poni,
    path_candidates,
    private_path,
    rms_px,
    significant,
    standard_from_name,
    suffix,
    text_hints,
    wavelength_angstrom,
)
from .models import PERMISSION_CONFIRM, ToolInputError, ToolOutcome

IMAGE_ENDINGS = (".cbf", ".tif", ".tiff", ".nxs", ".h5", ".hdf5", ".edf")


def file_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def _local_time(seconds: Optional[float]) -> Optional[str]:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(seconds)) if seconds else None


OPEN_LEVELS = 2
"""Folders above the frame whose whole tree may be read without asking (usually the beamtime folder)."""


class CalibrationToolsMixin:
    """Tool handlers mixed into ``ToolCatalog`` (which provides the ports and results)."""

    def _setup_calibration(self) -> None:
        self._known_files: set[str] = set()
        self._file_geometries: dict[str, dict] = {}
        self._open_roots: list[str] = []
        self._user_paths: list[dict] = []
        self._access_ready = False

    def _frame(self) -> dict:
        status = self.results.status or self.workbench.status()
        return status

    # -- access ------------------------------------------------------------------------

    def _prepare_access(self) -> None:
        """Files and folders the person named in the notes become readable."""
        if self._access_ready or self.explorer is None:
            return
        self._access_ready = True
        for text in (self.goals.instructions, getattr(self.goals, "standing_instructions", "")):
            for options in path_candidates(text or ""):
                for candidate in options:
                    kind = self.explorer.kind(candidate)
                    if kind is not None:
                        self._grant(candidate, kind)  # the person named it: trusted
                        break

    def _grant(self, path: str, kind: str) -> None:
        if kind == "file":
            self._known_files.add(file_key(path))
        else:
            self._open_roots.append(file_key(path))
        self._user_paths.append({"path": str(path), "kind": kind})

    def access_notes(self) -> list[str]:
        """Lines for the task message: what the person named, now readable."""
        self._prepare_access()
        return [f"{item['kind']} named by the user, readable and searchable: {item['path']}" for item in self._user_paths]

    def _open_folders(self) -> list[str]:
        roots = list(self._open_roots)
        frame = self._frame().get("path")
        if frame:
            base = Path(frame).resolve().parent
            for _ in range(OPEN_LEVELS):
                base = base.parent
            roots.append(file_key(base))
        return roots

    def _inside(self, key: str, roots: list[str]) -> bool:
        return self._root_of(key, roots) is not None

    def _base_for(self, key: str) -> Optional[str]:
        """The open folder holding ``key``, else the folder it shares with the frame."""
        root = self._root_of(key, self._open_folders())
        if root is not None:
            return root
        frame = self._frame().get("path")
        if not frame:
            return None
        try:
            return os.path.commonpath([file_key(Path(frame).resolve().parent), key])
        except ValueError:  # another drive
            return None

    @staticmethod
    def _root_of(key: str, roots: list[str]) -> Optional[str]:
        for root in roots:
            if key == root or key.startswith(root.rstrip("\\/") + os.sep):
                return root
        return None

    def _approve(self, what: str, path: str) -> None:
        if self.goals.permission != PERMISSION_CONFIRM:
            return
        question = f"{what} {path}? It is away from the folders around the frame."
        if self.confirmer is None or not self.confirmer.confirm("Claude wants to look at your files", question):
            raise ToolInputError("The user did not allow this; ask for another file or folder, or for the values.")

    def _allow(self, path: str) -> str:
        if self.explorer is None:
            raise ToolInputError("Reading files is not available in this session.")
        self._prepare_access()
        key = file_key(path)
        frame = self._frame().get("path")
        if key in self._known_files or (frame and key == file_key(frame)):
            return key
        if self.explorer.kind(path) != "file":
            raise ToolInputError(f"There is no file at {path}. search_files on its folder finds similar names.")
        root = self._root_of(key, self._open_folders())
        if private_path(key, self._base_for(key)):
            raise ToolInputError("Files in hidden or system folders (.ssh, AppData, Windows …) are never read.")
        if suffix(path) not in READABLE_SUFFIXES:
            raise ToolInputError(f"Only detector images, .poni, .json and text or log files are read, not {suffix(path) or 'this file'}.")
        if root is None:
            self._approve("Read", path)
        self._known_files.add(key)
        return key

    def _allow_folder(self, folder: str) -> str:
        if self.explorer is None:
            raise ToolInputError("Searching files is not available in this session.")
        self._prepare_access()
        if self.explorer.kind(folder) != "folder":
            raise ToolInputError(f"There is no folder at {folder}.")
        key = file_key(folder)
        frame = self._frame().get("path")
        above = bool(frame) and self._inside(file_key(Path(frame).resolve().parent), [key])
        root = self._root_of(key, self._open_folders())
        if not above and private_path(key, self._base_for(key)):
            raise ToolInputError("Hidden or system folders (.ssh, AppData, Windows …) are never searched.")
        if not above and root is None:
            self._approve("Search", folder)
        return key

    # -- tools -----------------------------------------------------------------------

    def _tool_find_calibration_files(self, max_results: int = 30) -> ToolOutcome:
        if self.explorer is None:
            raise ToolInputError("Searching files is not available in this session.")
        self._prepare_access()
        frame = self._frame()
        if not frame.get("path"):
            raise ToolInputError("No frame is open in Analyze.")
        listing = self.explorer.related_files(frame["path"])
        frame_time = (listing.get("frame") or {}).get("mtime")
        payload, summary = self._describe_listing(listing, frame_time, max_results)
        payload = {"frame": {"path": frame["path"], "modified": _local_time(frame_time), "header": frame.get("header")}, **payload}
        if self._user_paths:
            payload["named_by_user"] = [item["path"] for item in self._user_paths]
        return self._ok(payload, summary)

    def _tool_search_files(self, folder: str, name_contains=None, max_depth=None) -> ToolOutcome:
        self._allow_folder(folder)
        depth = 4 if max_depth is None else int(max_depth)
        listing = self.explorer.search(folder, name_contains, depth)
        frame = self._frame()
        frame_time = None
        if frame.get("path"):
            try:
                frame_time = os.path.getmtime(frame["path"])
            except OSError:
                frame_time = None
        if name_contains:
            matches = []
            for entry in listing.get("entries", ())[: 200]:
                self._known_files.add(file_key(entry["path"]))
                matches.append(self._record(entry, frame_time))
            payload = {"matches": matches[:80], "folders_searched": listing.get("folders", []), "search_truncated": bool(listing.get("truncated"))}
            return self._ok(payload, f"{len(matches)} file(s) named like '{name_contains}' in {Path(folder).name or folder}")
        payload, summary = self._describe_listing(listing, frame_time, 40)
        return self._ok(payload, f"in {Path(folder).name or folder}: {summary}")

    def _record(self, entry: dict, frame_time: Optional[float]) -> dict:
        return {
            "path": entry["path"],
            "folder": entry["relation"],
            "modified": _local_time(entry.get("mtime")),
            "hours_from_frame": significant((entry["mtime"] - frame_time) / 3600.0, 3) if frame_time and entry.get("mtime") else None,
            "size_kb": round(entry.get("size", 0) / 1024.0, 1),
        }

    def _frame_kind(self) -> tuple[str, bool]:
        """The frame's file type and whether it is one module of a NeXus series."""
        name = Path(str(self._frame().get("path") or "")).name
        return suffix(name), module_series(name) is not None

    def _describe_listing(self, listing: dict, frame_time: Optional[float], max_results: int) -> tuple[dict, str]:
        frame_path = self._frame().get("path")
        frame_suffix, frame_series = self._frame_kind()
        groups: dict[str, list] = {POINT_OF_NORMAL_INCIDENCE: [], GIMAP_CALIBRATION: [], STANDARD_IMAGE: [], LOG: []}
        series_seen: dict[str, dict] = {}
        for entry in listing.get("entries", ()):
            kind = classify(entry["relative"])
            if kind is None or (frame_path and file_key(entry["path"]) == file_key(frame_path)):
                continue
            series = module_series(Path(entry["path"]).name)
            if series is not None:
                # One stitched image: list the series once, by its first module file.
                key = file_key(str(Path(entry["path"]).parent / series[0]))
                self._known_files.add(file_key(entry["path"]))
                if key in series_seen:
                    series_seen[key]["modules"] += 1
                    if series[1] < series_seen[key]["_first"]:
                        series_seen[key].update(path=entry["path"], _first=series[1])
                    continue
            record = self._record(entry, frame_time)
            if series is not None:
                record.update(modules=1, _first=series[1])
                series_seen[file_key(str(Path(entry["path"]).parent / series[0]))] = record
            if kind == STANDARD_IMAGE:
                record["same_kind_as_frame"] = bool(
                    suffix(entry["path"]) == frame_suffix and (series is not None) == frame_series
                )
            if kind in (STANDARD_IMAGE, POINT_OF_NORMAL_INCIDENCE, GIMAP_CALIBRATION):
                standard = standard_from_name(entry["relative"])
                record["standard"] = STANDARD_NAMES.get(standard) if standard else None
                record["standard_key"] = standard
            if kind == STANDARD_IMAGE:
                record["gimap_can_fit"] = bool(standard in SUPPORTED_STANDARDS and suffix(entry["path"]) in READABLE_IMAGE_SUFFIXES)
            groups[kind].append(record)
            self._known_files.add(file_key(entry["path"]))

        for record in series_seen.values():
            record.pop("_first", None)

        def closeness(record: dict) -> tuple:
            hours = record.get("hours_from_frame")
            if hours is None:
                return (1e9,)
            return (abs(hours) * (1.0 if hours <= 0 else 3.0),)  # measured before the frame is preferred

        def rank(record: dict) -> tuple:
            name = record["path"].lower()
            return (
                not record.get("gimap_can_fit"),
                not record.get("same_kind_as_frame"),
                not any(word in name for word in ("giwaxs", "waxs")),
                -name_preference(Path(record["path"]).parent.name + "/" + Path(record["path"]).name),
                *closeness(record),
            )

        limit = max(5, min(int(max_results), 80))
        results = sorted(groups[POINT_OF_NORMAL_INCIDENCE] + groups[GIMAP_CALIBRATION], key=closeness)
        images = sorted(groups[STANDARD_IMAGE], key=rank)
        logs = sorted(groups[LOG], key=closeness)
        payload = {
            "calibration_results": results[:limit],
            "standard_images": images[:limit],
            "ranking": (
                "standard images are ranked: GIMaP can fit the standard, same detector kind as the frame "
                "(file type and module series), 'giwaxs'/'waxs' in the name, 'final'/'redone' names, then "
                "closeness in time (before the frame preferred); a multi-module NeXus series is listed once "
                "(modules = number of files)"
            ),
            "logs": logs[:limit],
            "folders_searched": listing.get("folders", [])[:30],
            "search_truncated": bool(listing.get("truncated")),
        }
        if not (results or images or logs):
            payload["note"] = (
                "Nothing named like calibration material or a log was found. search_files can look further "
                "up or down, or for a name (e.g. 'lab6', 'calib', '.poni')."
            )
        summary = f"{len(results)} calibration file(s), {len(images)} standard image(s), {len(logs)} log(s)"
        return payload, summary

    def _tool_inspect_file(self, path: str) -> ToolOutcome:
        if self.explorer is None:
            raise ToolInputError("Reading files is not available in this session.")
        key = self._allow(path)
        ending = suffix(path)
        name = Path(path).name
        if ending == ".poni":
            geometry = parse_poni(self.explorer.read_text(path))
            values = {
                "distance_mm": geometry.distance_mm,
                "beam_center_x_px": geometry.beam_center_x_px,
                "beam_center_y_px": geometry.beam_center_y_px,
                "wavelength_angstrom": geometry.wavelength_angstrom,
                "pixel_size_x_m": geometry.pixel_size_x_m,
                "pixel_size_y_m": geometry.pixel_size_y_m,
            }
            self._file_geometries[key] = values
            payload = {
                "kind": "pyFAI calibration (.poni)", **values,
                "energy_kev": energy_kev(geometry.wavelength_angstrom),
                "detector": geometry.detector, "tilt_deg": geometry.tilt_deg, "notes": list(geometry.notes),
            }
            return self._ok(payload, f"{name}: .poni, distance {geometry.distance_mm:.1f} mm")
        if ending == ".json":
            text = self.explorer.read_text(path, 2_000_000)
            try:
                data = json.loads(text)
            except ValueError:
                data = None
            if isinstance(data, dict) and data.get("format") == "gimap-geometry-calibration" and self.calibrator is not None:
                result = self.calibrator.read_result(path)
                self._file_geometries[key] = {
                    "distance_mm": result["distance_mm"],
                    "beam_center_x_px": result["beam_center_px"][0],
                    "beam_center_y_px": result["beam_center_px"][1],
                    "wavelength_angstrom": result["wavelength_angstrom"],
                    "pixel_size_x_m": result["pixel_size_um"][0] * 1e-6,
                    "pixel_size_y_m": result["pixel_size_um"][1] * 1e-6,
                }
                return self._ok({"kind": "GIMaP calibration", **result, "assessment": assessment(result)}, f"{name}: GIMaP calibration")
            keys = sorted(data)[:40] if isinstance(data, dict) else None
            return self._ok({"kind": "json", "keys": keys, "lines": text_hints(text)}, f"{name}: json")
        if ending in IMAGE_ENDINGS:
            header = self.explorer.image_header(path)
            frame_shape = (self._frame().get("shape") or None)
            header["same_detector_size_as_frame"] = bool(frame_shape and list(header.get("shape") or []) == list(frame_shape))
            header["standard"] = standard_from_name(path)
            return self._ok({"kind": "detector image", **header}, f"{name}: {header.get('shape')}")
        text = self.explorer.read_text(path)
        lines = text_hints(text)
        payload = {"kind": "text", "lines": lines}
        if len(lines) < 5:
            payload["first_lines"] = [line[:200] for line in text.splitlines()[:12]]
        return self._ok(payload, f"{name}: {len(lines)} relevant line(s)")

    def _tool_calibrate_geometry(
        self, path: str, standard: str, energy_kev: float, distance_mm=None, pixel_size_um=None,
    ) -> ToolOutcome:
        if self.calibrator is None:
            raise ToolInputError("Geometry calibration is not available in this session.")
        self._allow(path)
        if suffix(path) not in READABLE_IMAGE_SUFFIXES:
            raise ToolInputError(f"GIMaP reads {', '.join(READABLE_IMAGE_SUFFIXES)} images, not {suffix(path) or 'this file'}.")
        if not 1.0 <= float(energy_kev) <= 200.0:
            raise ToolInputError("energy_kev must be between 1 and 200 keV.")
        hint = float(distance_mm) if distance_mm else None
        options = {
            "energy_kev": float(energy_kev),
            "distance_mm": hint,
            "pixel_size_m": float(pixel_size_um) * 1e-6 if pixel_size_um else None,
            "cancelled": self.cancelled,
        }
        if standard != "compare":
            result = self.calibrator.calibrate(path, standard=standard, **options)
            self.results.calibrations.append(result)
            payload = {**result, "calibration_index": len(self.results.calibrations) - 1, "assessment": assessment(result)}
            summary = (
                f"{result.get('standard')}: {result.get('matched_rings')} rings, rms {result.get('rms_residual_px', 0):.2g} px, "
                f"{result.get('distance_mm', 0):.1f} mm"
            )
            return self._ok(payload, summary)
        fits, failures = [], []
        for key in SUPPORTED_STANDARDS:
            if self.cancelled():
                break
            try:
                result = self.calibrator.calibrate(path, standard=key, **options)
            except Exception as exc:  # a standard that does not fit at all is one answer of the comparison
                failures.append({"standard": key, "error": str(exc) or type(exc).__name__})
                continue
            self.results.calibrations.append(result)
            fits.append({**result, "calibration_index": len(self.results.calibrations) - 1})
        if not fits:
            raise ToolInputError("No standard could be fitted: " + "; ".join(f"{item['standard']}: {item['error']}" for item in failures))
        fits.sort(key=lambda item: (
            line_quality(item) if line_quality(item) is not None else 1.0,
            -(item.get("matched_rings") or 0),
            rms_px(item),
        ))
        verdict = compare_verdict(fits, hint)
        rows = [
            {
                "calibration_index": item["calibration_index"], "standard": item.get("standard"),
                "matched_rings": item.get("matched_rings"), "rms_residual_px": item.get("rms_residual_px"),
                "confidence": item.get("confidence"), "distance_mm": item.get("distance_mm"),
                "beam_center_px": item.get("beam_center_px"), "assessment": assessment(item),
                "line_q_error_percent": None if line_quality(item) is None else round(100 * line_quality(item), 3),
            }
            for item in fits
        ]
        payload = {"comparison": rows, "failed": failures, "best_calibration_index": fits[0]["calibration_index"], "verdict": verdict}
        return self._ok(payload, f"compared {len(fits)} standard(s): {verdict}"[:160])

    def _tool_ask_user(self, question: str, options: list, allow_text: bool = False) -> ToolOutcome:
        if self.chooser is None:
            return self._ok({"answered": False, "message": "Nobody can be asked in this session."}, "no one to ask")
        answer = self.chooser.choose(question, list(options), bool(allow_text))
        if not answer:
            return self._ok({"answered": False, "message": "The person closed the question without an answer."}, "not answered")
        index = answer.get("index")
        chosen = options[index] if isinstance(index, int) and 0 <= index < len(options) else None
        text = answer.get("text", "")
        if self.explorer is not None:  # a path typed in the answer becomes readable like one in the notes
            for candidates in path_candidates(text):
                for candidate in candidates:
                    kind = self.explorer.kind(candidate)
                    if kind:
                        self._grant(candidate, kind)
                        break
        payload = {"answered": True, "option": chosen, "index": index if chosen else None, "text": text}
        return self._ok(payload, f"answer: {chosen['label'] if chosen else answer.get('text', '')}"[:120])

    def _geometry_values(self, arguments: dict) -> tuple[dict, str]:
        source = arguments["source"]
        if source == "calibration":
            if not self.results.calibrations:
                raise ToolInputError("There is no calibration yet: run calibrate_geometry first.")
            index = arguments.get("calibration_index")
            index = len(self.results.calibrations) - 1 if index is None else int(index)
            if not 0 <= index < len(self.results.calibrations):
                raise ToolInputError(f"calibration_index must be 0…{len(self.results.calibrations) - 1}.")
            result = self.results.calibrations[index]
            values = {
                "distance_mm": result["distance_mm"],
                "beam_center_x_px": result["beam_center_px"][0],
                "beam_center_y_px": result["beam_center_px"][1],
                "wavelength_angstrom": result["wavelength_angstrom"],
                "pixel_size_x_m": result["pixel_size_um"][0] * 1e-6,
                "pixel_size_y_m": result["pixel_size_um"][1] * 1e-6,
            }
            origin = (
                f"the saved calibration {Path(result.get('source_image', '')).name}" if result.get("from_file") else
                f"{result.get('standard')} calibration of {Path(result.get('source_image', '')).name} "
                f"({result.get('matched_rings')} rings, rms {result.get('rms_residual_px', 0):.2g} px)"
            )
        elif source == "file":
            path = arguments.get("path")
            if not path:
                raise ToolInputError("path is needed for source 'file'.")
            values = self._file_geometries.get(file_key(path))
            if values is None:
                raise ToolInputError("Inspect the .poni or calibration file with inspect_file first.")
            values = dict(values)
            origin = Path(path).name
        else:
            missing = [name for name in ("distance_mm", "beam_center_x_px", "beam_center_y_px") if arguments.get(name) is None]
            wavelength = arguments.get("wavelength_A") or wavelength_angstrom(arguments.get("energy_kev"))
            if missing or not wavelength:
                raise ToolInputError(f"Values needed: {', '.join(missing + ([] if wavelength else ['energy_kev or wavelength_A']))}.")
            values = {
                "distance_mm": float(arguments["distance_mm"]),
                "beam_center_x_px": float(arguments["beam_center_x_px"]),
                "beam_center_y_px": float(arguments["beam_center_y_px"]),
                "wavelength_angstrom": float(wavelength),
            }
            origin = arguments.get("note") or "values given to the assistant"
        if values.get("wavelength_angstrom") is None:
            wavelength = arguments.get("wavelength_A") or wavelength_angstrom(arguments.get("energy_kev"))
            if not wavelength:
                raise ToolInputError("The file has no wavelength: give energy_kev or wavelength_A.")
            values["wavelength_angstrom"] = float(wavelength)
        if arguments.get("pixel_size_um"):
            values["pixel_size_x_m"] = values["pixel_size_y_m"] = float(arguments["pixel_size_um"]) * 1e-6
        if arguments.get("incidence_deg") is not None:
            values["incidence_deg"] = float(arguments["incidence_deg"])
        if not (values["distance_mm"] > 0 and values["wavelength_angstrom"] > 0):
            raise ToolInputError("The distance and the wavelength must be positive.")
        return values, origin

    def _tool_use_geometry(self, source: str, **arguments) -> ToolOutcome:
        values, origin = self._geometry_values({"source": source, **arguments})
        status = self.workbench.use_geometry(values, arguments.get("profile_name"), f"assistant: {origin}")
        self.results.geometry_used = {**values, "source": origin}
        self.results.status = status
        kind = (status.get("measurement") or "no reduction").upper()
        return self._ok(status, f"geometry from {origin} → {kind}"[:160])

    def _describe_geometry_write(self, arguments: dict) -> str:
        try:
            values, origin = self._geometry_values(arguments)
        except ToolInputError as exc:
            return f"Use a geometry for this detector ({exc})"
        return (
            f"Save the geometry from {origin} as the instrument profile of this detector and re-analyse: "
            f"distance {values['distance_mm']:.1f} mm, beam centre ({values['beam_center_x_px']:.1f}, "
            f"{values['beam_center_y_px']:.1f}) px, λ {values['wavelength_angstrom']:.4f} Å."
        )


__all__ = ["CalibrationToolsMixin", "OPEN_LEVELS", "file_key"]
