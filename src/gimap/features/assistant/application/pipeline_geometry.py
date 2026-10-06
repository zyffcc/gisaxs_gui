"""How the standard procedure finds the frame's geometry: the person's calibrations first, then the search.

The order, first that fits wins: the calibration of an earlier frame of the batch; the calibration given
(``calibration``) and the calibration files the notes name; the detector's instrument profile; the
automatic search around the frame (ready calibration files, then images of a standard GIMaP can fit).
A ready file (.poni, GIMaP calibration) is used as saved when its pixel size and shape fit; an image is
fitted and accepted when the standard's lines land within 0.5 % of their q (``good`` or ``usable``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

from ..domain import READABLE_IMAGE_SUFFIXES, STANDARD_NAMES, assessment, standard_from_name, suffix
from .calibration_tools import file_geometry, file_key

MAX_FILES = 2
MAX_IMAGES = 3
"""Calibration files and standard images the automatic search tries before giving up."""
ACCEPTED = ("good", "usable")
NAMED = "named by the user (--calibration, then the notes): tried before the instrument profile and the search"


class GeometryStepsMixin:
    """Needs ``catalog``, ``options``, ``_run``, ``_call``, ``_decide``, ``_attention`` and the values
    ``_energy``, ``_incidence``, ``_pixel``, ``_pixel_given``, ``_noted_standard``, ``_given_rejected``,
    ``_declined`` (``StandardPipeline``)."""

    def _incidence_missing(self) -> None:
        self._attention(
            "incidence angle αi",
            "Neither the options, the notes nor an instrument profile give αi, so 0° is used. Ring "
            "positions |q| barely change, but qz shifts by about k·sin αi (≈0.04 Å⁻¹ at 0.4° and 12 keV) "
            "and the missing wedge moves.",
            "incidence_deg",
            "Beamtime notes, the logbook or the slides usually state it (typically 0.1–0.5°); otherwise ask.",
        )

    def _geometry(self, status: dict) -> bool:
        """The calibration of an earlier frame of the batch; else the calibrations the person named (option,
        notes), the detector's instrument profile, then the automatic search — the first that fits."""
        options = self.options
        existing = status.get("geometry")
        if options.geometry is not None:
            self.catalog.results.calibrations.append(options.geometry)
            origin = Path(str(options.geometry.get("source_image") or "")).name
            return self._use_calibration(len(self.catalog.results.calibrations) - 1, f"{origin} (reused from the first frame of this batch)")
        named = self._named()
        found = self._first_fit(named)
        if not found and not self._declined and existing and not options.recalibrate:
            found = self._keep_profile(existing)
        if not found and not self._declined:
            found = self._first_fit(self._candidates({file_key(item["path"]) for item in named}))
        if self._declined:
            self._attention(
                "geometry", "The person declined saving the geometry, so nothing is in q.", None,
                "Approve the geometry, or calibrate in Tools ▸ Geometry Calibration.",
            )
            return False
        if found:
            if self._given_rejected:
                self._attention(
                    "calibration", f"The calibration given was not used ({self._given_rejected}); the geometry comes "
                    f"from {self._run.geometry_source} instead.", "calibration",
                    "Check that file (detector, energy, standard), or give the one used at the beamtime.",
                )
            return True
        if self._given_rejected:  # say why the person's own calibration was not used, not only that nothing fit
            self._attention(
                "calibration", f"The calibration given was not used ({self._given_rejected}), and no other "
                "calibration gave a good geometry.", "calibration",
                "Check that file (detector, energy, standard, pixel size), or give the one used at the beamtime.",
            )
        if not any(item["item"].startswith(("calibration", "X-ray energy", "pixel size")) for item in self._run.attention):
            self._attention(
                "calibration",
                "No calibration candidate gave a good geometry (see decisions for each one).",
                "calibration",
                "Name the calibration image or file used at the beamtime; the notes usually say which.",
            )
        return False

    def _first_fit(self, candidates: Iterable[dict]) -> bool:
        """Try ``candidates`` in order until one gives the geometry; stop when the person declines."""
        for candidate in candidates:
            decided = len(self._run.decisions)
            if self._try(candidate):
                return True
            if self._declined:
                return False
            if candidate.get("given"):
                last = self._run.decisions[-1] if len(self._run.decisions) > decided else None
                self._given_rejected = f"{last['decision']}: {last['why']}" if last else "see the other open items"
        return False

    def _named(self) -> list[dict]:
        """The calibration given (``calibration``), then the calibration files the notes name: tried first."""
        given = [self.options.calibration] if self.options.calibration else []
        noted = self.catalog.named_calibration_files(self.options.notes) if self.options.notes else []
        paths = given + [path for path in noted if file_key(path) not in {file_key(item) for item in given}]
        if paths:
            self._decide("calibration candidates", ", ".join(Path(path).name for path in paths), NAMED)
        return [
            {"path": path, "standard_key": standard_from_name(Path(path).name), "given": index < len(given)}
            for index, path in enumerate(paths)
        ]

    def _keep_profile(self, existing: dict) -> bool:
        name = existing.get("instrument_profile") or "saved"
        profile_pixel = (existing.get("pixel_size_um") or [None])[0]
        if self._pixel_given and profile_pixel and abs(profile_pixel - self._pixel) > 0.01 * profile_pixel:
            self._decide(
                "geometry", f"did not keep the instrument profile '{name}'",
                f"made for {profile_pixel:g} µm pixels, the given pixel size is {self._pixel:g} µm",
            )
            return False
        self._run.geometry_source = f"instrument profile '{name}'"
        self._decide(
            "geometry", f"kept the instrument profile '{name}' this detector already has",
            "a saved profile is the person's own calibration (recalibrate to replace it)",
        )
        if self._incidence is not None:
            self._call("set_incidence_angle", {"degrees": self._incidence})
        elif not existing.get("incidence_deg"):
            self._incidence_missing()
        return True

    def _fittable(self, image: dict) -> bool:
        """A standard image GIMaP can fit: named like a supported standard, or named like no standard while the
        notes name the calibrant."""
        return bool(image.get("gimap_can_fit") or (
            self._noted_standard and not image.get("standard_key") and suffix(image["path"]) in READABLE_IMAGE_SUFFIXES
        ))

    def _candidates(self, tried: set) -> Iterator[dict]:
        """The automatic search: ready calibration files, then images of a standard (none tried already)."""
        listing = self._call("find_calibration_files", {"max_results": 12})
        if listing is None:
            self._attention("calibration", "The folders around the frame could not be searched.", "calibration", "Name the calibration file.")
            return
        files = [dict(item, file=True) for item in listing.get("calibration_results", []) if file_key(item["path"]) not in tried]
        images = [item for item in listing.get("standard_images", []) if self._fittable(item) and file_key(item["path"]) not in tried]
        files, images = files[:MAX_FILES], images[:MAX_IMAGES]
        if not files and not images:
            searched = len(listing.get("folders_searched", []))
            self._attention(
                "calibration",
                f"No calibration file and no image of a standard GIMaP can fit was found ({searched} folders searched).",
                "calibration",
                "An image of AgBh, LaB6, CeO2 or a LaB6+CeO2 mixture taken with this detector, or a .poni / "
                "GIMaP calibration file; a log or the beamtime notes usually name it.",
            )
        ranked = ", ".join(Path(item["path"]).name for item in files + images)
        if ranked:
            self._decide("calibration candidates", ranked, str(listing.get("ranking", "ranked by the search")))
        yield from files + images

    def _standard_for(self, candidate: dict, info: dict, name: str) -> str:
        """The standard to fit: the option, else the one calibrant the notes name, else the image's name, else
        compared (also when the notes and the name disagree)."""
        if self.options.standard:
            return self.options.standard
        from_name, noted = candidate.get("standard_key") or info.get("standard"), self._noted_standard
        if noted and from_name and from_name != noted:
            self._decide("calibration standard", f"compared ({name})", f"the notes name {noted}, the file name {from_name}")
            return "compare"
        if noted:
            self._decide("calibration standard", STANDARD_NAMES.get(noted, noted), "from the notes (the only calibrant they name)")
            return noted
        return from_name or "compare"

    def _try(self, candidate: dict) -> bool:
        path, name = candidate["path"], Path(candidate["path"]).name
        info = self._call("inspect_file", {"path": path})
        if info is None:
            self._decide("calibration", f"skipped {name}", "it could not be read")
            return False
        frame = self.catalog.results.status or {}
        if candidate.get("file") or info.get("kind") in ("pyFAI calibration (.poni)", "GIMaP calibration"):
            return self._use_file(path, name, info, frame)
        if info.get("kind") != "detector image":
            self._decide("calibration", f"skipped {name}", f"it is a {info.get('kind')} file, not a calibration")
            return False
        if info.get("same_detector_size_as_frame") is False:
            self._decide("calibration", f"skipped {name}", f"{info.get('shape')} pixels, the frame has {frame.get('shape')}: another detector")
            return False
        energy = info.get("energy_kev") or self._energy
        if not energy:
            self._attention(
                "X-ray energy",
                "Neither the images' headers, the options nor the notes give the energy; calibration needs it.",
                "energy_kev",
                "Beamtime notes or the logbook; P03 GIWAXS is often 11.8 or 12.4 keV but never guess.",
            )
            return False
        standard = self._standard_for(candidate, info, name)
        arguments = {"path": path, "standard": standard, "energy_kev": float(energy)}
        if self._pixel and (self._pixel_given or not (info.get("pixel_size_um") or [None])[0]):
            arguments["pixel_size_um"] = self._pixel  # a pixel size given beats the image header
        fitted = self._call("calibrate_geometry", arguments)
        if fitted is None:
            failure = self._run.steps[-1]["summary"]
            self._decide("calibration", f"{name} ({standard}) failed", failure)
            if "pixel size" in failure.lower():
                self._attention(
                    "pixel size",
                    f"{name} has no pixel size in its header, so it cannot be calibrated.",
                    "pixel_size_um",
                    "The detector's pixel size: Pilatus 172 µm, Eiger 75 µm, Lambda 55 µm; the notes or the "
                    "detector name usually say which.",
                )
            return False
        if standard == "compare":
            verdict = str(fitted.get("verdict", ""))
            best = next((row for row in fitted.get("comparison", []) if row["calibration_index"] == fitted.get("best_calibration_index")), None)
            if best is None or not verdict.startswith(("clear", "probably")) or not str(best.get("assessment", "")).startswith(ACCEPTED):
                self._decide("calibration", f"rejected {name}", verdict or "no standard fitted")
                if verdict.startswith("ambiguous"):
                    self._attention("calibration standard", verdict, "standard", "The notes or the file name usually say which standard it is.")
                return False
            self._decide("calibration standard", str(best.get("standard")), verdict)
            index = int(fitted["best_calibration_index"])
        else:
            quality = str(fitted.get("assessment", ""))
            if not quality.startswith(ACCEPTED):
                self._decide("calibration", f"rejected {name} ({standard})", quality)
                return False
            index = int(fitted["calibration_index"])
        return self._use_calibration(index, f"{name}")

    def _use_calibration(self, index: int, origin: str) -> bool:
        result = self.catalog.results.calibrations[index]
        arguments: dict = {"source": "calibration", "calibration_index": index}
        fit_energy = result.get("energy_kev")
        if self._energy and fit_energy and abs(fit_energy - self._energy) > 2e-3 * self._energy:
            # The calibrant was measured at another energy: its distance and centre hold, the frame's energy is used.
            arguments = {
                "source": "values", "distance_mm": result["distance_mm"],
                "beam_center_x_px": result["beam_center_px"][0], "beam_center_y_px": result["beam_center_px"][1],
                "energy_kev": self._energy, "pixel_size_um": result["pixel_size_um"][0],
                "note": f"{origin} (calibrated at {fit_energy:g} keV; the frame's {self._energy:g} keV used)",
            }
        if self._incidence is not None:
            arguments["incidence_deg"] = self._incidence
        if self._call("use_geometry", arguments) is None:
            self._decide("geometry", f"could not use {origin}", self._run.steps[-1]["summary"])
            return False
        self._run.calibration = result
        self._run.geometry_source = origin
        if result.get("from_file"):
            self._decide("geometry", f"from {origin}", "a saved calibration (not re-checked against a standard image)")
        else:
            self._decide("geometry", f"calibrated from {origin}", assessment(result))
        if self._incidence is None:
            self._incidence_missing()
        return True

    def _use_file(self, path: str, name: str, info: dict, frame: dict) -> bool:
        header = frame.get("header") or {}
        frame_pixel = self._pixel if self._pixel_given else (header.get("pixel_size_um") or [None])[0]
        file_pixel = info.get("pixel_size_x_m")
        file_pixel = file_pixel * 1e6 if file_pixel else (info.get("pixel_size_um") or [None])[0]
        if frame_pixel and file_pixel and abs(file_pixel - frame_pixel) > 0.01 * frame_pixel:
            why = (
                f"made for {file_pixel:g} µm pixels, the given pixel size is {frame_pixel:g} µm" if self._pixel_given
                else f"made for {file_pixel:g} µm pixels, the frame has {frame_pixel:g} µm: another detector"
            )
            self._decide("calibration", f"skipped {name}", why)
            return False
        if info.get("shape") and frame.get("shape") and list(info["shape"]) != list(frame["shape"]):
            self._decide("calibration", f"skipped {name}", f"made for {info['shape']} pixels, the frame has {frame['shape']}")
            return False
        energy = self._energy or info.get("energy_kev")
        if not energy:
            self._attention("X-ray energy", f"{name} and the frame give no energy.", "energy_kev", "Beamtime notes or the logbook.")
            return False
        if not file_pixel and self._pixel_given:
            file_pixel = self._pixel
        values = file_geometry(path, info, file_pixel)
        arguments = {"source": "file", "path": path, "energy_kev": float(energy)}
        file_energy = info.get("energy_kev")
        if self._energy and file_energy and abs(file_energy - self._energy) > 2e-3 * self._energy:
            # Measured at another energy: the distance and centre hold, the frame's energy is used.
            arguments = {
                "source": "values", "distance_mm": values["distance_mm"],
                "beam_center_x_px": values["beam_center_px"][0], "beam_center_y_px": values["beam_center_px"][1],
                "energy_kev": self._energy, "note": f"{name} (made at {file_energy:g} keV; the frame's {self._energy:g} keV used)",
            }
        if file_pixel and (arguments["source"] == "values" or self._pixel_given):
            arguments["pixel_size_um"] = float(file_pixel)
        if self._incidence is not None:
            arguments["incidence_deg"] = self._incidence
        if self._call("use_geometry", arguments) is None:
            self._decide("geometry", f"could not use {name}", self._run.steps[-1]["summary"])
            return False
        self._run.geometry_source = name
        self._run.calibration = values
        self._decide("geometry", f"from {name}", str(info.get("assessment") or info.get("kind")))
        if self._incidence is None:
            self._incidence_missing()
        return True


__all__ = ["ACCEPTED", "GeometryStepsMixin", "MAX_FILES", "MAX_IMAGES"]
