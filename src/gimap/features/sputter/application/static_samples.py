"""The samples of After Deposition: folders of frames found under a folder the user adds, read by their names.

**What is a sample.** A folder with CBF frames anywhere below it (files named GIWAXS left out, as the package leaves
them out) or one CBF file. A folder whose subfolders hold frames, and at least one of them is named with a thickness,
is a folder *of* samples: each subfolder is one sample however deep its frames lie (measurement folders, ``run1`` …).
A folder with a single subfolder that tells nothing (an archive unpacked into a folder of the same name) is looked
into, at most three levels deep.

**Read from the name.** The nominal thickness is a number followed by ``nm`` or by a metal symbol (``10Au``, ``6.5nm``,
``6p5_Au``); ``H`` or ``HiPIMS`` right before it marks a HiPIMS deposition, ``DC`` a DC one. What stands before it is
the substrate prefix (``850`` of ``850_H10Au``). A folder beside the samples named just as such a prefix, without a
thickness (``850``), is the bare substrate of those samples; samples without one are taken as the metal directly on
Si. Samples of one prefix and one deposition method are a *series*, analysed into a folder of their own. A sample
whose name gives no thickness, or a metal other than Au, is listed but left out until the user says otherwise.
Nothing is guessed: what the name does not say stays empty.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Optional

LOOK_DEEPER = 3
FRAMES = ".cbf"
LEFT_OUT = "giwaxs"
STATIC_METALS = ("Au",)
"""Metals the package's static analysis knows."""
HIPIMS, DC = "HiPIMS", "DC"
_THICKNESS = re.compile(
    r"(?:^|(?<=[^A-Za-z0-9]))(?P<method>hipims|h|dc)?[-_ ]?(?P<number>\d+(?:[.,p]\d+)?)"
    r"(?:\s*(?P<unit>nm)(?![A-Za-z]))?[-_ ]?(?P<metal>Au|Ag|Cu|Pt|Pd|Ti|Al|Ni)?(?![A-Za-z0-9])", re.IGNORECASE)

NO_SAMPLE_FRAMES = "No CBF frames in this folder"
NO_THICKNESS = "The name gives no thickness: enter it"
OTHER_METAL = "Only Au films can be analysed after their deposition"
SAMPLE_NOTES = (NO_SAMPLE_FRAMES, NO_THICKNESS, OTHER_METAL)
"""Why a sample is left out (said in the interface language)."""


@dataclass(frozen=True)
class NameParts:
    nominal_nm: Optional[float] = None
    method: str = ""
    metal: str = ""
    prefix: str = ""


@dataclass(frozen=True)
class StaticSample:
    """One sample of the list: where its frames are, what it is called and what the analysis needs to know."""

    source: str
    """A folder of frames or one CBF file."""
    name: str
    nominal_nm: Optional[float] = None
    method: str = ""
    prefix: str = ""
    reference: str = ""
    """The folder of frames of the bare substrate ("": the metal directly on Si)."""
    frames: int = 0
    included: bool = True
    note: str = ""
    """Why the sample cannot be analysed as it is ("" when it can)."""

    @property
    def series(self) -> str:
        """The series the sample is analysed into: its prefix and its deposition method."""
        parts = [part for part in (self.prefix or self.name, self.method) if part]
        return " · ".join(parts)

    def ready(self) -> bool:
        return self.included and not self.note and self.nominal_nm is not None and self.nominal_nm > 0

    def to_dict(self) -> dict:
        return {"source": self.source, "name": self.name, "nominal_nm": self.nominal_nm, "method": self.method,
                "prefix": self.prefix, "reference": self.reference, "frames": self.frames,
                "included": self.included, "note": self.note}

    @classmethod
    def from_dict(cls, values) -> Optional["StaticSample"]:
        if not isinstance(values, dict) or not values.get("source"):
            return None
        nominal = values.get("nominal_nm")
        return cls(source=str(values["source"]), name=str(values.get("name") or Path(str(values["source"])).stem),
                   nominal_nm=float(nominal) if isinstance(nominal, (int, float)) and nominal > 0 else None,
                   method=str(values.get("method") or ""), prefix=str(values.get("prefix") or ""),
                   reference=str(values.get("reference") or ""), frames=int(values.get("frames") or 0),
                   included=bool(values.get("included", True)), note=str(values.get("note") or ""))


def name_parts(name: str) -> NameParts:
    """What a folder name says: the nominal thickness, the deposition method, the metal and the prefix before them."""
    for match in _THICKNESS.finditer(name):
        if not (match.group("unit") or match.group("metal")):
            continue
        number = float(match.group("number").replace(",", ".").replace("p", ".").replace("P", "."))
        method = (match.group("method") or "").lower()
        metal = (match.group("metal") or "").capitalize()
        prefix = re.sub(r"[-_ ]+$", "", name[:match.start()]).strip()
        return NameParts(number if number > 0 else None, HIPIMS if method in ("h", "hipims") else
                         DC if method == "dc" else "", metal, prefix)
    return NameParts()


def series_slug(series: str) -> str:
    """The folder name of a series."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", series).strip("_")[:60] or "series"


def _frames_below(folder: Path) -> int:
    try:
        return sum(1 for path in folder.rglob("*")
                   if path.suffix.lower() == FRAMES and LEFT_OUT not in path.name.lower())
    except OSError:
        return 0


def _folders(folder: Path) -> list:
    try:
        return sorted((path for path in folder.iterdir() if path.is_dir()), key=lambda path: path.name.lower())
    except OSError:
        return []


def _holder(folder: Path) -> Path:
    """The folder that holds the samples: ``folder``, or the folder inside it when it holds just one that tells
    nothing (``850_samples/850_samples``)."""
    for _ in range(LOOK_DEEPER):
        inner = _folders(folder)
        if (len(inner) != 1 or name_parts(folder.name).nominal_nm is not None
                or name_parts(inner[0].name).nominal_nm is not None or _frames_here(folder)):
            return folder
        folder = inner[0]
    return folder


def _frames_here(folder: Path) -> bool:
    try:
        return any(path.is_file() and path.suffix.lower() == FRAMES for path in folder.iterdir())
    except OSError:
        return False


def _sample(path: Path, frames: int) -> StaticSample:
    parts = name_parts(path.stem if path.is_file() else path.name)
    note = NO_SAMPLE_FRAMES if frames == 0 else NO_THICKNESS if parts.nominal_nm is None else \
        OTHER_METAL if parts.metal and parts.metal not in STATIC_METALS else ""
    return StaticSample(source=str(path), name=path.stem if path.is_file() else path.name,
                        nominal_nm=parts.nominal_nm, method=parts.method, prefix=parts.prefix, frames=frames,
                        included=not note, note=note)


def find_samples(path) -> list:
    """The samples ``path`` holds (one folder of samples, one sample folder or one CBF file), with their bare
    substrates and their series (``complete``)."""
    path = Path(path)
    if path.is_file():
        return [_sample(path, 1 if path.suffix.lower() == FRAMES else 0)] if path.suffix.lower() == FRAMES else []
    if not path.is_dir():
        return []
    holder = _holder(path)
    inner = [(folder, _frames_below(folder)) for folder in _folders(holder)]
    inner = [(folder, count) for folder, count in inner if count]
    if any(name_parts(folder.name).nominal_nm is not None for folder, _count in inner):
        found = [_sample(folder, count) for folder, count in inner]
        bare = {sample.name.lower(): sample.source for sample in found if sample.note == NO_THICKNESS}
        found = [sample for sample in found if not (sample.note == NO_THICKNESS and _is_bare(sample, found))]
        return complete(found, bare)
    sample = _sample(holder, _frames_below(holder))
    siblings = {folder.name.lower(): str(folder) for folder in _folders(holder.parent) if folder != holder}
    return complete([sample], siblings)


def _is_bare(candidate: StaticSample, found: Iterable[StaticSample]) -> bool:
    """A folder without a thickness named as the prefix of other samples is their bare substrate."""
    return any(sample.prefix.lower() == candidate.name.lower() for sample in found if sample is not candidate)


def complete(samples: list, bare: dict) -> list:
    """Each sample with the bare substrate of its prefix (``bare``: lower-case folder name → folder) unless it has
    one, and with DC named where samples of its prefix were also made by HiPIMS."""
    hipims = {sample.prefix.lower() for sample in samples if sample.method == HIPIMS}
    out = []
    for sample in samples:
        method = sample.method or (DC if sample.prefix.lower() in hipims and sample.nominal_nm is not None else "")
        reference = sample.reference or (bare.get(sample.prefix.lower(), "") if sample.prefix else "")
        out.append(replace(sample, method=method, reference=reference))
    return out


def merged(listed: list, added: list) -> tuple:
    """(the list with ``added`` after it, how many of ``added`` were in it already: the same frames)."""
    known = {_key(sample.source) for sample in listed}
    new = [sample for sample in added if _key(sample.source) not in known]
    return listed + new, len(added) - len(new)


def _key(source: str) -> str:
    try:
        return str(Path(source).resolve()).casefold()
    except OSError:
        return str(source).casefold()


def series_of(samples: list) -> list:
    """The series in the order their first sample is listed — those whose samples are all left out last —, each with
    its samples by nominal thickness."""
    order: dict = {}
    for sample in samples:
        order.setdefault(sample.series, []).append(sample)
    named = [(name, sorted(group, key=lambda sample: (sample.nominal_nm is None, sample.nominal_nm or 0.0,
                                                       sample.name.lower())))
             for name, group in order.items()]
    return sorted(named, key=lambda item: all(sample.note for sample in item[1]))


__all__ = ["DC", "HIPIMS", "NO_SAMPLE_FRAMES", "NO_THICKNESS", "NameParts", "OTHER_METAL", "SAMPLE_NOTES",
           "STATIC_METALS", "StaticSample", "complete", "find_samples", "merged", "name_parts", "series_of",
           "series_slug"]
