"""NXcanSAS files of reduced 1D curves (HDF5): the one writer and reader of GIMaP.

Analyze, the command line and Fitting write them; Fitting reads them, also those of other programs (SasView,
Mantid, DAWN …). The layout follows the NeXus application definition NXcanSAS, version 1.1
(https://manual.nexusformat.org/classes/applications/NXcanSAS.html), with one SASentry per curve: SasView
makes one data set of each SASentry and names it by the entry's title, so every curve keeps its own title.

::

    /                       @NX_class=NXroot @default=sasentry01 @file_name @file_time @creator
                            @HDF5_Version @h5py_version
      sasentry01            @NX_class=NXentry @canSAS_class=SASentry @version="1.1" @default=sasdata01
                            @gimap_role="data" | "model" (when the curve says which: Fitting's fits)
        definition          "NXcanSAS"
        title               what the curve is, and which q its Q is (a horizontal cut: q_y, signed)
        run                 the frame the curve is of
        sasdata01           @NX_class=NXdata @canSAS_class=SASdata @signal=I @I_axes=Q @Q_indices=[0]
                            @mask=mask
          Q                 @units=1/angstrom (Analyze) or 1/nm (Fitting)
          I                 @units=arbitrary @uncertainties=Idev
          Idev              @units=arbitrary: the σ of each point (written when the curve has one)
          mask              uint8, 1 = a point without a finite Q or I (0 everywhere else)
        sasinstrument       @NX_class=NXinstrument @canSAS_class=SASinstrument  (when anything is known)
          sassource         @NX_class=NXsource @canSAS_class=SASsource: probe "x-ray",
                            incident_wavelength @units=angstrom
          sasdetector       @NX_class=NXdetector @canSAS_class=SASdetector: name, SDD @units=m,
                            x_pixel_size / y_pixel_size @units=m
        sassample           @NX_class=NXsample @canSAS_class=SASsample: name; for grazing incidence pitch
                            @units=degree = the incidence angle αi (NeXus axes: z along the beam, y up, x
                            horizontal; the sample surface horizontal) and details that say so
        sasprocess01        @NX_class=NXprocess @canSAS_class=SASprocess: name, date, description
          gimap_record      @NX_class=NXnote: type "application/json", description, data = the JSON record

Text is written as fixed-length ASCII strings where it is ASCII (every reader takes those), else as UTF-8.
The reader takes what other programs write too: entries by ``canSAS_class`` (or an ``NXentry`` whose
``definition`` is NXcanSAS), the signal, Q and uncertainty datasets by the attributes that name them, 1D data
only; units as written (``q_unit`` names the usual spellings).

``gimap_role`` (an attribute NeXus allows beside those of the definition; other programs ignore it) tells a measured
curve (``data``) from a model curve (``model``) in a file that holds both: Fitting offers the measured curves when
such a file is opened again (``SasCurve.role``; ``ROLES``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .version import SOFTWARE

SUFFIXES = (".h5",)
"""What GIMaP names an NXcanSAS file (SasView's reader lists ``.h5``)."""
READ_SUFFIXES = (".h5", ".hdf5", ".hdf", ".nxs")
"""What the reader takes: other programs name the same files these ways."""
PER_ANGSTROM, PER_NM, PER_M = "1/angstrom", "1/nm", "1/m"
"""The units of Q the definition allows."""
ARBITRARY = "arbitrary"
VERSION = "1.1"
ROLES = ("data", "model")
"""What a curve is (``SasCurve.role``): measured points, or a model at their Q."""
NOT_NXCANSAS = "{name} holds no NXcanSAS curve (no SASentry with a 1D SASdata group)."
NO_SUCH_CURVE = "{name} holds {count} NXcanSAS curves; there is no curve {number}."


@dataclass(frozen=True)
class Instrument:
    """What is known of the measurement (``None``: not known, not written)."""

    wavelength_angstrom: Optional[float] = None
    distance_m: Optional[float] = None
    detector_name: str = ""
    pixel_size_m: Optional[tuple[float, float]] = None
    incidence_deg: Optional[float] = None
    """The grazing incidence angle αi (SASsample/pitch); ``None`` or 0: transmission."""

    @property
    def known(self) -> bool:
        return any(value is not None for value in (self.wavelength_angstrom, self.distance_m, self.pixel_size_m))


@dataclass(frozen=True)
class SasCurve:
    """One curve of an NXcanSAS file: Q, I and the σ of I (``sigma``: Idev), with what describes it."""

    title: str
    q: np.ndarray
    intensity: np.ndarray
    sigma: Optional[np.ndarray] = None
    q_unit: str = PER_ANGSTROM
    intensity_unit: str = ARBITRARY
    run: str = ""
    sample: str = ""
    instrument: Instrument = field(default_factory=Instrument)
    process: str = ""
    description: str = ""
    record: str = ""
    """The JSON record of the program that wrote the curve (SASprocess / NXnote ``data``)."""
    date: str = ""
    mask: Optional[np.ndarray] = None
    """Read: True where the file's mask leaves a point out (``None``: the file has no mask)."""
    group: str = ""
    """Read: the path of the SASdata group in the file."""
    role: str = ""
    """``data`` or ``model`` (``ROLES``; the SASentry's ``gimap_role``); empty: not said."""


def q_unit(units: Any) -> Optional[str]:
    """``"angstrom"``, ``"nm"`` or ``"m"`` for a unit of Q as files spell it (``1/angstrom``, ``1/A``, ``A^-1``,
    ``1/nm``, ``nm^-1``, ``1/m`` …); ``None`` for anything else."""
    text = _text(units).strip().replace(" ", "").replace("Å", "A").replace("⁻¹", "^-1").lower()
    for name, spellings in (("angstrom", ("a", "ang", "angstrom", "angstroms")), ("nm", ("nm",)), ("m", ("m",))):
        for word in spellings:
            if text in (f"1/{word}", f"{word}^-1", f"{word}-1", f"{word}**-1", f"{word}^(-1)"):
                return name
    return None


# -- writing ---------------------------------------------------------------------------------


def write_nxcansas(path: str | Path, curves: Sequence[SasCurve], *, creator: str = SOFTWARE) -> Path:
    """Every curve as its own SASentry of a new file at ``path`` (written under another name and renamed when
    complete: a program watching the folder never reads half a file). ``ValueError`` without a curve."""
    import h5py

    if not curves:
        raise ValueError("No curve to write.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    unfinished = path.with_name(f".{path.name}.part")
    width = max(2, len(str(len(curves))))
    names = [f"sasentry{number:0{width}d}" for number in range(1, len(curves) + 1)]
    try:
        with h5py.File(unfinished, "w") as root:
            _attributes(root, NX_class="NXroot", default=names[0], file_name=path.name,
                        file_time=datetime.now(timezone.utc).isoformat(timespec="seconds"), creator=creator,
                        HDF5_Version=h5py.version.hdf5_version, h5py_version=h5py.version.version)
            for name, curve in zip(names, curves):
                _write_entry(root.create_group(name), curve)
        os.replace(unfinished, path)
    finally:
        if unfinished.exists():
            unfinished.unlink()
    return path


def _write_entry(entry, curve: SasCurve) -> None:
    _attributes(entry, NX_class="NXentry", canSAS_class="SASentry", version=VERSION, default="sasdata01")
    if curve.role:
        if curve.role not in ROLES:
            raise ValueError(f"{curve.title}: a curve is data or model, not {curve.role!r}.")
        _attributes(entry, gimap_role=curve.role)
    _field(entry, "definition", "NXcanSAS")
    _field(entry, "title", curve.title)
    _field(entry, "run", curve.run or curve.title)
    data = entry.create_group("sasdata01")
    _attributes(data, NX_class="NXdata", canSAS_class="SASdata", signal="I", I_axes="Q", mask="mask")
    data.attrs["Q_indices"] = np.array([0], dtype=np.int32)
    q = np.asarray(curve.q, dtype=np.float64).ravel()
    intensity = np.asarray(curve.intensity, dtype=np.float64).ravel()
    if q.size != intensity.size:
        raise ValueError(f"{curve.title}: Q and I differ in length ({q.size}, {intensity.size}).")
    _attributes(data.create_dataset("Q", data=q), units=curve.q_unit)
    signal = data.create_dataset("I", data=intensity)
    _attributes(signal, units=curve.intensity_unit)
    if curve.sigma is not None:
        sigma = np.asarray(curve.sigma, dtype=np.float64).ravel()
        if sigma.size != q.size:
            raise ValueError(f"{curve.title}: Idev and Q differ in length ({sigma.size}, {q.size}).")
        _attributes(signal, uncertainties="Idev")
        _attributes(data.create_dataset("Idev", data=sigma), units=curve.intensity_unit)
    data.create_dataset("mask", data=(~(np.isfinite(q) & np.isfinite(intensity))).astype(np.uint8))
    _write_instrument(entry, curve.instrument)
    _write_sample(entry, curve)
    if curve.process or curve.description or curve.record:
        process = entry.create_group("sasprocess01")
        _attributes(process, NX_class="NXprocess", canSAS_class="SASprocess")
        _field(process, "name", curve.process or "GIMaP")
        _field(process, "date", curve.date or datetime.now(timezone.utc).isoformat(timespec="seconds"))
        if curve.description:
            _field(process, "description", curve.description)
        if curve.record:
            note = process.create_group("gimap_record")
            _attributes(note, NX_class="NXnote")
            _field(note, "type", "application/json")
            _field(note, "description", "The record of the program that wrote this curve: settings, geometry, "
                                        "corrections and every input file with its SHA-256")
            _field(note, "data", curve.record)


def _write_instrument(entry, instrument: Instrument) -> None:
    if not instrument.known:
        return
    group = entry.create_group("sasinstrument")
    _attributes(group, NX_class="NXinstrument", canSAS_class="SASinstrument")
    if instrument.wavelength_angstrom is not None:
        source = group.create_group("sassource")
        _attributes(source, NX_class="NXsource", canSAS_class="SASsource")
        _field(source, "probe", "x-ray")
        _number(source, "incident_wavelength", instrument.wavelength_angstrom, "angstrom")
    if instrument.distance_m is not None or instrument.pixel_size_m is not None:
        detector = group.create_group("sasdetector")
        _attributes(detector, NX_class="NXdetector", canSAS_class="SASdetector")
        _field(detector, "name", instrument.detector_name or "unknown")
        if instrument.distance_m is not None:
            _number(detector, "SDD", instrument.distance_m, "m")
        if instrument.pixel_size_m is not None:
            _number(detector, "x_pixel_size", instrument.pixel_size_m[0], "m")
            _number(detector, "y_pixel_size", instrument.pixel_size_m[1], "m")


def _write_sample(entry, curve: SasCurve) -> None:
    incidence = curve.instrument.incidence_deg
    grazing = incidence is not None and incidence > 0
    if not (curve.sample or grazing):
        return
    sample = entry.create_group("sassample")
    _attributes(sample, NX_class="NXsample", canSAS_class="SASsample")
    _field(sample, "name", curve.sample or curve.run or "sample")
    if grazing:
        _number(sample, "pitch", incidence, "degree")
        _field(sample, "details", "Grazing incidence: pitch is the incidence angle alpha_i between the beam and the "
                                  "sample surface (NeXus axes: z along the beam, y up; the surface horizontal).")


def _plain(value: str):
    """Fixed-length ASCII where the text is ASCII (every HDF5 reader takes it), else a UTF-8 string."""
    text = str(value)
    try:
        return np.bytes_(text.encode("ascii"))
    except UnicodeEncodeError:
        return text


def _attributes(node, **values) -> None:
    for key, value in values.items():
        node.attrs[key] = _plain(value)


def _field(group, name: str, value: str) -> None:
    group.create_dataset(name, data=_plain(value))


def _number(group, name: str, value: float, units: str) -> None:
    _attributes(group.create_dataset(name, data=np.float64(value)), units=units)


# -- reading ---------------------------------------------------------------------------------


def read_nxcansas(path: str | Path) -> list[SasCurve]:
    """Every 1D curve of the file, in the order of its entries and data groups (by name). ``OSError`` for a
    file that is not HDF5; an empty list for an HDF5 file without NXcanSAS curves."""
    import h5py

    curves: list[SasCurve] = []
    with h5py.File(Path(path), "r") as root:
        for entry in _children(root, h5py.Group):
            if not _is_entry(entry):
                continue
            for data in _children(entry, h5py.Group):
                if _class(data) == "SASdata" or (not _class(data) and _text(data.attrs.get("NX_class")) == "NXdata"):
                    curve = _read_data(entry, data)
                    if curve is not None:
                        curves.append(curve)
    return curves


def read_curve(path: str | Path, index: int = 0) -> SasCurve:
    """The curve ``index`` (from 0) of the file; ``ValueError`` when it has no such curve."""
    curves = read_nxcansas(path)
    name = Path(path).name
    if not curves:
        raise ValueError(NOT_NXCANSAS.format(name=name))
    if not 0 <= int(index) < len(curves):
        raise ValueError(NO_SUCH_CURVE.format(name=name, count=len(curves), number=int(index) + 1))
    return curves[int(index)]


def is_nxcansas(path: str | Path) -> bool:
    """The file is HDF5 and holds at least one NXcanSAS entry (never raises)."""
    try:
        import h5py

        with h5py.File(Path(path), "r") as root:
            return any(_is_entry(entry) for entry in _children(root, h5py.Group))
    except (OSError, ValueError, KeyError, RuntimeError):
        return False


def _children(group, kind) -> list:
    out = []
    for name in sorted(group.keys()):
        try:
            item = group[name]
        except (KeyError, OSError):  # a broken external or soft link
            continue
        if isinstance(item, kind):
            out.append(item)
    return out


def _class(node) -> str:
    return _text(node.attrs.get("canSAS_class"))


def _is_entry(group) -> bool:
    if _class(group) == "SASentry":
        return True
    definition = group.get("definition")
    return _text(group.attrs.get("NX_class")) == "NXentry" and definition is not None and \
        _text(_value(definition)) == "NXcanSAS"


def _read_data(entry, data) -> Optional[SasCurve]:
    import h5py

    signal = _text(data.attrs.get("signal")) or "I"
    axes = _names(data.attrs.get("I_axes")) or ["Q"]
    indices = np.atleast_1d(data.attrs.get("Q_indices", [0])).astype(int).tolist()
    q_name = axes[indices[0]] if indices and 0 <= indices[0] < len(axes) else axes[0]
    intensity, q = data.get(signal), data.get(q_name)
    if not isinstance(intensity, h5py.Dataset) or not isinstance(q, h5py.Dataset):
        return None
    if intensity.ndim != 1 or q.ndim != 1 or intensity.shape != q.shape:
        return None  # 2D data, or several frames: not a curve
    uncertainty = (_names(intensity.attrs.get("uncertainties")) or _names(intensity.attrs.get("uncertainty"))
                   or (["Idev"] if "Idev" in data else []))
    sigma = data.get(uncertainty[0]) if uncertainty else None
    sigma = np.asarray(sigma[()], dtype=float) if isinstance(sigma, h5py.Dataset) and sigma.shape == q.shape else None
    mask_name = _text(data.attrs.get("mask"))
    mask = data.get(mask_name) if mask_name else None
    mask = np.asarray(mask[()]).astype(bool) if isinstance(mask, h5py.Dataset) and mask.shape == q.shape else None
    process = _first(entry, "SASprocess")
    note = None
    if process is not None:
        note = next((item for item in _children(process, h5py.Group)
                     if _text(item.attrs.get("NX_class")) == "NXnote" and "data" in item), None)
    return SasCurve(
        title=_field_text(entry, "title"), q=np.asarray(q[()], dtype=float), intensity=np.asarray(intensity[()], dtype=float),
        sigma=sigma, q_unit=_units(q), intensity_unit=_units(intensity), run=_field_text(entry, "run"),
        sample=_field_text(_first(entry, "SASsample"), "name"), instrument=_read_instrument(entry),
        process=_field_text(process, "name"), description=_field_text(process, "description"),
        record=_field_text(note, "data"), date=_field_text(process, "date"), mask=mask, group=data.name,
        role=_text(entry.attrs.get("gimap_role")) if _text(entry.attrs.get("gimap_role")) in ROLES else "",
    )


def _read_instrument(entry) -> Instrument:
    instrument = _first(entry, "SASinstrument")
    source = _first(instrument, "SASsource") if instrument is not None else None
    detector = _first(instrument, "SASdetector") if instrument is not None else None
    sample = _first(entry, "SASsample")
    wavelength = _length(source, "incident_wavelength", {"angstrom": 1.0, "a": 1.0, "nm": 10.0, "m": 1e10})
    distance = _length(detector, "SDD", {"m": 1.0, "mm": 1e-3, "cm": 1e-2})
    pitch = _length(sample, "pitch", {"degree": 1.0, "degrees": 1.0, "deg": 1.0})
    sizes = tuple(_length(detector, name, {"m": 1.0, "mm": 1e-3, "um": 1e-6}) for name in ("x_pixel_size", "y_pixel_size"))
    return Instrument(wavelength_angstrom=wavelength, distance_m=distance, detector_name=_field_text(detector, "name"),
                      pixel_size_m=None if None in sizes else sizes, incidence_deg=pitch)


def _first(group, canSAS_class: str):  # noqa: N803 - the name in the definition
    if group is None:
        return None
    import h5py

    return next((item for item in _children(group, h5py.Group) if _class(item) == canSAS_class), None)


def _length(group, name: str, scales: dict) -> Optional[float]:
    if group is None or name not in group:
        return None
    try:
        value = float(np.asarray(group[name][()]).ravel()[0])
    except (TypeError, ValueError, IndexError):
        return None
    scale = scales.get(_text(group[name].attrs.get("units")).strip().lower())
    return None if scale is None else value * scale


def _units(dataset) -> str:
    return _text(dataset.attrs.get("units", dataset.attrs.get("unit", "")))


def _field_text(group, name: str) -> str:
    if group is None or name not in group:
        return ""
    return _text(_value(group[name]))


def _value(dataset):
    try:
        return dataset[()]
    except (TypeError, ValueError, OSError):
        return ""


def _names(value) -> list[str]:
    """An attribute that names datasets: one string, a comma-separated string or an array of strings."""
    if value is None:
        return []
    if isinstance(value, np.ndarray) and value.ndim > 0:
        names = [_text(item) for item in value.ravel()]
    else:
        names = _text(value).split(",")
    return [name.strip() for name in names if name.strip()]


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return ",".join(_text(item) for item in value.ravel())
        value = value.ravel()[0]
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8", errors="replace").rstrip("\x00")
    return str(value)


__all__ = ["ARBITRARY", "Instrument", "NOT_NXCANSAS", "NO_SUCH_CURVE", "PER_ANGSTROM", "PER_M", "PER_NM",
           "READ_SUFFIXES", "SUFFIXES", "SasCurve", "VERSION", "is_nxcansas", "q_unit", "read_curve",
           "read_nxcansas", "write_nxcansas"]
